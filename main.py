import numpy as np
from collections import deque

class BCEGO_H:
    """
    Hierarchical Batched Coherent Entropy‐Guided Optimizer (BCEGO-H)

    Implements a two‐phase, batch‐based optimizer that uses:
      1. Batched sampling from a Gaussian belief (mu, Sigma).
      2. An optional "refocus" step: if enough samples achieve near‐optimal f ≈ 0,
         the belief is re‐centered and its covariance recomputed on those samples.
      3. Otherwise, a hierarchical clustering:
         a) Macro‐cluster: keep the largest connected component in the full batch
            based on Gaussian similarity thresholded at τ_macro.
         b) Micro‐cluster: within the macro cluster, apply a stricter threshold τ_micro
            to extract the tightest coherent subset.
      4. Fit a weighted Gaussian to the micro‐cluster with weights ∝ exp(η·f)
         and update the belief (mu, Sigma).
    """

    def __init__(self, f, dim, mu0, Sigma0,
                 eta=1.0,
                 batch_size=100,
                 tau_macro=0.5,
                 tau_micro=0.8,
                 refocus_thresh=1e-2,
                 refocus_min_samples=5,
                 random_seed=None):
        """
        Initialize optimizer parameters and belief.

        Parameters
        ----------
        f : callable
            Objective function mapping an (N×dim) array X to an (N,) array of f(x).
        dim : int
            Dimensionality of the search space.
        mu0 : array_like, shape (dim,)
            Initial mean of the Gaussian belief.
        Sigma0 : array_like, shape (dim, dim)
            Initial covariance matrix of the Gaussian belief.
        eta : float, optional
            Scaling factor for fitness when computing weights (default 1.0).
        batch_size : int, optional
            Number of samples drawn each iteration (default 100).
        tau_macro : float, optional
            Fraction of median similarity to threshold for macro‐clustering (default 0.5).
        tau_micro : float, optional
            Fraction of median similarity within macro cluster for micro‐clustering (default 0.8).
        refocus_thresh : float, optional
            Absolute threshold on |f(x)| below which a sample is considered near‐optimal (default 1e-2).
        refocus_min_samples : int, optional
            Minimum number of near‐optimal samples required to trigger refocus (default 5).
        random_seed : int or None, optional
            Seed for reproducible random sampling (default None).
        """
        self.f = f
        self.dim = dim
        self.mu = mu0.copy()            # Current mean of belief
        self.Sigma = Sigma0.copy()      # Current covariance of belief
        self.eta = eta
        self.batch_size = batch_size
        self.tau_macro = tau_macro
        self.tau_micro = tau_micro
        self.refocus_thresh = refocus_thresh
        self.refocus_min_samples = refocus_min_samples

        # Seed numpy's RNG for reproducibility if requested
        if random_seed is not None:
            np.random.seed(random_seed)

    def _cluster_mask(self, X, tau):
        """
        Compute a boolean mask for the largest connected component in X based on similarity.

        Parameters
        ----------
        X : ndarray, shape (N, dim)
            Batch of N samples.
        tau : float
            Fraction of the median Gaussian similarity to use as a threshold.

        Returns
        -------
        mask : ndarray, shape (N,), dtype bool
            True for indices in the largest connected component.
        """
        N = X.shape[0]
        # Compute pairwise squared distances between samples
        diffs = X[:, None, :] - X[None, :, :]
        sq = np.sum(diffs**2, axis=2)

        # Set kernel width h to sqrt(median distance)
        h = np.sqrt(np.median(sq)) + 1e-12

        # Gaussian similarity matrix S_ij = exp(-d^2/(2h^2))
        S = np.exp(-sq / (2*h*h))

        # Threshold similarity at tau * median(S)
        thresh = tau * np.median(S)

        # Build adjacency list for graph connectivity
        adj = [set() for _ in range(N)]
        for i in range(N):
            for j in range(i+1, N):
                if S[i, j] >= thresh:
                    adj[i].add(j)
                    adj[j].add(i)

        # Find largest connected component via breadth-first search
        visited = [False] * N
        best_comp = []
        for i in range(N):
            if not visited[i]:
                comp = []
                queue = deque([i])
                visited[i] = True
                while queue:
                    u = queue.popleft()
                    comp.append(u)
                    for v in adj[u]:
                        if not visited[v]:
                            visited[v] = True
                            queue.append(v)
                # Keep track of the largest component found
                if len(comp) > len(best_comp):
                    best_comp = comp

        # Create boolean mask for the best component
        mask = np.zeros(N, dtype=bool)
        mask[best_comp] = True
        return mask

    def _try_refocus(self, X, fvals):
        """
        Check for near-optimal samples and refocus belief if criteria met.

        Parameters
        ----------
        X : ndarray, shape (N, dim)
            Batch of N samples.
        fvals : ndarray, shape (N,)
            Objective values for each sample.

        Returns
        -------
        did_refocus : bool
            True if the belief was refocused on near-optimal samples.
        best_f_in_refocus : float or None
            Maximum f-value among the refocus samples, if refocused; otherwise None.
        """
        # Identify samples with |f(x)| <= refocus_thresh
        near_zero = np.abs(fvals) <= self.refocus_thresh
        count = np.sum(near_zero)

        # If enough samples are near-optimal, recalibrate belief on that subset
        if count >= self.refocus_min_samples:
            X0 = X[near_zero]
            mu0 = X0.mean(axis=0)
            Sigma0 = np.cov(X0, rowvar=False) + 1e-8 * np.eye(self.dim)
            self.mu, self.Sigma = mu0, Sigma0
            return True, np.max(fvals[near_zero])

        return False, None

    def step(self):
        """
        Perform one optimization step.

        1. Sample batch from current Gaussian belief (mu, Sigma).
        2. Attempt refocus on near-optimal samples.
        3. If no refocus, apply macro- and micro-clustering.
        4. Compute fitness‐weighted update of (mu, Sigma).

        Returns
        -------
        mu : ndarray, shape (dim,)
            Updated mean of the belief.
        detS : float
            Determinant of the updated covariance.
        best_f : float
            Best objective value found in this step (or in refocus).
        refocused : bool
            True if the step was a refocus, False otherwise.
        """
        # 1. Draw batch from N(mu, Sigma)
        X = np.random.multivariate_normal(self.mu, self.Sigma, self.batch_size)
        fvals = self.f(X)

        # 2. Refocus check
        did_refocus, best_refocus = self._try_refocus(X, fvals)
        if did_refocus:
            # Return immediately if refocused
            return self.mu.copy(), np.linalg.det(self.Sigma), best_refocus, True

        # 3. Macro‐cluster: keep largest component in full batch
        mask_macro = self._cluster_mask(X, self.tau_macro)
        Xm, fm = X[mask_macro], fvals[mask_macro]
        if Xm.size == 0:
            # Fallback: use entire batch if clustering fails
            Xm, fm = X, fvals

        # 4. Micro‐cluster: refine within the macro cluster
        mask_micro = self._cluster_mask(Xm, self.tau_micro)
        Xc, fc = Xm[mask_micro], fm[mask_micro]
        if Xc.size == 0:
            # Fallback: use macro cluster if micro clustering fails
            Xc, fc = Xm, fm

        # 5. Compute fitness‐based weights w ∝ exp(eta * f)
        log_w = self.eta * fc
        log_w -= np.max(log_w)   # stabilize by subtracting max
        w = np.exp(log_w)
        w /= np.sum(w)

        # 6. Update belief: weighted mean and covariance of micro‐cluster
        new_mu = np.sum(w[:, None] * Xc, axis=0)
        C = Xc - new_mu
        new_Sigma = (C.T * w) @ C + 1e-8 * np.eye(self.dim)

        self.mu, self.Sigma = new_mu, new_Sigma
        return self.mu.copy(), np.linalg.det(self.Sigma), np.max(fc), False

    def optimize(self, iters=50):
        """
        Run the optimizer for a fixed number of iterations.

        Parameters
        ----------
        iters : int
            Number of optimization steps to perform.

        Returns
        -------
        history : list of tuples
            Each tuple contains (mu, detS, best_f, refocused) for each iteration.
        """
        history = []
        for _ in range(iters):
            mu, detS, best_f, refocused = self.step()
            history.append((mu, detS, best_f, refocused))
        return history
