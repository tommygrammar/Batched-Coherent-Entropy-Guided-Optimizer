# Entropy Guided Optimizer

This repo implements a stochastic optimizer combining Gaussian sampling, hierarchical clustering, and adaptive refocusing. The optimizer is designed for functions with complex landscapes where exploration-exploitation trade-offs are critical — specifically, problems where the search space is large, multimodal, and where a naive random search would waste most of its budget on unpromising regions.

---

## Performance

This implementation has been profiled and optimised. The baseline and optimised measurements are below, collected with `perf stat -r 5` on the same hardware:

| Metric | Baseline | Optimised | Change |
|---|---|---|---|
| Time elapsed | 46.42 s | 2.75 s | **16.88× faster** |
| Cycles | 109,811,112,204 |  6,522,346,731 | -94.06% |
| Instructions | 212,066,397,386 | 13,227,031,578 | -93.76% |
| Cache references | 867,725,950 | 272,639,031 | -68.58% |
| Cache misses | 465,455,453 | 33,138,387 | −92.88% |
| Cache miss rate | 53.64% | 12.15% | −41 pp |
| Branch misses | 276,558,379 | 9,432,838 | -96.58% |
| IPC | 1.93 | 2.03 | +5.18% |

**What drove the improvement:**

The baseline had a 53.64% cache miss rate — the CPU was stalling on memory more than half the time. Flamegraph analysis identified three interacting bottlenecks:

1. **`std::vector<bool>` bit-packing overhead** — cluster masks were stored as `vector<bool>`, which is a specialised bit-packed container in C++. Every access required bit-shift and masking operations through a proxy iterator, generating significant instruction overhead. Replaced with `vector<uint8_t>` — one byte per boolean, direct load, no bit manipulation. This alone removed a large fraction of the branch miss and instruction overhead.

2. **Full O(N²) pairwise distance and similarity computation** — the original clustering computed all N×(N-1)/2 pairwise distances every iteration, then ran `std::nth_element` over all of them to find the median. For batch=1000 that is ~500,000 distances per clustering call, and the clustering runs twice per step (macro then micro). Replaced with approximate median sampling: for large N, a random sample of 2048 pairs is drawn and the median is estimated from that sample. This reduced the data volume feeding `nth_element` by orders of magnitude and eliminated most of the O(N²) similarity matrix computation.

3. **Hot-path vector allocation and page faults** — temporary vectors (`Xm`, `fm`, `Xc`, `fc`, `dist_samples`, `sim_samples`, adjacency lists) were being constructed, filled, and destroyed inside every call to `step()`. Repeated `operator new` calls and first-touch page faults were visible in the flamegraph's kernel frames. These were eliminated by designing the data flow so that intermediate buffers are either reused across calls or sized to the approximate sample budget upfront.

The 92.88 reduction in absolute cache misses is the primary driver of the 16.88× speedup — the CPU spends dramatically less time stalled on memory, which explains both the throughput improvement and the IPC increase.

---

## Core Idea

The optimizer maintains a belief over the search space in the form of a multivariate Gaussian — a mean vector and a covariance matrix. Each iteration it:

1. Draws a batch of samples from the current belief using multivariate normal sampling.
2. Evaluates an objective function on every sample.
3. Checks whether enough samples are in a promising region to trigger a refocus — if so, it immediately narrows the belief toward those samples and skips the clustering step.
4. If no refocus, it finds the most dominant connected region in the sample cloud using macro clustering — a Gaussian kernel similarity graph where the largest connected component is extracted via BFS.
5. Within that macro region, it finds the densest local sub-region using micro clustering — the same process applied to the macro samples only.
6. Computes softmax weights over the micro samples, biased toward higher objective values.
7. Updates the belief mean and covariance using the weighted micro samples.
8. Repeats until iteration budget is exhausted.

The intuition is that the macro cluster collapses the search to the most densely populated promising region, and the micro cluster within it collapses further to the locally strongest sub-region. This two-stage spatial collapse focuses the belief faster than a single clustering step would.

---
## How to Build

This project uses CMake. Build from the repository root:

```bash
git clone https://github.com/tommygrammar/entropy-guided-optimization
cd entropy-guided-optimization
mkdir build
cd build
cmake ..
make
```

The binary will be at `build/optimizer`.

---

## How to Run

```bash
./optimizer
```

This runs the optimizer with the parameters configured in `main()`. Output is one line per iteration:

```
Iter 0 best_f=-45.231 mu=(0.12, -0.34, ...) refocused=no
Iter 1 best_f=-38.107 mu=(0.08, -0.21, ...) refocused=no
...
Iter 23 best_f=-1.204 mu=(0.01, -0.02, ...) refocused=yes
```

---

## Algorithm Design

### Template Parameters

```cpp
template<size_t dim, size_t batch>
class Optimizer
```

The class is generic over `dim` (dimensionality of the input space) and `batch` (number of samples drawn per iteration). Both are `size_t` template parameters so that array sizes are known at compile time. This drives the key performance choice: internal arrays like `mu0`, `sigma0`, and `X_flat` are `std::array` rather than `std::vector` as their sizes are statically determined, they live on the stack, heap allocation overhead is zero, and memory layout is fully contiguous.

---

### Class Members

```cpp
std::array<double, dim>       mu0;     // mean of the sampling distribution
std::array<double, dim*dim>   sigma0;  // covariance matrix, flattened row-major to 1D
double eta;                            // weight scaling factor — controls how strongly the optimizer biases toward high-objective samples
double tau_macro;                      // similarity threshold for macro clustering
double tau_micro;                      // similarity threshold for micro clustering
double refocus_thres;                  // objective value threshold that triggers refocus
int    refocus_min_samples;            // minimum qualifying samples needed to trigger refocus
int    random_seed;                    // RNG seed for reproducibility
double refocus_change_rate;            // how much refocus_thres advances after each refocus
std::mt19937 rng;                      // Mersenne Twister RNG, seeded once in run()
```

`sigma0` is stored as a flat 1D array of size `dim*dim` rather than a 2D array. The navigation rule is `sigma0[r*dim + c]` for row `r`, column `c`. This keeps the entire covariance matrix in a single contiguous memory block which is cache-friendly for the Cholesky decomposition which reads it row by row.

---

### `mu0_generation()` and `sigma_generation()`

```cpp
inline std::array<double, dim>      mu0_generation();
inline std::array<double, dim*dim>  sigma_generation();
```

Both are called once in `run()` before the optimisation loop begins. The rationale for generating them here rather than inside `step()` is that if they were generated per-step, each iteration would start from scratch with a zero mean and identity covariance which would destroy the accumulated belief. They are initialized once, then updated in place by `step()` and `_try_refocus()`.

`mu0_generation` returns a zero-initialized array of size `dim`. This places the initial belief at the origin.

`sigma_generation` returns a diagonal covariance matrix with 2.0 on the diagonal. Originally implemented as 2D, changed to a flat 1D layout for cache efficiency and to avoid the overhead of 2D array navigation. A diagonal initial covariance means the optimizer starts with uncorrelated, moderate-spread sampling which is broad enough to explore but not so broad that samples are immediately outside any useful region.

---


### `multivariate.h` — Sampling Infrastructure

This header provides three functions that form the sampling layer of
the optimizer. Everything that produces the batch of candidate samples
lives here.

---

#### `clamp_normal_value(double z, double limit = 8.0)`

```cpp
inline double clamp_normal_value(double z, double limit = 8.0)
```

**What it does:**
Clamps an input double to the range `[-limit, limit]`. If the value
is NaN or inf it is replaced with 0.0. If it is outside the clamp
range it is hard-limited to the boundary.

**Why it exists:**
Normal distributions have infinite tails. The Mersenne Twister feeding
our sampler can produce extreme values — not frequently, but it can.
When those extreme values reach the Cholesky transformation step they
get multiplied by L matrix entries and can produce catastrophic
overflow. A clamp of 8.0 standard deviations covers essentially the
entire practical range of a normal distribution (probability of
exceeding 8σ is ~6×10⁻¹⁶) while preventing any downstream numerical
damage from tail outliers.

We inline this function because it sits inside the innermost sampling
loop — `batch * dim` calls per step. The function body is small enough
that the inline is legitimate rather than just optimistic.

---

#### `next_standard_normal(rng, has_spare, spare)`

```cpp
inline double next_standard_normal(
    std::mt19937& rng,
    bool& has_spare,
    double& spare
)
```

**What it does:**
Generates one standard normal sample N(0,1) using Box-Muller
transform. Maintains a cached spare so that every two uniform draws
produce two normal values — the second is stored in `spare` and
returned on the next call without any additional RNG work.

**Why this instead of `std::normal_distribution`:**
`std::normal_distribution` carries internal state, branches on whether
it has a cached value, and involves function call overhead per sample.
In the original implementation, `std::normal_distribution<double>` was
being called `batch * dim` times per step inside the sampling loop —
1000 × 10 = 10,000 calls per iteration, 1,000,000 calls over 100
iterations. That overhead is measurable.

Box-Muller with an explicit spare cache removes the distribution
object entirely. The transform itself is two `generate_canonical`
calls, one `sqrt`, one `log`, one `cos`, one `sin` — and every pair
of draws produces two usable normal values. The spare mechanism means
we amortise the transcendental function cost across two samples rather
than paying it once per sample.

**Why `u1 > 0.0` is enforced:**
`log(0)` is undefined (−∞). `generate_canonical` can theoretically
return exactly 0.0 on some RNG states. The `do { } while (u1 <= 0.0)`
loop retries until u1 is strictly positive. In practice this almost
never loops more than once — the probability of generating exactly 0.0
from a 53-bit uniform is 2⁻⁵³ ≈ 10⁻¹⁶.

**The `has_spare` and `spare` parameters are passed by reference**
because the spare cache must persist across calls within the same
batch. They are declared in `multivariate()` and passed down — the
caller owns the state, not the function.

---

#### `safe_cholesky(Sigma, L, initial_jitter, max_attempts)`

```cpp
template<size_t dim>
bool safe_cholesky(
    const std::array<double, dim*dim>& Sigma,
    std::array<double, dim*dim>& L,
    double initial_jitter = 1e-12,
    int max_attempts = 8
)
```

**What it does:**
Computes the lower-triangular Cholesky factor L of a symmetric
positive definite matrix Sigma such that L * L^T = Sigma. Returns
true on success, false if all attempts fail.

**Why standard Cholesky is not enough:**
Covariance matrices degenerate toward singularity as the optimizer
tightens its belief — when the sampled distribution collapses into a
tight region, samples become nearly collinear and the covariance matrix
loses rank. Standard Cholesky on a singular or near-singular matrix
produces a zero or negative diagonal element, which causes a sqrt of a
non-positive number and a failed decomposition. Without a recovery
mechanism, one bad covariance update would terminate the entire
optimization run.

**The jitter mechanism:**
A small positive value is added to the diagonal before attempting
decomposition. This is equivalent to assuming a tiny amount of
independent noise in each dimension — it regularises the matrix enough
to restore positive definiteness. If decomposition fails, jitter grows
by 10× and the attempt repeats, up to `max_attempts` times. Starting
at `1e-12` and growing to at most `1e-12 × 10^7 = 1e-5` before giving
up. This range covers near-singular cases without distorting a
genuinely well-conditioned matrix.

**Implementation decisions:**
- Invalid entries (NaN, inf) in Sigma are detected before any
  computation begins and the function returns false immediately.
  There is no point attempting a decomposition on a matrix containing
  garbage — it will fail in an unpredictable way otherwise.
- L is filled with zeros before writing. Only the lower triangle is
  computed and written. The upper triangle stays zero and is never
  touched, which halves the memory writes in the factorisation loop.
- The `ok` flag breaks the inner loop the moment a failure is detected
  rather than continuing through remaining columns. On a bad matrix
  this saves the remaining column iterations.
- `Sigma` is passed as `const std::array<double, dim*dim>&` — a
  reference, not a value. Avoids copying 100 doubles (800 bytes for
  dim=10) on every call to safe_cholesky.

**When it returns false:**
The caller falls back to diagonal sampling — diagonal elements of
Sigma are extracted, sqrt is taken, and each dimension is sampled
independently. This is a genuine degradation in sampling quality
(independent dimensions, no covariance structure) but it is always
numerically safe and keeps the optimizer running rather than crashing.

---

#### `multivariate(rng, mu0, Sigma)`

```cpp
template<size_t dim, size_t batch>
std::array<double, batch*dim> multivariate(
    std::mt19937& rng,
    const std::array<double, dim>& mu0,
    const std::array<double, dim*dim>& Sigma
)
```

**What it does:**
Produces `batch` samples from the multivariate normal N(mu0, Sigma).
Returns a flat array of size `batch * dim` in row-major order: sample
`b` occupies positions `[b*dim, b*dim+dim)`.

**The full process, step by step:**

**Step 1 — Validate Sigma upfront.**
Before anything else, every element of Sigma is checked for finiteness.
If any element is NaN or inf, sampling is impossible — the Cholesky
factorisation would produce garbage and the resulting samples would be
meaningless. The fallback is to return `batch` copies of `mu0`. This
is a degenerate output but it is always finite and keeps the optimizer
alive for the next iteration.

**Step 2 — Symmetrise Sigma.**
Floating-point arithmetic in the covariance update (weighted outer
products, jitter addition, symmetry enforcement) can leave Sigma
slightly asymmetric — off-diagonal pairs that should be equal may
differ by a small epsilon. `safe_cholesky` requires a symmetric input
to function correctly. We symmetrise explicitly by averaging each
off-diagonal pair: `Sigma_sym[i*dim+j] = 0.5*(Sigma[i*dim+j] +
Sigma[j*dim+i])`. This costs `dim*dim` operations once per step and
eliminates the class of Cholesky failures caused by accumulated
asymmetry.

**Step 3 — Cholesky factorisation.**
`safe_cholesky` is called on the symmetrised matrix. If it returns
false (degenerate covariance), L is populated with the diagonal fallback
— `L[i*dim+i] = sqrt(max(Sigma_sym[i*dim+i], 1e-12))` for each
dimension, zeros elsewhere. This preserves the per-dimension scale
information while discarding the covariance structure.

**Step 4 — Generate samples in a tight sequential loop.**
`has_spare` and `spare` are declared once before the batch loop and
passed into `next_standard_normal` on each call. This means the Box-
Muller spare cache persists across the entire batch — every two
dimensions consume one pair of uniform draws, so for a 10-dimensional
batch of 1000, we make 5000 Box-Muller pairs producing 10,000 normal
values rather than 10,000 separate distribution calls.

The inner loop structure for each sample:
```
for each sample b in [0, batch):
    generate z[0..dim-1] using next_standard_normal
    clamp each z[d] to [-8, 8]
    for each dimension i:
        acc = sum_{j <= i} L[i*dim+j] * z[j]   // lower-triangular transform
        val = mu0[i] + acc
        if val is non-finite: val = mu0[i]      // last-resort fallback
        X_flat[b*dim+i] = val
```

The lower-triangular loop `sum_{j <= i}` uses the structure of L
directly — no branch needed to skip the upper triangle, the loop
simply does not iterate past `j = i`. This is the correct and efficient
way to apply a Cholesky factor.

**Step 5 — Non-finite output guard.**
After computing each sample coordinate, if the result is non-finite it
is replaced with `mu0[i]`. This is the last line of defence against NaN
propagation. The Cholesky fallback and the clamp should prevent this
from triggering in normal operation, but if both fail — for example if
mu0 itself contains a very large value that causes overflow in the
accumulation — this guard catches it. A cerr warning is emitted so
these events are visible during debugging.

**The second overload:**
```cpp
template<size_t dim, size_t batch>
std::array<double, batch*dim> multivariate(
    int seed,
    const std::array<double, dim>& mu0,
    const std::array<double, dim*dim>& Sigma
)
```
Constructs a local `std::mt19937` from the given seed and delegates to
the primary overload. This exists for testing and reproducibility — a
caller that wants a deterministic sample from a known seed can use this
without managing an RNG externally. The primary overload is used by the
optimizer where the RNG is seeded once in `run()` and reused across all
steps.

### `step()`

This is the main optimisation step. It is called once per iteration and returns a `std::pair<bool, double>` where the bool indicates whether a refocus occurred and the double is the best objective value seen in this step.

#### Sampling

```cpp
std::array<double, batch*dim> X_flat = multivariate<dim, batch>(rng, mu0, sigma0);
```

Draws `batch` samples from the current belief. Stored in a flat stack-allocated array — no heap allocation, full cache locality. After sampling, any non-finite values in X_flat are replaced with the corresponding `mu0` component to prevent NaN propagation through the rest of the step.

#### Objective Evaluation

```cpp
std::array<double, batch> fvals = fobj<dim, batch>(X_flat);
```

Evaluates the objective function on all `batch` samples simultaneously. `fobj` is a template function in `fobj.h` — it takes the flat sample array and returns a flat array of function values.

#### Refocus Check

```cpp
auto refocus_result = _try_refocus(X_flat, fvals, refocus_thres, refocus_min_samples);
if (refocus_result.first) { return refocus_result; }
```

Before any clustering, `_try_refocus` scans the function values for samples above `refocus_thres`. If at least `refocus_min_samples` qualify, it computes the mean and covariance of those qualifying samples and commits them directly to `mu0` and `sigma0`, bypassing the clustering step entirely. The rationale: if enough samples are already in a promising region, the clustering machinery is unnecessary — just focus directly on those samples. After each successful refocus, `refocus_thres` advances by `refocus_change_rate` so that subsequent refocuses require progressively better samples.

#### Macro Clustering

`cluster_mask_macro` identifies the largest connected component in the sample cloud under a Gaussian kernel similarity measure.

**Step 1 — Estimate median squared distance.** For batch=1000 the full pairwise set is ~500,000 distances. Rather than computing all of them, a random sample of up to 2048 pairs is drawn and the median is estimated from that sample. For small N where the total pairs are under 2048, all pairs are computed exactly. This is the key optimisation that replaced the original O(N²) computation — the estimated median is close enough to the true median to drive the kernel width correctly.

**Step 2 — Compute kernel width.** `h = sqrt(median_sq) + 1e-12`. The `1e-12` prevents division by zero when all samples are identical. `denom = 2 * h^2` is the Gaussian kernel bandwidth.

**Step 3 — Estimate median similarity.** The same sampled distances are converted to similarities via `exp(-d2 / denom)`. The median of those similarities becomes the connectivity threshold: `thresh = tau * median_S`. Pairs with similarity >= thresh are connected.

**Step 4 — Build adjacency list on demand.** The full N×N similarity matrix is never materialised. Instead, for each pair (i,j), the similarity is computed on demand and if it exceeds the threshold, both directions are added to the adjacency list. This eliminates the 1,000,000-entry similarity matrix from the original implementation.

**Step 5 — BFS for largest connected component.** Standard BFS over the adjacency list, tracking which component is largest. The result is a `vector<uint8_t>` mask of size N — 1 for samples in the largest component, 0 otherwise. `uint8_t` rather than `bool` avoids the `vector<bool>` bit-packing specialisation and its associated iterator overhead.

#### Micro Clustering

`cluster_mask_micro` applies the same process to the macro cluster samples only (`Xm`, the samples that passed the macro mask). This produces a tighter sub-region within the dominant macro region. The sample budget for the approximate median is 1024 rather than 2048 since the input is already a subset.

Fallback chain: if micro clustering produces an empty set, fall back to the full macro cluster. If macro clustering produced an empty set, fall back to the full batch. This ensures `step()` always has samples to work with regardless of clustering outcomes.

#### Weight Computation

Weights are computed over the micro cluster samples using a softmax biased toward higher objective values:

```cpp
log_w[i] = eta * fc[i];   // log-space weight
w[i] = exp(log_w[i] - max_log_w);  // exponentiate with log-sum-exp stability
w[i] /= sum_w;            // normalise
```

The log-sum-exp trick — subtracting `max_log_w` before exponentiating — prevents overflow when `eta * fc[i]` is large. Without it, `exp(eta * fc[i])` can overflow to infinity for large positive function values. `eta` controls how sharply the distribution focuses on the best samples: small eta gives nearly uniform weights, large eta concentrates weight on the single best sample.

If `sum_w` is not finite or not positive (numerical edge case), weights fall back to uniform — every micro sample gets equal weight.

#### Belief Update

The new mean `new_mu` is the weighted sum of micro cluster samples. The new covariance `new_Sigma` is the weighted outer product of deviations from the new mean. After computation:

- Jitter (`1e-8`) is added to the diagonal for numerical stability — prevents the covariance from degenerating to a singular matrix as the belief tightens.
- The matrix is explicitly symmetrised by averaging off-diagonal pairs — floating-point arithmetic can make it slightly asymmetric.
- Finite check: if any element of `new_Sigma` is non-finite, the update is skipped and the current `mu0`/`sigma0` are preserved.

If all checks pass, `mu0` and `sigma0` are updated in place.

---

### `_try_refocus(X_flat, fvals, refocus_threshold, refocus_min_samples)`

```cpp
std::pair<bool, double> _try_refocus(
    const std::array<double, batch*dim>& X_flat,
    const std::array<double, batch>& fvals,
    double& refocus_threshold,
    int& refocus_min_samples
)
```

The refocus mechanism handles the case where clustering might miss a genuinely good region because the overall sample distribution is still broad. It scans all samples for those with `fvals[i] >= refocus_threshold` and if at least `refocus_min_samples` qualify, it computes their mean and covariance directly.

The covariance computation uses the same outer-product formula as the main belief update, with a diagonal regularisation of `1e-8`. For the degenerate case of exactly one qualifying sample, the covariance is set to `1e-8 * I` — a very tight identity matrix that forces the next iteration to sample tightly around that single point.

After a successful refocus, `refocus_threshold` advances by `refocus_change_rate` and `refocus_min_samples` decrements by 1 (minimum 1). This adaptive adjustment means each subsequent refocus requires a progressively better region — the optimizer has to keep improving to keep triggering refocuses.

Parameters `refocus_threshold` and `refocus_min_samples` are passed by reference because the adaptive updates need to persist across calls.

---

### `optimize(int iters)`

```cpp
std::vector<HistoryEntry> optimize(int iters)
```

Runs `step()` for `iters` iterations, collecting the result of each step into a `HistoryEntry`:

```cpp
struct HistoryEntry {
    std::array<double, dim> mu;  // mean vector at this iteration
    double best_f;               // best objective value seen in this step
    bool refocused;              // whether refocus was triggered
};
```

`history.reserve(iters)` prevents reallocation as entries are pushed. Entries are moved rather than copied: `history.push_back(std::move(h))`. The history is returned and printed in `run()`.

---



## How to Change Parameters

All parameters are set in `main()` in `Implementation.cpp`:

```cpp
Optimizer<10, 1000> session;
//         ^    ^
//         |    batch: number of samples drawn per iteration
//         dim: dimensionality of the search space
```

The template parameters `dim` and `batch` control the fixed array sizes. Changing them requires recompilation.

The runtime parameters are passed to `session.run()`:

```cpp
double eta = 0.09;
// Weight scaling factor. Controls how sharply the optimizer focuses on
// high-objective samples when updating the belief.
// Small eta (~0.01–0.1): nearly uniform weights, broad exploration.
// Large eta (>1.0): concentrates weight on the single best sample, fast
// but greedy. Start with 0.05–0.1 for most problems.

double tau_macro = 0.5;
// Similarity threshold for macro clustering.
// Higher values require stronger similarity to be considered connected —
// produces smaller, tighter macro clusters.
// Lower values connect more samples — larger, looser macro clusters.
// Range: 0.1–0.9. Start at 0.5.

double tau_micro = 0.3;
// Similarity threshold for micro clustering within the macro cluster.
// Same semantics as tau_macro but applied to the macro subset only.
// Should typically be lower than tau_macro to avoid an empty micro cluster.
// Range: 0.1–0.6. Start at 0.3.

double refocus_thresh = -1.0;
// Objective value above which a sample is considered "promising" for refocus.
// For negated Rastrigin: starts large-negative, progresses toward 0.
// Set this based on what a "good enough" region looks like for your problem.
// If refocus never triggers, lower this value.
// If refocus triggers too early, raise it.

int refocus_min_samples = 20;
// Minimum number of qualifying samples (above refocus_thresh) required to
// trigger a refocus. Too low: refocuses on noise. Too high: rarely triggers.
// Range: 10–100. Start at 20.

int random_seed = 42;
// RNG seed. Set to any fixed integer for reproducible runs.
// Set to -1 to use std::random_device for non-deterministic runs.

int iterations = 100;
// Total number of optimisation steps.

double refocus_change_rate = 0.03;
// How much refocus_thresh advances after each successful refocus.
// Larger values: each subsequent refocus requires a significantly better region.
// Smaller values: the optimizer can trigger multiple refocuses at similar quality levels.
// Range: 0.01–0.1. Start at 0.03.
```

To change the objective function, edit `fobj.h`. The current default is the 10D negated Rastrigin function. The function signature must match:

```cpp
template<size_t dim, size_t batch>
std::array<double, batch> fobj(const std::array<double, batch*dim>& X_flat)
```

---

## Performance Profiling

To reproduce the profiling results:

```bash
# Build with RelWithDebInfo for symbols + optimisation
cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo
make

# Hardware counter measurement (5 runs)
perf stat -r 20 \
  -e task-clock,cycles,instructions,cache-references,\
cache-misses,branches,branch-misses \
  ./optimizer

# Flamegraph (requires Brendan Gregg's FlameGraph tools)
perf record -g ./optimizer
perf script | stackcollapse-perf.pl | flamegraph.pl > flamegraph.svg
```

---

## License

MIT License © 2026 Tom Muga
