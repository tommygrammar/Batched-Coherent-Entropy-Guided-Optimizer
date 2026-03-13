#include <vector> //dynamic arrays
#include <iostream> 
#include <array> //fixed size arrays, whose size is static and memory is coontiguous, its knoown at compile time and not run time, faster than vector
#include <algorithm>//nth_element, move
#include <queue> //deque used for bfs
#include "multivariate.h" //multivariate function
#include "fobj.h" //objective function





template<size_t dim, size_t batch> //makes the class generic ver dim which is dimensions of input variables and batch which is number of samples drawn per iteration
class Optimizer{ //optimization class

        struct HistoryEntry {
        std::array<double,dim> mu;   // copy of mean at iteration
        double best_f;
        bool refocused;
    };

    private:
        std::array<double,dim> mu0; //mean vector of sampling distribution
        std::array<double, dim*dim> sigma0; //covariance matrix ->flattened describing the spread
        double eta; //scaling factor when computing weighted mean
        double tau_macro; //clustering threshold
        double tau_micro; //clustering threshold
        double refocus_thres; //parametrs to decide when to "refocus n promising samples"
        int refocus_min_samples;
        int random_seed;//seed random number for reproducibility
        // add to class members
        double refocus_change_rate; //refocus change rate, after ever successfufl refocus, we change the refocus so that the next refocus if an can increasingly focus on better regions
        std::mt19937 rng;

        std::array<double,dim> mu0_generation(){
            //returns a vector of zeros, this is specifically the initial mean of sampling distributions

            std::array<double,dim> mu0{};  //initialize with 0
                for(int i=0;i<dim;i++){ // all elements are converted to 0 in the array of size dim
                    mu0[i]=0.0;
                }
            return mu0;
        }

        std::array<double, dim*dim> sigma_generation(){
            //returns a diagonal covariance matrix with matrix 2.0, flattened to a 1d array
            std::array<double, dim*dim> sigma0{};
            for (int r = 0; r < dim; ++r)
                for (int c = 0; c < dim; ++c)
                    sigma0[r*dim + c] = (r==c ? 2.0 : 0.0);//converts 2d to 1d index, for every column it sets 0 and if row number = ccolumn number it sets 2
            return sigma0;
        }




        std::vector<bool> cluster_mask_macro(const std::array<double, batch*dim> &X_flat, double tau) {
            //creates macro clusters from X-flat, the tau is the threshold for similarity
            const size_t N = static_cast<size_t>(batch);
            if (N == 0) return {};

            // computes squarewise distances - we went with vector as compared to array due to triggered segmentation faults due to a stack overflow
            //computes euclidean squared distances between all pairs of samples

            std::vector<double> sq;
            sq.assign(N * N, 0.0);
            
            for (size_t i = 0; i < N; ++i) {
                for (size_t j = i + 1; j < N; ++j) {
                    double d2 = 0.0;
                    const size_t base_i = i * dim;
                    const size_t base_j = j * dim;
                    for (size_t k = 0; k < dim; ++k) {
                        double diff = X_flat[base_i + k] - X_flat[base_j + k];
                        d2 += diff * diff;
                    }
                    sq[i * N + j] = d2;
                    sq[j * N + i] = d2;
                }
                sq[i * N + i] = 0.0;
            }

            // computes median distance -> kernel width h
            std::vector<double> sq_copy = sq; // copy to sq_copy to compute median
            auto mid_it = sq_copy.begin() + (sq_copy.size() / 2);
            std::nth_element(sq_copy.begin(), mid_it, sq_copy.end()); //auto ensures that it is dropped after use
            double median_sq = *mid_it;
            double h = std::sqrt(median_sq) + 1e-12; //1e-12 ensures prevention of division by zero

            // compute similarity matrix
            //it converts distance to similarity matrix using a gaussian kernel
            //close points = 1, distant closer to 0
            std::vector<double> S;
            S.resize(N * N);
            const double denom = 2.0 * h * h;

            for (size_t i = 0; i < N; ++i) {
                for (size_t j = 0; j < N; ++j) {
                    
                    S[i * N + j] = std::exp(-sq[i * N + j] / denom);
                }
            }

            // find the median
            std::vector<double> S_copy = S;
            auto mid2 = S_copy.begin() + (S_copy.size() / 2);
            std::nth_element(S_copy.begin(), mid2, S_copy.end());
            double median_S = *mid2;
            double thresh = tau * median_S;

            // adjacency of neighbours -> it calculates the neighbours of each point
            std::vector<std::vector<int>> adj(N);
            for (size_t i = 0; i < N; ++i) {
                for (size_t j = i + 1; j < N; ++j) {
                    if (S[i * N + j] >= thresh) {
                        adj[i].push_back(static_cast<int>(j));
                        adj[j].push_back(static_cast<int>(i));
                    }
                }
            }

            //this one works to find the largest connected component in the list of neighbours
            std::vector<char> visited(N, 0);
            std::vector<int> best_comp;
            std::vector<int> comp;
            comp.reserve(N);

            std::deque<int> q;
            for (size_t i = 0; i < N; ++i) {
                if (visited[i]) continue;
                // BFS from i
                comp.clear();
                q.clear();
                visited[i] = 1;
                q.push_back(static_cast<int>(i));
                while (!q.empty()) {
                    int u = q.front(); q.pop_front();
                    comp.push_back(u);
                    for (int v : adj[u]) {
                        if (!visited[static_cast<size_t>(v)]) {
                            visited[static_cast<size_t>(v)] = 1;
                            q.push_back(v);
                        }
                    }
                }
                if (comp.size() > best_comp.size()) best_comp = comp;
            }

            // build boolean mask of size N
            std::vector<bool> mask(N, false);
            for (int idx : best_comp) mask[static_cast<size_t>(idx)] = true;

            return mask;
        }

        std::vector<bool> cluster_mask_micro(const std::vector<double> &Xm, double tau)
        {
            //this one does not take X_flat, it takes Xm, which are the macro clusters collected by macro mask and attempts to find the most connected micro clusters
            int N = Xm.size() / dim;

            // Pairwise squared distance matrix
            std::vector<double> sq(N * N, 0.0);
            
            for (int i = 0; i < N; i++)
            {
                for (int j = i; j < N; j++)
                {
                    double d2 = 0.0;
                    for (int k = 0; k < dim; k++)
                    {
                        double diff = Xm[i*dim + k] - Xm[j*dim + k];
                        d2 += diff * diff;
                    }

                    sq[i*N + j] = d2;
                    sq[j*N + i] = d2;
                }
            }

            // median distance
            std::vector<double> sq_copy = sq;
            std::nth_element(sq_copy.begin(),
                            sq_copy.begin() + sq_copy.size()/2,
                            sq_copy.end());

            double median_sq = sq_copy[sq_copy.size()/2];
            double h = std::sqrt(median_sq) + 1e-12;

            // similarity matrix
            std::vector<double> S(N * N);

            for (int i = 0; i < N; i++)
            {
                for (int j = 0; j < N; j++)
                {
                    S[i*N + j] = std::exp(-sq[i*N + j] / (2*h*h));
                }
            }

            // similarity threshold
            std::vector<double> S_copy = S;
            std::nth_element(S_copy.begin(),
                            S_copy.begin() + S_copy.size()/2,
                            S_copy.end());

            double median_S = S_copy[S_copy.size()/2];
            double thresh = tau * median_S;

            // adjacency list
            std::vector<std::vector<int>> adj(N);

            for (int i = 0; i < N; i++)
            {
                for (int j = i+1; j < N; j++)
                {
                    if (S[i*N + j] >= thresh)
                    {
                        adj[i].push_back(j);
                        adj[j].push_back(i);
                    }
                }
            }

            // BFS for largest connected component
            std::vector<bool> visited(N, false);
            std::vector<int> best_comp;

            for (int i = 0; i < N; i++)
            {
                if (!visited[i])
                {
                    std::vector<int> comp;
                    std::deque<int> queue;

                    queue.push_back(i);
                    visited[i] = true;

                    while (!queue.empty())
                    {
                        int u = queue.front();
                        queue.pop_front();

                        comp.push_back(u);

                        for (int v : adj[u])
                        {
                            if (!visited[v])
                            {
                                visited[v] = true;
                                queue.push_back(v);
                            }
                        }
                    }

                    if (comp.size() > best_comp.size())
                        best_comp = comp;
                }
            }

            std::vector<bool> mask(N, false);
            for (int idx : best_comp)
                mask[idx] = true;

            return mask;
        }


        //sometimes optimization converges prematurely, this function tries to detect possible optimums and refocus the search
        //returns, first which is whether refocus happened and second which is the best objective value

        std::pair<bool,double> _try_refocus(std::array<double,batch*dim> &X_flat,std::array<double,batch> &fvals,double refocus_threshold,int refocus_min_samples )
        { 
            int count = 0;
            // gather indices of near-zero samples
            std::vector<int> indices;
            for (int i = 0; i < batch; i++) {
                if (fvals[i] >= refocus_threshold) {
                    indices.push_back(i);
                }
            }
            
            count = indices.size();


            if (count >= refocus_min_samples) {
                //if count samples are more than or equal to refocus min samples, we  refocus, this prevents tiny clusters from messing arround

                //builds X0 which is an array of promising samples, vectoor is used here because we do not know the size that will be required, we also initialize it to 0.0
                std::vector<double> X0(count * dim, 0.0); //size is count * dim, so for each one, we have all the dimensions of the problem
                for (int r = 0; r < count; ++r) {
                    int row = indices[r];
                    for (int j = 0; j < dim; ++j) X0[r*dim + j] = X_flat[row*dim + j]; //this one then copies thoose specific ones, using indices so that they are specifically the rows that are extracted and put in X0
                }

                // computes a mean of promising samples, think of it as mu generation of the new samples -> shifts center to promising regions
                std::array<double, dim> mu0_new{};
                mu0_new.fill(0.0);
                for (int r = 0; r < count; ++r)
                    for (int j = 0; j < dim; ++j) mu0_new[j] += X0[r*dim + j];
                for (int j = 0; j < dim; ++j) mu0_new[j] /= static_cast<double>(count);

                // computes  covariance of the promising points, the spread of the new ones 
                std::array<double, dim*dim> sigma0_new{};//initializes to 0

                if (count > 1) {
                    for (int r = 0; r < count; ++r) {
                        for (int j = 0; j < dim; ++j) {
                            double dj = X0[r*dim + j] - mu0_new[j];
                            for (int t = 0; t < dim; ++t) {
                                double dk = X0[r*dim + t] - mu0_new[t];
                                sigma0_new[j*dim + t] += dj * dk;
                            }
                        }
                    }
                    double denom = static_cast<double>(count - 1);
                    for (int k = 0; k < dim*dim; ++k) sigma0_new[k] /= denom;
                } else {
                    // degenerate: single sample -> tiny covariance
                    for (int d = 0; d < dim; ++d) sigma0_new[d*dim + d] = 1e-8;
                }

                for (int d = 0; d < dim; ++d) sigma0_new[d*dim + d] += 1e-8;
                // commit
                mu0 = mu0_new; // this updates the mu0 and sigma0 with the new ones specifically
                sigma0 = sigma0_new;

                // computes the best objective in the refocused set and returns true with best f
                double bestf = -std::numeric_limits<double>::infinity();
                for (int idx : indices) if (fvals[idx] > bestf) bestf = fvals[idx];
                refocus_thres+=refocus_change_rate; //new additions - creates an adaptive refocus
                refocus_min_samples-=1; //reduces amount of samples required
                return {true, bestf};
            }

            return {false, 0.0}; //if not enough promising samples, returns this and what happens is the optimizer continues normally
        }

        std::pair<bool,double> step() {


            // draw batch samples using multivariate which is then stored in a flat 1d array if static size
            std::array<double, batch*dim> X_flat = multivariate<dim, batch>(rng, mu0, sigma0);

            //this one prevents numerical issues that may create inf values, it replaces any non finite with current means for stability, thus no breaking clustering and weights later
            for (size_t b = 0; b < batch; ++b) {
                bool row_bad = false;
                for (size_t d = 0; d < dim; ++d) {
                    double &v = X_flat[b*dim + d];
                    if (!std::isfinite(v)) {
                        v = mu0[d];          //fallback to current mean for that coordinate
                        row_bad = true;
                    }
                }
                if (row_bad) {
                    std::cerr << "Warning: repaired non-finite sample at b=" << b << " using mu0\n";
                }
            }

            // //evaluates the objective function for each sample in the batch, this is the feedback signal for updating mu0 and sigma0
            std::array<double, batch> fvals = fobj<dim, batch>(X_flat);

            // calls try refocus and if triggered it does it and updates mu0 and sigma0 and if not, just continues
            auto refocus_result = _try_refocus(X_flat, fvals, refocus_thres, refocus_min_samples); //always dropped after use
            if (refocus_result.first){ return refocus_result;}

            // macro cluster - creates promising cluster samples
            std::vector<bool> mask_macro = cluster_mask_macro(X_flat, tau_macro);
            int macro_count = 0; //how many samples survived macro
            for (bool b : mask_macro) if (b) ++macro_count;
            //this is fallback which falls back to entire batch
            std::vector<double> Xm;
            std::vector<double> fm;
            if (macro_count == 0) {
                Xm.assign(X_flat.begin(), X_flat.end());
                fm.assign(fvals.begin(), fvals.end());
                macro_count = batch;
            } else {
                Xm.resize(macro_count * dim);
                fm.resize(macro_count);
                int idx = 0;
                for (int i = 0; i < batch; ++i) {
                    if (mask_macro[i]) {
                        for (int j = 0; j < dim; ++j) Xm[idx*dim + j] = X_flat[i*dim + j];
                        fm[idx] = fvals[i];
                        ++idx;
                    }
                }
            }

            // creates micro samples within the macro cluster
            std::vector<bool> mask_micro = cluster_mask_micro(Xm, tau_micro);
            int micro_count = 0;
            for (bool b : mask_micro) if (b) ++micro_count;

            // fallback to macro cluster if micro empties
            std::vector<double> Xc;
            std::vector<double> fc;
            if (micro_count == 0) {
                Xc = Xm;
                fc = fm;
                micro_count = macro_count;
            } else {
                Xc.resize(micro_count * dim);
                fc.resize(micro_count);
                int idx2 = 0;
                for (int i = 0; i < macro_count; ++i) {
                    if (mask_micro[i]) {
                        for (int j = 0; j < dim; ++j) Xc[idx2*dim + j] = Xm[i*dim + j];
                        fc[idx2] = fm[i];
                        ++idx2;
                    }
                }
            }

            // If still empty, fallback to full batch (last-resort)
            if (micro_count == 0) {
                Xc.assign(X_flat.begin(), X_flat.end());
                fc.assign(fvals.begin(), fvals.end());
                micro_count = batch;
            }

            //all these ensure you never have an empty set


            // favours high obective values
            std::vector<double> log_w(micro_count), w(micro_count);
            double max_log_w = -std::numeric_limits<double>::infinity();
            
            for (int i = 0; i < micro_count; ++i) {

                if (!std::isfinite(fc[i])) log_w[i] = -1e300; //prevents overflow
                else log_w[i] = eta * fc[i];
                if (log_w[i] > max_log_w) max_log_w = log_w[i];
            }

                //creates normalized probabilities
            double sum_w = 0.0;
            
            for (int i = 0; i < micro_count; ++i) {
                double ex = std::exp(log_w[i] - max_log_w);
                w[i] = ex;
                sum_w += ex;
            }

            // fallback to uniform if numerical issues
            if (!(sum_w > 0.0) || !std::isfinite(sum_w)) {
                double uniform = 1.0 / static_cast<double>(micro_count);
                for (int i = 0; i < micro_count; ++i) w[i] = uniform;
            } else {
                for (int i = 0; i < micro_count; ++i) w[i] /= sum_w;
            }

            // computes new mean
            std::array<double, dim> new_mu{};
            new_mu.fill(0.0); // explicit

            
            for (int i = 0; i < micro_count; ++i) {
                double wi = w[i];
                const double* row = &Xc[i*dim];
                for (int j = 0; j < dim; ++j) new_mu[j] += wi * row[j];
            }

            //computes new sigma
            std::array<double, dim*dim> new_Sigma{};
            for (int i = 0; i < dim*dim; ++i) new_Sigma[i] = 0.0;

            
            for (int i = 0; i < micro_count; ++i) {
                double wi = w[i];
                const double* row = &Xc[i*dim];
                for (int j = 0; j < dim; ++j) {
                    double diff_j = row[j] - new_mu[j];
                    for (int k = 0; k < dim; ++k) {
                        double diff_k = row[k] - new_mu[k];
                        new_Sigma[j*dim + k] += wi * diff_j * diff_k;
                    }
                }
            }

            // Add jitter, ensure symmetry -> numerical safety
            const double jitter = 1e-8;
            for (int i = 0; i < dim; ++i) {
                new_Sigma[i*dim + i] += jitter;
            }
            //
            for (int i = 0; i < dim; ++i)
                for (int j = i+1; j < dim; ++j) {
                    double a = new_Sigma[i*dim + j];
                    double b = new_Sigma[j*dim + i];
                    double avg = 0.5 * (a + b);
                    new_Sigma[i*dim + j] = new_Sigma[j*dim + i] = avg;
                }

            // Validate finite covariance - if all is okay, it updates mu0 and sigma0
            bool cov_ok = true;
            for (int i = 0; i < dim*dim; ++i) {
                if (!std::isfinite(new_Sigma[i]) || new_Sigma[i] != new_Sigma[i]) { cov_ok = false; break; }
            }
            if (!cov_ok) {
                std::cerr << "Warning: new_Sigma invalid; skipping update\n";
                return {false, -std::numeric_limits<double>::infinity()};
            }

            // Commit update
            mu0 = new_mu;
            sigma0 = new_Sigma;

            // Return best f in micro cluster -> this one is for watching progress
            double best_f = -std::numeric_limits<double>::infinity();
            for (int i = 0; i < micro_count; ++i) if (fc[i] > best_f) best_f = fc[i];


            return {false, best_f};
        }



        std::vector<HistoryEntry> optimize(int iters){
        std::vector<HistoryEntry> history;
        history.reserve(iters);
            for(int i = 0;i<iters;i++){
                auto r = step();
                HistoryEntry h;
                h.mu=mu0; //records all these for every iteration
                h.best_f = r.second; 
                h.refocused=r.first;
                history.push_back(std::move(h));
            }
            return history;

        };


public:
    void run(
        double eta_, 
        double tau_macro_,
        double tau_micro_,
        double refocus_thresh_,
        int refocus_min_samples_,
        int random_seed_,
        int iterations,
        double refocus_change_rate_
    ) {
        // assign params
        eta = eta_;
        tau_macro = tau_macro_;
        tau_micro = tau_micro_;
        refocus_thres = refocus_thresh_;
        refocus_min_samples = refocus_min_samples_;
        random_seed = random_seed_;
        refocus_change_rate=refocus_change_rate_;

        // initialize mu0 & sigma0 once to avoid initializing per step
        mu0 = mu0_generation();
        sigma0 = sigma_generation();

        // seed RNG once here (deterministic when desired)
        if (random_seed >= 0) rng.seed(static_cast<uint32_t>(random_seed));
        else rng.seed(std::random_device{}());

        auto history = optimize(iterations);
        for (size_t i = 0; i < history.size(); ++i) {
            const auto& h = history[i];
            std::cout << "Iter " << i<< " best_f=" << h.best_f << " mu=(";
            for (int d = 0; d < dim; ++d) {
                std::cout << h.mu[d] << (d+1<dim? ", ":"");
            }
            std::cout << ") refocused=" << (h.refocused? "yes":"no") << "\n";
        }
        }

};


int main(){

Optimizer<10,1000> session;

//parameters
double eta = 0.09;
double tau_macro=0.3;
double tau_micro=0.5;
double refocus_thresh=-0.98;
int refocus_min_samples=200;
int random_seed=42;
int iterations=2;
double refocus_change_rate=0.04;






session.run(
        eta, // eta 
        tau_macro, //tau macro
        tau_micro, //tau micro
        refocus_thresh, //refooocus threshold
        refocus_min_samples, //refocus min samples
        random_seed, //random seed
        iterations, //iterations
        refocus_change_rate 
    )   ;
return 0;
}