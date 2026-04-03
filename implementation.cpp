#include <vector> //dynamic arrays
#include <iostream> 
#include <array> //fixed size arrays, whose size is static and memory is coontiguous, its knoown at compile time and not run time, faster than vector
#include <algorithm>//nth_element, move
#include <queue> //deque used for bfs
#include "include/multivariate.h" //multivariate function
#include "include/fobj.h" //objective function

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

        inline std::array<double,dim> mu0_generation(){
            //returns a vector of zeros, this is specifically the initial mean of sampling distributions

            std::array<double,dim> mu0{};  //initialize with 0
                for(int i=0;i<dim;i++){ // all elements are converted to 0 in the array of size dim
                    mu0[i]=0.0;
                }
            return mu0;
        }

        inline std::array<double, dim*dim> sigma_generation(){
            //returns a diagonal covariance matrix with matrix 2.0, flattened to a 1d array
            std::array<double, dim*dim> sigma0{};
            for (int r = 0; r < dim; ++r)
                for (int c = 0; c < dim; ++c)
                    sigma0[r*dim + c] = (r==c ? 2.0 : 0.0);//converts 2d to 1d index, for every column it sets 0 and if row number = ccolumn number it sets 2
            return sigma0;
        }

        std::vector<uint8_t> cluster_mask_macro(const std::array<double, batch * dim>& X_flat, double tau) {

            auto sqdist = [&](size_t i, size_t j) -> double {
                double d2 = 0.0;
                const size_t base_i = i * dim;
                const size_t base_j = j * dim;

                for (size_t k = 0; k < dim; ++k) {
                    const double diff = X_flat[base_i + k] - X_flat[base_j + k];
                    d2 += diff * diff;
                }
                return d2;
            };

            //computes median of pairwise distances
            auto median_of = [](std::vector<double>& v) -> double {
                const size_t m = v.size() / 2;
                auto mid = v.begin() + static_cast<std::ptrdiff_t>(m);
                std::nth_element(v.begin(), mid, v.end());
                return *mid;
            };

            // Estimate median squared distance
            // Exact for small N, sampled for large N.
            constexpr size_t sample_budget = 2048; //upper bund on how many distances will be computed
            const size_t total_pairs = (batch * (batch - 1)) / 2;

            std::vector<double> dist_samples;
            dist_samples.reserve(std::min(total_pairs, sample_budget)); //avoids repeated allocations during pushback - preallocate enough memory to hld whichever is  2048 or less than


                // Approximate median via random sampling of pairs.
                std::mt19937 rng(123456789u); // deterministic seed for reproducibility of results
                std::uniform_int_distribution<size_t> pick(0, batch - 1); //picks a random integer from here

                //keeps sampled distance values less than sample budget
                while (dist_samples.size() < sample_budget) {
                    //pick two independent random samples from the dataset
                    size_t i = pick(rng);
                    size_t j = pick(rng);
                    if (i == j) continue; //rejects all self pairs
                    if (i > j) std::swap(i, j); //enforce ordering to avoid redundancy - (i,j) would be same as (j,i) but would be treated as different samples
                    dist_samples.push_back(sqdist(i, j)); // appends to vector the squared distance
                }
            
            //computes median of sampled distances
            double median_sq = median_of(dist_samples);
            if (median_sq < 0.0) median_sq = 0.0;

            const double h = std::sqrt(median_sq) + 1e-12; //converts it back to a distance scale  and prevents h rom being exactly 0
            const double denom = 2.0 * h * h; // controls how fast similarity decays with distance, larger - slower decay, smaller - sharper decay

            // Estimate median similarity
            // we convert each distance into a similarity score -  small - similarity close to 1, big - similarity close to 0.
            std::vector<double> sim_samples;
            sim_samples.reserve(dist_samples.size());
            for (double d2 : dist_samples) {
                sim_samples.push_back(std::exp(-d2 / denom));
            }

            //take the median os similarit values - this gives a typical similarity level
            const double median_S = median_of(sim_samples);
            const double thresh = tau * median_S; //multiply by our threshold, larger tau-more strict, smaller - more permissive

            // Build adjacency matrix in a single flat vector
            std::vector<int> adj(batch * batch, 0);

            for (size_t i = 0; i < batch; ++i) {
                for (size_t j = i + 1; j < batch; ++j) {
                    const double d2 = sqdist(i, j);
                    const double s = std::exp(-d2 / denom);

                    if (s >= thresh) {
                        adj[i * batch + j] = 1; // there is an edge here
                        adj[j * batch + i] = 1; // there is an edge here
                    }
                }
            }

            //Find the largest connected component
            std::vector<uint8_t> visited(batch, 0); // stores the ones visited
            std::vector<int> best_comp; // stores the largest component found so far
            std::vector<int> comp; //temporary storage for the current component we are exploring
            comp.reserve(batch); //preallocate memory

            std::deque<int> q; //BFS queue

            for (size_t i = 0; i < batch; ++i) { // start a traversal from every node
                if (visited[i]) continue; // skip if already visited

                comp.clear(); //reset current component
                q.clear(); //reset bfs queue

                visited[i] = 1; //mark starting node as visited
                q.push_back(static_cast<int>(i)); // start BFS from node i

                while (!q.empty()) { // as long as there are unprocessed nodes, continue processing
                    const int u = q.front(); // take the next node to process - reads it
                    q.pop_front(); // mark as processed b removing it from queue

                    comp.push_back(u); // add this node to the current cnnected component 

                    for (size_t v = 0; v < batch; ++v) { // chek all possible nodes to see if the are neighbours of u - scans a flat matrix
                        const size_t uv = static_cast<size_t>(u) * batch + v; // converts (u,v) into a flat index
                        if (adj[uv] && !visited[v]) { // is there an edge between u and v and have we not explored v already
                            visited[v] = 1; // marked as discovered
                            q.push_back(static_cast<int>(v)); //schedule v to be processed later
                        }
                    }
                }
                //compare current cmponent vs best seen so far, if bigger, it overwrites it
                if (comp.size() > best_comp.size()) {
                    best_comp = comp;
                }
}
            

            // Return a normal byte-based mask
            std::vector<uint8_t> mask(batch, 0);
            for (int idx : best_comp) {
                mask[static_cast<size_t>(idx)] = 1;
            }

            return mask;
        }

        std::vector<uint8_t> cluster_mask_micro(const std::vector<double>& Xm, double tau)
        {
            // Xm contains flattened macro-cluster samples: N rows, each row has `dim` values.
            auto sqdist = [&](size_t i, size_t j) -> double {
                double d2 = 0.0;
                const size_t base_i = i * dim;
                const size_t base_j = j * dim;

                for (size_t k = 0; k < dim; ++k) {
                    const double diff = Xm[base_i + k] - Xm[base_j + k];
                    d2 += diff * diff;
                }
                return d2;
            };

            auto median_of = [](std::vector<double>& v) -> double {
                const size_t m = v.size() / 2;
                auto mid = v.begin() + static_cast<std::ptrdiff_t>(m);
                std::nth_element(v.begin(), mid, v.end());
                return *mid;
            };

            // Estimate median squared distance
            constexpr size_t sample_budget = 1024;
            const size_t total_pairs = (batch * (batch - 1)) / 2;

            std::vector<double> dist_samples;
            dist_samples.reserve(std::min(total_pairs, sample_budget));


                // Approximate for larger N: sample random pairs.
                std::mt19937 rng(123456789u);
                std::uniform_int_distribution<size_t> pick(0, batch - 1);

                while (dist_samples.size() < sample_budget) {
                    size_t i = pick(rng);
                    size_t j = pick(rng);
                    if (i == j) continue;
                    if (i > j) std::swap(i, j);
                    dist_samples.push_back(sqdist(i, j));
                }
            

            double median_sq = median_of(dist_samples);
            if (median_sq < 0.0) median_sq = 0.0;

            const double h = std::sqrt(median_sq) + 1e-12;
            const double denom = 2.0 * h * h;

            // Estimate median similarity
            std::vector<double> sim_samples;
            sim_samples.reserve(dist_samples.size());

            for (double d2 : dist_samples) {
                sim_samples.push_back(std::exp(-d2 / denom));
            }

            const double median_S = median_of(sim_samples);
            const double thresh = tau * median_S;

            // Build adjacency matrix in a single flat vector
            std::vector<int> adj(batch * batch, 0);

            for (size_t i = 0; i < batch; ++i) {
                for (size_t j = i + 1; j < batch; ++j) {
                    const double d2 = sqdist(i, j);
                    const double s = std::exp(-d2 / denom);

                    if (s >= thresh) {
                        adj[i * batch + j] = 1; // there is an edge here
                        adj[j * batch + i] = 1; // there is an edge here
                    }
                }
            }

            // Find largest connected component
            std::vector<uint8_t> visited(batch, 0);
            std::vector<int> best_comp;
            std::vector<int> comp;
            comp.reserve(batch);

            std::deque<int> q;

            for (size_t i = 0; i < batch; ++i) {
                if (visited[i]) continue;

                comp.clear();
                q.clear();

                visited[i] = 1;
                q.push_back(static_cast<int>(i));

                while (!q.empty()) {
                    const int u = q.front();
                    q.pop_front();

                    comp.push_back(u);

                    for (size_t v = 0; v < batch; ++v) {
                        const size_t uv = static_cast<size_t>(u) * batch + v;
                        if (adj[uv] && !visited[v]) {
                            visited[v] = 1;
                            q.push_back(static_cast<int>(v));
                        }
                    }
                }

                if (comp.size() > best_comp.size()) {
                    best_comp = comp;
                }
}
            

            // Build byte mask
            std::vector<uint8_t> mask(batch, 0);
            for (int idx : best_comp) {
                mask[static_cast<size_t>(idx)] = 1;
            }

            return mask;
        }

        //sometimes optimization converges prematurely, this function tries to detect possible optimums and refocus the search
        //returns, first which is whether refocus happened and second which is the best objective value
        std::pair<bool, double> _try_refocus(const std::array<double,batch * dim>& X_flat,const std::array<double, batch>& fvals,double& refocus_threshold,int& refocus_min_samples)
        {
            // Count qualifying samples, compute their mean, and track best objective value.
            std::size_t count = 0;
            std::array<double, dim> mu0_new{};
            mu0_new.fill(0.0);

            double bestf = -std::numeric_limits<double>::infinity();
            
            for (std::size_t i = 0; i < batch; ++i) {
                if (fvals[i] >= refocus_threshold) { // is the sample good enough
                    ++count; //increase number o qualifing samples
                    bestf = std::max(bestf, fvals[i]); //tracks best objective

                    const std::size_t base = i * dim;
                    for (std::size_t j = 0; j < dim; ++j) {
                        mu0_new[j] += X_flat[base + j];
                    }
                }
            }

            if (count < static_cast<std::size_t>(refocus_min_samples)) {
                return {false, 0.0};
            }

            for (std::size_t j = 0; j < dim; ++j) {
                mu0_new[j] /= static_cast<double>(count);
            }

            // Covariance of the qualifying samples.
            std::array<double, dim * dim> sigma0_new{};
            sigma0_new.fill(0.0);

            if (count > 1) {
                for (std::size_t i = 0; i < batch; ++i) {
                    if (fvals[i] < refocus_threshold) {
                        continue;
                    }

                    const std::size_t base = i * dim;
                    for (std::size_t j = 0; j < dim; ++j) {
                        const double dj = X_flat[base + j] - mu0_new[j];
                        for (std::size_t t = 0; t < dim; ++t) {
                            const double dk = X_flat[base + t] - mu0_new[t];
                            sigma0_new[j * dim + t] += dj * dk;
                        }
                    }
                }

                const double denom = static_cast<double>(count - 1);
                for (double& x : sigma0_new) {
                    x /= denom;
                }
            } else {
                // Degenerate case: one sample only.
                for (std::size_t d = 0; d < dim; ++d) {
                    sigma0_new[d * dim + d] = 1e-8;
                }
            }

            // Diagonal regularization.
            for (std::size_t d = 0; d < dim; ++d) {
                sigma0_new[d * dim + d] += 1e-8;
            }

            // Commit the new refocus state.
            mu0 = mu0_new;
            sigma0 = sigma0_new;

            // Adaptive update.
            refocus_threshold += refocus_change_rate;
            refocus_min_samples = std::max(1, refocus_min_samples - 1);

            return {true, bestf};
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
            std::vector<uint8_t> mask_macro = cluster_mask_macro(X_flat, tau_macro);
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
            std::vector<uint8_t> mask_micro = cluster_mask_micro(Xm, tau_micro);
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
double tau_macro=0.5;
double tau_micro=0.3;
double refocus_thresh=-0.98;
int refocus_min_samples=20;
int random_seed=42;
int iterations=100;
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
