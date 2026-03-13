#pragma once //prevents header from being included multiple times in the same translatin unit
#include <random> //normal distribution and std::mt19937 - gaussian sampling
#include <array> //for arrays
#include <cmath> //isfinite and sqrt for numerical stability
#include <limits> //numeric limits
#include <algorithm> //max
#include <iostream> //diagnostics cerr

//limits extreme gaussian values, normal distriibution has infinite tails to prevent overflow - effectivel prevents catasrophic values
inline double clamp_normal_value(double z, double limit = 8.0) {
    if (!std::isfinite(z)) return 0.0; //if rnd returns with Nan/inf, replace with safe values
    if (z >  limit) return  limit;
    if (z < -limit) return -limit;
    return z; //otherwise dont change
}


template <size_t dim> //matrix size is a compile time constant
bool safe_cholesky(const std::array<double, dim*dim>& Sigma,//sigma is flattened with a row major layout
                   std::array<double, dim*dim>& L, //cholesky factor(lower triangle)
                   double initial_jitter = 1e-12, //tin diagnal added too avoid singularity in matrices
                   int max_attempts = 8// max retries for increasing jitter
                   )
{
    // detects invalid covariance entries like Nan, Inf, if present decomposition impossible, so it returns failure immediately
    for (size_t i = 0; i < dim*dim; ++i) {
        if (!std::isfinite(Sigma[i])) return false;
    }

    double jitter = initial_jitter; //this will grow if decompoosition fails
    for (int attempt = 0; attempt < max_attempts; ++attempt) {//covariance matrices from optimization often become nearl singular so we have to stabilize the diagonal b inccreasing the jitter
        L.fill(0.0); //zero fill - oonl lower triangle will be re rewritten
        bool ok = true; //success flag to break loops if early failure detected

        for (size_t i = 0; i < dim && ok; ++i) { //rows
            for (size_t j = 0; j <= i; ++j) {//columns of lower triangle, upper triangle skipped 
                double sum = Sigma[i*dim + j];//starts the accumulation
                if (i == j) sum += jitter;  //added to diagonal
                for (size_t k = 0; k < j; ++k) {//subtracts previously computed contributions
                    sum -= L[i*dim + k] * L[j*dim + k];
                }
                if (i == j) {
                    if (!(sum > 0.0) || !std::isfinite(sum)) { ok = false; break; }//checks positivit as cholesk requires positive diagonal
                    L[i*dim + j] = std::sqrt(sum);
                } else {
                    double denom = L[j*dim + j]; //diagonal element for normalization
                    if (!(denom != 0.0) || !std::isfinite(denom)) { ok = false; break; }//prevent division by zero
                    L[i*dim + j] = sum / denom;
                }
            }
        }

        if (ok) return true; //decomposition succeded, exit immediately
        jitter = std::max(jitter * 10.0, 1e-16); //this gradually stabilizes matrix b increasing jitter
    }

    return false; //all failed
}

// PRIMARY multivariate sampler (preferred):


template <size_t dim, size_t batch>
std::array<double, batch*dim> multivariate(std::mt19937 &rng,
                                           const std::array<double, dim>& mu0,
                                           const std::array<double, dim*dim>& Sigma)
{
    std::array<double, batch*dim> X_flat{};
    std::normal_distribution<double> nd(0.0, 1.0);

    // if covariance contains invalid values, sampling is impossible, so the fallback is it returns mu0(means)
    for (size_t i = 0; i < dim*dim; ++i) { 
        if (!std::isfinite(Sigma[i])) {
            std::cerr << "multivariate: Sigma contains non-finite values - returning means\n";
            for (size_t b = 0; b < batch; ++b)
                for (size_t d = 0; d < dim; ++d)
                    X_flat[b*dim + d] = mu0[d];
            return X_flat;
        }
    }

    // Symmetrize Sigma into local copy to avoid small asymmetries - floating point rounding can make matrix slightly assmetric, cholesk requires symmetry
    std::array<double, dim*dim> Sigma_sym{};
    for (size_t i = 0; i < dim; ++i)
        for (size_t j = 0; j < dim; ++j)
            Sigma_sym[i*dim + j] = 0.5 * (Sigma[i*dim + j] + Sigma[j*dim + i]);

    // Compute robust Cholesky with jitter attempts
    std::array<double, dim*dim> L{};
    bool chol_ok = safe_cholesky<dim>(Sigma_sym, L, 1e-12, 8); //compute lower triangle

    if (!chol_ok) {
        // fallback to diagonal covariance
        const double min_diag = 1e-12; //prevents 0 vairance
        std::cerr << "multivariate: Cholesky failed; falling back to diagonal sqrt of diag(Sigma)\n";
        for (size_t i = 0; i < dim; ++i) {
            double v = Sigma_sym[i*dim + i];
            if (!std::isfinite(v) || v < min_diag) v = min_diag;
            L[i*dim + i] = std::sqrt(v);
            // other entries remain zero
        }
    }

    // generate samples
    for (size_t b = 0; b < batch; ++b) { //generates each sample
        // sample z vector
        std::array<double, dim> z{}; //temporary vector of standard normals
        for (size_t d = 0; d < dim; ++d) {//generate and clamp
            double raw = nd(rng);
            z[d] = clamp_normal_value(raw, 8.0); 
        }

        // transform x = mu + L * z (L lower-triangular)
        for (size_t i = 0; i < dim; ++i) { //compute each coordinate
            double acc = 0.0; //stores L*z
            for (size_t j = 0; j <= i; ++j) {
                double term = L[i*dim + j] * z[j]; //contribution from dimension j
                if (!std::isfinite(term)) term = 0.0; //avoid propagation of invalid numbers
                acc += term;
            }
            double val = mu0[i] + acc; // final cordinate, mean
            if (!std::isfinite(val)) {
                // defensive fallback: use mean
                std::cerr << "multivariate: non-finite sample at b=" << b << " i=" << i
                          << " (using mu)\n";
                val = mu0[i];
            }
            X_flat[b*dim + i] = val; //row major storage(flattened to 1D)
        }
    }

    return X_flat;
}

// CONVENIENCE overload: build a local RNG from seed and delegate to the rng-ref version.

template <size_t dim, size_t batch>
std::array<double, batch*dim> multivariate(int seed,
                                           const std::array<double, dim>& mu0,
                                           const std::array<double, dim*dim>& Sigma)
{
    std::mt19937 rng(static_cast<unsigned int>(seed));
    return multivariate<dim, batch>(rng, mu0, Sigma);
}



