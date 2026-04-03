#pragma once

#include <random>
#include <array>
#include <cmath>
#include <limits>
#include <algorithm>
#include <iostream>
#include <cstddef>

// limits extreme values
inline double clamp_normal_value(double z, double limit = 8.0) {
    if (!std::isfinite(z)) return 0.0; // if Non or inf, replace with 0
    if (z >  limit) return  limit;
    if (z < -limit) return -limit;
    return z;
}


// generates a standard normal with Box Muller transform, caches one sample for efficiency
inline double next_standard_normal(std::mt19937& rng, bool& has_spare, double& spare) {
    if (has_spare) { // i a prebiously computed spare exists, reuse it and avoid computatin
        has_spare = false;
        return spare;
    }

    // u1 must be strictly > 0 to avoid log(0).
    double u1 = 0.0;
    do {
        u1 = std::generate_canonical<double, 53>(rng); // first sample
    } while (u1 <= 0.0);

    const double u2 = std::generate_canonical<double, 53>(rng); // second sample

    constexpr double two_pi = 6.283185307179586476925286766559; // 2 pi used in polar coordinate transform
    const double r = std::sqrt(-2.0 * std::log(u1)); // converts unifirm to polar coordinates
    const double theta = two_pi * u2;

    const double z0 = r * std::cos(theta);
    const double z1 = r * std::sin(theta);
    //two inpependent normal samples
    spare = z1;
    has_spare = true;
    return z0;
}


//computes cholesk decomposition of covariance matrix
template <std::size_t dim>
bool safe_cholesky(const std::array<double, dim * dim>& Sigma,
                   std::array<double, dim * dim>& L,
                   double initial_jitter = 1e-12,
                   int max_attempts = 8)
{
    for (std::size_t i = 0; i < dim * dim; ++i) { // reject invalid ones early
        if (!std::isfinite(Sigma[i])) return false;
    }

    double jitter = initial_jitter; // added fr numerical stability

    for (int attempt = 0; attempt < max_attempts; ++attempt) { // retries with increasing jitter if decomposition fails
        L.fill(0.0); // clear output matrix
        bool ok = true;

        for (std::size_t i = 0; i < dim && ok; ++i) { //iterate over rows
            for (std::size_t j = 0; j <= i; ++j) { // only lower triangle
                double sum = Sigma[i * dim + j]; // accumulate
                if (i == j) sum += jitter; //add numerical stabilization on diagnonal

                //remove previsuly computed contributions
                for (std::size_t k = 0; k < j; ++k) {
                    sum -= L[i * dim + k] * L[j * dim + k];
                }
                //check validity - must be positive and finite
                if (i == j) {
                    if (!(sum > 0.0) || !std::isfinite(sum)) {
                        ok = false;
                        break;
                    }
                    L[i * dim + j] = std::sqrt(sum); // diagnonal element
                } else {
                    const double denom = L[j * dim + j];
                    if (!(denom != 0.0) || !std::isfinite(denom)) {//ensures no division by 0 or invalid values
                        ok = false;
                        break;
                    }
                    L[i * dim + j] = sum / denom; // computes lower triangle entry
                }
            }
        }

        if (ok) return true;
        jitter = std::max(jitter * 10.0, 1e-16); //progressively stabilize harder matrices
    }

    return false;
}

//generate samples from multivariate normal distribution
template <std::size_t dim, std::size_t batch>
std::array<double, batch * dim> multivariate(std::mt19937& rng,
                                             const std::array<double, dim>& mu0,
                                             const std::array<double, dim * dim>& Sigma)
{
    std::array<double, batch * dim> X_flat{};

    // detect invalid covariance first
    for (std::size_t i = 0; i < dim * dim; ++i) {
        if (!std::isfinite(Sigma[i])) {
            std::cerr << "multivariate: Sigma contains non-finite values - returning means\n";
            for (std::size_t b = 0; b < batch; ++b) {
                for (std::size_t d = 0; d < dim; ++d) {
                    X_flat[b * dim + d] = mu0[d];
                }
            }
            return X_flat;
        }
    }

    // Symmetrize sigma - needed for cholesky.
    std::array<double, dim * dim> Sigma_sym{};
    for (std::size_t i = 0; i < dim; ++i) {
        for (std::size_t j = 0; j < dim; ++j) {
            Sigma_sym[i * dim + j] = 0.5 * (Sigma[i * dim + j] + Sigma[j * dim + i]);
        }
    }

    // Cholesky factorization.
    std::array<double, dim * dim> L{};
    bool chol_ok = safe_cholesky<dim>(Sigma_sym, L, 1e-12, 8);

    if (!chol_ok) {// if failed, fallback
        std::cerr << "multivariate: Cholesky failed; falling back to diagonal sqrt of diag(Sigma)\n";
        L.fill(0.0);

        const double min_diag = 1e-12;
        for (std::size_t i = 0; i < dim; ++i) {
            double v = Sigma_sym[i * dim + i];
            if (!std::isfinite(v) || v < min_diag) v = min_diag;
            L[i * dim + i] = std::sqrt(v);
        }
    }

    // Generate samples in a tight sequential loop
    bool has_spare = false;
    double spare = 0.0;

    for (std::size_t b = 0; b < batch; ++b) {
        std::array<double, dim> z{};

        for (std::size_t d = 0; d < dim; ++d) {
            const double raw = next_standard_normal(rng, has_spare, spare);
            z[d] = clamp_normal_value(raw, 8.0); // clamp extreme values
        }

        for (std::size_t i = 0; i < dim; ++i) {
            double acc = 0.0;

            for (std::size_t j = 0; j <= i; ++j) {
                double term = L[i * dim + j] * z[j];
                if (!std::isfinite(term)) term = 0.0;
                acc += term;
            }

            double val = mu0[i] + acc;
            if (!std::isfinite(val)) {
                std::cerr << "multivariate: non-finite sample at b=" << b
                          << " i=" << i << " (using mu)\n";
                val = mu0[i];
            }

            X_flat[b * dim + i] = val;
        }
    }

    return X_flat;
}

template <std::size_t dim, std::size_t batch>
std::array<double, batch * dim> multivariate(int seed,
                                             const std::array<double, dim>& mu0,
                                             const std::array<double, dim * dim>& Sigma)
{
    std::mt19937 rng(static_cast<unsigned int>(seed));
    return multivariate<dim, batch>(rng, mu0, Sigma);
}
