#pragma once
#include <array>
#include <cmath>


template<size_t dim, size_t batch>
std::array<double, batch> fobj(const std::array<double, batch*dim>& X_flat)
{
    constexpr double A = 10.0;
    constexpr double PI = 3.14159265358979323846;

    std::array<double, batch> fvals{};

    for(size_t b = 0; b < batch; b++)
    {
        double sum = 0.0;

        for(size_t d = 0; d < dim; d++)
        {
            double xi = X_flat[b*dim + d];
            sum += xi*xi - A * std::cos(2.0 * PI * xi);
        }

        fvals[b] = -(A*dim + sum);
    }

    return fvals;
}