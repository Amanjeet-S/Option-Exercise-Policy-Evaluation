// Driftless finite-difference propagation between cash/exercise events.
// Original implementation of the tridiagonal elimination derived in the report.
// C ABI: arrays have n interior nodes, n+2 values and k substeps respectively.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <new>
#include <vector>

extern "C" int qf_funded_advance(
    const double* diffusion, std::size_t n, const double* initial,
    const double* widths, const double* theta, const double* left,
    const double* right, std::size_t k, double* output) noexcept {
    if (!diffusion || !initial || !widths || !theta || !left || !right ||
        !output || n == 0 || k == 0) return 1;
    try {
        for (std::size_t i = 0; i < n; ++i)
            if (!std::isfinite(diffusion[i]) || diffusion[i] < 0.0) return 1;
        for (std::size_t i = 0; i < n + 2; ++i)
            if (!std::isfinite(initial[i])) return 1;
        for (std::size_t j = 0; j < k; ++j)
            if (!std::isfinite(widths[j]) || widths[j] <= 0.0 ||
                !std::isfinite(theta[j]) || theta[j] < 0.5 || theta[j] > 1.0 ||
                !std::isfinite(left[j]) || !std::isfinite(right[j])) return 1;

        std::vector<double> values(initial, initial + n + 2);
        std::vector<double> upper(n), reduced(n);
        for (std::size_t j = 0; j < k; ++j) {
            for (std::size_t i = 0; i < n; ++i) {
                const double a = theta[j] * widths[j] * diffusion[i];
                double rhs = values[i + 1] + (1.0 - theta[j]) * widths[j] *
                    diffusion[i] * (values[i] - 2.0 * values[i + 1] + values[i + 2]);
                if (i == 0) rhs += a * left[j];
                if (i + 1 == n) rhs += a * right[j];
                const double pivot = 1.0 + 2.0 * a + (i == 0 ? 0.0 : a * upper[i - 1]);
                if (!std::isfinite(pivot) || pivot <= 0.0) return 2;
                upper[i] = i + 1 == n ? 0.0 : -a / pivot;
                reduced[i] = (rhs + (i == 0 ? 0.0 : a * reduced[i - 1])) / pivot;
                if (!std::isfinite(reduced[i])) return 2;
            }
            values[n] = reduced[n - 1];
            for (std::size_t i = n - 1; i > 0; --i)
                values[i] = reduced[i - 1] - upper[i - 1] * values[i + 1];
            values[0] = left[j];
            values[n + 1] = right[j];
            for (double value : values)
                if (!std::isfinite(value)) return 2;
        }
        std::copy(values.begin(), values.end(), output);
        return 0;
    } catch (const std::bad_alloc&) {
        return 3;
    } catch (...) {
        return 4;
    }
}
