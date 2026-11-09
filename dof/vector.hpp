#pragma once

#include <cmath>
#include <cstddef>
#include <numbers>

/// Uniform proposal on S^(N-1), N >= 1. Samples a uniform height, then a
/// uniform direction on the remaining sphere; the result is exactly uniform.
/// Also serves as the random Wolff reflection axis for N > 1.
template <std::size_t N, class T = double>
struct UniformSphereN
{
    static_assert(N >= 1, "UniformSphereN needs at least one component");

    /// Writes a uniformly distributed unit vector to out; cur is ignored.
    template <class R>
    void sample(const T* /*cur*/, R& rng, T* out) const
    {
        if constexpr (N == 1) {
            out[0] = rng.uni() < 0.5 ? T(-1) : T(1);
        } else if constexpr (N == 2) {
            const T phi = T(2) * std::numbers::pi_v<T> * static_cast<T>(rng.uni());
            out[0] = std::cos(phi);
            out[1] = std::sin(phi);
        } else {
            const T z = T(2) * static_cast<T>(rng.uni()) - T(1);
            const T r = std::sqrt(T(1) - z * z);
            UniformSphereN<N - 1, T>{}.sample(nullptr, rng, out);
            for (std::size_t k = 0; k < N - 1; ++k) out[k] *= r;
            out[N - 1] = z;
        }
    }

    /// Uniform over the sphere: q(old|new) = q(new|old).
    double log_ratio(const T*, const T*) const noexcept { return 0.0; }
};
