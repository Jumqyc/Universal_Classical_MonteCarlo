#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numbers>

#include <dof/dof.hpp>

/// Representation of an N-component vector degree of freedom.
template <std::size_t N>
struct VectorDof
{
    static_assert(N >= 1, "VectorDof needs at least one component");

    using dtype = double;
    static constexpr std::size_t size = N;
};

static_assert(Dof<VectorDof<3>>);

/// Uniform proposal on S^(N-1), N >= 1. Samples a uniform height, then a
/// uniform direction on the remaining sphere; the result is exactly uniform.
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

/// Local proposal on S^(N-1), N >= 2: picks a uniform oriented 2-plane and
/// rotates by a uniform angle in [-delta, delta] in that plane. The rotation
/// preserves the norm; the kernel depends only on the angle between old and
/// new and is symmetric, so the log ratio is 0.
///
/// For N = 3 this is the usual random-axis rotation. For N = 2 the plane is the
/// whole space, so it is a rotation of the circle.
template <std::size_t N, class T = double>
struct LocalRotation
{
    static_assert(N >= 2, "LocalRotation needs at least two components");

    T delta;   // maximum rotation angle in radians; the only quantity to tune

    /// Rotates cur by a small random angle into out, keeping |out| = |cur|.
    template <class R>
    void sample(const T* cur, R& rng, T* out) const
    {
        std::array<T, N> u{};
        std::array<T, N> v{};

        UniformSphereN<N, T>{}.sample(nullptr, rng, u.data());

        // Draw a second sphere point, project it orthogonal to u, and normalize;
        // this gives a uniform oriented 2-frame (u, v).
        const T eps = T(64) * std::numeric_limits<T>::epsilon();
        for (;;) {
            UniformSphereN<N, T>{}.sample(nullptr, rng, v.data());
            T dot = 0;
            for (std::size_t k = 0; k < N; ++k) dot += v[k] * u[k];
            T r2 = 0;
            for (std::size_t k = 0; k < N; ++k) {
                v[k] -= dot * u[k];
                r2 += v[k] * v[k];
            }
            if (r2 > eps) {
                const T inv = T(1) / std::sqrt(r2);
                for (std::size_t k = 0; k < N; ++k) v[k] *= inv;
                break;
            }
        }

        const T th = delta * (T(2) * static_cast<T>(rng.uni()) - T(1));
        const T c  = std::cos(th);
        const T s  = std::sin(th);

        T cu = 0;
        T cv = 0;
        for (std::size_t k = 0; k < N; ++k) {
            cu += u[k] * cur[k];
            cv += v[k] * cur[k];
        }
        for (std::size_t k = 0; k < N; ++k)
            out[k] = cur[k] + (c - T(1)) * (cu * u[k] + cv * v[k])
                            + s * (cu * v[k] - cv * u[k]);
    }

    /// Rotation kernel is symmetric: q(old|new) = q(new|old).
    double log_ratio(const T*, const T*) const noexcept { return 0.0; }
};
