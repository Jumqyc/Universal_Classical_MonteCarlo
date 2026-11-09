#pragma once

#include <cmath>
#include <numbers>

/// Uniform proposal on S^2: draws a direction uniformly by area, ignoring the
/// current value. Samples a uniform z and a uniform azimuth; do NOT sample three
/// numbers in a cube and normalize, which over-samples the body diagonals
/// (measured: the polar caps hold only 60% of the correct point count, while
/// <x^2> still comes out right). The density does not depend on the current
/// value, so the log ratio is 0.
template <class T>
struct UniformSphere
{
    /// Writes a uniformly distributed unit vector to out; cur is ignored.
    template <class R>
    void sample(const T* /*cur*/, R& rng, T* out) const
    {
        const T z   = T(2) * static_cast<T>(rng.uni()) - T(1);
        const T phi = T(2) * std::numbers::pi_v<T> * static_cast<T>(rng.uni());
        const T r   = std::sqrt(T(1) - z * z);
        out[0] = r * std::cos(phi);
        out[1] = r * std::sin(phi);
        out[2] = z;
    }

    /// Uniform over the sphere: q(old|new) = q(new|old) = 1/4pi.
    double log_ratio(const T*, const T*) const noexcept { return 0.0; }
};

/// Local proposal on S^2: rotates by a uniform angle in [-delta, delta] about a
/// uniform random axis. The rotation preserves the norm, so the result stays on
/// S^2. The kernel depends only on the angle between
/// the old and new directions and is symmetric, so the log ratio is 0.
/// out must be a different array from cur.
template <class T>
struct LocalRotation
{
    T delta;   // maximum rotation angle in radians; the only quantity to tune

    /// Rotates cur by a small random angle into out, keeping |out| = |cur|.
    template <class R>
    void sample(const T* cur, R& rng, T* out) const
    {
        const T z   = T(2) * static_cast<T>(rng.uni()) - T(1);
        const T phi = T(2) * std::numbers::pi_v<T> * static_cast<T>(rng.uni());
        const T r   = std::sqrt(T(1) - z * z);
        const T a[3] = { r * std::cos(phi), r * std::sin(phi), z };   // uniform axis

        const T th = delta * (T(2) * static_cast<T>(rng.uni()) - T(1));
        const T c  = std::cos(th);
        const T s  = std::sin(th);
        // Rodrigues: out = c*cur + s*(a x cur) + (1-c)(a.cur)a
        const T axc[3] = { a[1]*cur[2] - a[2]*cur[1],
                           a[2]*cur[0] - a[0]*cur[2],
                           a[0]*cur[1] - a[1]*cur[0] };
        const T ad = a[0]*cur[0] + a[1]*cur[1] + a[2]*cur[2];
        for (int k = 0; k < 3; ++k)
            out[k] = c*cur[k] + s*axc[k] + (T(1) - c)*ad*a[k];
    }

    /// Rotation kernel is symmetric: q(old|new) = q(new|old).
    double log_ratio(const T*, const T*) const noexcept { return 0.0; }
};
