#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>

#include <dof/dof.hpp>

/// One SU(N) link variable: an N x N complex matrix, row-major. The group
/// operations are the only ones the rest of the code uses; no Lie-algebra basis
/// is fixed here.
template <std::size_t N>
struct SULink
{
    static_assert(N >= 2, "SU(N) needs N >= 2");

    static constexpr std::size_t dim = N;
    using Mat = std::array<std::complex<double>, N * N>;

    Mat m{};

    SULink() = default;
    explicit SULink(const Mat& mat) : m(mat) {}

    static SULink identity() noexcept
    {
        SULink u;
        for (std::size_t i = 0; i < N; ++i) u.m[i * N + i] = 1.0;
        return u;
    }

    SULink dagger() const noexcept
    {
        SULink u;
        for (std::size_t i = 0; i < N; ++i)
            for (std::size_t j = 0; j < N; ++j)
                u.m[i * N + j] = std::conj(m[j * N + i]);
        return u;
    }

    /// Re Tr U; the plaquette action uses this divided by N.
    double real_trace() const noexcept
    {
        double t = 0.0;
        for (std::size_t i = 0; i < N; ++i) t += m[i * N + i].real();
        return t;
    }

    /// max_ij |(U^dagger U - I)_ij|, for tests and diagnostics.
    double unitarity_error() const noexcept
    {
        const SULink u = dagger() * (*this);
        double e = 0.0;
        for (std::size_t i = 0; i < N; ++i)
            for (std::size_t j = 0; j < N; ++j) {
                const std::complex<double> d =
                    u.m[i * N + j] - (i == j ? 1.0 : 0.0);
                e = std::max(e, std::abs(d));
            }
        return e;
    }

    friend SULink operator*(const SULink& a, const SULink& b) noexcept
    {
        SULink r;
        for (std::size_t i = 0; i < N; ++i)
            for (std::size_t j = 0; j < N; ++j) {
                std::complex<double> s = 0.0;
                for (std::size_t k = 0; k < N; ++k) s += a.m[i * N + k] * b.m[k * N + j];
                r.m[i * N + j] = s;
            }
        return r;
    }
};

/// Re Tr(U0 U1 U2^dagger U3^dagger) / N, the oriented plaquette observable.
template <std::size_t N>
[[nodiscard]] double plaquette(const SULink<N>& u0, const SULink<N>& u1,
                               const SULink<N>& u2, const SULink<N>& u3) noexcept
{
    return (u0 * u1 * u2.dagger() * u3.dagger()).real_trace() / static_cast<double>(N);
}

namespace sun_detail
{
template <std::size_t N>
using Mat = std::array<std::complex<double>, N * N>;

template <std::size_t N>
Mat<N> mul(const Mat<N>& a, const Mat<N>& b) noexcept
{
    Mat<N> r{};
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = 0; j < N; ++j) {
            std::complex<double> s = 0.0;
            for (std::size_t k = 0; k < N; ++k) s += a[i * N + k] * b[k * N + j];
            r[i * N + j] = s;
        }
    return r;
}

template <std::size_t N>
double norm(const Mat<N>& a) noexcept
{
    double n = 0.0;
    for (std::size_t i = 0; i < N; ++i) {
        double row = 0.0;
        for (std::size_t j = 0; j < N; ++j) row += std::abs(a[i * N + j]);
        n = std::max(n, row);
    }
    return n;
}

/// exp(A) by scaling and squaring plus a Taylor series. Intended for the small
/// anti-Hermitian A = i X of a local SU(N) update, where it converges quickly.
template <std::size_t N>
SULink<N> expm(const Mat<N>& a) noexcept
{
    double n = norm<N>(a);
    int squarings = 0;
    while (n > 0.5) {
        n *= 0.5;
        ++squarings;
    }

    const double scale = std::ldexp(1.0, -squarings);
    Mat<N> b = a;
    for (auto& z : b) z *= scale;

    Mat<N> result{};
    Mat<N> term{};
    for (std::size_t i = 0; i < N; ++i) result[i * N + i] = 1.0;
    for (std::size_t i = 0; i < N; ++i) term[i * N + i] = 1.0;

    for (std::size_t k = 1; k <= 30; ++k) {
        term = mul<N>(term, b);
        for (auto& z : term) z /= static_cast<double>(k);
        for (std::size_t i = 0; i < N * N; ++i) result[i] += term[i];
        if (norm<N>(term) < 1e-16) break;
    }

    for (int k = 0; k < squarings; ++k) result = mul<N>(result, result);
    return SULink<N>(result);
}
}  // namespace sun_detail

/// Representation of one SU(N) link degree of freedom.
template <std::size_t N>
struct SUDof
{
    using dtype = SULink<N>;
    static constexpr std::size_t size = 1;
};

static_assert(Dof<SUDof<2>>);
static_assert(Dof<SUDof<3>>);

/// Symmetric local proposal for an SU(N) link: U -> U exp(i X), where X is a
/// random traceless Hermitian matrix with independent uniform entries scaled by
/// width. The distribution of X is symmetric, so the proposal ratio is 1.
template <std::size_t N>
struct SUShift
{
    static_assert(N >= 2, "SUShift needs N >= 2");

    double width = 0.3;

    template <class R>
    void sample(const SULink<N>* cur, R& rng, SULink<N>* out) const
    {
        using Mat = sun_detail::Mat<N>;

        Mat x{};
        double tr = 0.0;
        for (std::size_t i = 0; i < N; ++i) {
            const double d = width * (2.0 * rng.uni() - 1.0);
            x[i * N + i] = d;
            tr += d;
        }

        for (std::size_t i = 0; i < N; ++i)
            for (std::size_t j = i + 1; j < N; ++j) {
                const double a = width * (2.0 * rng.uni() - 1.0);
                const double b = width * (2.0 * rng.uni() - 1.0);
                x[i * N + j] = {a, b};
                x[j * N + i] = {a, -b};
            }

        for (std::size_t i = 0; i < N; ++i)
            x[i * N + i] -= tr / static_cast<double>(N);

        // A = i X is anti-Hermitian, so exp(A) is unitary.
        Mat a{};
        for (std::size_t k = 0; k < N * N; ++k)
            a[k] = {-x[k].imag(), x[k].real()};

        *out = (*cur) * sun_detail::expm<N>(a);
    }

    /// Symmetric local kernel: q(old|new) = q(new|old).
    double log_ratio(const SULink<N>*, const SULink<N>*) const noexcept { return 0.0; }
};

