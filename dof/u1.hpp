#pragma once

#include <cmath>
#include <cstddef>
#include <numbers>

#include <dof/dof.hpp>

/// One U(1) link variable, stored as an angle in radians. The group element is
/// exp(i theta); the identity is theta = 0.
struct U1Link
{
    double theta = 0.0;

    static constexpr U1Link identity() noexcept { return {}; }
};

/// Representation of one U(1) link degree of freedom.
struct U1Dof
{
    using dtype = U1Link;
    static constexpr std::size_t size = 1;
};

static_assert(Dof<U1Dof>);

/// Symmetric Metropolis proposal for a U(1) link: theta -> theta + uniform
/// displacement in [-width, width], wrapped to [-pi, pi].
struct U1Shift
{
    double width = 0.5;

    /// Writes the shifted angle to out; cur is the current link.
    template <class R>
    void sample(const U1Link* cur, R& rng, U1Link* out) const
    {
        constexpr double two_pi = 2.0 * std::numbers::pi_v<double>;
        const double shift = width * (2.0 * rng.uni() - 1.0);
        out->theta = std::remainder(cur->theta + shift, two_pi);
    }

    /// Uniform displacement is symmetric, so q(old|new) / q(new|old) = 1.
    double log_ratio(const U1Link*, const U1Link*) const noexcept { return 0.0; }
};
