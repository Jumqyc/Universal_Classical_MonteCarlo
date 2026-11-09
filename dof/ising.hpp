#pragma once

#include <cstddef>
#include <cstdint>

#include <dof/dof.hpp>

/// Ising degree of freedom: one sign per site, stored as int8_t.
using IsingSpin = std::int8_t;

/// Representation of the Ising degree of freedom.
struct IsingDof
{
    using dtype = IsingSpin;
    static constexpr std::size_t size = 1;
};

static_assert(Dof<IsingDof>);

/// Symmetric Ising proposal: flip the sign.
struct Flip
{
    /// Writes -cur to out; the current value is the only input.
    template <class R>
    void sample(const IsingSpin* cur, R&, IsingSpin* out) const
    {
        *out = static_cast<IsingSpin>(-*cur);
    }

    /// The flip is its own inverse, so the proposal densities cancel.
    double log_ratio(const IsingSpin*, const IsingSpin*) const noexcept { return 0.0; }
};

/// Wolff reflection axis for Ising: r = 1.
struct FixedAxis
{
    /// Writes the fixed axis (1) to out; cur is ignored.
    template <class R>
    void sample(const IsingSpin*, R&, IsingSpin* out) const
    {
        out[0] = IsingSpin(1);
    }
};
