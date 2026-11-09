#pragma once

#include <cstddef>
#include <cstdint>

#include <dof/ising.hpp>

/// Uniform Ising pair interaction: E = -J s_i s_j.
struct IsingTerm
{
    using dtype = IsingDof::dtype;
    using energy_type = std::int32_t;
    static constexpr std::size_t arity = 2;

    energy_type J = 1;

    /// Pair energy of the two endpoint states.
    energy_type energy(const dtype* a, const dtype* b) const noexcept
    {
        return -J * energy_type(*a) * energy_type(*b);
    }

    /// Energy change when one endpoint goes from old to nw; fixed is the other.
    energy_type delta(const dtype* old, const dtype* nw, const dtype* fixed) const noexcept
    {
        return -J * (energy_type(*nw) - energy_type(*old)) * energy_type(*fixed);
    }
};
