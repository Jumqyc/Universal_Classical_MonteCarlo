#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>

#include <dof/ising.hpp>
#include <term/cell.hpp>

/// Uniform Ising pair interaction: E = -J s_i s_j. A pair is a cell of arity 2,
/// so this satisfies CellTerm rather than a separate pair concept.
struct IsingTerm
{
    using dtype = IsingDof::dtype;
    using energy_type = std::int32_t;
    static constexpr std::size_t arity = 2;

    energy_type J = 1;

    /// Pair energy of the two endpoint states, links[0] and links[1].
    energy_type energy(const dtype* const* links) const noexcept
    {
        return -J * dot(*links[0], *links[1]);
    }

    /// Energy change when links[which] goes from old to nw; the other is fixed.
    energy_type delta(std::size_t which, const dtype* old, const dtype* nw,
                      const dtype* const* links) const noexcept
    {
        assert(which < arity);
        const dtype* fixed = links[which == 0 ? 1 : 0];
        return -J * (dot(*nw, *fixed) - dot(*old, *fixed));
    }
};

static_assert(CellTerm<IsingTerm, IsingDof>);
