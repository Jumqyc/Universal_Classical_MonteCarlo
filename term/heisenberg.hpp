#pragma once

#include <cassert>
#include <cstddef>

#include <dof/vector.hpp>
#include <term/cell.hpp>

/// Uniform N-component Heisenberg pair interaction: E = -J S_i . S_j. A pair is
/// a cell of arity 2, so this satisfies CellTerm.
template <std::size_t N>
struct HeisenbergTerm
{
    using dtype = VectorDof<N>::dtype;
    using energy_type = double;
    static constexpr std::size_t arity = 2;

    double J = 1.0;

    /// Pair energy of links[0] . links[1].
    energy_type energy(const dtype* const* links) const noexcept
    {
        return -J * dot<N>(links[0], links[1]);
    }

    /// Energy change when links[which] goes from old to nw; the other is fixed.
    energy_type delta(std::size_t which, const dtype* old, const dtype* nw,
                      const dtype* const* links) const noexcept
    {
        assert(which < arity);
        const dtype* fixed = links[which == 0 ? 1 : 0];
        return -J * (dot<N>(nw, fixed) - dot<N>(old, fixed));
    }
};

static_assert(CellTerm<HeisenbergTerm<3>, VectorDof<3>>);
