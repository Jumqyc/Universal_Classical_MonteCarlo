#pragma once

#include <array>
#include <cassert>
#include <cstddef>

#include <dof/sun.hpp>
#include <term/cell.hpp>

/// SU(N) plaquette interaction. Links are ordered as U0 U1 U2^dagger U3^dagger;
/// the lattice builder is responsible for that order. The energy is
///   S_p = beta * (1 - Re Tr(U_p) / N).
template <std::size_t N>
struct SUPlaquetteTerm
{
    static_assert(N >= 2, "SUPlaquetteTerm needs N >= 2");

    using dtype = SULink<N>;
    using energy_type = double;
    static constexpr std::size_t arity = 4;

    double beta = 1.0;

    /// Re Tr(U_p) / N, the plaquette observable.
    double real_plaquette(const dtype* const* links) const noexcept
    {
        return plaquette(*links[0], *links[1], *links[2], *links[3]);
    }

    energy_type energy(const dtype* const* links) const noexcept
    {
        return beta * (1.0 - real_plaquette(links));
    }

    /// Energy change when links[which] changes from old to nw; the other three
    /// links come from links.
    energy_type delta(std::size_t which, const dtype* /*old*/, const dtype* nw,
                      const dtype* const* links) const noexcept
    {
        assert(which < arity);
        std::array<const dtype*, arity> p{links[0], links[1], links[2], links[3]};
        p[which] = nw;
        const double before = plaquette(*links[0], *links[1], *links[2], *links[3]);
        const double after = plaquette(*p[0], *p[1], *p[2], *p[3]);
        return beta * (before - after);
    }
};

static_assert(CellTerm<SUPlaquetteTerm<2>, SUDof<2>>);
static_assert(CellTerm<SUPlaquetteTerm<3>, SUDof<3>>);
