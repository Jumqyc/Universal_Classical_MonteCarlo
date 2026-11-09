#pragma once

#include <array>
#include <cassert>
#include <cstddef>

#include <dof/u1.hpp>
#include <term/cell.hpp>

/// U(1) plaquette interaction. The four links are ordered so that the oriented
/// plaquette angle is theta0 + theta1 - theta2 - theta3; the lattice builder is
/// responsible for that order.
struct U1PlaquetteTerm
{
    using dtype = U1Dof::dtype;
    using energy_type = double;
    static constexpr std::size_t arity = 4;

    double beta = 1.0;

    /// S_p = beta * (1 - cos(theta_p)).
    energy_type energy(const dtype* const* links) const noexcept
    {
        return beta * (1.0 - cos_plaquette(links));
    }

    /// cos of the oriented plaquette angle, useful for the mean plaquette.
    double cos_plaquette(const dtype* const* links) const noexcept
    {
        return plaquette(links[0]->theta, links[1]->theta, links[2]->theta,
                         links[3]->theta);
    }

    /// Energy change when links[which] changes from old to nw; other links are
    /// read from links.
    energy_type delta(std::size_t which, const dtype* /*old*/, const dtype* nw,
                      const dtype* const* links) const noexcept
    {
        assert(which < arity);
        std::array<const dtype*, arity> p{links[0], links[1], links[2], links[3]};
        p[which] = nw;
        const double before = plaquette(links[0]->theta, links[1]->theta,
                                        links[2]->theta, links[3]->theta);
        const double after = plaquette(p[0]->theta, p[1]->theta, p[2]->theta,
                                       p[3]->theta);
        return beta * (before - after);
    }
};

static_assert(CellTerm<U1PlaquetteTerm, U1Dof>);
