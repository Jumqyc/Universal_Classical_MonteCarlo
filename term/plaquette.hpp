#pragma once

#include <cassert>
#include <cmath>
#include <cstddef>

#include <term/cell.hpp>
#include <dof/u1.hpp>

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
        return std::cos(angle(links));
    }

    /// Energy change when links[which] changes from old to nw; other links are
    /// read from links.
    energy_type delta(std::size_t which,
                      const dtype* old,
                      const dtype* nw,
                      const dtype* const* links) const noexcept
    {
        assert(which < arity);
        const double before = angle_with(links, which, old);
        const double after = before + sign(which) * (nw->theta - old->theta);
        return beta * (std::cos(before) - std::cos(after));
    }

private:
    static constexpr int sign(std::size_t which) noexcept
    {
        return which < 2 ? 1 : -1;
    }

    static double angle_with(const dtype* const* links, std::size_t which,
                             const dtype* value) noexcept
    {
        double theta = 0.0;
        for (std::size_t k = 0; k < arity; ++k)
            theta += sign(k) * (k == which ? value->theta : links[k]->theta);
        return theta;
    }

    static double angle(const dtype* const* links) noexcept
    {
        double theta = 0.0;
        for (std::size_t k = 0; k < arity; ++k) theta += sign(k) * links[k]->theta;
        return theta;
    }
};

static_assert(CellTerm<U1PlaquetteTerm, U1Dof>);
