#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include <lattice/cell_list.hpp>

/// Periodic four-dimensional hypercubic lattice. Sites are oriented links,
/// indexed mu*L^4 + linear(c); cells are the 6*L^4 plaquettes, ordered so the
/// oriented angle is theta_mu(c) + theta_nu(c+e_mu) - theta_mu(c+e_nu) - theta_nu(c).
[[nodiscard]] inline CellList<4> hypercubic_plaquettes(std::size_t L)
{
    constexpr std::size_t max = std::numeric_limits<std::size_t>::max();
    if (L == 0)
        throw std::invalid_argument("hypercubic_plaquettes: L must be positive");

    std::size_t L4 = 1;
    for (int k = 0; k < 4; ++k) {
        if (L4 > max / L)
            throw std::overflow_error("hypercubic_plaquettes: L^4 overflows size_t");
        L4 *= L;
    }
    if (L4 > max / 4)
        throw std::overflow_error("hypercubic_plaquettes: 4*L^4 overflows size_t");
    if (L4 > max / 6)
        throw std::overflow_error("hypercubic_plaquettes: 6*L^4 overflows size_t");

    const std::size_t sites = 4 * L4;
    constexpr std::size_t max32 = std::numeric_limits<std::uint32_t>::max();
    if (sites > max32)
        throw std::overflow_error("hypercubic_plaquettes: site count does not fit uint32_t");

    const auto index = [L, L4](std::size_t mu, const std::array<std::size_t, 4>& c) {
        return mu * L4 + (((c[0] * L + c[1]) * L + c[2]) * L + c[3]);
    };

    std::vector<Cell<4>> cells;
    cells.reserve(6 * L4);

    for (std::size_t mu = 0; mu < 4; ++mu)
        for (std::size_t nu = mu + 1; nu < 4; ++nu)
            for (std::size_t t = 0; t < L; ++t)
                for (std::size_t z = 0; z < L; ++z)
                    for (std::size_t y = 0; y < L; ++y)
                        for (std::size_t x = 0; x < L; ++x) {
                            const std::array<std::size_t, 4> c{t, z, y, x};
                            auto cm = c;
                            auto cn = c;
                            cm[mu] = (cm[mu] + 1) % L;
                            cn[nu] = (cn[nu] + 1) % L;

                            cells.push_back(Cell<4>{{
                                static_cast<std::uint32_t>(index(mu, c)),
                                static_cast<std::uint32_t>(index(nu, cm)),
                                static_cast<std::uint32_t>(index(mu, cn)),
                                static_cast<std::uint32_t>(index(nu, c))}});
                        }

    return CellList<4>(sites, cells);
}
