#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include <lattice/pair_graph.hpp>

/// Builds a periodic Lx-by-Ly square lattice with site id y*Lx+x. One edge is
/// emitted per horizontal and vertical bond; PairGraph stores both directions.
/// Lx==1 drops horizontal bonds and Ly==1 drops vertical bonds. Lx==2 or Ly==2
/// keeps both periodic images, so those bonds appear twice.
[[nodiscard]] inline PairGraph<> square_lattice(std::size_t Lx, std::size_t Ly)
{
    if (Lx == 0 || Ly == 0)
        throw std::invalid_argument("square_lattice: Lx and Ly must be positive");
    if (Lx > std::numeric_limits<std::size_t>::max() / Ly)
        throw std::overflow_error("square_lattice: site count overflows size_t");

    const std::size_t n = Lx * Ly;
    if (n > std::numeric_limits<std::uint32_t>::max())
        throw std::overflow_error("square_lattice: site count exceeds uint32_t");

    std::vector<PairGraph<>::Edge> edges;
    edges.reserve(2 * n);
    for (std::size_t y = 0; y < Ly; ++y) {
        for (std::size_t x = 0; x < Lx; ++x) {
            const auto i = static_cast<std::uint32_t>(y * Lx + x);
            if (Lx > 1)
                edges.push_back({i, static_cast<std::uint32_t>(y * Lx + (x + 1) % Lx)});
            if (Ly > 1)
                edges.push_back({i, static_cast<std::uint32_t>(((y + 1) % Ly) * Lx + x)});
        }
    }
    return PairGraph<>(n, edges);
}
