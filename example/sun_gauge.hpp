#pragma once

#include <array>
#include <cstddef>
#include <vector>

#include <dof/sun.hpp>
#include <lattice/cell_complex.hpp>
#include <lattice/hypercubic.hpp>

/// Pure SU(N) lattice gauge theory on a periodic four-dimensional lattice.
///
/// The table supplies the four links of every plaquette; the action is written
/// directly with the local plaquette primitive:
///   S = beta * sum_p (1 - Re Tr(U_p) / N).
template <std::size_t N>
class SUGauge
{
public:
    using value_type = SULink<N>;
    static constexpr std::size_t size = 1;

    /// L is the extent in every direction; beta is the inverse coupling.
    explicit SUGauge(std::size_t L, double beta = 1.0)
        : cells_(hypercubic_plaquettes(L)), beta_(beta), L_(L),
          links_(cells_.sites(), SULink<N>::identity())
    {}

    std::size_t sites() const noexcept { return links_.size(); }
    std::size_t cells() const noexcept { return cells_.cells(); }
    std::size_t length() const noexcept { return L_; }

    const SULink<N>* data() const noexcept { return links_.data(); }
    SULink<N>* data() noexcept { return links_.data(); }

    /// Energy change when link i takes p, all other links fixed.
    double energy_diff_at(std::size_t i, const SULink<N>* p) const noexcept
    {
        double d = 0.0;
        for (const auto inc : cells_.incident(i)) {
            const auto s = cells_.cell(inc.cell);
            std::array<const SULink<N>*, 4> links{
                links_.data() + s[0], links_.data() + s[1],
                links_.data() + s[2], links_.data() + s[3]};
            const double before = plaquette(*links[0], *links[1], *links[2], *links[3]);
            links[inc.which] = p;
            const double after = plaquette(*links[0], *links[1], *links[2], *links[3]);
            d += beta_ * (before - after);
        }
        return d;
    }

    void update(std::size_t i, const SULink<N>* p) noexcept { links_[i] = *p; }

    /// Full Wilson action, useful for finite-difference checks.
    double total_energy() const noexcept
    {
        double e = 0.0;
        for (std::size_t c = 0; c < cells_.cells(); ++c)
            e += beta_ * (1.0 - plaquette_at(c));
        return e;
    }

    /// Mean of Re Tr(U_p) / N over all 6*L^4 plaquettes.
    double mean_plaquette() const noexcept
    {
        double sum = 0.0;
        for (std::size_t c = 0; c < cells_.cells(); ++c) sum += plaquette_at(c);
        return sum / static_cast<double>(cells_.cells());
    }

private:
    double plaquette_at(std::size_t c) const noexcept
    {
        const auto s = cells_.cell(c);
        return plaquette(links_[s[0]], links_[s[1]], links_[s[2]], links_[s[3]]);
    }

    CellComplex<4> cells_;
    double beta_;
    std::size_t L_;
    std::vector<SULink<N>> links_;
};

using SU2Gauge = SUGauge<2>;
using SU3Gauge = SUGauge<3>;
