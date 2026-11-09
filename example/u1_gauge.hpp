#pragma once

#include <array>
#include <cstddef>

#include <lattice/cell_complex.hpp>
#include <lattice/hypercubic.hpp>
#include <model/cell.hpp>
#include <term/plaquette.hpp>

/// Pure U(1) lattice gauge theory on a periodic four-dimensional lattice.
///
/// Every link is one site. The model is a CellModel whose cells are the
/// plaquettes: CellComplex supplies which four links form each plaquette and which
/// plaquettes contain a given link, and U1PlaquetteTerm supplies the Wilson
/// energy. The action is
///   S = beta * sum_p (1 - cos theta_p).
class U1Gauge
    : public CellModel<U1Dof, U1PlaquetteTerm, CellComplex<4>>
{
    using Base = CellModel<U1Dof, U1PlaquetteTerm, CellComplex<4>>;

public:
    /// L is the extent in every direction; beta is the inverse coupling.
    explicit U1Gauge(std::size_t L, double beta = 1.0)
        : Base(hypercubic_plaquettes(L), U1PlaquetteTerm{beta}), L_(L)
    {}

    /// Linear extent; all four directions have the same length.
    std::size_t length() const noexcept { return L_; }

    /// Mean of cos(theta_p) over all 6*L^4 plaquettes.
    double mean_plaquette() const noexcept
    {
        double sum = 0.0;
        for (std::size_t c = 0; c < this->cells(); ++c) {
            const auto sites = this->cell_sites(c);
            std::array<const U1Link*, 4> links{};
            for (std::size_t k = 0; k < 4; ++k) links[k] = this->data() + sites[k];
            sum += this->term().cos_plaquette(links.data());
        }
        return sum / static_cast<double>(this->cells());
    }

private:
    std::size_t L_;
};
