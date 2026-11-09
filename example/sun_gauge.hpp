#pragma once

#include <array>
#include <cstddef>

#include <dof/sun.hpp>
#include <lattice/cell_complex.hpp>
#include <lattice/hypercubic.hpp>
#include <model/cell.hpp>
#include <term/sun_plaquette.hpp>

/// Pure SU(N) lattice gauge theory on a periodic four-dimensional lattice.
///
/// Every link is one site. The model is a CellModel whose cells are the
/// plaquettes; SUPlaquetteTerm supplies the Wilson energy
///   S = beta * sum_p (1 - Re Tr(U_p) / N).
template <std::size_t N>
class SUGauge
    : public CellModel<SUDof<N>, SUPlaquetteTerm<N>, CellComplex<4>>
{
    using Base = CellModel<SUDof<N>, SUPlaquetteTerm<N>, CellComplex<4>>;

public:
    /// L is the extent in every direction; beta is the inverse coupling.
    explicit SUGauge(std::size_t L, double beta = 1.0)
        : Base(hypercubic_plaquettes(L), SUPlaquetteTerm<N>{beta}), L_(L)
    {
        const auto id = SULink<N>::identity();
        for (std::size_t i = 0; i < this->sites(); ++i) this->update(i, &id);
    }

    /// Linear extent; all four directions have the same length.
    std::size_t length() const noexcept { return L_; }

    /// Mean of Re Tr(U_p) / N over all 6*L^4 plaquettes.
    double mean_plaquette() const noexcept
    {
        double sum = 0.0;
        for (std::size_t c = 0; c < this->cells(); ++c) {
            const auto sites = this->cell_sites(c);
            std::array<const SULink<N>*, 4> links{};
            for (std::size_t k = 0; k < 4; ++k) links[k] = this->data() + sites[k];
            sum += this->term().real_plaquette(links.data());
        }
        return sum / static_cast<double>(this->cells());
    }

private:
    std::size_t L_;
};

using SU2Gauge = SUGauge<2>;
using SU3Gauge = SUGauge<3>;
