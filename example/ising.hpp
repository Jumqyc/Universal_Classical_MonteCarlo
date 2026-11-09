#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>

#include <lattice/graph.hpp>
#include <term/ising_term.hpp>
#include <term/pair_model.hpp>

/// Ising model on a topology-only bond graph: the generic pair model with the
/// Ising dof and a uniform Ising term.
class Ising : public PairModel<IsingDof, IsingTerm, CsrGraph<>>
{
    using Base = PairModel<IsingDof, IsingTerm, CsrGraph<>>;

public:
    /// Takes ownership of graph and starts every spin up.
    explicit Ising(CsrGraph<> graph, std::int32_t J = 1)
        : Base(std::move(graph), IsingTerm{J})
    {
        std::fill(this->data(), this->data() + this->sites(), IsingSpin(1));
    }

    /// Convenience magnetization density component at i.
    IsingSpin order_param_at(std::size_t i) const noexcept { return data()[i]; }
};
