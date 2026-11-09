#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>

#include <lattice/pair_graph.hpp>
#include <term/ising.hpp>
#include <model/pair.hpp>

/// Ising model on a topology-only bond graph: the generic pair model with the
/// Ising dof and a uniform Ising term.
class Ising : public PairModel<IsingDof, IsingTerm, PairGraph<>>
{
    using Base = PairModel<IsingDof, IsingTerm, PairGraph<>>;

public:
    /// Takes ownership of graph and starts every spin up.
    explicit Ising(PairGraph<> graph, std::int32_t J = 1)
        : Base(std::move(graph), IsingTerm{J})
    {
        std::fill(this->data(), this->data() + this->sites(), IsingSpin(1));
    }

    /// Convenience magnetization density component at i.
    IsingSpin order_param_at(std::size_t i) const noexcept { return data()[i]; }
};
