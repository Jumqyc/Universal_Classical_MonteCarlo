#pragma once

#include <cstddef>
#include <utility>

#include <lattice/pair_graph.hpp>
#include <term/heisenberg.hpp>
#include <model/pair.hpp>

/// N-component spin on a topology-only bond graph: the generic pair model with
/// the vector dof and a uniform Heisenberg term. For Wolff use a random axis
/// (UniformSphereN<N>); a fixed axis is not ergodic for N > 1.
template <std::size_t N>
class VectorSpin : public PairModel<VectorDof<N>, HeisenbergTerm<N>, PairGraph<>>
{
    using Base = PairModel<VectorDof<N>, HeisenbergTerm<N>, PairGraph<>>;

public:
    /// Takes ownership of graph and starts every spin along the last axis.
    explicit VectorSpin(PairGraph<> graph, double J = 1.0)
        : Base(std::move(graph), HeisenbergTerm<N>{J})
    {
        for (std::size_t i = 0; i < this->sites(); ++i)
            this->data()[i * N + (N - 1)] = 1.0;
    }
};

using XY         = VectorSpin<2>;
using Heisenberg = VectorSpin<3>;
