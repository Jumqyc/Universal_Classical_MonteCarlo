#pragma once

#include <algorithm>
#include <cstddef>
#include <utility>
#include <vector>

#include <dof/vector.hpp>
#include <lattice/pair_graph.hpp>

/// N-component spin written directly against the topology table:
/// E = -J sum_<ij> S_i . S_j. For Wolff use a random axis; a fixed axis is not
/// ergodic for N > 1.
template <std::size_t N>
class VectorSpin
{
public:
    using value_type = double;
    using energy_type = double;
    using bond_type = PairGraph<>::bond_type;
    static constexpr std::size_t size = N;

    /// Takes ownership of the graph and starts every spin along the last axis.
    explicit VectorSpin(PairGraph<> graph, double J = 1.0)
        : graph_(std::move(graph)), J_(J), spins_(graph_.sites() * N, 0.0)
    {
        for (std::size_t i = 0; i < graph_.sites(); ++i)
            spins_[i * N + (N - 1)] = 1.0;
    }

    std::size_t sites() const noexcept { return graph_.sites(); }
    const double* data() const noexcept { return spins_.data(); }
    double* data() noexcept { return spins_.data(); }

    std::size_t degree(std::size_t i) const noexcept { return graph_.degree(i); }
    const bond_type* bonds(std::size_t i) const noexcept { return graph_.bonds(i); }

    /// Energy of all bonds incident to i, each counted once.
    energy_type energy_at(std::size_t i) const noexcept
    {
        const double* si = spins_.data() + i * N;
        energy_type e = 0.0;
        const bond_type* bs = graph_.bonds(i);
        for (std::size_t a = 0; a < graph_.degree(i); ++a)
            e += -J_ * dot<N>(si, spins_.data() + bs[a].j * N);
        return e;
    }

    /// Energy change when i takes p, all other sites fixed.
    energy_type energy_diff_at(std::size_t i, const double* p) const noexcept
    {
        const double* old = spins_.data() + i * N;
        energy_type d = 0.0;
        const bond_type* bs = graph_.bonds(i);
        for (std::size_t a = 0; a < graph_.degree(i); ++a) {
            const double* sj = spins_.data() + bs[a].j * N;
            d += -J_ * (dot<N>(p, sj) - dot<N>(old, sj));
        }
        return d;
    }

    /// Energy change when bond.j takes p, with i fixed.
    energy_type energy_diff_bond(std::size_t i, const bond_type& bond,
                                 const double* p) const noexcept
    {
        const double* fixed = spins_.data() + i * N;
        const double* old = spins_.data() + bond.j * N;
        return -J_ * (dot<N>(p, fixed) - dot<N>(old, fixed));
    }

    void update(std::size_t i, const double* p) noexcept
    {
        std::copy_n(p, N, spins_.data() + i * N);
    }

private:
    PairGraph<> graph_;
    double J_;
    std::vector<double> spins_;
};

using XY         = VectorSpin<2>;
using Heisenberg = VectorSpin<3>;
