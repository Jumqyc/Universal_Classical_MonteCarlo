#pragma once

#include <algorithm>
#include <cstddef>
#include <utility>
#include <vector>

#include <lattice/graph.hpp>

/// N-component spin on a topology-only bond graph. J is uniform; the graph
/// payload is unused. Ising is kept separate because it fits in int8_t. For
/// Wolff use a random axis (UniformSphereN<N>): a fixed axis is not ergodic
/// for N > 1.
template <std::size_t N>
class VectorSpin
{
public:
    static_assert(N >= 1, "VectorSpin needs at least one component");

    using value_type = double;
    static constexpr std::size_t dof = N;
    using bond_type = Bond<NoPayload>;

    /// Takes ownership of graph and starts every spin along the last axis.
    explicit VectorSpin(CsrGraph<> graph, double J = 1.0)
        : graph_(std::move(graph)), spin_(graph_.sites() * N, 0.0), J_(J)
    {
        for (std::size_t i = 0; i < graph_.sites(); ++i)
            spin_[i * N + (N - 1)] = 1.0;
    }

    std::size_t sites() const noexcept { return graph_.sites(); }
    const double* data() const noexcept { return spin_.data(); }
    double* data() noexcept { return spin_.data(); }

    std::size_t degree(std::size_t i) const noexcept { return graph_.degree(i); }
    const bond_type* bonds(std::size_t i) const noexcept { return graph_.bonds(i); }

    /// Energy of all bonds incident to i, each counted once.
    double energy_at(std::size_t i) const noexcept
    {
        double e = 0.0;
        const auto* bs = graph_.bonds(i);
        const double* si = spin_.data() + i * N;
        for (std::size_t a = 0; a < graph_.degree(i); ++a) {
            const double* sj = spin_.data() + bs[a].j * N;
            double dot = 0.0;
            for (std::size_t k = 0; k < N; ++k) dot += si[k] * sj[k];
            e += -J_ * dot;
        }
        return e;
    }

    /// Last spin component, so the scalar order-parameter interface works.
    double order_param_at(std::size_t i) const noexcept
    {
        return spin_[i * N + (N - 1)];
    }

    double energy_diff_at(std::size_t i, const double* p) const noexcept
    {
        double d = 0.0;
        const auto* bs = graph_.bonds(i);
        const double* si = spin_.data() + i * N;
        for (std::size_t a = 0; a < graph_.degree(i); ++a) {
            const double* sj = spin_.data() + bs[a].j * N;
            for (std::size_t k = 0; k < N; ++k) d += (p[k] - si[k]) * sj[k];
        }
        return -J_ * d;
    }

    /// Energy change when b.j takes p, with i fixed.
    double energy_diff_bond(std::size_t i, const bond_type& b, const double* p) const noexcept
    {
        const double* si = spin_.data() + i * N;
        const double* sj = spin_.data() + b.j * N;
        double d = 0.0;
        for (std::size_t k = 0; k < N; ++k) d += (p[k] - sj[k]) * si[k];
        return -J_ * d;
    }

    void update(std::size_t i, const double* p) noexcept
    {
        std::copy_n(p, N, spin_.data() + i * N);
    }

private:
    CsrGraph<> graph_;
    std::vector<double> spin_;
    double J_;
};

using XY         = VectorSpin<2>;
using Heisenberg = VectorSpin<3>;
