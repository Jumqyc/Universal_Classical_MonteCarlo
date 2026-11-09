#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include <dof/ising.hpp>
#include <lattice/pair_graph.hpp>

/// Ising model written directly against the topology table: E = -J sum_<ij> s_i s_j.
/// The graph only supplies the bonds; the Hamiltonian and dE are ordinary C++.
class Ising
{
public:
    using value_type = IsingSpin;
    using energy_type = std::int32_t;
    using bond_type = PairGraph<>::bond_type;
    static constexpr std::size_t size = 1;

    /// Takes ownership of the graph and starts every spin up.
    explicit Ising(PairGraph<> graph, std::int32_t J = 1)
        : graph_(std::move(graph)), J_(J), spins_(graph_.sites(), IsingSpin(1))
    {}

    std::size_t sites() const noexcept { return spins_.size(); }
    const IsingSpin* data() const noexcept { return spins_.data(); }
    IsingSpin* data() noexcept { return spins_.data(); }

    std::size_t degree(std::size_t i) const noexcept { return graph_.degree(i); }
    const bond_type* bonds(std::size_t i) const noexcept { return graph_.bonds(i); }

    /// Energy of all bonds incident to i, each counted once.
    energy_type energy_at(std::size_t i) const noexcept
    {
        const IsingSpin si = spins_[i];
        energy_type e = 0;
        const bond_type* bs = graph_.bonds(i);
        for (std::size_t a = 0; a < graph_.degree(i); ++a)
            e += -J_ * dot(si, spins_[bs[a].j]);
        return e;
    }

    /// Energy change when i takes p, all other sites fixed.
    energy_type energy_diff_at(std::size_t i, const IsingSpin* p) const noexcept
    {
        const IsingSpin old = spins_[i];
        energy_type d = 0;
        const bond_type* bs = graph_.bonds(i);
        for (std::size_t a = 0; a < graph_.degree(i); ++a) {
            const IsingSpin sj = spins_[bs[a].j];
            d += -J_ * (dot(*p, sj) - dot(old, sj));
        }
        return d;
    }

    /// Energy change when bond.j takes p, with i fixed.
    energy_type energy_diff_bond(std::size_t i, const bond_type& bond,
                                 const IsingSpin* p) const noexcept
    {
        const IsingSpin fixed = spins_[i];
        const IsingSpin old = spins_[bond.j];
        return -J_ * (dot(*p, fixed) - dot(old, fixed));
    }

    void update(std::size_t i, const IsingSpin* p) noexcept { spins_[i] = *p; }

    /// Convenience magnetization component at i.
    IsingSpin order_param_at(std::size_t i) const noexcept { return spins_[i]; }

private:
    PairGraph<> graph_;
    std::int32_t J_;
    std::vector<IsingSpin> spins_;
};
