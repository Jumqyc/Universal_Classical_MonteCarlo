#pragma once

#include <cstddef>
#include <utility>
#include <vector>

#include <dof/ising.hpp>
#include <lattice/graph.hpp>

/// Ising model on a topology-only bond graph. J is uniform; the graph payload
/// is unused.
class Ising
{
public:
    using value_type = IsingSpin;
    static constexpr std::size_t dof = 1;
    using bond_type = Bond<NoPayload>;

    /// Takes ownership of graph and starts every spin up.
    explicit Ising(CsrGraph<> graph, double J = 1.0)
        : graph_(std::move(graph)), spin_(graph_.sites(), IsingSpin(1)), J_(J)
    {}

    std::size_t sites() const noexcept { return graph_.sites(); }
    const IsingSpin* data() const noexcept { return spin_.data(); }
    IsingSpin* data() noexcept { return spin_.data(); }

    std::size_t degree(std::size_t i) const noexcept { return graph_.degree(i); }
    const bond_type* bonds(std::size_t i) const noexcept { return graph_.bonds(i); }

    /// Energy of all bonds incident to i, each counted once.
    double energy_at(std::size_t i) const noexcept
    {
        double e = 0.0;
        const auto* bs = graph_.bonds(i);
        for (std::size_t a = 0; a < graph_.degree(i); ++a)
            e += -J_ * double(spin_[i]) * double(spin_[bs[a].j]);
        return e;
    }

    IsingSpin order_param_at(std::size_t i) const noexcept { return spin_[i]; }

    double energy_diff_at(std::size_t i, const IsingSpin* p) const noexcept
    {
        double d = 0.0;
        const auto* bs = graph_.bonds(i);
        for (std::size_t a = 0; a < graph_.degree(i); ++a)
            d += -J_ * double(*p - spin_[i]) * double(spin_[bs[a].j]);
        return d;
    }

    /// Energy change when b.j takes p, with i fixed.
    double energy_diff_bond(std::size_t i, const bond_type& b, const IsingSpin* p) const noexcept
    {
        return -J_ * double(*p - spin_[b.j]) * double(spin_[i]);
    }

    void update(std::size_t i, const IsingSpin* p) noexcept { spin_[i] = *p; }

private:
    CsrGraph<> graph_;
    std::vector<IsingSpin> spin_;
    double J_;
};
