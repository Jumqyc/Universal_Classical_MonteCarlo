#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <utility>
#include <vector>

#include <dof/dof.hpp>
#include <lattice/pair_graph.hpp>
#include <term/cell.hpp>

/// A model built from a degree of freedom, one pair interaction and a bond
/// graph. A pair is a cell of arity 2: the term sees the two endpoint states in
/// a two-pointer array, not a special pair API. The graph supplies geometry and
/// per-bond payload; bonds()/degree() remain exposed because Wolff needs them.
template <class D, class Term, class G>
    requires Dof<D> && CellTerm<Term, D> && PairGraphLike<G> && (Term::arity == 2)
class PairModel
{
public:
    using dtype = typename D::dtype;
    using value_type = dtype;
    using energy_type = typename Term::energy_type;
    static constexpr std::size_t size = D::size;
    using bond_type = typename G::bond_type;

    /// Takes ownership of graph and term; every component starts at zero.
    PairModel(G graph, Term term)
        : graph_(std::move(graph)), term_(std::move(term)), field_(graph_.sites() * size)
    {}

    std::size_t sites() const noexcept { return graph_.sites(); }
    const dtype* data() const noexcept { return field_.data(); }
    dtype* data() noexcept { return field_.data(); }

    std::size_t degree(std::size_t i) const noexcept { return graph_.degree(i); }
    const bond_type* bonds(std::size_t i) const noexcept { return graph_.bonds(i); }

    /// Energy of all bonds incident to i, each counted once.
    energy_type energy_at(std::size_t i) const noexcept
    {
        energy_type e = 0;
        const dtype* si = field_.data() + i * size;
        const bond_type* bs = graph_.bonds(i);
        for (std::size_t a = 0; a < graph_.degree(i); ++a) {
            const std::array<const dtype*, 2> links{si, field_.data() + bs[a].j * size};
            e += term_.energy(links.data());
        }
        return e;
    }

    /// Energy change when i takes p, all other sites fixed.
    energy_type energy_diff_at(std::size_t i, const dtype* p) const noexcept
    {
        energy_type d = 0;
        const dtype* si = field_.data() + i * size;
        const bond_type* bs = graph_.bonds(i);
        for (std::size_t a = 0; a < graph_.degree(i); ++a) {
            const std::array<const dtype*, 2> links{si, field_.data() + bs[a].j * size};
            d += term_.delta(0, si, p, links.data());
        }
        return d;
    }

    /// Energy change when b.j takes p, with i fixed.
    energy_type energy_diff_bond(std::size_t i, const bond_type& b, const dtype* p) const noexcept
    {
        const dtype* sj = field_.data() + b.j * size;
        const std::array<const dtype*, 2> links{sj, field_.data() + i * size};
        return term_.delta(0, sj, p, links.data());
    }

    void update(std::size_t i, const dtype* p) noexcept
    {
        std::copy_n(p, size, field_.data() + i * size);
    }

private:
    G graph_;
    Term term_;
    std::vector<dtype> field_;
};
