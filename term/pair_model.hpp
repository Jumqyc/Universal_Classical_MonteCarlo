#pragma once

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <utility>
#include <vector>

#include <dof/dof.hpp>

/// A graph that exposes one row of bonds per site, with the bond payload part
/// of the row type. CsrGraph satisfies it.
template <class G>
concept BondGraph = requires(const G g, std::size_t i) {
    typename G::bond_type;
    { g.sites() } -> std::convertible_to<std::size_t>;
    { g.degree(i) } -> std::convertible_to<std::size_t>;
    { g.bonds(i) } -> std::convertible_to<const typename G::bond_type*>;
};

/// Local pair interaction over a degree of freedom: the pair energy and the
/// energy change when one endpoint moves, the other held fixed.
template <class Term, class D>
concept PairTerm = Dof<D> &&
    requires {
        typename Term::dtype;
        typename Term::energy_type;
        { Term::arity } -> std::convertible_to<std::size_t>;
        requires (Term::arity == 2);
        requires std::same_as<typename Term::dtype, typename D::dtype>;
    } &&
    requires(const Term term,
             const typename D::dtype* old,
             const typename D::dtype* nw,
             const typename D::dtype* fixed) {
        { term.energy(old, fixed) } -> std::convertible_to<typename Term::energy_type>;
        { term.delta(old, nw, fixed) } -> std::convertible_to<typename Term::energy_type>;
    };

/// A model built from a degree of freedom, one pair interaction and a bond
/// graph. It owns the state and delegates the local physics to the term; the
/// graph supplies geometry and per-bond payload.
template <class D, class Term, class G>
    requires Dof<D> && PairTerm<Term, D> && BondGraph<G>
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
        for (std::size_t a = 0; a < graph_.degree(i); ++a)
            e += term_.energy(si, field_.data() + bs[a].j * size);
        return e;
    }

    /// Energy change when i takes p, all other sites fixed.
    energy_type energy_diff_at(std::size_t i, const dtype* p) const noexcept
    {
        energy_type d = 0;
        const dtype* si = field_.data() + i * size;
        const bond_type* bs = graph_.bonds(i);
        for (std::size_t a = 0; a < graph_.degree(i); ++a)
            d += term_.delta(si, p, field_.data() + bs[a].j * size);
        return d;
    }

    /// Energy change when b.j takes p, with i fixed.
    energy_type energy_diff_bond(std::size_t i, const bond_type& b, const dtype* p) const noexcept
    {
        return term_.delta(field_.data() + b.j * size, p, field_.data() + i * size);
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
