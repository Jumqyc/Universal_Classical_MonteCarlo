#pragma once

#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

#include <dof/dof.hpp>
#include <term/cell_term.hpp>

/// A cell collection that exposes, for every site, the cells containing it and
/// the site's position in each cell. CellList satisfies it.
template <class L>
concept CellListLike = requires(const L list, std::size_t i, std::size_t c) {
    { L::arity } -> std::convertible_to<std::size_t>;
    { list.sites() } -> std::convertible_to<std::size_t>;
    { list.cells() } -> std::convertible_to<std::size_t>;
    { list.cell(c) } -> std::convertible_to<std::span<const std::uint32_t, L::arity>>;
    { list.incident(i) } -> std::convertible_to<std::span<const typename L::Incident>>;
};

/// A model with one cell interaction per cell. The state is AoS; the cell list
/// supplies geometry and incidence, the term supplies the local energy. There
/// is deliberately no per-site energy: a plaquette energy belongs to the cell,
/// not to one link.
template <class D, class Term, class List>
    requires Dof<D> && CellTerm<Term, D> && CellListLike<List> &&
             (Term::arity == List::arity)
class CellModel
{
public:
    using dtype = typename D::dtype;
    using value_type = dtype;
    using energy_type = typename Term::energy_type;
    static constexpr std::size_t size = D::size;
    static constexpr std::size_t arity = List::arity;

    /// Takes ownership of the cell list and term; every component starts at zero.
    CellModel(List list, Term term)
        : list_(std::move(list)), term_(std::move(term)),
          field_(list_.sites() * size)
    {}

    std::size_t sites() const noexcept { return list_.sites(); }
    std::size_t cells() const noexcept { return list_.cells(); }

    const dtype* data() const noexcept { return field_.data(); }
    dtype* data() noexcept { return field_.data(); }

    const Term& term() const noexcept { return term_; }

    std::span<const std::uint32_t, arity> cell_sites(std::size_t c) const
    {
        return list_.cell(c);
    }

    /// Energy of one cell, e.g. one plaquette.
    energy_type cell_energy(std::size_t c) const
    {
        const auto ls = links(c);
        return term_.energy(ls.data());
    }

    /// Sum over all cells; each cell is counted once.
    energy_type total_energy() const
    {
        energy_type e = 0;
        for (std::size_t c = 0; c < cells(); ++c) e += cell_energy(c);
        return e;
    }

    /// Energy change when site i takes p, all other sites fixed. Only the cells
    /// containing i can change.
    energy_type energy_diff_at(std::size_t i, const dtype* p) const
    {
        energy_type d = 0;
        const dtype* old = field_.data() + i * size;
        std::array<const dtype*, arity> ls{};
        for (const auto inc : list_.incident(i)) {
            const auto sites = list_.cell(inc.cell);
            for (std::size_t k = 0; k < arity; ++k)
                ls[k] = field_.data() + sites[k] * size;
            d += term_.delta(inc.which, old, p, ls.data());
        }
        return d;
    }

    void update(std::size_t i, const dtype* p) noexcept
    {
        for (std::size_t k = 0; k < size; ++k) field_[i * size + k] = p[k];
    }

private:
    /// Builds the A link pointers of cell c.
    std::array<const dtype*, arity> links(std::size_t c) const
    {
        const auto sites = list_.cell(c);
        std::array<const dtype*, arity> ls{};
        for (std::size_t k = 0; k < arity; ++k)
            ls[k] = field_.data() + sites[k] * size;
        return ls;
    }

    List list_;
    Term term_;
    std::vector<dtype> field_;
};
