#pragma once

#include <concepts>
#include <cstddef>

#include <dof/dof.hpp>

/// Local interaction on a cell of arity sites, e.g. a plaquette of four links.
/// The term only knows the cell's sites, not the lattice geometry.
template <class Term, class D>
concept CellTerm = Dof<D> &&
    requires {
        typename Term::dtype;
        typename Term::energy_type;
        { Term::arity } -> std::convertible_to<std::size_t>;
        requires (Term::arity >= 1);
        requires std::same_as<typename Term::dtype, typename D::dtype>;
    } &&
    requires(const Term term,
             const typename D::dtype* const* links,
             const typename D::dtype* old,
             const typename D::dtype* nw) {
        { term.energy(links) } -> std::convertible_to<typename Term::energy_type>;
        { term.delta(std::size_t{}, old, nw, links) } -> std::convertible_to<typename Term::energy_type>;
    };
