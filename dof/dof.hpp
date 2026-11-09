#pragma once

#include <concepts>
#include <cstddef>

/// Representation of a degree of freedom: its storage type and how many
/// components one site holds. Nothing here presumes an interaction; a gauge
/// link and a spin both satisfy this.
template <class D>
concept Dof = requires {
    typename D::dtype;
    { D::size } -> std::convertible_to<std::size_t>;
};
