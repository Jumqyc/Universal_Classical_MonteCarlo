#pragma once

#include <cstddef>

#include <dof/vector.hpp>

/// Uniform N-component Heisenberg pair interaction: E = -J S_i . S_j.
template <std::size_t N>
struct HeisenbergTerm
{
    using dtype = VectorDof<N>::dtype;
    using energy_type = double;
    static constexpr std::size_t arity = 2;

    double J = 1.0;

    /// Pair energy of the two endpoint states.
    energy_type energy(const dtype* a, const dtype* b) const noexcept
    {
        energy_type dot = 0.0;
        for (std::size_t k = 0; k < N; ++k) dot += a[k] * b[k];
        return -J * dot;
    }

    /// Energy change when one endpoint goes from old to nw; fixed is the other.
    energy_type delta(const dtype* old, const dtype* nw, const dtype* fixed) const noexcept
    {
        energy_type d = 0.0;
        for (std::size_t k = 0; k < N; ++k) d += (nw[k] - old[k]) * fixed[k];
        return -J * d;
    }
};
