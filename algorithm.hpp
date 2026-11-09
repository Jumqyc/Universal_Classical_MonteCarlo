#pragma once

#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <vector>

#include <dof/proposal.hpp>

/// One single-site Metropolis sweep: visits every site in order, proposes a new
/// value, accepts it with min(1, exp(-beta*dE) * q(old|new)/q(new|old)), and
/// writes back only on acceptance. The comparison is done in log space so a
/// large dE cannot underflow to a wrong decision.
template <class M, class Q, class R>
    requires Proposal<Q, typename M::value_type, R>
void local_update(M& model, const Q& proposal, R& rng, double beta)
{
    using T = typename M::value_type;
    std::array<T, M::dof> candidate;

    for (std::size_t i = 0; i < model.sites(); ++i) {
        const T* current = model.data() + i * M::dof;
        proposal.sample(current, rng, candidate.data());

        const double dE    = model.energy_diff_at(i, candidate.data());
        const double log_p = -beta * dE + proposal.log_ratio(current, candidate.data());
        if (log_p >= 0.0 || std::log(rng.uni()) < log_p)
            model.update(i, candidate.data());
    }
}

/// Scratch space for wolff_update, allocated once per lattice and reused across
/// calls. Kept outside the update because one sweep issues up to one call per
/// site, and allocating two vectors per call would dominate the cost.
struct WolffWorkspace
{
    std::vector<char>        in_cluster;
    std::vector<std::size_t> stack;

    /// Allocates for a lattice with the given number of sites.
    explicit WolffWorkspace(std::size_t sites)
        : in_cluster(sites, 0)
    {
        stack.reserve(sites);
    }
};

/// One Wolff cluster flip: grows a cluster from a random seed by reflecting
/// spins about a random axis, then flips the whole cluster at once. Returns the
/// number of sites flipped. There is no global accept/reject step: a bond
/// inside the cluster keeps its energy under one common reflection, so only the
/// boundary changes and the growth probabilities already satisfy detailed
/// balance.
///
/// Requires a bond-only Hamiltonian. A field term, or any term that is not a
/// pair interaction, makes the boundary argument invalid and the chain
/// converges to the wrong distribution.
///
/// The axis must be a unit vector of M::dof components and axis.sample(nullptr,
/// rng, out) must ignore cur, which every uniform sampler on the sphere does.
template <class M, class A, class R>
std::size_t wolff_update(M& model, const A& axis, R& rng, double beta,
                         WolffWorkspace& ws)
{
    using T = typename M::value_type;
    static constexpr std::size_t D = M::dof;
    const std::size_t n = model.sites();

    std::array<T, D> r;
    axis.sample(nullptr, rng, r.data());

    std::fill(ws.in_cluster.begin(), ws.in_cluster.end(), 0);
    ws.stack.clear();

    const std::size_t seed =
        static_cast<std::size_t>(rng.uni() * static_cast<double>(n));
    ws.in_cluster[seed] = 1;
    ws.stack.push_back(seed);

    std::array<T, D> prop;   // reflected value of the site being grown from
    std::array<T, D> nb;     // reflected value of a candidate neighbour
    std::array<std::size_t, M::max_neighbors> nbrs;

    std::size_t cluster = 0;
    while (!ws.stack.empty()) {
        const std::size_t i = ws.stack.back();
        ws.stack.pop_back();
        ++cluster;

        const T* s = model.data() + i * D;
        T dot = 0;
        for (std::size_t k = 0; k < D; ++k) dot += s[k] * r[k];
        for (std::size_t k = 0; k < D; ++k) prop[k] = s[k] - T(2) * dot * r[k];

        // Explore before writing i back: energy_diff_between reads the lattice,
        // and the growth test needs the bond as it is before either end flips.
        const std::size_t deg = model.neighbors(i, nbrs.data());
        assert(deg <= M::max_neighbors);
        for (std::size_t a = 0; a < deg; ++a) {
            const std::size_t j = nbrs[a];
            if (ws.in_cluster[j]) continue;

            const T* t = model.data() + j * D;
            T d = 0;
            for (std::size_t k = 0; k < D; ++k) d += t[k] * r[k];
            for (std::size_t k = 0; k < D; ++k) nb[k] = t[k] - T(2) * d * r[k];

            // p = 1 - exp(-beta*dE), clamped from below at 0: a negative dE
            // gives a negative p, and rng.uni() is never negative.
            const double dE = model.energy_diff_between(j, i, nb.data());
            if (rng.uni() < 1.0 - std::exp(-beta * dE)) {
                ws.in_cluster[j] = 1;
                ws.stack.push_back(j);
            }
        }
        model.update(i, prop.data());
    }
    return cluster;
}
