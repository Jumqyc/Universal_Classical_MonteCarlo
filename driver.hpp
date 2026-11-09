#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <memory>
#include <utility>

#include <dof/proposal.hpp>
#include <model.hpp>

/// Drives one Markov chain with single-site Metropolis updates. Owns the
/// proposal, the random state and the temperature, so a call site only says how
/// many sweeps it wants.
template <ModelLike M, class Q, class R>
    requires Proposal<Q, typename M::value_type, R>
class Metropolis
{
public:
    Metropolis(M& model, Q proposal, R rng, double beta)
        : model_(model), proposal_(std::move(proposal)), rng_(std::move(rng)),
          beta_(beta)
    {}

    /// One pass over every site: propose, accept with
    /// min(1, exp(-beta*dE) * q(old|new)/q(new|old)), write back.
    void sweep()
    {
        using T = typename M::value_type;
        std::array<T, M::dof> candidate;

        for (std::size_t i = 0; i < model_.sites(); ++i) {
            const T* current = model_.data() + i * M::dof;
            proposal_.sample(current, rng_, candidate.data());

            const double dE    = model_.energy_diff_at(i, candidate.data());
            const double log_p = -beta_ * dE + proposal_.log_ratio(current, candidate.data());
            if (log_p >= 0.0 || std::log(rng_.uni()) < log_p)
                model_.update(i, candidate.data());
        }
    }

    /// n passes over every site.
    void run(std::size_t sweeps)
    {
        for (std::size_t k = 0; k < sweeps; ++k) sweep();
    }

private:
    M&     model_;
    Q      proposal_;
    R      rng_;
    double beta_;
};

/// Drives one Wolff chain. Owns the two arrays the growth needs, so no
/// workspace type is exposed and a call site only says how many flips it wants.
/// Requires a bond-only Hamiltonian: the cluster's internal bonds keep their
/// energy under one common reflection, which is what makes the flip
/// rejection-free; a field term or any non-pair term breaks that.
template <ModelLike M, class A, class R>
class Wolff
{
public:
    Wolff(M& model, A axis, R rng, double beta)
        : model_(model), axis_(std::move(axis)), rng_(std::move(rng)),
          beta_(beta), sites_(model.sites()),
          in_cluster_(new char[sites_]()),
          stack_(new std::size_t[sites_])
    {}

    /// One cluster flip; returns how many sites it flipped.
    std::size_t flip()
    {
        using T = typename M::value_type;
        static constexpr std::size_t D = M::dof;

        std::array<T, D> r;
        axis_.sample(nullptr, rng_, r.data());

        std::fill_n(in_cluster_.get(), sites_, 0);

        const std::size_t seed =
            static_cast<std::size_t>(rng_.uni() * static_cast<double>(sites_));
        in_cluster_[seed] = 1;
        stack_[0]        = seed;
        std::size_t top  = 1;

        std::array<T, D> prop;   // reflected value of the site being grown from
        std::array<T, D> nb;     // reflected value of a candidate neighbour
        std::array<std::size_t, M::max_neighbors> nbrs;

        std::size_t cluster = 0;
        while (top > 0) {
            const std::size_t i = stack_[--top];
            ++cluster;

            const T* s = model_.data() + i * D;
            T dot = 0;
            for (std::size_t k = 0; k < D; ++k) dot += s[k] * r[k];
            for (std::size_t k = 0; k < D; ++k) prop[k] = s[k] - T(2) * dot * r[k];

            // Explore before writing i back: energy_diff_between reads the
            // lattice, and the growth test needs the bond before either end
            // flips.
            const std::size_t deg = model_.neighbors(i, nbrs.data());
            for (std::size_t a = 0; a < deg; ++a) {
                const std::size_t j = nbrs[a];
                if (in_cluster_[j]) continue;

                const T* t = model_.data() + j * D;
                T d = 0;
                for (std::size_t k = 0; k < D; ++k) d += t[k] * r[k];
                for (std::size_t k = 0; k < D; ++k) nb[k] = t[k] - T(2) * d * r[k];

                // p = 1 - exp(-beta*dE), clamped from below at 0: a negative
                // dE gives a negative p and rng.uni() is never negative.
                const double dE = model_.energy_diff_between(j, i, nb.data());
                if (rng_.uni() < 1.0 - std::exp(-beta_ * dE)) {
                    in_cluster_[j] = 1;
                    stack_[top++]  = j;
                }
            }
            model_.update(i, prop.data());
        }

        ++flips_;
        size_sum_ += double(cluster);
        return cluster;
    }

    /// Flips clusters until the flipped sites add up to one lattice, i.e. until
    /// every site has been touched about once. Meant for the critical region:
    /// far above it the clusters are single sites and this costs one buffer
    /// clear per site.
    void sweep()
    {
        std::size_t touched = 0;
        do { touched += flip(); } while (touched < sites_);
    }

    /// n sweeps.
    void run(std::size_t sweeps)
    {
        for (std::size_t k = 0; k < sweeps; ++k) sweep();
    }

    /// Mean cluster size over every flip so far, in sites.
    double mean_cluster_size() const
    {
        return flips_ == 0 ? 0.0 : size_sum_ / double(flips_);
    }

private:
    M&     model_;
    A      axis_;
    R      rng_;
    double beta_;
    const std::size_t sites_;

    std::unique_ptr<char[]>        in_cluster_;
    std::unique_ptr<std::size_t[]> stack_;

    std::size_t flips_    = 0;
    double      size_sum_ = 0.0;
};
