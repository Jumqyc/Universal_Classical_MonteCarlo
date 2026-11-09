#pragma once

#include <concepts>

/// A proposal supplies one candidate value and the log density ratio
/// log q(old|new) - log q(new|old).
/// Symmetric proposals (uniform sampling, rotation about a random axis) return 0.
/// A biased proposal must return the true ratio: otherwise detailed balance
/// breaks, the chain converges to the wrong distribution, and nothing reports it.
template <class Q, class T, class R>
concept Proposal = requires(const Q& q, const T* cur, T* out, R& rng) {
    { q.sample(cur, rng, out) };
    { q.log_ratio(cur, out) } -> std::convertible_to<double>;
};
