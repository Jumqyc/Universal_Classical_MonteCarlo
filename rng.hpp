#pragma once

#include <cstdint>

/// PCG-XSH-RR 32-bit generator (O'Neill, "PCG: A Family of Simple Fast Space-
/// Efficient Statistically Good Algorithms for Random Number Generation", 2014).
/// A 64-bit LCG state and a 32-bit permuted output; the two-word state is
/// trivially copyable, so the engine survives a move into a driver and can be
/// copied to a device unchanged.
///
/// Deterministic: equal (seed, stream) gives the same sequence. Different
/// streams give sequences that are independent over the full 2^64 period, so
/// parallel chains should vary the stream rather than adding a constant to the
/// seed (seeds that differ by small amounts start from nearby states).
class Pcg32
{
public:
    /// Sets the state from the seed and chooses one of the 2^63 odd stream
    /// increments from stream. The two dummy advances are part of PCG's seeding
    /// recipe and are required for a well-mixed starting state.
    explicit Pcg32(std::uint64_t seed, std::uint64_t stream = 0) noexcept
        : state_(0), inc_((stream << 1u) | 1u)
    {
        next();
        state_ += seed;
        next();
    }

    /// Uniform in [0, 1): one 32-bit draw scaled by 2^-32, so the result is
    /// exactly k / 2^32 and has 32 random bits. Never returns 1.
    [[nodiscard]] double uni() noexcept
    {
        return next() * 0x1.0p-32;
    }

private:
    /// Advances the LCG and returns the permuted 32-bit output of the old state.
    std::uint32_t next() noexcept
    {
        const std::uint64_t old = state_;
        state_                  = old * 6364136223846793005ULL + inc_;

        const auto xorshifted =
            static_cast<std::uint32_t>(((old >> 18u) ^ old) >> 27u);
        const auto rot = static_cast<std::uint32_t>(old >> 59u);
        return (xorshifted >> rot) | (xorshifted << ((32u - rot) & 31u));
    }

    std::uint64_t state_;
    std::uint64_t inc_;
};
