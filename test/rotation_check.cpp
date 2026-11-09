#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>

#include <dof/proposal.hpp>
#include <dof/vector.hpp>
#include <rng.hpp>

static_assert(Proposal<LocalRotation<2>, double, Pcg32>);
static_assert(Proposal<LocalRotation<3>, double, Pcg32>);
static_assert(Proposal<LocalRotation<4>, double, Pcg32>);

template <std::size_t N>
static void check(const char* name)
{
    std::array<double, N> cur{};
    cur[0] = 1.0;
    std::array<double, N> out{};

    LocalRotation<N> rot{0.3};
    Pcg32 rng(99, static_cast<std::uint64_t>(N));

    double moved = 0.0;
    for (int k = 0; k < 20000; ++k) {
        rot.sample(cur.data(), rng, out.data());
        double norm = 0.0;
        for (std::size_t j = 0; j < N; ++j) {
            assert(std::isfinite(out[j]));
            norm += out[j] * out[j];
        }
        assert(std::abs(norm - 1.0) < 1e-10);
        moved = std::max(moved, std::abs(out[0] - cur[0]));
        cur = out;
    }

    assert(moved > 1e-3);
    std::printf("rotation %s: ok\n", name);
}

int main()
{
    check<2>("N=2");
    check<3>("N=3");
    check<4>("N=4");
    std::puts("rotation_check: ok");
}
