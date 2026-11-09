#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>

#include <dof/proposal.hpp>
#include <dof/sun.hpp>
#include <driver.hpp>
#include <example/sun_gauge.hpp>
#include <model.hpp>
#include <rng.hpp>

static_assert(ModelLike<SU2Gauge>);
static_assert(ModelLike<SU3Gauge>);
static_assert(!EnergyModel<SU2Gauge>);
static_assert(!WolffModel<SU2Gauge>);
static_assert(Proposal<SUShift<2>, SULink<2>, Pcg32>);
static_assert(Proposal<SUShift<3>, SULink<3>, Pcg32>);

template <std::size_t N>
void check_group()
{
    Pcg32 rng(11 + N, 2);
    SUShift<N> shift{0.7};
    SULink<N> u = SULink<N>::identity();
    for (int k = 0; k < 1000; ++k) shift.sample(&u, rng, &u);
    assert(u.unitarity_error() < 1e-12);

    const SULink<N> id = SULink<N>::identity();
    const SULink<N> d = u.dagger() * u;
    double err = 0.0;
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = 0; j < N; ++j)
            err = std::max(err, std::abs(d.m[i * N + j] - id.m[i * N + j]));
    assert(err < 1e-12);
}

/// energy_diff_at must match a full-action finite difference for every link.
template <std::size_t N>
void check_energy_diff()
{
    Pcg32 rng(21 + N, 3);
    SUShift<N> shift{0.5};
    SUGauge<N> g(2, 1.3);
    for (std::size_t i = 0; i < g.sites(); ++i) {
        SULink<N> p;
        shift.sample(g.data() + i, rng, &p);
        g.update(i, &p);
    }

    for (std::size_t j = 0; j < g.sites(); ++j) {
        SULink<N> p;
        shift.sample(g.data() + j, rng, &p);

        const double dE = g.energy_diff_at(j, &p);
        const double before = g.total_energy();
        const auto old = g.data()[j];
        g.update(j, &p);
        const double after = g.total_energy();
        g.update(j, &old);

        assert(std::abs(dE - (after - before)) < 1e-9);
    }
}

template <std::size_t N>
double run_chain(double beta, bool hot, std::size_t sweeps, std::uint64_t seed,
                 std::size_t L)
{
    SUGauge<N> g(L, beta);
    Pcg32 init(seed, 1);
    if (hot) {
        SUShift<N> big{3.0};
        for (std::size_t i = 0; i < g.sites(); ++i) {
            SULink<N> p;
            for (int k = 0; k < 5; ++k) {
                big.sample(g.data() + i, init, &p);
                g.update(i, &p);
            }
        }
    }

    Metropolis<SUGauge<N>, SUShift<N>, Pcg32> m(g, SUShift<N>{0.3}, Pcg32(seed, 2), beta);
    m.run(100);

    double sum = 0.0;
    std::size_t n = 0;
    m.sample(sweeps, [&](const SUGauge<N>& model) {
        sum += model.mean_plaquette();
        ++n;
    });
    assert(n == sweeps);

    for (std::size_t i = 0; i < g.sites(); ++i)
        assert(g.data()[i].unitarity_error() < 1e-10);

    return sum / static_cast<double>(n);
}

int main()
{
    check_group<2>();
    check_group<3>();
    check_energy_diff<2>();
    check_energy_diff<3>();

    const double su2_hot0 = run_chain<2>(0.0, true, 200, 10, 4);
    const double su2_cold0 = run_chain<2>(0.0, false, 200, 20, 4);
    const double su2_hot2 = run_chain<2>(2.0, true, 200, 30, 4);
    const double su2_cold2 = run_chain<2>(2.0, false, 200, 40, 4);
    const double su2_cold10 = run_chain<2>(10.0, false, 200, 50, 4);

    const double su3_hot0 = run_chain<3>(0.0, true, 200, 60, 3);
    const double su3_cold0 = run_chain<3>(0.0, false, 200, 70, 3);
    const double su3_hot4 = run_chain<3>(4.0, true, 200, 80, 3);
    const double su3_cold4 = run_chain<3>(4.0, false, 200, 90, 3);

    std::fprintf(stderr, "SU2 beta=0 hot=%.5f cold=%.5f\n", su2_hot0, su2_cold0);
    std::fprintf(stderr, "SU2 beta=2 hot=%.5f cold=%.5f\n", su2_hot2, su2_cold2);
    std::fprintf(stderr, "SU2 beta=10 cold=%.5f\n", su2_cold10);
    std::fprintf(stderr, "SU3 beta=0 hot=%.5f cold=%.5f\n", su3_hot0, su3_cold0);
    std::fprintf(stderr, "SU3 beta=4 hot=%.5f cold=%.5f\n", su3_hot4, su3_cold4);

    assert(std::abs(su2_hot0) < 0.05);
    assert(std::abs(su2_cold0) < 0.05);
    assert(std::abs(su2_hot2 - su2_cold2) < 0.03);
    assert(su2_cold10 > 0.9);

    assert(std::abs(su3_hot0) < 0.05);
    assert(std::abs(su3_cold0) < 0.05);
    assert(std::abs(su3_hot4 - su3_cold4) < 0.05);

    std::puts("su_check: ok");
}
