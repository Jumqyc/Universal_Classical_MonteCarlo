#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <numbers>

#include <dof/u1.hpp>
#include <driver.hpp>
#include <example/u1_gauge.hpp>
#include <model.hpp>
#include <rng.hpp>

static_assert(ModelLike<U1Gauge>);
static_assert(!EnergyModel<U1Gauge>);
static_assert(!WolffModel<U1Gauge>);
static_assert(Proposal<U1Shift, U1Link, Pcg32>);

/// S = beta * sum_p (1 - cos theta_p), with 6*L^4 plaquettes.
double action(const U1Gauge& g, double beta)
{
    const double L = static_cast<double>(g.length());
    return beta * 6.0 * L * L * L * L * (1.0 - g.mean_plaquette());
}

/// Equilibrates and returns the mean plaquette averaged over sweeps. A hot
/// start fills every link with a uniform random angle.
double run_chain(double beta, bool hot, std::size_t sweeps, std::uint64_t seed)
{
    U1Gauge g(4, beta);
    Pcg32 init(seed, 1);
    if (hot) {
        for (std::size_t i = 0; i < g.sites(); ++i) {
            U1Link p;
            p.theta = 2.0 * std::numbers::pi_v<double> * init.uni();
            g.update(i, &p);
        }
    }

    Metropolis metro(g, U1Shift{0.7}, Pcg32(seed, 2), beta);
    metro.run(100);

    double sum = 0.0;
    std::size_t n = 0;
    metro.sample(sweeps, [&](const U1Gauge& m) {
        sum += m.mean_plaquette();
        ++n;
    });
    assert(n == sweeps);
    return sum / static_cast<double>(n);
}

int main()
{
    // energy_diff_at must match a full-action finite difference.
    {
        U1Gauge g(3, 1.3);
        Pcg32 rng(3, 4);
        U1Shift shift{2.0};
        for (std::size_t i = 0; i < g.sites(); ++i) {
            U1Link p;
            shift.sample(g.data() + i, rng, &p);
            g.update(i, &p);
        }

        for (std::size_t j = 0; j < g.sites(); ++j) {
            U1Link p;
            shift.sample(g.data() + j, rng, &p);

            const double dE = g.energy_diff_at(j, &p);
            const double before = action(g, 1.3);
            g.update(j, &p);
            const double after = action(g, 1.3);
            assert(std::abs(dE - (after - before)) < 1e-10);
        }
    }

    const double cold0 = run_chain(0.0, false, 400, 10);
    const double cold10 = run_chain(10.0, false, 400, 20);
    const double hot05 = run_chain(0.5, true, 400, 50);
    const double cold05 = run_chain(0.5, false, 400, 60);
    const double hot2 = run_chain(2.0, true, 400, 70);
    const double cold2 = run_chain(2.0, false, 400, 80);

    std::fprintf(stderr, "beta=0 cold hot=%.5f\n", cold0);
    std::fprintf(stderr, "beta=10 cold=%.5f\n", cold10);
    std::fprintf(stderr, "beta=0.5 hot=%.5f cold=%.5f\n", hot05, cold05);
    std::fprintf(stderr, "beta=2 hot=%.5f cold=%.5f\n", hot2, cold2);

    assert(std::abs(cold0) < 0.05);
    assert(std::abs(hot05 - cold05) < 0.02);
    assert(std::abs(hot2 - cold2) < 0.02);
    assert(cold10 > 0.9);

    std::puts("ok");
}
