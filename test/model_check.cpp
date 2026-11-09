#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <tuple>

#include <dof/ising.hpp>
#include <dof/vector.hpp>
#include <driver.hpp>
#include <example/ising.hpp>
#include <example/vector_spin.hpp>
#include <lattice/square.hpp>
#include <model.hpp>
#include <rng.hpp>

static_assert(ModelLike<Ising>);
static_assert(WolffModel<Ising>);
static_assert(EnergyModel<Ising>);
static_assert(ModelLike<VectorSpin<3>>);
static_assert(WolffModel<VectorSpin<3>>);
static_assert(EnergyModel<VectorSpin<3>>);

/// A model with only what a single-site updater needs: no energy_at and no
/// order_param_at. ModelLike must accept it; EnergyModel must not.
struct MinimalModel
{
    using value_type = double;
    static constexpr std::size_t size = 1;

    std::size_t sites() const noexcept { return 1; }
    const double* data() const noexcept { return &x_; }
    double energy_diff_at(std::size_t, const double*) const noexcept { return 0.0; }
    void update(std::size_t, const double* p) noexcept { x_ = *p; }

    double x_ = 0.0;
};

static_assert(ModelLike<MinimalModel>);
static_assert(!EnergyModel<MinimalModel>);
static_assert(Proposal<Flip, IsingSpin, Pcg32>);
static_assert(Proposal<UniformSphereN<3>, double, Pcg32>);

/// Total energy: energy_at counts every bond once at each end, so sum and halve.
template <class M>
double total_energy(const M& model)
{
    double sum = 0.0;
    for (std::size_t i = 0; i < model.sites(); ++i) sum += model.energy_at(i);
    return 0.5 * sum;
}

/// Samples S^(D-1), checks unit norm and zero mean on every component.
template <std::size_t D>
void check_sphere()
{
    Pcg32 rng(9 + D, 1);
    std::array<double, D> v{};
    std::array<double, D> sum{};
    double norm = 0.0;

    constexpr int samples = 100000;
    for (int k = 0; k < samples; ++k) {
        UniformSphereN<D>{}.sample(nullptr, rng, v.data());
        for (std::size_t j = 0; j < D; ++j) {
            sum[j] += v[j];
            norm += v[j] * v[j];
        }
    }
    assert(std::abs(norm / samples - 1.0) < 0.002);
    for (std::size_t j = 0; j < D; ++j) assert(std::abs(sum[j] / samples) < 0.01);
}

int main()
{
    constexpr std::size_t L = 4;
    constexpr std::size_t N = L * L;

    check_sphere<2>();
    check_sphere<3>();
    check_sphere<4>();

    const auto graph = square_lattice(L, L);
    assert(graph.sites() == N);
    for (std::size_t i = 0; i < N; ++i) assert(graph.degree(i) == 4);

    for (const auto& [Lx, Ly, degree] :
         {std::tuple{std::size_t(1), std::size_t(1), std::size_t(0)},
          std::tuple{std::size_t(1), std::size_t(4), std::size_t(2)},
          std::tuple{std::size_t(2), std::size_t(2), std::size_t(4)}}) {
        const auto g = square_lattice(Lx, Ly);
        assert(g.sites() == Lx * Ly);
        for (std::size_t i = 0; i < g.sites(); ++i) assert(g.degree(i) == degree);
    }

    // Ising: ground state and a finite-difference check of both energy changes.
    {
        Ising model(graph);
        assert(std::abs(total_energy(model) / N + 2.0) < 1e-12);

        Pcg32 rng(7, 3);
        for (std::size_t i = 0; i < N; ++i) {
            const IsingSpin s = rng.uni() < 0.5 ? IsingSpin(-1) : IsingSpin(1);
            model.update(i, &s);
        }

        const IsingSpin p = IsingSpin(-model.data()[0]);
        const double dE = model.energy_diff_at(0, &p);
        const double before = total_energy(model);
        model.update(0, &p);
        assert(std::abs(dE - (total_energy(model) - before)) < 1e-12);
    }
    {
        // One bond, so energy_diff_bond accounts for the whole energy change.
        const std::array<PairGraph<>::Edge, 1> one_edge{PairGraph<>::Edge{0, 1}};
        const auto edge_graph = PairGraph<>(2, one_edge);
        Ising model(edge_graph);
        Pcg32 rng(7, 4);
        for (std::size_t i = 0; i < 2; ++i) {
            const IsingSpin s = rng.uni() < 0.5 ? IsingSpin(-1) : IsingSpin(1);
            model.update(i, &s);
        }

        const auto bond = edge_graph.bonds(0)[0];
        const IsingSpin p = IsingSpin(-model.data()[bond.j]);
        const double dE = model.energy_diff_bond(0, bond, &p);
        const double before = total_energy(model);
        model.update(bond.j, &p);
        assert(std::abs(dE - (total_energy(model) - before)) < 1e-12);
    }

    // VectorSpin: ground state and the same finite-difference checks in 3D.
    {
        VectorSpin<3> model(graph);
        assert(std::abs(total_energy(model) / N + 2.0) < 1e-12);

        Pcg32 rng(8, 5);
        double v[3];
        for (std::size_t i = 0; i < N; ++i) {
            UniformSphereN<3>{}.sample(nullptr, rng, v);
            model.update(i, v);
        }

        double p[3];
        UniformSphereN<3>{}.sample(nullptr, rng, p);
        const double dE = model.energy_diff_at(0, p);
        const double before = total_energy(model);
        model.update(0, p);
        assert(std::abs(dE - (total_energy(model) - before)) < 1e-12);
    }
    {
        const std::array<PairGraph<>::Edge, 1> one_edge{PairGraph<>::Edge{0, 1}};
        const auto edge_graph = PairGraph<>(2, one_edge);
        VectorSpin<3> model(edge_graph);
        Pcg32 rng(8, 6);
        double v[3];
        for (std::size_t i = 0; i < 2; ++i) {
            UniformSphereN<3>{}.sample(nullptr, rng, v);
            model.update(i, v);
        }

        const auto bond = edge_graph.bonds(0)[0];
        double p[3];
        UniformSphereN<3>{}.sample(nullptr, rng, p);
        const double dE = model.energy_diff_bond(0, bond, p);
        const double before = total_energy(model);
        model.update(bond.j, p);
        assert(std::abs(dE - (total_energy(model) - before)) < 1e-12);
    }

    // Ising at beta=0.3 on 4x4: exact E/site is -0.844054..., compare both chains.
    constexpr double ising_exact = -0.8440540736;
    {
        Ising model(graph);
        Metropolis metro(model, Flip{}, Pcg32(11, 1), 0.3);
        metro.run(2000);

        double sum = 0.0;
        metro.sample(4000, [&](const Ising& m) { sum += total_energy(m); });
        const double avg = sum / (4000.0 * double(N));
        std::printf("Ising Metropolis: %.5f\n", avg);
        std::fflush(stdout);
        assert(std::abs(avg - ising_exact) < 0.03);
    }
    {
        Ising model(graph);
        Wolff wolf(model, FixedAxis{}, Pcg32(12, 1), 0.3);
        for (int k = 0; k < 2000; ++k) wolf.flip();

        double sum = 0.0;
        wolf.sample(20000, [&](const Ising& m) { sum += total_energy(m); });
        const double avg = sum / (20000.0 * double(N));
        std::printf("Ising Wolff:      %.5f (cluster %.2f)\n", avg, wolf.mean_cluster_size());
        std::fflush(stdout);
        assert(std::abs(avg - ising_exact) < 0.05);
    }

    // Heisenberg at beta=0.3 on 4x4: two independent algorithms must agree.
    double heis_metro = 0.0;
    {
        VectorSpin<3> model(graph);
        Metropolis metro(model, UniformSphereN<3>{}, Pcg32(13, 1), 0.3);
        metro.run(2000);

        double sum = 0.0;
        metro.sample(4000, [&](const VectorSpin<3>& m) { sum += total_energy(m); });
        heis_metro = sum / (4000.0 * double(N));
        std::printf("Heisenberg Metropolis: %.5f\n", heis_metro);
        std::fflush(stdout);
    }
    {
        VectorSpin<3> model(graph);
        Wolff wolf(model, UniformSphereN<3>{}, Pcg32(14, 1), 0.3);
        for (int k = 0; k < 2000; ++k) wolf.flip();

        double sum = 0.0;
        wolf.sample(20000, [&](const VectorSpin<3>& m) { sum += total_energy(m); });
        const double avg = sum / (20000.0 * double(N));
        std::printf("Heisenberg Wolff:      %.5f (cluster %.2f)\n", avg, wolf.mean_cluster_size());
        std::fflush(stdout);
        assert(std::abs(avg - heis_metro) < 0.05);
    }

    std::puts("ok");
}
