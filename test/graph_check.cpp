#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <numbers>
#include <stdexcept>
#include <utility>
#include <vector>

#include <driver.hpp>
#include <lattice/pair_graph.hpp>
#include <rng.hpp>

/// Deterministic Ising proposal: flip the sign, symmetric so log_ratio is 0.
struct Flip
{
    template <class R>
    void sample(const double* cur, R&, double* out) const
    {
        *out = *cur > 0.0 ? -1.0 : 1.0;
    }

    double log_ratio(const double*, const double*) const { return 0.0; }
};

/// Fixed reflection axis r = (1, 0, 0), enough for an Ising model.
struct FixedAxis
{
    template <class R>
    void sample(const double*, R&, double* out) const { out[0] = 1.0; }
};

class IsingGraph
{
public:
    using value_type = double;
    static constexpr std::size_t size = 1;
    using bond_type = typename PairGraph<double>::bond_type;

    explicit IsingGraph(PairGraph<double> graph)
        : graph_(std::move(graph)), spin_(graph_.sites(), 1.0)
    {}

    std::size_t sites() const noexcept { return graph_.sites(); }
    const double* data() const noexcept { return spin_.data(); }
    double* data() noexcept { return spin_.data(); }

    std::size_t degree(std::size_t i) const noexcept { return graph_.degree(i); }
    const bond_type* bonds(std::size_t i) const noexcept { return graph_.bonds(i); }

    double energy_at(std::size_t i) const noexcept
    {
        double e = 0.0;
        const auto* bs = graph_.bonds(i);
        for (std::size_t a = 0; a < graph_.degree(i); ++a)
            e += -bs[a].c * spin_[i] * spin_[bs[a].j];
        return e;
    }

    double order_param_at(std::size_t i) const noexcept { return spin_[i]; }

    double energy_diff_at(std::size_t i, const double* p) const noexcept
    {
        double d = 0.0;
        const auto* bs = graph_.bonds(i);
        for (std::size_t a = 0; a < graph_.degree(i); ++a)
            d += -bs[a].c * (*p - spin_[i]) * spin_[bs[a].j];
        return d;
    }

    double energy_diff_bond(std::size_t i, const bond_type& b, const double* p) const noexcept
    {
        return -b.c * (*p - spin_[b.j]) * spin_[i];
    }

    void update(std::size_t i, const double* p) noexcept { spin_[i] = *p; }

private:
    PairGraph<double> graph_;
    std::vector<double> spin_;
};

static_assert(ModelLike<IsingGraph>);
static_assert(WolffModel<IsingGraph>);
static_assert(sizeof(Bond<NoPayload>) == 4);

int main()
{
    std::setbuf(stdout, nullptr);

    using G = PairGraph<double>;
    constexpr int L = 16;
    constexpr std::size_t N = std::size_t(L) * L;

    auto id = [](int x, int y) {
        return static_cast<std::uint32_t>(y * L + x);
    };

    std::vector<G::Edge> edges;
    edges.reserve(2 * N);
    for (int y = 0; y < L; ++y) {
        for (int x = 0; x < L; ++x) {
            edges.push_back({id(x, y), id((x + 1) % L, y), 1.0});
            edges.push_back({id(x, y), id(x, (y + 1) % L), 1.0});
        }
    }

    G graph(N, edges);
    assert(graph.sites() == N);
    for (std::size_t i = 0; i < N; ++i) assert(graph.degree(i) == 4);

    IsingGraph model(std::move(graph));

    // The graph rows must reproduce the explicit edge sum.
    Pcg32 config_rng(77, 3);
    for (std::size_t i = 0; i < N; ++i) {
        const double v = config_rng.uni() < 0.5 ? -1.0 : 1.0;
        model.update(i, &v);
    }
    double row_energy = 0.0;
    for (std::size_t i = 0; i < N; ++i) row_energy += model.energy_at(i);
    row_energy *= 0.5;
    double edge_energy = 0.0;
    for (const auto& e : edges)
        edge_energy += -e.c * model.data()[e.i] * model.data()[e.j];
    assert(std::abs(row_energy - edge_energy) < 1e-9);

    // energy_diff_at must be the finite difference of the total energy.
    for (std::size_t i = 0; i < 8; ++i) {
        const double orig = model.data()[i];
        const double p = -orig;
        const double d_at = model.energy_diff_at(i, &p);
        model.update(i, &p);
        double flipped = 0.0;
        for (std::size_t k = 0; k < N; ++k) flipped += model.energy_at(k);
        flipped *= 0.5;
        const double d_fd = flipped - row_energy;
        model.update(i, &orig);
        assert(std::abs(d_at - d_fd) < 1e-9);
    }

    // All spins up: 2N bonds, each -1.
    for (std::size_t i = 0; i < N; ++i) {
        const double v = 1.0;
        model.update(i, &v);
    }
    double all_up = 0.0;
    for (std::size_t i = 0; i < N; ++i) all_up += model.energy_at(i);
    all_up *= 0.5;
    assert(std::abs(all_up + 2.0 * double(N)) < 1e-9);

    // Onsager energy per spin at beta = 0.3, J = 1; K(0.906057) = 2.3076245774.
    constexpr double beta = 0.30;
    const double tb = std::tanh(2.0 * beta);
    const double u = -(1.0 / std::tanh(2.0 * beta)) *
                     (1.0 + (2.0 / std::numbers::pi) * (2.0 * tb * tb - 1.0) *
                                2.307624577366968);
    std::printf("exact E/site = %.5f\n", u);

    Pcg32 rng_mc(2026, 1);
    Metropolis<IsingGraph, Flip, Pcg32> mc(model, Flip{}, rng_mc, beta);
    mc.run(300);
    double sum = 0.0;
    int samples = 0;
    for (int k = 0; k < 2000; ++k) {
        mc.sweep();
        double e = 0.0;
        for (std::size_t i = 0; i < N; ++i) e += model.energy_at(i);
        sum += 0.5 * e;
        ++samples;
    }
    const double e_mc = sum / samples / double(N);
    std::printf("Metropolis E/site = %.5f\n", e_mc);
    assert(std::abs(e_mc - u) < 0.03);

    Pcg32 rng_w(2026, 2);
    Wolff<IsingGraph, FixedAxis, Pcg32> w(model, FixedAxis{}, rng_w, beta);
    for (int k = 0; k < 200; ++k) w.flip();
    sum = 0.0;
    samples = 0;
    for (int k = 0; k < 20000; ++k) {
        w.flip();
        double e = 0.0;
        for (std::size_t i = 0; i < N; ++i) e += model.energy_at(i);
        sum += 0.5 * e;
        ++samples;
    }
    const double e_w = sum / samples / double(N);
    std::printf("Wolff E/site = %.5f, mean cluster = %.2f\n", e_w, w.mean_cluster_size());
    assert(std::abs(e_w - u) < 0.03);
    assert(std::abs(e_mc - e_w) < 0.04);

    using G0 = PairGraph<>;
    std::vector<G0::Edge> dup{{0, 1, {}}, {0, 1, {}}};
    G0 g0(2, dup);
    assert(g0.degree(0) == 2 && g0.degree(1) == 2);

    std::vector<G0::Edge> none;
    G0 empty(3, none);
    assert(empty.sites() == 3 && empty.degree(0) == 0 && empty.bonds(0) == nullptr);

    std::vector<G0::Edge> one{{0, 1, {}}};
    G0 directed(2, one, false);
    assert(directed.degree(0) == 1 && directed.degree(1) == 0);

    bool threw = false;
    try {
        std::vector<G0::Edge> loop{{0, 0, {}}};
        G0 bad(2, loop);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);

    std::puts("graph_check: ok");
}
