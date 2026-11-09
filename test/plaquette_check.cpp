#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdio>

#include <term/plaquette.hpp>

int main()
{
    constexpr double beta = 1.3;
    U1PlaquetteTerm term{beta};

    std::array<U1Link, 4> links{{{0.2}, {0.4}, {-0.1}, {0.7}}};
    std::array<const U1Link*, 4> ptrs{
        &links[0], &links[1], &links[2], &links[3]};

    const double theta = links[0].theta + links[1].theta
                       - links[2].theta - links[3].theta;
    const double expected = beta * (1.0 - std::cos(theta));
    assert(std::abs(term.energy(ptrs.data()) - expected) < 1e-12);

    for (std::size_t which = 0; which < 4; ++which) {
        const U1Link old = links[which];
        U1Link nw = old;
        nw.theta += 0.37;

        const double before = term.energy(ptrs.data());
        links[which] = nw;
        const double after = term.energy(ptrs.data());
        links[which] = old;

        const double dE = term.delta(which, &old, &nw, ptrs.data());
        assert(std::abs(dE - (after - before)) < 1e-12);
    }

    std::puts("plaquette_check: ok");
}
