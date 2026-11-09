# universal_monte_carlo

Header-only C++23 lattice Monte Carlo. The library does not ship a fixed
Hamiltonian; it gives you a topology table, local primitives, RNG and updaters,
and you write the model in ordinary C++.

## Design philosophy

### The table only answers topology

A `PairGraph` or `CellComplex` answers one question:

> which sites form a local interaction cell, and which cells contain a given site?

It does **not** store `J`, `beta`, a bond type, or an `interaction id`. Those
belong to the Hamiltonian the user writes. If a model has several interactions,
build several tables (or write a model that owns them) and add their `dE`s.
There is no runtime term registry and no type erasure in the inner loop.

### The user writes the Hamiltonian

A model is a plain C++ type with the few members `ModelLike` needs:

```cpp
value_type, size, sites(), data(),
energy_diff_at(i, proposal), update(i, proposal)
```

`energy_at(i)` is optional (`EnergyModel`). The examples in `example/` are
written this way: they own their field, loop over the table, and compute `dE`
directly.

Local operator helpers keep that code short:

```cpp
- J * dot(si, sj)                         // Ising / vector spin
beta * (1 - plaquette(t0, t1, t2, t3))    // U(1) gauge
beta * (1 - plaquette(U0, U1, U2, U3))    // SU(N) gauge
```

`term/` and `model/` contain optional reusable adapters, but they are not
required by the drivers and the examples do not depend on them.

### Compile-time types, runtime data

The interaction type, arity, `dtype`, `energy_type` and the number of tables are
fixed at compile time. Couplings, temperatures, proposal widths, the graph
itself and per-bond/per-cell parameters are runtime data.

### Wolff is a special case

Wolff is only valid for a bond-only Hamiltonian whose `dE` decomposes into
pair `dE`s and whose cluster move satisfies detailed balance. `WolffModel` is a
separate concept in `driver.hpp`; generic `CellModel`s do not expose bonds.
Frustrated J1-J2, Kitaev and gauge models should use Metropolis unless you
know the cluster move is valid.

## Layout

```text
dof/        data types and local proposals
lattice/    PairGraph, CellComplex, square/hypercubic builders
model/      optional PairModel / CellModel adapters
term/       optional Ising / Heisenberg / plaquette terms
example/    complete models written directly
driver.hpp  Metropolis and Wolff
rng.hpp     PCG32
test/       checks and short Monte Carlo runs
```

## Current state

- `rng.hpp`: PCG32.
- `dof/`: Ising (`int8_t`), N-vector spin, U(1) link, SU(N) link; proposals
  `Flip`, `FixedAxis`, `UniformSphereN`, `LocalRotation<N>`, `U1Shift`,
  `SUShift<N>`; local `dot` and `plaquette` helpers.
- `lattice/`: `PairGraph` (CSR, per-bond payload), `CellComplex<A>` (cells plus
  incidence), `square_lattice`, `hypercubic_plaquettes`.
- `driver.hpp`: Metropolis and Wolff, both with `sample(n, observe)`.
- `example/`: `Ising`, `VectorSpin<N>` (`XY`, `Heisenberg`), `U1Gauge`,
  `SUGauge<N>` (`SU2Gauge`, `SU3Gauge`).
- `test/`: graph/model checks and short full Monte Carlo runs for Ising,
  Heisenberg, U(1) and SU(N). `plaquette_check` and `rotation_check` are local
  unit checks; the others run actual chains.

Currently the 4D gauge builders materialize the full cell table and its
incidence. On-the-fly neighbor generation for regular lattices is not
implemented yet.

## Building and running tests

There is no CMake file. From the repository root:

```bash
for t in test/*.cpp; do
  name=$(basename "$t" .cpp)
  g++ -std=c++23 -O2 -Wall -Wextra -Wpedantic \
    -fsanitize=address,undefined -I. "$t" -o "/tmp/$name" &&
  "/tmp/$name"
done
```

The full-chain tests print their measured values and end with `ok`. `tmp/` is
scratch space and is gitignored.
