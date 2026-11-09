#pragma once

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

/// No per-bond data: the bond is just a neighbour index.
struct NoPayload {};

/// One directed adjacency entry. c is empty for a topology-only graph, and
/// [[no_unique_address]] keeps such a bond at one uint32_t.
template <class P = NoPayload>
struct Bond
{
    std::uint32_t j;
    [[no_unique_address]] P c;
};

/// Compressed sparse row graph: one contiguous row of bonds per site.
/// Undirected edges are stored in both rows, so the graph is symmetric. A
/// duplicated edge stays duplicated: periodic images and multi-bonds are
/// physically distinct, and the energy must count each one. P must be default
/// constructible and copyable; indices are uint32_t, so N and 2E must fit.
template <class P = NoPayload>
class PairGraph
{
public:
    using bond_type = Bond<P>;

    /// Input edge; c is copied into both directed entries when symmetric.
    struct Edge
    {
        std::uint32_t i;
        std::uint32_t j;
        P c{};
    };

    PairGraph() = default;

    /// Builds from n sites and an edge list. symmetric=true adds (j in row i)
    /// and (i in row j). Throws std::out_of_range for an endpoint >= n,
    /// std::invalid_argument for a self-loop, and std::overflow_error if n or
    /// the stored bond count exceeds uint32_t.
    PairGraph(std::size_t n, std::span<const Edge> edges, bool symmetric = true)
    {
        if (n > std::numeric_limits<std::uint32_t>::max())
            throw std::overflow_error("PairGraph: site count exceeds uint32_t");

        std::vector<std::size_t> degree(n, 0);
        for (const Edge& e : edges) {
            if (e.i >= n || e.j >= n)
                throw std::out_of_range("PairGraph: edge endpoint out of range");
            if (e.i == e.j)
                throw std::invalid_argument("PairGraph: self-loop");
            ++degree[e.i];
            if (symmetric) ++degree[e.j];
        }

        std::vector<std::uint32_t> offset(n + 1, 0);
        std::uint64_t total = 0;
        for (std::size_t i = 0; i < n; ++i) {
            offset[i] = static_cast<std::uint32_t>(total);
            total += degree[i];
            if (total > std::numeric_limits<std::uint32_t>::max())
                throw std::overflow_error("PairGraph: bond count exceeds uint32_t");
        }
        offset[n] = static_cast<std::uint32_t>(total);

        std::vector<bond_type> bonds(static_cast<std::size_t>(total));
        std::vector<std::uint32_t> pos(offset.begin(), offset.end() - 1);
        for (const Edge& e : edges) {
            bonds[pos[e.i]++] = bond_type{e.j, e.c};
            if (symmetric) bonds[pos[e.j]++] = bond_type{e.i, e.c};
        }

        offset_ = std::move(offset);
        bonds_  = std::move(bonds);
    }

    /// Number of sites.
    std::size_t sites() const noexcept
    {
        return offset_.empty() ? 0 : offset_.size() - 1;
    }

    /// Number of bonds in row i; i must be < sites().
    std::size_t degree(std::size_t i) const noexcept
    {
        return static_cast<std::size_t>(offset_[i + 1] - offset_[i]);
    }

    /// Start of row i, or nullptr when the graph has no bonds. The row is
    /// [bonds(i), bonds(i) + degree(i)); i must be < sites().
    const bond_type* bonds(std::size_t i) const noexcept
    {
        return bonds_.empty() ? nullptr : bonds_.data() + offset_[i];
    }

private:
    std::vector<std::uint32_t> offset_;
    std::vector<bond_type>     bonds_;
};

/// A graph that exposes one row of bonds per site. PairGraph satisfies it.
template <class G>
concept PairGraphLike = requires(const G g, std::size_t i) {
    typename G::bond_type;
    { g.sites() } -> std::convertible_to<std::size_t>;
    { g.degree(i) } -> std::convertible_to<std::size_t>;
    { g.bonds(i) } -> std::convertible_to<const typename G::bond_type*>;
};
