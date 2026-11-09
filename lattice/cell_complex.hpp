#pragma once

#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

/// One interaction cell: A site indices, in the order the term expects.
template <std::size_t A>
struct Cell
{
    std::array<std::uint32_t, A> sites;
};

/// Cells plus their inverse incidence. A site may belong to many cells (a link
/// belongs to 6 plaquettes in four dimensions), and a cell may repeat a site.
template <std::size_t A>
class CellComplex
{
    static_assert(A >= 1 && A <= 255, "CellComplex arity must fit uint8_t");

public:
    struct Incident
    {
        std::uint32_t cell;
        std::uint8_t which;
    };

    static constexpr std::size_t arity = A;

    /// Validates the site indices and builds the site -> incident-cell rows.
    explicit CellComplex(std::size_t sites, std::span<const Cell<A>> cells)
        : sites_(sites), cells_(cells.begin(), cells.end())
    {
        constexpr std::size_t max = std::numeric_limits<std::uint32_t>::max();
        if (sites > max || cells_.size() > max)
            throw std::overflow_error("CellComplex: too many sites or cells");
        if (cells_.size() > std::numeric_limits<std::size_t>::max() / A)
            throw std::overflow_error("CellComplex: incident count overflows size_t");

        offset_.assign(sites_ + 1, 0);
        incident_.assign(cells_.size() * A, Incident{});

        for (const auto& cell : cells_)
            for (const auto s : cell.sites) {
                if (s >= sites_) throw std::out_of_range("CellComplex: site out of range");
                ++offset_[s + 1];
            }

        for (std::size_t i = 0; i < sites_; ++i) offset_[i + 1] += offset_[i];

        auto cursor = offset_;
        for (std::size_t c = 0; c < cells_.size(); ++c)
            for (std::size_t k = 0; k < A; ++k) {
                const std::size_t s = cells_[c].sites[k];
                incident_[cursor[s]++] = {static_cast<std::uint32_t>(c),
                                          static_cast<std::uint8_t>(k)};
            }
    }

    std::size_t sites() const noexcept { return sites_; }
    std::size_t cells() const noexcept { return cells_.size(); }

    /// The A site indices of cell c, in term order.
    std::span<const std::uint32_t, A> cell(std::size_t c) const
    {
        if (c >= cells_.size()) throw std::out_of_range("CellComplex: cell out of range");
        return cells_[c].sites;
    }

    /// The cells containing site i, with the site's position in each cell.
    std::span<const Incident> incident(std::size_t i) const
    {
        if (i >= sites_) throw std::out_of_range("CellComplex: site out of range");
        return {incident_.data() + offset_[i], offset_[i + 1] - offset_[i]};
    }

private:
    std::size_t sites_;
    std::vector<Cell<A>> cells_;
    std::vector<std::size_t> offset_;
    std::vector<Incident> incident_;
};

/// A cell collection: sites, cells, and the inverse incidence site -> cells.
template <class L>
concept CellComplexLike = requires(const L list, std::size_t i, std::size_t c) {
    { L::arity } -> std::convertible_to<std::size_t>;
    { list.sites() } -> std::convertible_to<std::size_t>;
    { list.cells() } -> std::convertible_to<std::size_t>;
    { list.cell(c) } -> std::convertible_to<std::span<const std::uint32_t, L::arity>>;
    { list.incident(i) } -> std::convertible_to<std::span<const typename L::Incident>>;
};
