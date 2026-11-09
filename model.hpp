#pragma once

#include <algorithm>
#include <concepts>
#include <cstddef>

/// Owning buffer for one lattice: N scalars, allocated once, moved but never
/// copied. A model holds one of these and exposes it through data().
template <typename T, std::size_t N>
class Field
{
public:
    Field() : data_(new T[N]) {}
    ~Field() { delete[] data_; }

    Field(const Field&)            = delete;
    Field& operator=(const Field&) = delete;

    Field(Field&& other) noexcept
        : data_(other.data_)
    {
        other.data_ = nullptr;
    }

    Field& operator=(Field&& other) noexcept
    {
        if (this != &other) {
            delete[] data_;
            data_       = other.data_;
            other.data_ = nullptr;
        }
        return *this;
    }

    const T* data() const noexcept { return data_; }
    T*       data()       noexcept { return data_; }

private:
    T* data_;
};

/// Base for a model whose state is one field: hands out sites(), data() and
/// the write-back the updaters need, so a concrete model only writes its
/// energy terms. Nothing here is virtual; a model that caches local fields
/// defines its own update(), which hides this one.
template <typename T, std::size_t Sites, std::size_t Dof>
class Model
{
public:
    std::size_t sites() const noexcept { return Sites; }
    const T*    data()  const noexcept { return field_.data(); }
    T*          data()        noexcept { return field_.data(); }

    /// Writes p into loc, leaving every other site untouched.
    void update(std::size_t loc, const T* proposal)
    {
        std::copy_n(proposal, Dof, field_.data() + loc * Dof);
    }

private:
    Field<T, Sites * Dof> field_;
};

/// What an updater needs from a model. Every member is resolved at compile
/// time: the model stays a plain struct with no table pointer, so the same
/// definition can be copied to a device and driven by a kernel, and a missing
/// member is a compile error rather than a run-time surprise.
template <class M>
concept ModelLike = requires(const M cm, 
                            M m, 
                            std::size_t loc, 
                            std::size_t* out,
                            const typename M::value_type* proposal) {
    typename M::value_type;
    { M::dof }           -> std::convertible_to<std::size_t>;
    { M::max_neighbors } -> std::convertible_to<std::size_t>;

    { cm.sites() }       -> std::convertible_to<std::size_t>;
    { cm.data() }        -> std::convertible_to<const typename M::value_type*>;

    /// Neighbour indices of loc, periodic; at most max_neighbors of them.
    { cm.neighbors(loc, out) } -> std::convertible_to<std::size_t>;

    /// Total energy of one site, and its contribution to the order parameter.
    { cm.energy_at(loc) }       -> std::convertible_to<double>;
    { cm.order_param_at(loc) }  -> std::convertible_to<typename M::value_type>;

    /// Energy change when loc takes proposal, all other sites fixed.
    { cm.energy_diff_at(loc, proposal) } -> std::convertible_to<double>;

    /// Energy change of the single bond between the two sites when the first
    /// takes proposal and the second keeps its value.
    { cm.energy_diff_between(loc, loc, proposal) } -> std::convertible_to<double>;

    /// Writes proposal into loc; a model caching local fields updates them here.
    { m.update(loc, proposal) } -> std::same_as<void>;
};
