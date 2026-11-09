#pragma once

#include <cstddef>
#include <cstring>

template <typename T,
          std::size_t latt_size,
          std::size_t dof_size>
class Model
{
public:
    using value_type                = T;
    static constexpr std::size_t dof = dof_size;

    Model() // constructor
        : field_(new T[latt_size * dof_size])
    {}

    virtual ~Model() // free
    {
        delete[] field_;
    }

    Model(const Model&)            = delete;
    Model& operator=(const Model&) = delete;

    Model(Model&& other) noexcept
        : field_(other.field_)
    {
        other.field_ = nullptr;
    }

    Model& operator=(Model&& other) noexcept
    {
        if (this != &other) {
            delete[] field_;
            field_     = other.field_;
            other.field_ = nullptr;
        }
        return *this;
    }


    virtual double energy_at(std::size_t loc) const = 0;
    virtual T order_param_at(std::size_t loc) const = 0;

    /// Change of the total energy when self_loc takes proposal, every other site fixed.
    virtual double energy_diff_at(std::size_t self_loc, const T *proposal) const = 0;

    /// Change of the bond energy between self_loc and other_loc when self_loc
    /// takes proposal and other_loc keeps its value. Only that bond, not the
    /// whole site: cluster growth must not count the other bonds twice.
    virtual double energy_diff_between(std::size_t self_loc, std::size_t other_loc, const T *proposal) const = 0;

    /// Writes the periodic neighbour indices of loc into out and returns how many.
    /// out has room for at least max_neighbors entries.
    virtual std::size_t neighbors(std::size_t loc, std::size_t *out) const = 0;

    std::size_t sites() const noexcept { return latt_size; }
    const T*    data() const noexcept { return field_; }

    /// Writes a new value for one site. Virtual so that a model caching local
    /// fields can keep them in step; the default only copies the bytes.
    virtual void update(std::size_t loc,const T *proposal) noexcept {
        std::memcpy(field_ + loc * dof_size,
                    proposal,
                    dof_size*sizeof(T));
    }
protected:
    T* field_;
};
