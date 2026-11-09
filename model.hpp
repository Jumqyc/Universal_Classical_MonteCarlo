#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <numeric>

template <typename T,
          std::size_t latt_size,
          std::size_t dof_size>
class Model
{
public:
    Model()
        : field_(new T[latt_size * dof_size])
    {}

    ~Model()
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

private:
    T* field_;
};
