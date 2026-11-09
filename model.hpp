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
    Model::Model();
    ~Model::Model();

private:
    T *field_
};