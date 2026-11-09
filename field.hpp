#pragma once

#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <functional>
#include <memory>
#include <numeric>
#include <utility>

/// 对齐字节数。主机侧分配与 CUDA 的 cudaMalloc(256 B) 保持同一个假设，
/// 以后把同一块内存交给设备时可以复用同一套下标计算。
inline constexpr std::size_t kAlignment = 64;

/// 释放按 kAlignment 对齐分配的数组；必须与 operator new[](n, align_val_t) 配对。
template <class T>
struct AlignedDelete
{
    void operator()(T* p) const noexcept
    { ::operator delete[](p, std::align_val_t{kAlignment}); }
};

/// 非拥有的格点位形视图：裸指针 + 总数 + shape，可平凡复制。
/// 线性下标约定（行主序，最后一维连续）：
///   i = x[0] + e[0] * (x[1] + e[1] * (x[2] + ...))
/// 内核只接受这个类型，因此同一份代码在主机和设备上签名一致。
template <class T, std::size_t Dof, std::size_t Dim>
struct FieldView
{
    T*                           data = nullptr;
    std::size_t                  n    = 0;
    std::array<std::size_t, Dim> ext{};

    FieldView() = default;

    FieldView(T* d, std::size_t sites, std::array<std::size_t, Dim> e)
        : data(d), n(sites), ext(e)
    {}

    /// 可写视图到只读视图的隐式转换（T* -> const T*），反方向被约束挡住。
    template <class U>
        requires std::convertible_to<U*, T*>
    FieldView(const FieldView<U, Dof, Dim>& other)
        : data(other.data), n(other.n), ext(other.ext)
    {}

    /// 第 i 个格点的首地址；Dof 个分量连续存放。
    [[nodiscard]] T* site(std::size_t i) const noexcept { return data + i * Dof; }
};

/// 拥有一块 n*Dof 个 T 的格点位形；只可移动。
template <class T, std::size_t Dof, std::size_t Dim>
class Field
{
public:
    /// 构造 n = Π ext 个格点的位形，每个分量填 fill。
    explicit Field(std::array<std::size_t, Dim> ext, T fill = T{});

    /// 移动后源对象变空：sites() == 0，data() == nullptr。
    Field(Field&& other) noexcept;
    Field& operator=(Field&& other) noexcept;

    Field(const Field&)            = delete;
    Field& operator=(const Field&) = delete;

    [[nodiscard]] std::size_t sites() const noexcept { return n_; }

    /// 缓冲区字节数，等于 n_ * Dof * sizeof(T)。
    [[nodiscard]] std::size_t bytes() const noexcept { return n_ * Dof * sizeof(T); }

    [[nodiscard]] const std::array<std::size_t, Dim>& extents() const noexcept { return ext_; }

    [[nodiscard]] T*       data() noexcept { return data_.get(); }
    [[nodiscard]] const T* data() const noexcept { return data_.get(); }

    /// 第 i 个格点的首地址；Dof 个分量连续存放。
    [[nodiscard]] T*       site(std::size_t i) noexcept { return data_.get() + i * Dof; }
    [[nodiscard]] const T* site(std::size_t i) const noexcept { return data_.get() + i * Dof; }

    /// 非拥有视图；内核按值接收它。
    [[nodiscard]] FieldView<T, Dof, Dim> view() noexcept { return {data_.get(), n_, ext_}; }
    [[nodiscard]] FieldView<const T, Dof, Dim> view() const noexcept
    { return {data_.get(), n_, ext_}; }

private:
    std::array<std::size_t, Dim>           ext_;
    std::size_t                            n_;
    std::unique_ptr<T[], AlignedDelete<T>> data_;
};

template <class T, std::size_t Dof, std::size_t Dim>
Field<T, Dof, Dim>::Field(std::array<std::size_t, Dim> ext, T fill)
    : ext_(ext)
    , n_(std::accumulate(ext.begin(), ext.end(), std::size_t{1}, std::multiplies<>{}))
    , data_(static_cast<T*>(::operator new[](n_ * Dof * sizeof(T),
                                              std::align_val_t{kAlignment})))
{
    std::fill_n(data_.get(), n_ * Dof, fill);
}

template <class T, std::size_t Dof, std::size_t Dim>
Field<T, Dof, Dim>::Field(Field&& other) noexcept
    : ext_(other.ext_)
    , n_(other.n_)
    , data_(std::move(other.data_))
{
    other.ext_.fill(0);
    other.n_ = 0;
}

template <class T, std::size_t Dof, std::size_t Dim>
Field<T, Dof, Dim>& Field<T, Dof, Dim>::operator=(Field&& other) noexcept
{
    if (this != &other) {
        ext_    = other.ext_;
        n_      = other.n_;
        data_   = std::move(other.data_);
        other.ext_.fill(0);
        other.n_ = 0;
    }
    return *this;
}
