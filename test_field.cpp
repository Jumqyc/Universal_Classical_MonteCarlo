#include "field.hpp"

#include <cstdint>
#include <cstdio>
#include <type_traits>
#include <utility>

using F = Field<double, 3, 3>;

static_assert(std::is_trivially_copyable_v<FieldView<double, 3, 3>>,
              "视图必须是平凡可复制的，才能作为内核参数按值传递");
static_assert(std::is_trivially_copyable_v<FieldView<const double, 3, 3>>);
static_assert(sizeof(Field<double, 3, 3>) == 40, "3*8 字节 shape + 8 字节 n + 8 字节指针");

static int failures = 0;

static void check(bool ok, const char* what)
{
    if (!ok) {
        std::printf("FAIL: %s\n", what);
        ++failures;
    }
}

/// 平均磁化强度：只接受只读视图，因此主机与设备可以共用同一个函数体。
static double magnetization(FieldView<const double, 3, 3> v)
{
    double m = 0;
    for (std::size_t i = 0; i < v.n; ++i) m += v.site(i)[0];
    return m / static_cast<double>(v.n);
}

int main()
{
    F f{{8, 8, 8}, 0.0};
    check(f.sites() == 512, "8^3 = 512 个格点");
    check(f.bytes() == 512 * 3 * 8, "512 * 3 * 8 = 12288 字节");
    check(f.extents()[0] == 8 && f.extents()[2] == 8, "shape 原样保存");
    check(reinterpret_cast<std::uintptr_t>(f.data()) % kAlignment == 0,
          "缓冲区按 64 B 对齐");

    f.site(5)[2] = 7.0;
    check(&f.site(5)[2] == f.data() + 17, "site(5)[2] 落在 data()[5*3+2]");
    check(f.site(4)[2] == 0.0, "fill 把其余分量置零");
    check(f.view().site(5)[2] == 7.0, "视图与所有者看到同一块内存");

    check(magnetization(f.view()) == 0.0, "磁化强度为 0（只有第 5 个格点非零，且落在第 2 分量）");

    F g{std::move(f)};
    check(g.sites() == 512 && g.site(5)[2] == 7.0, "移动后数据完整转移到 g");
    check(f.sites() == 0 && f.data() == nullptr, "移动后源对象为空");

    F h{{4, 4, 4}, 2.5};
    double sum = 0;
    for (std::size_t i = 0; i < h.sites(); ++i)
        for (std::size_t c = 0; c < 3; ++c) sum += h.site(i)[c];
    check(sum == 64.0 * 3 * 2.5, "fill 覆盖全部 64*3 个分量");

    if (failures == 0) std::printf("ALL CHECKS PASSED\n");
    return failures != 0;
}
