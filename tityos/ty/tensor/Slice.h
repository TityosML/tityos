#pragma once

#include <optional>

namespace ty {
struct Slice {
    std::optional<ptrdiff_t> start;
    std::optional<ptrdiff_t> stop;
    ptrdiff_t step;

    Slice(std::optional<ptrdiff_t> start = std::nullopt,
          std::optional<ptrdiff_t> stop = std::nullopt, ptrdiff_t step = 1)
        : start(start), stop(stop), step(step) {}
};
} // namespace ty