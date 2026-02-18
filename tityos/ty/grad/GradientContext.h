#pragma once

#include <functional>
#include <memory>

namespace ty {
  class Tensor;
namespace internal {
    class GradientContextStorage;

    using GradFn = std::function<void(const GradientContextStorage&)>;

    class GradientContextStorage {
      private:
        std::vector<Tensor> tensors_;
        size_t numTensors_;

      public:
        template <typename... Args> GradientContextStorage(Args&&... args) {
            numTensors_ = sizeof...(Args);

            tensors_ = {std::forward<Args>(args)...};
        }
        ~GradientContextStorage() = default;

        const Tensor& operator[](size_t index) const { return tensors_[index]; }
    };

    class GradientContext {
      private:
        GradientContextStorage tensors_;
        GradFn gradFn_;

      public:
        GradientContext(const GradientContextStorage& tensors, GradFn gradFn)
            : tensors_(tensors), gradFn_(gradFn) {}
        ~GradientContext() = default;

        void backward() const { gradFn_(tensors_); }
    };
} // namespace internal
} // namespace ty