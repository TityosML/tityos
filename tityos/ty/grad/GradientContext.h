#pragma once

#include <functional>
#include <memory>

namespace ty {
namespace internal {
    class BaseTensor;
    class GradientContextStorage;

    using GradFn = std::function<BaseTensor(const GradientContextStorage&)>;

    class GradientContextStorage {
      private:
        std::vector<std::shared_ptr<BaseTensor>> tensors_;
        size_t numTensors_;

      public:
        template <typename... Args> GradientContextStorage(Args&&... args) {
            numTensors_ = sizeof...(Args);

            tensors_ = {std::forward<Args>(args)...};
        }
        ~GradientContextStorage() = default;

        const BaseTensor& operator[](size_t index) const {
            return *tensors_[index];
        }
    };

    class GradientContext {
      private:
        GradientContextStorage tensors_;
        GradFn gradFn_;

      public:
        GradientContext(const GradientContextStorage& tensors, GradFn gradFn)
            : tensors_(tensors), gradFn_(gradFn) {}
        ~GradientContext() = default;
    };
} // namespace internal
} // namespace ty