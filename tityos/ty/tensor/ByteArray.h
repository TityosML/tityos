#include "tityos/ty/tensor/Dtype.h"
#include <cstdlib>

namespace ty {
    namespace internal {
        class ByteArray {
          private:
            void *startPointer_;
            size_t size_;

          public:
            ByteArray(size_t numBytes) : size_(numBytes) {
                startPointer_ = std::malloc(numBytes);

                if (startPointer_ == nullptr) {
                    // TODO: Logging
                }
            }

            ~ByteArray() {
                std::free(startPointer_);
            }

            void *at(size_t index) {
                return static_cast<char *>(startPointer_) + index;
            }

            size_t getSize() const {
                return size_;
            }
        };
    } // namespace internal
} // namespace ty