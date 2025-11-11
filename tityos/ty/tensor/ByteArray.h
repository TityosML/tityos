#include <cstdlib>
#include "tityos/ty/tensor/Dtype.h"

class ByteArray {
    private:
        void* startPointer;
        size_t size;

    public:
        ByteArray(size_t numBytes) : size(numBytes) {
            startPointer = std::malloc(numBytes);
        }

        ~ByteArray() {
            std::free(startPointer);
        }

        void* at(size_t index) {
            return static_cast<char*>(startPointer) + index;
        }

        size_t getSize() const {
            return size;
        }
};