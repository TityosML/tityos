#pragma once

#include "tityos/ty/tensor/Tensor.h"
#include "tityos/ty/ops/cpu/TensorView.h"
#include "tityos/ty/backend/CPUBackend.h"
#include "tityos/ty/ops/cpu/avx2/add.h"

#include <cstdint>
#include <stdexcept>
#include <omp.h>