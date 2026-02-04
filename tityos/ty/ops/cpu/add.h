#pragma once

#include "tityos/ty/backend/CPUBackend.h"
#include "tityos/ty/ops/cpu/TensorView.h"
#include "tityos/ty/tensor/BaseTensor.h"
#include "tityos/ty/ops/cpu/avx/add.h"
#include "tityos/ty/ops/empty.h"

#include <cstdint>
#include <stdexcept>
#include <omp.h>