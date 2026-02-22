#pragma once

#include "tityos/ty/backend/CPUBackend.h"
#include "tityos/ty/ops/cpu/TensorView.h"
#include "tityos/ty/ops/cpu/avx/add.h"
#include "tityos/ty/ops/dispatchDType.h"
#include "tityos/ty/ops/empty.h"
#include "tityos/ty/tensor/BaseTensor.h"

#include <cstdint>
#include <omp.h>
#include <stdexcept>