#pragma once

#include "tityos/ty/export.h"

namespace ty {
TITYOS_EXPORT enum class DeviceType { CPU, CUDA };

TITYOS_EXPORT class Device {
  private:
    DeviceType type_;
    int index_;

  public:
    Device(DeviceType type, int index = 0) : type_(type), index_(index) {}

    bool operator==(const Device& other) const {
        return other.type_ == type_ && other.index_ == index_;
    }

    bool isCpu() const { return type_ == DeviceType::CPU; }
    bool isCuda() const { return type_ == DeviceType::CUDA; }
};
} // namespace ty