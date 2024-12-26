#pragma once

#include "VulkanDevice.h"
#include "VulkanBuffer.h"
#include "HashTable.hpp"

namespace gpu {
    class HashSet : public HashTable {
    public:
        HashSet() = default;

        HashSet(VulkanDevice &device, VulkanDescriptorPool& descriptorPool, uint32_t capacity)
        :HashTable(device, descriptorPool, capacity, true){}

    protected:
        std::vector<uint32_t> insert_shader_source() final;

        std::vector<uint32_t> find_shader_source() final;

        std::vector<uint32_t> remove_shader_source() override;
    };
}