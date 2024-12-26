#pragma once

#include "VulkanDevice.h"
#include "VulkanBuffer.h"
#include "HashTable.hpp"

namespace gpu {
    class HashMap : public HashTable {
    public:
        HashMap() = default;

        HashMap(VulkanDevice &device, VulkanDescriptorPool& descriptorPool, uint32_t capacity)
        :HashTable(device, descriptorPool, capacity){}

    protected:
        std::vector<uint32_t> insert_shader_source() override;

        std::vector<uint32_t> find_shader_source() override;

        std::vector<uint32_t> remove_shader_source() override;

    };
}