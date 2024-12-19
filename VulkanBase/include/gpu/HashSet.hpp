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
        std::string insertShaderPath() override {
            return "data/shaders/data_structure/cuckoo_hash_set_insert.comp.spv";
        }

        std::string findShaderPath() override {
            return "data/shaders/data_structure/cuckoo_hash_set_query.comp.spv";
        }

    };
}