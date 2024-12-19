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
        std::string insertShaderPath() override {
            return "data/shaders/data_structure/cuckoo_hash_map_insert.comp.spv";
        }

        std::string findShaderPath() override {
            return "data/shaders/data_structure/cuckoo_hash_map_query.comp.spv";
        }


    };
}