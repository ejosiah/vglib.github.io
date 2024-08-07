#pragma once

#include "common.h"
#include "RefCounted.hpp"
#include "VulkanRAII.h"

#include <atomic>

struct VulkanDescriptorPool : RefCounted {


    VulkanDescriptorPool() = default;

    VulkanDescriptorPool(VkDevice device, VkDescriptorPool pool)
    : RefCounted((ResourceHandle)pool, [device, pool](ResourceHandle){ vkDestroyDescriptorPool(device, pool, VK_NULL_HANDLE); }, "VkDescriptorPool")
    , device(device)
    , pool(pool)
    {}

    VulkanDescriptorPool(const VulkanDescriptorPool& source)
    : RefCounted(source)
    , device(source.device)
    , pool(source.pool)
    {}

    VulkanDescriptorPool(VulkanDescriptorPool&& source) noexcept {
        operator=(static_cast<VulkanDescriptorPool&&>(source));
    }

    ~VulkanDescriptorPool() override = default;

    VulkanDescriptorPool& operator=(const VulkanDescriptorPool& source) {
        if(this == &source) return *this;

        copyRef(source);
        this->device = source.device;
        this->pool = source.pool;
    }

    VulkanDescriptorPool& operator=(VulkanDescriptorPool&& source) noexcept {
        if(this == &source) return *this;

        if(pool){
            this->~VulkanDescriptorPool();
        }

        moveRef(static_cast<RefCounted&&>(source));
        this->device = std::exchange(source.device, VK_NULL_HANDLE);
        this->pool = std::exchange(source.pool, VK_NULL_HANDLE);

        return *this;
    }

    [[nodiscard]]
    inline std::vector<VkDescriptorSet> allocate(const std::vector<VulkanDescriptorSetLayout>& layouts) const {
        std::vector<VkDescriptorSetLayout> handles{};
        for(const auto& layout : layouts) handles.push_back(layout.handle);
        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = pool;
        allocInfo.descriptorSetCount = COUNT(layouts);
        allocInfo.pSetLayouts = handles.data();

        std::vector<VkDescriptorSet> sets(layouts.size());
        vkAllocateDescriptorSets(device, &allocInfo, sets.data());
        allocationCount += layouts.size();

        return sets;
    }

    template<typename DescriptorSets>
    inline void allocate(const std::vector<VulkanDescriptorSetLayout>& layouts, DescriptorSets& descriptorSets){
        assert(descriptorSets.size() >= layouts.size());

        std::vector<VkDescriptorSetLayout> handles{};
        for(const auto& layout : layouts) handles.push_back(layout.handle);
        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = pool;
        allocInfo.descriptorSetCount = COUNT(layouts);
        allocInfo.pSetLayouts = handles.data();

        vkAllocateDescriptorSets(device, &allocInfo, descriptorSets.data());

        allocationCount += layouts.size();
    }

    inline void free(VkDescriptorSet set) const {
       vkFreeDescriptorSets(device, pool, 1, &set);
       --allocationCount;
    }

    inline void free(const std::vector<VkDescriptorSet>& sets) const {
        vkFreeDescriptorSets(device, pool ,COUNT(sets), sets.data());
        allocationCount += sets.size();
    }

    inline void reset() const {
        vkResetDescriptorPool(device, pool, 0);
        allocationCount = 0;
    }


    operator VkDescriptorPool() const {
        return pool;
    }

    VkDevice device = VK_NULL_HANDLE;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    mutable std::atomic_uint32_t allocationCount{};
};