#pragma once

#include "common.h"
#include <vulkan/vulkan.h>
#include "vk_mem_alloc.h"

#include <span>

struct BufferRegion;

struct VulkanBuffer{

    VulkanBuffer() = default;

    inline VulkanBuffer(VmaAllocator allocator, VkBuffer buffer, VmaAllocation allocation, VkDeviceSize size, const std::string name = "", bool mappable = false)
    : allocator(allocator)
    , buffer(buffer)
    , allocation(allocation)
    , size(size)
    , name(name)
    , mappable(mappable)
    {
        refCounts[buffer] = 1;
    }

    VulkanBuffer(const VulkanBuffer& source){
        operator=(source);
    }

    VulkanBuffer& operator=(const VulkanBuffer& source){
        if(&source == this) return *this;

        allocator = source.allocator;
        buffer = source.buffer;
        allocation = source.allocation;
        name = source.name;
        size = source.size;
        if(buffer) {
            incrementRef(buffer);
        }
        return *this;
    }

    VulkanBuffer(VulkanBuffer&& source) noexcept {
        operator=(static_cast<VulkanBuffer&&>(source));
    }

    VulkanBuffer& operator=(VulkanBuffer&& source) noexcept {
        if(&source == this) return *this;

        this->~VulkanBuffer();

        allocator = std::exchange(source.allocator, VK_NULL_HANDLE);
        buffer = std::exchange(source.buffer, VK_NULL_HANDLE);
        allocation = std::exchange(source.allocation, VK_NULL_HANDLE);
        name = std::exchange(source.name, "");
        size = std::exchange(source.size, 0);
        op_handle = std::exchange(source.op_handle, {});

        return *this;
    }

    template<typename T>
    void copy(std::vector<T> source, uint32_t offset = 0) const {
        assert(!source.empty());
        copy(source.data(), sizeof(T) * source.size(), offset); // FIXME offset should be offset * sizeof(T)
    }

    void copy(const void* source, VkDeviceSize size, uint32_t offset = 0) const {
        assert(size + offset <= this->size);
        void* dest;
        vmaMapMemory(allocator, allocation, &dest);
        dest = static_cast<char*>(dest) + offset;
        memcpy(dest, source, size);
        vmaUnmapMemory(allocator, allocation);
    }

    template<typename T>
    void map(std::function<void(T*)> use){
        auto ptr = map();
        use(reinterpret_cast<T*>(ptr));
        unmap();
    }

    template<typename T>
    void use(std::function<void(T)> func){
        map<T>([&](auto* ptr){
            int n = size/(sizeof(T));
           for(auto i = 0; i < n; i++){
               func(*(ptr+i));
           }
        });
    }

    ~VulkanBuffer(){
        if(buffer){
            auto itr = refCounts.find(buffer);
            assert(itr != refCounts.end());
            if(itr->second == 1) {
                spdlog::debug("no more references to VkBuffer[{}], destroying now ...", (uint64_t)buffer);
                refCounts.erase(itr);
                if (mapped) {
                    unmap();
                }
                if(op_handle.has_value()) {
#ifdef WIN32
                    CloseHandle(op_handle.value());
#else
                    if(op_handle.value() != -1){
                        close(op_handle.value());
                        op_handle = -1;
                    }
#endif
                }
                vmaDestroyBuffer(allocator, buffer, allocation);
            }else{
                decrementRef(buffer);
            }
        }
    }

#ifdef WIN32
    HANDLE getHandle(VkDevice device) const {
        if(op_handle.has_value()){
            return op_handle.value();
        }else {
            VkMemoryGetWin32HandleInfoKHR getMemoryInfo{VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR};
            getMemoryInfo.memory = allocationInfo().deviceMemory;
            getMemoryInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;

            HANDLE handle;
            vkGetMemoryWin32HandleKHR(device, &getMemoryInfo, &handle);
            op_handle = handle;
            return handle;
        }

    }
#else
    int getHandle(VkDevice device) const {
        if(op_handle.has_value()){
            return op_handle.value();
        }else{
            VmaAllocationInfo info;
            vmaGetAllocationInfo(allocator, allocation, &info);
            auto memory = info.deviceMemory;
            VkMemoryGetFdInfoKHR getMemoryInfo{ VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR };
            getMemoryInfo.memory = info.deviceMemory;
            getMemoryInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;

            int handle;
            vkGetMemoryFdKHR(device, &getMemoryInfo, &handle);
            op_handle = handle;
            return handle;
        }
    }
#endif


    [[nodiscard]]
    inline VmaAllocationInfo allocationInfo() const {
        VmaAllocationInfo info;
        vmaGetAllocationInfo(allocator, allocation, &info);
        return info;
    }

    operator VkBuffer() const {
        return buffer;
    }

    operator VkBuffer*() {
        return &buffer;
    }

    operator bool() const {
        return buffer != VK_NULL_HANDLE;
    }

    void* map() const {
        if(mapped) return mapped;
        vmaMapMemory(allocator, allocation, &mapped);
        return mapped;
    }

    void unmap() const {
        if(!mapped) return;
        vmaUnmapMemory(allocator, allocation);
        mapped = nullptr;
    }

    template<typename T>
    T get(int index){
        if(mapped){
            return reinterpret_cast<T*>(mapped)[index];
        }
        // TODO check if mappable & bounds
        T res;
        map<T>([&](auto ptr){
            res = ptr[index];
        });
        return res;
    }

    void clear(VkCommandBuffer commandBuffer) const {
        vkCmdFillBuffer(commandBuffer, buffer, 0, size, 0);
    }

    static void incrementRef(VkBuffer buffer){
        ensureRef(buffer);
        refCounts[buffer]++;
        spdlog::debug("{} current references to VkBuffer[{}]", refCounts[buffer].load(), (uint64_t)buffer);
    }

    static void decrementRef(VkBuffer buffer){
        ensureRef(buffer);
        refCounts[buffer]--;
        spdlog::debug("{} current references to VkBuffer[{}]", refCounts[buffer].load(), (uint64_t)buffer);
    }

    static void ensureRef(VkBuffer buffer){
        assert(refCounts.find(buffer) != refCounts.end());
    }

    template<typename T>
    size_t sizeAs() const {
        return size/sizeof(T);
    }

    template<typename T>
    std::span<T> span(size_t aSize = std::numeric_limits<size_t>::max()) const {
        aSize = glm::min(aSize, sizeAs<T>());
        return { reinterpret_cast<T*>(map()), aSize } ;
    }

    BufferRegion region(VkDeviceSize start, VkDeviceSize end);

    VmaAllocator allocator = VK_NULL_HANDLE;
    VkBuffer buffer = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    VkDeviceSize  size = 0;
    std::string name{};
    mutable void* mapped = nullptr;
    bool isMapped = false;
    bool mappable = false;
    static std::map<VkBuffer, std::atomic_uint32_t> refCounts;
#ifdef WIN32
    mutable std::optional<HANDLE> op_handle {};
#else
    std::optional<int> op_handle = -1;
#endif
};

struct BufferRegion {
    VulkanBuffer* buffer{};
    VkDeviceSize offset{0};
    VkDeviceSize end{VK_WHOLE_SIZE};

    inline VkDeviceSize size() const {
        return end - offset;
    }

    template<typename T>
    [[nodiscard]] size_t sizeAs() const {
        return size()/sizeof(T);
    }

    [[nodiscard]] auto map() const {
        return reinterpret_cast<char*>(buffer->map()) + offset;
    }

    void unmap() const {
        buffer->unmap();
    }

    template<typename T>
    std::span<T> span() const {
        return { reinterpret_cast<T*>(map()), sizeAs<T>() };
    }

    void upload(const void* data)  {
        buffer->copy(data, size(), offset);
    }

    void copy(const void* source, VkDeviceSize size, uint32_t rOffset) const {
        buffer->copy(source, size, offset + rOffset);
    }
};