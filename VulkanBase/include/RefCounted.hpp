#pragma once

#include <cstdint>
#include <unordered_map>
#include <atomic>
#include <functional>
#include <utility>

using ResourceHandle = uint64_t;
using ResourceCleaner = std::function<void(ResourceHandle)>;

static ResourceCleaner VoidCleaner = [](ResourceHandle) {};

class RefCounted {
public:
    RefCounted() = default;

    RefCounted(ResourceHandle handle, ResourceCleaner cleaner)
    : _handle{handle}
    , _cleanup{ std::move(cleaner) } {
        if(!counts.contains(_handle)) {
            counts[_handle] = 0;
        }
        counts[_handle]++;
    }

    RefCounted(const RefCounted& source)
    : RefCounted(source._handle, source._cleanup)
    {}

    RefCounted(RefCounted&& source) noexcept
    : _handle( std::exchange(_handle, 0) )
    , _cleanup( std::exchange(_cleanup, VoidCleaner ))
    {}


    virtual ~RefCounted() {
        if(decrementRef() == 0) {
            _cleanup(_handle);
        }
    }

    void copyRef(const RefCounted& source) {
        if(this != &source) {
            _handle = source._handle;
            _cleanup = source._cleanup;
            incrementRef();
        }
    }

    void moveRef(RefCounted&& source) noexcept {
        _handle = std::exchange(source._handle, 0);
        _cleanup = std::exchange(source._cleanup, ResourceCleaner{});
    }

    static uint32_t references(ResourceHandle handle) {
        auto itr = counts.find(handle);
        return itr != counts.end() ? itr->second.load() : 0;
    }

private:
    void incrementRef() const {
        auto itr = counts.find(_handle);
        if(itr != counts.end()){
            ++itr->second;
        }
    }

    [[nodiscard]] uint32_t decrementRef() const {
        auto itr = counts.find(_handle);
        if(itr == counts.end()) {
            return -1;
        }
        auto count = --itr->second;
        if(count == 0){
            counts.erase(itr);
        }
        return count;
    }
private:
    ResourceHandle _handle;
    ResourceCleaner _cleanup;
    static std::unordered_map<ResourceHandle, std::atomic_uint32_t> counts;
};