#pragma once

#include <cstdint>
#include <unordered_map>
#include <atomic>
#include <functional>
#include <utility>
#include <spdlog/spdlog.h>
#include <string>
#include <mutex>

using ResourceHandle = uint64_t;
using ResourceCleaner = std::function<void(ResourceHandle)>;

static ResourceCleaner VoidCleaner = [](ResourceHandle) {};

class RefCounted {
public:
    RefCounted() = default;

    RefCounted(ResourceHandle handle, const ResourceCleaner& cleaner, const std::string& name = "")
    : _handle{handle}
    , _cleanup{ cleaner }
    , _name{ name }
    {
        if(_handle != 0) {
            counts[_handle]++; // TODO might race
            spdlog::debug("ref added, {} references to {}[{}]", counts[_handle].load(), _name, _handle);
        }
    }

    RefCounted(const RefCounted& source)
    : RefCounted(source._handle, source._cleanup, source._name)
    {}

    RefCounted(RefCounted&& source) noexcept
    : _handle( std::exchange(source._handle, 0) )
    , _cleanup( std::exchange(source._cleanup, VoidCleaner ))
    , _name( std::exchange(source._name, ""))
    {}


    virtual ~RefCounted() {
        releaseRef();
    }

    void copyRef(const RefCounted& source) {
        if(this != &source) {
            releaseRef();
            copyFrom(source);
            incrementRef();
        }
    }

    void moveRef(RefCounted&& source) noexcept {
        if(this == &source) {
            return;
        }
        releaseRef();
        _handle = std::exchange(source._handle, 0);
        _cleanup = std::exchange(source._cleanup, VoidCleaner);
        _name = std::exchange(source._name, "");
    }

    static uint32_t references(ResourceHandle handle) {
        auto itr = counts.find(handle);
        return itr != counts.end() ? itr->second.load() : 0;
    }

private:
    void copyFrom(const RefCounted& source) {
        _handle = source._handle;
        _cleanup = source._cleanup;
        _name = source._name;
    }

    void releaseRef() {
        if(_handle != 0 && decrementRef() == 0) {
            spdlog::debug("no more references to {}[{}], deletion in progress", _name, _handle);
            _cleanup(_handle);
            spdlog::debug("{}[{}] successfully deleted", _name, _handle);
        }
        _handle = 0;
        _cleanup = VoidCleaner;
        _name.clear();
    }

    void incrementRef() const {
        if(_handle != 0) {
            auto& count = counts[_handle];
            ++count;
            spdlog::debug("ref added, {} references to {}[{}]", count.load(), _name, _handle);
        }
    }

    [[nodiscard]] uint32_t decrementRef() const {
        if(!counts.contains(_handle)) {
            return ~0u;
        }
        auto count = --counts[_handle];
        if(count == 0u){
            auto itr = counts.find(_handle);
            counts.erase(itr);
        }else{
            spdlog::debug("ref removed, {} references to {}[{}]", count, _name, _handle);
        }
        return count;
    }
private:
    ResourceHandle _handle{};
    ResourceCleaner _cleanup{};
    std::string _name{};
    static std::unordered_map<ResourceHandle, std::atomic_uint32_t> counts;
};
