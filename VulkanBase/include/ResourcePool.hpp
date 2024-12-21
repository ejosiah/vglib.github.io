#pragma once

#include <vector>
#include <optional>
#include <functional>
#include <cassert>

template<typename T>
class ResourcePool {
public:
    using Borrowed = std::optional<std::reference_wrapper<T>>;

    ResourcePool() = default;

    template<typename Factory>
    explicit ResourcePool(Factory factory, size_t capacity);

    explicit ResourcePool(std::vector<T> data);

    Borrowed obtain();

    void release(Borrowed borrowed);

    void releaseAll();

private:
    std::vector<T> data_;
    std::vector<Borrowed> resources_;
};

template<typename T>
template<typename Factory>
ResourcePool<T>::ResourcePool(Factory factory, size_t capacity){
    for(auto i = 0; i < capacity; ++i) {
        data_.push_back(factory());
    }
    releaseAll();
}

template<typename T>
void ResourcePool<T>::release(ResourcePool::Borrowed borrowed) {
    resources_.push_back(std::move(borrowed));
    assert(resources_.size() <= data_.size());
}

template<typename T>
ResourcePool<T>::ResourcePool(std::vector<T> data)
: data_(std::move(data))
, resources_(data_.size())
{
    const auto capacity = data_.size();
    for(auto i = 0; i < capacity; ++i) {
        resources_[i] = std::ref(data_[i]);
    }
}

template<typename T>
ResourcePool<T>::Borrowed ResourcePool<T>::obtain() {
    if(resources_.empty()) {
        return {};
    }
    auto resource = std::move(resources_.back());
    resources_.pop_back();

    return resource;
}

template<typename T>
void ResourcePool<T>::releaseAll() {
    const auto capacity = data_.size();
    for(auto i = 0; i < capacity; ++i) {
        resources_.push_back(std::ref(data_[i]));
    }
}