#include "RefCounted.hpp"

std::unordered_map<ResourceHandle, std::atomic_uint32_t> RefCounted::counts{};
