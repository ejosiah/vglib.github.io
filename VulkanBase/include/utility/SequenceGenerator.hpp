#pragma once

#include <atomic>

template<typename T>
inline auto sequence(T start = 0) {
    return [next=start] () mutable {
        auto current = next;
        ++next;
        return current;
    };
}
