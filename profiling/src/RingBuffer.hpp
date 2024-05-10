#pragma once

#include <array>
#include <atomic>
#include <optional>

template<typename Entry, size_t Capacity>
class RingBuffer {
public:
    RingBuffer() = default;

    void push(Entry entry) {
        m_data[++m_writeIndex % Capacity] = entry;
    }

    bool size() const {
        return (m_writeIndex - m_readIndex) + 1;
    }

    bool empty() const {
        return m_writeIndex < m_readIndex;
    }

    std::optional<Entry> poll() {
        if(empty()){
            return {};
        }
        auto value = m_data[m_readIndex % Capacity];
        m_readIndex++;
        return value;
    }

private:
    std::array<Entry, Capacity> m_data;
    std::atomic_int m_writeIndex{-1};
    std::atomic_int m_readIndex{0};
};