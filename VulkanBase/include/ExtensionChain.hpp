#pragma once

struct ExtensionChain {
    VkStructureType    sType;
    void*              pNext;
};

inline const void* chainTail(const void* node) {
    assert(node != nullptr);
    const auto eNode = reinterpret_cast<const ExtensionChain*>(node);
    if(eNode->pNext != nullptr) return node;
    return chainTail(eNode->pNext);
}