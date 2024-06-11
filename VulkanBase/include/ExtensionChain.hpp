#pragma once

struct ExtensionChain {
    VkStructureType    sType{};
    void*              pNext{};
};

inline const void* chainTail(const void* node) {
    assert(node != nullptr);
    const auto eNode = reinterpret_cast<const ExtensionChain*>(node);
    if(eNode->pNext != nullptr) return node;
    return chainTail(eNode->pNext);
}

inline bool containsExtension(VkStructureType structType, const void* chain) {
    void* next = const_cast<void*>(chain);

    do{
        if(reinterpret_cast<ExtensionChain*>(next)->sType == structType) {
            return true;
        }
        next = reinterpret_cast<ExtensionChain*>(next)->pNext;
    }while(next != nullptr);

    return false;
}