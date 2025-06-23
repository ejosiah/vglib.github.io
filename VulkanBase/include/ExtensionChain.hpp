#pragma once

#include <optional>

struct ExtensionChain {
    VkStructureType    sType{};
    void*              pNext{};
};

inline const void* chainTail(const void* node) {
    assert(node != nullptr);
    const auto eNode = reinterpret_cast<const ExtensionChain*>(node);
    if(eNode->pNext != nullptr) return node;
    return chainTail(eNode->pNext); // FIXME changed to loop
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


template<typename Extension>
inline Extension* findExtension(VkStructureType type, void*& chain) {
    auto next = chain;
    while(next) {
        auto extension = reinterpret_cast<ExtensionChain*>(next);
        if(extension->sType == type) {
            return  reinterpret_cast<Extension*>(extension);
        }
        next = extension->pNext;
    }
    auto extension = new Extension{ type, chain };
    chain = extension;

    return extension;
}

inline void destroyExtensionChain(void* chain) {
    auto next = chain;
    while(next) {
        auto extension = reinterpret_cast<ExtensionChain*>(next);
        next = extension->pNext;
        delete extension;
    }
}

inline void logExtensions(void* chain) {
    auto next = chain;
    while(next) {
        auto extension = reinterpret_cast<ExtensionChain*>(next);
        next = extension->pNext;
    }
}