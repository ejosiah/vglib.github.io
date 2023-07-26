#pragma once

#include <string>
#include <vector>
#include "common.h"
#include <sstream>
#include <iomanip>
#include <iostream>


std::string hexdump(const std::string& path){
    auto rawData = loadFile(path);
    auto size = rawData.size()/sizeof(uint32_t);
    auto data  = reinterpret_cast<uint32_t*>(rawData.data());

    std::stringstream ss{};
    std::stringstream toHex{};

    for(int i = 0; i < size; ++i){
        toHex << std::hex << data[i];
        auto hexString = toHex.str();
        ss << "0x";
        ss << std::setfill('0') << std::setw(8) <<  hexString << ((i+1)%8 == 0 ? ",\n" : ", ");
        toHex.str("");
        toHex.clear();
    }
    return ss.str();
}