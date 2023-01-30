#include "dds.hpp"
#include <fstream>
#include <cassert>
#include <spdlog/spdlog.h>

namespace dds {

    StatusCode save(const SaveInfo &saveInfo, const BYTE *data) {
        if(saveInfo.type == Caps2::VOLUME){
            assert(saveInfo.depth > 1);
        }
        if(saveInfo.type == Caps2::CUBEMAP){
            assert(saveInfo.depth == 6);
        }
        File file{};
        file.header.dwFlags = Data::CAPS | Data::WIDTH | Data::HEIGHT | Data::PIXELFORMAT;
        file.header.dwFlags |= (saveInfo.depth > 1) ? Data::DEPTH : 0;
        file.header.dwCaps |= (saveInfo.type != Caps2::BASIC) ? Caps::COMPLEX : 0;
        file.header.dwCaps2 = saveInfo.type;
        file.header.dwWidth = saveInfo.width;
        file.header.dwHeight = saveInfo.height;
        file.header.dwDepth = saveInfo.depth;
        file.header.ddspf.dwRGBBitCount = saveInfo.numChannels * saveInfo.channelSize * 8;

        auto channels = saveInfo.numChannels;

        if(channels >= 1){
            file.header.ddspf.dwRBitMask = 0x000000FF;
        }
        if(channels >= 2){
            file.header.ddspf.dwGBitMask = 0x0000FF00;
        }
        if(channels >= 3){
            file.header.ddspf.dwFlags = DDPF::RGB;
            file.header.ddspf.dwBBitMask = 0x00FF0000;
        }
        if(channels >= 4){
            file.header.ddspf.dwFlags |= DDPF::ALPHAPIXELS;
            file.header.ddspf.dwABitMask = 0xFF000000;
        }

        file.bdata = const_cast<BYTE*>(data);
        
        auto outputPath = saveInfo.path;
        size_t size = saveInfo.numChannels * saveInfo.channelSize * saveInfo.width * saveInfo.height * saveInfo.depth;

        return save(file, size, outputPath);
    }

    StatusCode save(const File &file, size_t size, std::string &outputPath) {
        std::ofstream fout{ outputPath.data(), std::ios::binary };
        if(!fout.good()){
            spdlog::error("unable to open output file: {} for writing", outputPath);
            return ERROR_WRITING_OUTPUT;
        }

        auto& mFile = const_cast<File&>(file);

        fout.write(reinterpret_cast<char*>(&mFile.dwMagic), sizeof(DWORD));
        fout.write(reinterpret_cast<char*>(&mFile.header), file.header.dwSize);
        fout.write(file.bdata, size);
        fout.flush();
        fout.close();

        spdlog::info("dds successfully written to : {}", outputPath);

        return StatusCode::SUCCESS;
    }
}