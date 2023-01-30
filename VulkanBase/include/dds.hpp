#pragma once

#include <cstdint>
#include <string>

namespace dds{
    using DWORD = uint32_t;
    using BYTE = char;
    using Width = DWORD;
    using Height = DWORD;
    using Depth = DWORD;
    using Channels = DWORD;
    using Filename = std::string;

    enum StatusCode : int {
        SUCCESS = 0,
        MISSING_DIMENSION = 100,
        MISSING_CHANNEL = 200,
        ERROR_READING_INPUT = 300,
        ERROR_WRITING_OUTPUT = 400
    };

    enum Caps : DWORD {
        COMPLEX = 0x8,
        MIPMAP  = 0x400000,
        TEXTURE = 0x1000
    };

    enum Caps2 : DWORD {
        BASIC              = 0x0,
        CUBEMAP            = 0x200,
        CUBEMAP_POSITIVEX  = 0x400,
        CUBEMAP_NEGATIVEX  = 0x800,
        CUBEMAP_POSITIVEY  = 0x1000,
        CUBEMAP_NEGATIVEY  = 0x2000,
        CUBEMAP_POSITIVEZ  = 0x4000,
        CUBEMAP_NEGATIVEZ  = 0x8000,
        VOLUME             = 0x200000
    };

    enum Data : DWORD {
        CAPS = 0x1,
        HEIGHT = 0x2,
        WIDTH = 0x4,
        PITCH = 0x8,
        PIXELFORMAT = 0x1000,
        MIPMAPCOUNT = 0x20000,
        LINEARSIZE = 0x80000,
        DEPTH = 0x800000
    };

    enum DDPF : DWORD {
        ALPHAPIXELS = 0x1,
        ALPHA = 0x2,
        FOURCC = 0x4,
        RGB = 0x40,
        YUV = 0x200,
        LUMINANCE = 0x20000
    };

    struct PixelFormat {
        DWORD dwSize{32};
        DWORD dwFlags{DDPF::RGB | DDPF::ALPHAPIXELS};
        DWORD dwFourCC{0};
        DWORD dwRGBBitCount{0};
        DWORD dwRBitMask{0};
        DWORD dwGBitMask{0};
        DWORD dwBBitMask{0};
        DWORD dwABitMask{0};
    };

    struct Header {
        DWORD           dwSize{124};
        DWORD           dwFlags;
        DWORD           dwHeight;
        DWORD           dwWidth;
        DWORD           dwPitchOrLinearSize;
        DWORD           dwDepth;
        DWORD           dwMipMapCount;
        DWORD           dwReserved1[11];
        PixelFormat     ddspf;
        DWORD           dwCaps{Caps::TEXTURE};
        DWORD           dwCaps2;
        DWORD           dwCaps3;
        DWORD           dwCaps4;
        DWORD           dwReserved2;
    };

    struct File {
        DWORD dwMagic{0x20534444};
        Header header;
        BYTE* bdata{};
        BYTE* bdata2{};
    };

    struct SaveInfo{
        uint32_t width;
        uint32_t height;
        uint32_t depth{1};
        uint32_t channelSize{1};
        uint32_t numChannels{3};
        uint32_t mipmaps{0};
        std::string path;
        Caps2 type{Caps2::BASIC};
    };

    StatusCode save(const SaveInfo& saveInfo, const BYTE* data);

    StatusCode save(const File& file, size_t size, std::string& outputPath);
}