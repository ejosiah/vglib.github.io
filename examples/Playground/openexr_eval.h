#pragma once

//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

//-----------------------------------------------------------------------------
//
//	Code examples that show how class TiledInputFile and
//	class TiledOutputFile can be used to read and write
//	OpenEXR image files with an arbitrary set of channels.
//
//-----------------------------------------------------------------------------
#include <ImfArray.h>
#include <ImfChannelList.h>
#include <ImfMatrixAttribute.h>
#include <ImfStringAttribute.h>
#include <ImfTiledInputFile.h>
#include <ImfTiledOutputFile.h>

#include "drawImage.h"

#include <iostream>

#include "namespaceAlias.h"
using namespace IMF;
using namespace IMATH_NAMESPACE;

void
writeTiled1 (
        const char   fileName[],
        Array2D<GZ>& pixels,
        int          width,
        int          height,
        int          tileWidth,
        int          tileHeight)
{
    Header header (width, height);
    header.channels ().insert ("G", Channel (IMF::HALF));
    header.channels ().insert ("Z", Channel (IMF::FLOAT));

    header.setTileDescription (
            TileDescription (tileWidth, tileHeight, ONE_LEVEL));

    TiledOutputFile out (fileName, header);

    FrameBuffer frameBuffer;

    frameBuffer.insert (
            "G", // name
            Slice (
                    IMF::HALF,                       // type
                    (char*) &pixels[0][0].g,         // base
                    sizeof (pixels[0][0]) * 1,       // xStride
                    sizeof (pixels[0][0]) * width)); // yStride

    frameBuffer.insert (
            "Z", // name
            Slice (
                    IMF::FLOAT,                      // type
                    (char*) &pixels[0][0].z,         // base
                    sizeof (pixels[0][0]) * 1,       // xStride
                    sizeof (pixels[0][0]) * width)); // yStride

    out.setFrameBuffer (frameBuffer);
    out.writeTiles (0, out.numXTiles () - 1, 0, out.numYTiles () - 1);
}

void
readTiled1 (const char fileName[], Array2D<GZ>& pixels, int& width, int& height)
{
    TiledInputFile in (fileName);

    Box2i dw = in.header ().dataWindow ();
    width    = dw.max.x - dw.min.x + 1;
    height   = dw.max.y - dw.min.y + 1;
    int dx   = dw.min.x;
    int dy   = dw.min.y;

    pixels.resizeErase (height, width);

    FrameBuffer frameBuffer;

    frameBuffer.insert (
            "G", // name
            Slice (
                    IMF::HALF,                       // type
                    (char*) &pixels[-dy][-dx].g,     // base
                    sizeof (pixels[0][0]) * 1,       // xStride
                    sizeof (pixels[0][0]) * width)); // yStride

    frameBuffer.insert (
            "Z", // name
            Slice (
                    IMF::FLOAT,                      // type
                    (char*) &pixels[-dy][-dx].z,     // base
                    sizeof (pixels[0][0]) * 1,       // xStride
                    sizeof (pixels[0][0]) * width)); // yStride

    in.setFrameBuffer (frameBuffer);
    in.readTiles (0, in.numXTiles () - 1, 0, in.numYTiles () - 1);
}

void
generalInterfaceTiledExamples ()
{
    std::cout << "\nGZ (green, depth) tiled files\n" << std::endl;
    std::cout << "drawing image" << std::endl;

    int w = 800;
    int h = 600;

    Array2D<GZ> pixels (h, w);
    drawImage6 (pixels, w, h);

    std::cout << "writing file" << std::endl;

    writeTiled1 ("tiledgz1.exr", pixels, w, h, 64, 64);

    std::cout << "reading file" << std::endl;

    readTiled1 ("tiledgz1.exr", pixels, w, h);
}