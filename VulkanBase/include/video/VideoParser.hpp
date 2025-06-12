#pragma once

#include "VulkanDevice.h"
#include "Video.hpp"

#include <memory>
#include <filesystem>

namespace video {

    class VideoParser {
    public:
        explicit VideoParser(VulkanDevice& device);

        std::shared_ptr<Video> parse(const std::filesystem::path& path);

    private:
        VulkanDevice* m_device{};
    };
}