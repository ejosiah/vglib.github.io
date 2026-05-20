#pragma once

#include "common.h"

class FileManager{
public:
    explicit FileManager(std::vector<fs::path> searchPaths = {});

    void addSearchPath(const fs::path& searchPath);

    void addSearchPathFront(const fs::path& searchPath);

    [[nodiscard]]
    byte_string load(const std::string& resource, bool binary = true) const;

    [[nodiscard]]
    std::optional<fs::path> getFullPath(const std::string& resource) const;

    static FileManager& instance();

    static std::string resource(const std::string& name);

    static void save(const std::string& content, const fs::path& outputPath, bool saveAsBinary = true);

private:
    static FileManager createInstance(const std::vector<fs::path>& searchPath = {});


private:
    std::deque<fs::path> searchPaths_;
    static FileManager instance_;
};