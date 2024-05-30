#pragma once

#include "common.h"

class FileManager{
public:
    explicit FileManager(std::vector<fs::path> searchPaths = {});

    void addSearchPath(const fs::path& searchPath);

    void addSearchPathFront(const fs::path& searchPath);

    [[nodiscard]]
    byte_string load(const std::string& resource) const;

    [[nodiscard]]
    std::optional<fs::path> getFullPath(const std::string& resource) const;

    static FileManager& instance();

    static std::string resource(const std::string& name);

private:
    static FileManager createInstance(const std::vector<fs::path>& searchPath = {});

private:
    std::deque<fs::path> searchPaths_;
    static FileManager instance_;
};