#include "utility/filemanager.hpp"

#include <thread>
#include <utility>

FileManager::FileManager(std::vector<fs::path>  searchPaths)
:searchPaths_{begin(searchPaths), end(searchPaths)}{

}

void FileManager::addSearchPath(const fs::path &searchPath) {
    searchPaths_.push_back(searchPath);
}

void FileManager::addSearchPathFront(const fs::path &searchPath) {
    searchPaths_.push_front(searchPath);
}


byte_string FileManager::load(const std::string &resource, bool binary) const {
    auto maybePath = getFullPath(resource);
    if(!maybePath.has_value()){
        throw std::runtime_error{fmt::format("resource: {} does not exists", resource)};
    }
    return loadFile(maybePath->string(), binary);
}

std::optional<fs::path> FileManager::getFullPath(const std::string &resource) const {
    // TODO add recursive path search
    assert(!searchPaths_.empty());
    fs::path path = resource;
    auto itr = begin(searchPaths_);
    while(!exists(path) && itr != end(searchPaths_)){
        path = fmt::format("{}/{}", itr->string(), resource);  // FIXME use platform specific path
        itr++;
    }
    return exists(path) ? std::optional{path} : std::nullopt;
}

FileManager &FileManager::instance() {
    return instance_;
}

FileManager FileManager::createInstance(const std::vector<fs::path>& searchPaths) {
    static std::once_flag flag;
    static auto create = [&]{
        instance_ = FileManager{ searchPaths };
    };
    std::call_once(flag, create);
    return instance_;
}

void FileManager::save(const std::string &content, const fs::path &outputPath, bool saveAsBinary) {
    fs::path fullPath = fs::current_path() / outputPath;

    fs::create_directories(fullPath.parent_path());

    std::ofstream file(fullPath, saveAsBinary ? std::ios::binary : std::ios::out);
    if (!file.is_open()) {
        throw std::runtime_error{fmt::format("unable to open {} for writing", outputPath.string())};
    }

    file.write(content.data(), static_cast<std::streamsize>(content.size()));

}

FileManager FileManager::instance_ = createInstance();

std::string FileManager::resource(const std::string &name) {
    return instance_.getFullPath(name)->string();
}
