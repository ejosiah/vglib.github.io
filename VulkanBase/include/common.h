#pragma once

#ifndef NDEBUG
#define DEBUG_MODE
#else
#define RELEASE_MODE
#endif

#ifndef NDEBUG
constexpr bool debugMode = true;
#else
constexpr bool debugMode = false;
#endif

// TODO revite this include
#include "windows_include.h"

#include <string>
#include <string_view>
#include <array>
#include <vector>
#include <map>
#include <set>
#include <atomic>
#include <queue>
#include <thread>
#include <condition_variable>
#include <mutex>
#include <future>
#include <limits>
#include <algorithm>
#include <numeric>
#include <optional>
#include <variant>
#include <stdexcept>
#include <thread>
#include <chrono>
#include <random>
#include <functional>
#include <type_traits>
#include <memory>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_access.hpp>
#include <glm/gtc/epsilon.hpp>
#include <vulkan/vulkan.h>
#ifdef WIN32
#include <vulkan/vulkan_win32.h>
#endif
#include <fmt/chrono.h>
#include <fmt/ranges.h>
#include <spdlog/spdlog.h>
#include <chrono>
#include "vk_mem_alloc.h"
#include "xforms.h"
#include "glm_format.h"
#include <filesystem>
#include <fstream>
#include "color.hpp"

namespace chrono = std::chrono;
namespace fs = std::filesystem;

using real = float;
using uint = unsigned int;
using Flags = unsigned int;
using byte_string = std::vector<char>;
using ubyte_string = std::vector<unsigned char>;

constexpr float EPSILON = 0.000001;
constexpr float MAX_FLOAT = std::numeric_limits<float>::max();
constexpr float MIN_FLOAT = std::numeric_limits<float>::lowest();
constexpr std::chrono::seconds ONE_SECOND = std::chrono::seconds(1);

#define STRINGIZE(x) STRINGIZE2(x)
#define STRINGIZE2(x) #x
#define LINE_STRING STRINGIZE(__LINE__)
#define ERR_GUARD_VULKAN(expr) do { if((expr) < 0) { \
        assert(0 && #expr); \
        throw std::runtime_error(__FILE__ "(" LINE_STRING "): VkResult( " #expr " ) < 0"); \
    } } while(false)
#define COUNT(sequence) static_cast<uint32_t>(sequence.size())
#define BYTE_SIZE(sequence) static_cast<VkDeviceSize>(sizeof(sequence[0]) * sequence.size())

#define ASSERT(expr) if(!(expr)){ assert(expr); throw std::runtime_error(__FILE__ "(" LINE_STRING "): " #expr " not true"); }
#define ASSERT_MSG(expr, msg)               \
if(!(expr)){                                \
    assert(expr);                           \
    throw std::runtime_error(msg);          \
}

#ifndef UNUSED_VARIABLE
#   define UNUSED_VARIABLE(x) ((void)x)
#endif

using cstring = const char*;

inline bool closeEnough(float x, float y, float epsilon = 1E-3) {
    return fabs(x - y) <= epsilon * (abs(x) + abs(y) + 1.0f);
}

inline glm::quat fromAxisAngle(const glm::vec3& axis, const float angle) {
    float w = cos(glm::radians(angle) / 2);
    glm::vec3 xyz = axis * sin(glm::radians(angle) / 2);
    return glm::quat(w, xyz);
}

template <typename Predicate>
int findInterval(int size, const Predicate &pred) {
    int first = 0, len = size;
    while (len > 0) {
        int half = len >> 1, middle = first + half;
        // Bisect range based on value of _pred_ at _middle_
        if (pred(middle)) {
            first = middle + 1;
            len -= half + 1;
        } else
            len = half;
    }
    return std::clamp(first - 1, 0, size - 2);
}


template<typename VkObject, typename Provider>
inline std::vector<VkObject> get(Provider&& provider){
    uint32_t size;
    provider(&size, static_cast<VkObject*>(nullptr));
    std::vector<VkObject> objects(size);
    provider(&size, objects.data());
    return objects;
}

template<typename VkObject, typename Provider>
inline std::vector<VkObject> enumerate(Provider&& provider){
    uint32_t size;
    provider(&size, static_cast<VkObject*>(nullptr));
    std::vector<VkObject> objects(size);
    VkResult result;
    do {
        result = provider(&size, objects.data());
    }while(result == VK_INCOMPLETE);
    ERR_GUARD_VULKAN(result);
    return objects;
}

inline bool hasStencil(VkFormat format) {
    return format == VK_FORMAT_D32_SFLOAT_S8_UINT || format == VK_FORMAT_D24_UNORM_S8_UINT;
}

template<typename Func>
Func instanceProc(const std::string& procName, VkInstance instance){
    auto proc = reinterpret_cast<Func>(vkGetInstanceProcAddr(instance, procName.c_str()));
    if(!proc) throw std::runtime_error{procName + "Not found"};
    return proc;
}

template <typename T>
inline void dispose(T& t){
    T temp = std::move(t);
}

#define DISABLE_COPY(TypeName) \
TypeName(const TypeName&) = delete; \
TypeName& operator=(const TypeName&) = delete;

#define DISABLE_MOVE(TypeName) \
TypeName(TypeName&&) = delete; \
TypeName& operator=(TypeName&&) = delete;


constexpr uint32_t alignedSize(uint32_t value, uint32_t alignment){
    return (value + alignment - 1) & ~(alignment - 1);
}

// helper type for the visitor #4
template<class... Ts> struct overloaded : Ts... { using Ts::operator()...; };
// explicit deduction guide (not needed as of C++20)
template<class... Ts> overloaded(Ts...) -> overloaded<Ts...>;

inline byte_string loadFile(const std::string& path) {
    std::ifstream fin(path.data(), std::ios::binary | std::ios::ate);
    if(!fin.good()) throw std::runtime_error{"Failed to open file: " + path};

    auto size = fin.tellg();
    fin.seekg(0);
    std::vector<char> data(size);
    fin.read(data.data(), size);

    return data;
}

template<typename T>
inline std::function<T()> rngFunc(T lower, T upper, uint32_t seed = std::random_device{}()) {
    std::default_random_engine engine{ seed };
    if constexpr(std::is_integral_v<T>){
        std::uniform_int_distribution<T> dist{lower, upper};
        return std::bind(dist, engine);
    }else {
        std::uniform_real_distribution<T> dist{lower, upper};
        return std::bind(dist, engine);
    }
}

/**
 * Returns a random float within the range [0, 1)
 * @param seed used to initialize the random number generator
 * @return random float within the range [0, 1)
 */
inline std::function<float()> canonicalRng(uint32_t seed = std::random_device{}()){
    return rngFunc<float>(0.0f, 1.0f, seed);
}


template<glm::length_t L, typename T, glm::qualifier Q>
bool vectorEquals(glm::vec<L, T, Q> v0, glm::vec<L, T, Q> v1, float eps = 1e-4){
    return glm::all(glm::epsilonEqual(v0, v1, eps));
}

constexpr uint32_t nearestPowerOfTwo(uint32_t x) {
    if (x <= 1) return 2;
    x -= 1;
    x |= (x >> 1);
    x |= (x >> 2);
    x |= (x >> 4);
    x |= (x >> 8);
    x |= (x >> 16);
    x += 1;
    return x;
}

constexpr uint nearestMultiple(uint n, uint x) {
    uint nModx = n % x;
    return nModx == 0 ? n : n + x - nModx;
}

using Proc = std::function<void()>;

#define REPORT_ERROR(result, msg) if(result != VK_SUCCESS) throw std::runtime_error{msg}
#define offsetOf(s,m) static_cast<uint32_t>(offsetof(s, m))


constexpr float meter = 1;
constexpr float meters = 1;
constexpr float m = meters;
constexpr float centimetre = meter * 0.01;
constexpr float centimetres = centimetre;
constexpr float cm = centimetre;
constexpr float CM = centimetre;
constexpr float kilometer = meter * 1000;
constexpr float km = kilometer;
constexpr float KM = kilometer;

constexpr float kb = 1024;
constexpr float mb = kb * kb;
constexpr float gb = mb * kb;

inline auto remap(auto x, auto a, auto b, auto c, auto d){
    return glm::mix(c, d, (x - a)/(b - a));
}

template<typename T>
T* as(auto u) { return reinterpret_cast<T*>(u); }

template<typename T>
constexpr T to(auto u){ return static_cast<T>(u); }