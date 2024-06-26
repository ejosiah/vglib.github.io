#pragma once
#include "VulkanDevice.h"
#include <optional>
#include "Statistics.hpp"
#include "VulkanQuery.hpp"
#include <spdlog/spdlog.h>

class Profiler{
public:
    struct MovingAverage{
        float value{};
        uint32_t count{};
    };
    struct Query{
        std::string name;
        uint64_t startId{0};
        uint64_t endId{0};
        std::vector<uint64_t> runtimes{};
        MovingAverage movingAverage{};
    };

    struct QueryGroup{
        std::string name;
        std::vector<std::string> queries;
        std::vector<uint64_t> runtimes{};
    };

    Profiler() = default;

    inline explicit  Profiler(VulkanDevice* device, uint32_t queryCount = DEFAULT_QUERY_COUNT)
    : device(device)
    , queryCount(std::max(queryCount, DEFAULT_QUERY_COUNT))
    {
        queryPool = TimestampQueryPool{*device, queryCount};
    }

    inline void addQuery(const std::string& name){
        if(!isReady()) return;
        if(queries.size() >= queryCount){
            queryCount += queryCount * 0.25;
            queryPool = TimestampQueryPool{*device, queryCount};
        }
        Query query{name};
        query.startId = queries.size() * 2;
        query.endId = queries.size() * 2 + 1;
        queries.insert(std::make_pair(name, query));
    }

    template<typename Queries>
    inline void group(const std::string& groupName, Queries queries){
        if(!isReady()) return;
        QueryGroup queryGroup = queryGroups.find(groupName) == end(queryGroups) ? QueryGroup{groupName} : queryGroups[groupName];
        for(auto _query : queries){
            queryGroup.queries.push_back(_query);
        }
        queryGroups[groupName] = queryGroup;
    }

    inline void addGroup(const std::string& name, int queries){
        if(!isReady()) return;
        assert(queries > 0);
        std::vector<std::string> queryNames;
        for(auto i = 0; i < queries; i++){
            auto queryName = fmt::format("{}_{}", name, i);
            addQuery(queryName);
            queryNames.push_back(queryName);
        }
        group(name, queryNames);
    }

    template<typename Body>
    inline void profile(const std::string& name, VkCommandBuffer commandBuffer, Body&& body){
        if(!isReady()){
            body();
            return;
        }
        assert(queries.find(name) != end(queries));
        auto query = queries[name];
        vkCmdResetQueryPool(commandBuffer, queryPool, query.startId, 2);
        vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, queryPool, query.startId);
        body();
        vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, queryPool, query.endId);
    }

    inline void commit(){
        if(!isReady()) return;
        std::vector<uint64_t> counters(queries.size() * 2);

        VkQueryResultFlags flags = VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT;
        vkGetQueryPoolResults(*device, queryPool, 0, COUNT(counters), BYTE_SIZE(counters), counters.data(), sizeof(uint64_t), flags);

        auto timestampPeriod = device->timestampPeriod();
        for(auto& [_, query] : queries){
            auto start = counters[query.startId];
            auto end = counters[query.endId];
            auto runtime = (end - start) * timestampPeriod;
            query.runtimes.push_back(runtime);
        }

        for(auto& [_, group] : queryGroups){
            uint64_t runtime = 0;
            for(auto& queryName : group.queries){
                auto& query = queries[queryName];
                runtime += query.runtimes.back();
            }
            runtime /= group.queries.size();
            group.runtimes.push_back(runtime);
        }
    }

    inline void clearRunTimes()  {
        for(auto& [_, group] : queries) {
            group.runtimes.clear();
        }
    }

    inline void endFrame() {
        if(!isReady()) return;

        std::vector<uint64_t> counters(queries.size() * 2);

        VkQueryResultFlags flags = VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT;
        vkGetQueryPoolResults(*device, queryPool, 0, COUNT(counters), BYTE_SIZE(counters), counters.data(), sizeof(uint64_t), flags);

        auto timestampPeriod = device->timestampPeriod();
        for(auto& [name, query] : queries){
            auto start = counters[query.startId];
            auto end = counters[query.endId];
            auto newRuntime = static_cast<float>((end - start) * timestampPeriod);
            auto previousRuntime = query.movingAverage.value;
            if(previousRuntime != 0 && newRuntime/previousRuntime > 3) continue;
            auto numEntries = static_cast<float>(++query.movingAverage.count);
            query.movingAverage.value = glm::mix(previousRuntime, newRuntime, 1/numEntries );
        }
        // TODO group queries;
//        for(auto& [_, group] : queryGroups){
//            uint64_t runtime = 0;
//            for(auto& queryName : group.queries){
//                auto& query = queries[queryName];
//                runtime += query.runtimes.back();
//            }
//            runtime /= group.queries.size();
//            group.runtimes.push_back(runtime);
//        }
    }

    inline std::optional<QueryGroup> getGroup(const std::string& name){
        if(queryGroups.find(name) != end(queryGroups)){
            return queryGroups[name];
        }
        return {};
    }

    inline std::map<std::string, stats::Statistics<float>> groupStats(){
        std::map<std::string, stats::Statistics<float>> result;

        for(auto& [name, group] : queryGroups){
            auto runtimes = toMillis(group.runtimes);
            auto statistics = stats::summarize<float>(runtimes);
            result.insert(std::make_pair(name, statistics));
        }
        return result;
    }

    inline std::map<std::string, stats::Statistics<float>> queryStats(){
        std::map<std::string, stats::Statistics<float>> result;

        for(auto& [name, query] : queries){
            auto runtimes = toMillis(query.runtimes);
            auto statistics = stats::summarize<float>(runtimes);
            result.insert(std::make_pair(name, statistics));
        }
        return result;
    }

    static constexpr float toMillis(uint64_t duration){
        return static_cast<float>(duration) * 1e-6f;
    }

    static inline std::vector<float> toMillis(const std::vector<uint64_t>& durations){
        std::vector<float> result;
        result.reserve(durations.size());
        for(auto& duration : durations){
            result.push_back(toMillis(duration));
        }
        return result;
    }

    inline bool isReady() const {
        return static_cast<bool>(queryPool) && device && !paused;
    }

    std::map<std::string, Query> queries;

    mutable bool paused = false;

private:
    static constexpr uint32_t DEFAULT_QUERY_COUNT = 1024;

    TimestampQueryPool queryPool;
    VulkanDevice* device = nullptr;
    uint32_t queryCount = DEFAULT_QUERY_COUNT;
    std::map<std::string, QueryGroup> queryGroups;
};