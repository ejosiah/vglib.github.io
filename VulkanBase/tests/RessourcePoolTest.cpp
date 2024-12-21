#include "ResourcePool.hpp"

#include <gtest/gtest.h>
#include <numeric>

struct ResourcePoolSuit : public ::testing::Test {};

TEST(ResourcePoolSuit, resourcePoolObtain) {
    std::vector<int> data(10);
    std::iota(data.begin(), data.end(), 0);
    ResourcePool<int> pool{ data };

    for(auto i = 9; i >= 0; --i) {
        ASSERT_EQ(i, pool.obtain()->get());
    }
}

TEST(ResourcePoolSuit, resourcePoolReelase) {
    std::vector<int> data(10);
    std::iota(data.begin(), data.end(), 0);
    ResourcePool<int> pool{ data };

    auto resource = pool.obtain();
    auto expected = resource->get();

    pool.release(resource);

    ASSERT_EQ(expected, pool.obtain()->get()) << "expected to obtain last acquired resource";
}

TEST(ResourcePoolSuit, returnNothingWhenPoolEmpty) {
    std::vector<int> data(10);
    std::iota(data.begin(), data.end(), 0);
    ResourcePool<int> pool{ data };

    for(auto i = 0; i < 10; ++i) {
        pool.obtain();
    }

    ASSERT_FALSE(pool.obtain().has_value()) << "resource pool should be empty";

}

TEST(ResourcePoolSuit, releaseAllResources) {
    std::vector<int> data(10);
    std::iota(data.begin(), data.end(), 0);
    ResourcePool<int> pool{ data };

    for(auto i = 0; i < 10; ++i) {
        pool.obtain();
    }

    ASSERT_FALSE(pool.obtain().has_value());

    pool.releaseAll();

    ASSERT_TRUE(pool.obtain().has_value());
}