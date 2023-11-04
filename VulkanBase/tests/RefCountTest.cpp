#include <gtest/gtest.h>
#include "RefCounted.hpp"

#include <vector>
#include <thread>
#include <latch>

class RefCountFixture : public ::testing::Test {
protected:

    void SetUp() override {

    }

    void TearDown() override {

    }

};

struct TestResource : RefCounted {
public:
    TestResource() = default;

    TestResource(ResourceHandle handle)
    : RefCounted(handle, VoidCleaner)
    {}

    TestResource(const TestResource& source)
    : RefCounted(source)
    {}

    TestResource(TestResource&& source)
    : RefCounted(std::move(source))
    {}

    TestResource& operator=(const TestResource& source) {
        copyRef(source);
        return *this;
    }

private:
};

TEST_F(RefCountFixture, SingleResourceReference) {
    {
        TestResource resource{1};
        ASSERT_EQ(1, RefCounted::references(ResourceHandle{1}));
    }
    ASSERT_EQ(0, RefCounted::references(ResourceHandle{1}));
}

TEST_F(RefCountFixture, SingleResourceMultiplReferences) {

    {
        TestResource resource{1};
        std::vector<TestResource> copies;

        for(auto i = 0; i < 10; i++){
            copies.emplace_back(resource);
        }
        ASSERT_EQ(11, RefCounted::references(ResourceHandle{1}));
    }
    ASSERT_EQ(0, RefCounted::references(ResourceHandle{1}));
}

TEST_F(RefCountFixture, SingleResourceCopyAssignment) {

    {
        TestResource resource{2};
        std::vector<TestResource> copies(10);
        for(auto i = 0; i < 10; i++){
            copies[i] = resource;
        }
        ASSERT_EQ(11, RefCounted::references(ResourceHandle{2}));
    }
    ASSERT_EQ(0, RefCounted::references(ResourceHandle{2}));
}

TEST_F(RefCountFixture, moveResourceReference) {
    {
        TestResource resource{1};
        std::vector<TestResource> copies;

        for(auto i = 0; i < 10; i++){
            copies.emplace_back(resource);
        }
        ASSERT_EQ(11, RefCounted::references(ResourceHandle{1}));

        TestResource newResource{ static_cast<TestResource&&>(resource) };

        ASSERT_EQ(11, RefCounted::references(ResourceHandle{1}));
    }
    ASSERT_EQ(0, RefCounted::references(ResourceHandle{2}));
}

TEST_F(RefCountFixture, singleResourrceMultipleRefernceMT) {
    {
        TestResource resource{1};
        std::vector<std::thread> threads{};
        std::latch thread_start{3};
        std::latch sentinel{1};

        for(int i = 0; i < 3; i++) {

            threads.emplace_back([&] {
                std::vector<TestResource> copies;
                for(auto j = 0; j < 10; j++){
                    copies.emplace_back(resource);
                }
                thread_start.count_down();
                sentinel.wait();
            });
        }

        thread_start.wait();
        ASSERT_EQ(31, RefCounted::references(ResourceHandle{1}));

        sentinel.count_down();
        for(auto& thread : threads) thread.join();
    }
    ASSERT_EQ(0, RefCounted::references(ResourceHandle{1}));

}

TEST_F(RefCountFixture, multipleResourrceMultipleRefernceMT) {
    {
        std::vector<TestResource> resources{ TestResource{1}, TestResource{2}, TestResource{3} };
        std::vector<std::thread> threads{};
        std::latch thread_start{3};
        std::latch sentinel{1};

        for(auto i = 0; i < 3; i++){
            threads.emplace_back([&, id=i] {
                std::vector<TestResource> copies;
                for(auto j = 0; j < 10; j++){
                    copies.emplace_back(resources[id]);
                }
                thread_start.count_down();
                sentinel.wait();
            });
        }

        thread_start.wait();
        ASSERT_EQ(11, RefCounted::references(ResourceHandle{1}));
        ASSERT_EQ(11, RefCounted::references(ResourceHandle{2}));
        ASSERT_EQ(11, RefCounted::references(ResourceHandle{3}));

        sentinel.count_down();
        for(auto& thread : threads) thread.join();
    }

    ASSERT_EQ(0, RefCounted::references(ResourceHandle{1}));
    ASSERT_EQ(0, RefCounted::references(ResourceHandle{2}));
    ASSERT_EQ(0, RefCounted::references(ResourceHandle{3}));
}

TEST_F(RefCountFixture, constructSingleResourceMultipleTimes) {
    {
        std::vector<TestResource> resource;
        for (int i = 0; i < 100; i++){
            resource.emplace_back(ResourceHandle{1});
        }
        ASSERT_EQ(100, resource.size());
        ASSERT_EQ(100, RefCounted::references(ResourceHandle{1}));

        for(int i = 99; i >= 0; i--) {
            resource.erase(std::prev(resource.end()));
            ASSERT_EQ(i, RefCounted::references(ResourceHandle{1}));
        }
    }
    ASSERT_EQ(0, RefCounted::references(ResourceHandle{1}));

}