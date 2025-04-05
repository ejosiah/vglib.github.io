#include "VulkanFixture.hpp"
#include "PrefixSum.hpp"

class PrefixScanTest : public VulkanFixture{
protected:

    void postVulkanInit() override {
        _prefix_sum = PrefixSum{ &device, const_cast<VulkanCommandPool*>(&device.computeCommandPool()) };
        _prefix_sum.init();
    }

protected:
    PrefixSum _prefix_sum;
};


TEST_F(PrefixScanTest, ScanWithSingleWorkGroup){
    std::vector<int> data(8 << 10);
    auto rng = rngFunc<int>(0, 100, 1 << 20);
    std::generate(begin(data), end(data), [&]{ return rng(); });

    auto expected = data;
    std::exclusive_scan(begin(expected), end(expected), begin(expected), 0);

    _prefix_sum.scan(begin(data), end(data));

    for(int i = 0; i < data.size(); i++){
        ASSERT_EQ(expected[i], data[i]);
    }

}

TEST_F(PrefixScanTest, ScanDataItemsLessThanWorkGroupSize){
    std::vector<int> data(8 << 9);
    auto rng = rngFunc<int>(0, 100, 1 << 20);
    std::generate(begin(data), end(data), [&]{ return rng(); });

    auto expected = data;
    std::exclusive_scan(begin(expected), end(expected), begin(expected), 0);

    _prefix_sum.scan(begin(data), end(data));

    for(int i = 0; i < data.size(); i++){
        ASSERT_EQ(expected[i], data[i]);
    }
}

TEST_F(PrefixScanTest, ScanNonPowerOfDataItems){
    std::vector<int> data(55555);
    auto rng = rngFunc<int>(0, 100, 1 << 20);
    std::generate(begin(data), end(data), [&]{ return rng(); });

    auto expected = data;
    std::exclusive_scan(begin(expected), end(expected), begin(expected), 0);

    _prefix_sum.scan(begin(data), end(data));

    for(int i = 0; i < data.size(); i++){
        ASSERT_EQ(expected[i], data[i]);
    }
}

TEST_F(PrefixScanTest, ScanWithMutipleWorkGroups){
    std::vector<int> data((8 << 10) + 1);
    auto rng = rngFunc<int>(0, 100, 1 << 20);
    std::generate(begin(data), end(data), [&]{ return rng(); });

    auto expected = data;
    std::exclusive_scan(begin(expected), end(expected), begin(expected), 0);

    _prefix_sum.scan(begin(data), end(data));

    for(int i = 0; i < data.size(); i++){
        ASSERT_EQ(expected[i], data[i]);
    }
}

TEST_F(PrefixScanTest, scanLargeNumberOfItems){
    std::vector<int> data((15 << 20) );
    auto rng = rngFunc<int>(0, 100, 1 << 20);
    std::generate(begin(data), end(data), [&]{ return rng(); });

    auto expected = data;
    std::exclusive_scan(begin(expected), end(expected), begin(expected), 0);

    _prefix_sum.scan(begin(data), end(data));

    for(int i = 0; i < data.size(); i++){
        ASSERT_EQ(expected[i], data[i]);
    }
}

TEST_F(PrefixScanTest, incusiveScan) {
    std::vector<int> data((2 << 20) );
    auto rng = rngFunc<int>(0, 100, 1 << 20);
    std::generate(begin(data), end(data), [&]{ return rng(); });

    auto expected = data;
    std::inclusive_scan(begin(expected), end(expected), begin(expected));

    _prefix_sum.inclusive(begin(data), end(data));

    for(int i = 0; i < data.size(); i++){
        ASSERT_EQ(expected[i], data[i]);
    }
}

TEST_F(PrefixScanTest, accumulate) {
    std::vector<int> data((2 << 20) );
    auto rng = rngFunc<int>(0, 100, 1 << 20);
    std::generate(begin(data), end(data), [&]{ return rng(); });

    VulkanBuffer gpuData = device.createCpuVisibleBuffer(data.data(), BYTE_SIZE(data), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    int x = 0;
    VulkanBuffer gpuResult = device.createCpuVisibleBuffer(&x, sizeof(x), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    execute([&](auto commandBuffer){
       _prefix_sum.accumulate(commandBuffer, gpuData, gpuResult);
    });

    auto expected = std::accumulate(data.begin(), data.end(), 0);
    auto result = gpuResult.span<int>().front();

    ASSERT_EQ(expected, result);
}

TEST_F(PrefixScanTest, accumulateFloat) {
    std::vector<float> data((2 << 20) );
    auto rng = rngFunc<float>(0, 100, 1 << 20);
    std::generate(begin(data), end(data), [&]{ return rng(); });

    VulkanBuffer gpuData = device.createCpuVisibleBuffer(data.data(), BYTE_SIZE(data), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    int x = 0;
    VulkanBuffer gpuResult = device.createCpuVisibleBuffer(&x, sizeof(x), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    execute([&](auto commandBuffer){
        _prefix_sum.accumulate(commandBuffer, gpuData, gpuResult, Operation::Add, DataType::Float);
    });

    auto expected = std::accumulate(data.begin(), data.end(), 0.f);
    auto result = gpuResult.span<float>().front();

    ASSERT_EQ(expected, result);
}

TEST_F(PrefixScanTest, scanWithMaxOperation) {
    std::vector<int> data((2 << 20) );
    auto rng = rngFunc<int>(1, 100, 1 << 20);
    std::generate(begin(data), end(data), [&]{ return rng(); });

    VulkanBuffer gpuData = device.createCpuVisibleBuffer(data.data(), BYTE_SIZE(data), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    execute([&](auto commandBuffer){
        _prefix_sum(commandBuffer, gpuData, Operation::Max);
    });

    auto expected = data;
    std::exclusive_scan(expected.begin(), expected.end(), expected.begin(),
                        std::numeric_limits<int>::min(),
                        [](auto a, auto b){ return std::max(a, b); });

    auto result = gpuData.span<int>();

    for(int i = 0; i < data.size(); i++){
        ASSERT_EQ(expected[i], result[i]);
    }
}


TEST_F(PrefixScanTest, scanWithMinOperation) {
    std::vector<int> data((2 << 20) );
    auto rng = rngFunc<int>(1, 100, 1 << 20);
    std::generate(begin(data), end(data), [&]{ return rng(); });

    VulkanBuffer gpuData = device.createCpuVisibleBuffer(data.data(), BYTE_SIZE(data), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    execute([&](auto commandBuffer){
        _prefix_sum(commandBuffer, gpuData, Operation::Min);
    });

    auto expected = data;
    std::exclusive_scan(expected.begin(), expected.end(), expected.begin(),
                        std::numeric_limits<int>::max(),
                        [](auto a, auto b){ return std::min(a, b); });

    auto result = gpuData.span<int>();

    for(int i = 0; i < data.size(); i++){
        ASSERT_EQ(expected[i], result[i]);
    }
}


TEST_F(PrefixScanTest, scanWithMinOperationAndFloatData) {
    std::vector<float> data((2 << 20) );
    auto rng = rngFunc<float>(1, 100, 1 << 20);
    std::generate(begin(data), end(data), [&]{ return rng(); });

    VulkanBuffer gpuData = device.createCpuVisibleBuffer(data.data(), BYTE_SIZE(data), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    execute([&](auto commandBuffer){
        _prefix_sum(commandBuffer, gpuData, Operation::Min, DataType::Float);
    });

    auto expected = data;
    std::exclusive_scan(expected.begin(), expected.end(), expected.begin(),
                        std::numeric_limits<float>::max(),
                        [](auto a, auto b){ return std::min(a, b); });

    auto result = gpuData.span<float>();

    for(int i = 0; i < data.size(); i++){
        ASSERT_EQ(expected[i], result[i]);
    }
}

TEST_F(PrefixScanTest, min) {
    glm::ivec2 x{};
    std::vector<int> data((2 << 20) );
    auto rng = rngFunc<int>(-100, 100, 1 << 20);
    std::generate(begin(data), end(data), [&]{ return rng(); });

    VulkanBuffer gpuData = device.createCpuVisibleBuffer(data.data(), BYTE_SIZE(data), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    VulkanBuffer gpuResult = device.createCpuVisibleBuffer(&x, sizeof(x), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    execute([&](auto commandBuffer){ _prefix_sum.min(commandBuffer, gpuData.region(0), gpuResult); });

    auto expected = *std::min_element(data.begin(), data.end());
    auto result = gpuResult.span<int>();

    ASSERT_EQ(expected, result.front());
}

TEST_F(PrefixScanTest, max) {
    glm::ivec2 x{};
    std::vector<int> data((2 << 20) );
    auto rng = rngFunc<int>(-100, 100, 1 << 20);
    std::generate(begin(data), end(data), [&]{ return rng(); });

    VulkanBuffer gpuData = device.createCpuVisibleBuffer(data.data(), BYTE_SIZE(data), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    VulkanBuffer gpuResult = device.createCpuVisibleBuffer(&x, sizeof(x), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    execute([&](auto commandBuffer){ _prefix_sum.max(commandBuffer, gpuData.region(0), gpuResult); });

    auto expected = *std::max_element(data.begin(), data.end());
    auto result = gpuResult.span<int>();

    ASSERT_EQ(expected, result.front());
}


TEST_F(PrefixScanTest, minFloat) {
    float x{};
    std::vector<float> data((2 << 20) );
    auto rng = rngFunc<float>(-100, 100, 1 << 20);
    std::generate(begin(data), end(data), [&]{ return rng(); });

    VulkanBuffer gpuData = device.createCpuVisibleBuffer(data.data(), BYTE_SIZE(data), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    VulkanBuffer gpuResult = device.createCpuVisibleBuffer(&x, sizeof(x), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    execute([&](auto commandBuffer){ _prefix_sum.min(commandBuffer, gpuData.region(0), gpuResult, DataType::Float); });

    auto expected = *std::min_element(data.begin(), data.end());
    auto result = gpuResult.span<float>();

    ASSERT_EQ(expected, result.front());
}

TEST_F(PrefixScanTest, maxFloat) {
    float x{};
    std::vector<float> data((2 << 20) );
    auto rng = rngFunc<float>(-100, 100, 1 << 20);
    std::generate(begin(data), end(data), [&]{ return rng(); });

    VulkanBuffer gpuData = device.createCpuVisibleBuffer(data.data(), BYTE_SIZE(data), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    VulkanBuffer gpuResult = device.createCpuVisibleBuffer(&x, sizeof(x), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    execute([&](auto commandBuffer){ _prefix_sum.max(commandBuffer, gpuData.region(0), gpuResult, DataType::Float); });

    auto expected = *std::max_element(data.begin(), data.end());
    auto result = gpuResult.span<float>();

    ASSERT_EQ(expected, result.front());
}

TEST_F(PrefixScanTest, throwExceptionWhenDataExceedsMaxLimit) {
    std::vector<int> data(PrefixSum::MAX_NUM_ITEMS * 2 );

    ASSERT_THROW(_prefix_sum.scan(begin(data), end(data)), PrefixSum::DataSizeExceedsMaxSupported);

}