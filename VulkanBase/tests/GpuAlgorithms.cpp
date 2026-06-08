#include "VulkanFixture.hpp"
#include "gpu/algorithm.h"
#include "algorithm"

class GpuAlgorithms : public VulkanFixture{
protected:
    void postVulkanInit() override {
        gpu::init(device, _fileManager);
    }

    void TearDown() override {
        gpu::shutdown();
    }
};

TEST_F(GpuAlgorithms, averageLargeValues){

    std::vector<float> data(1 << 20);
    auto rng = rngFunc<float>(0.0f, 100.0f, 1 << 20);
    std::generate(begin(data), end(data), [&]{ return rng(); });

    auto sum = std::accumulate(begin(data), end(data), 0.0f);
    auto expected = sum/static_cast<float>(data.size());
    VulkanBuffer buffer = device.createDeviceLocalBuffer(data.data(), BYTE_SIZE(data), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto actual = gpu::average(buffer);

    ASSERT_NEAR(expected, actual, 0.01);
}

TEST_F(GpuAlgorithms, averageValuesNearZero){
    std::vector<float> data(1 << 20);
    auto rng = rngFunc<float>(-0.1f, 0.1f, 1 << 20);
    std::generate(begin(data), end(data), [&]{ return rng(); });

    auto sum = std::accumulate(begin(data), end(data), 0.0f);
    auto expected = sum/static_cast<float>(data.size());
    VulkanBuffer buffer = device.createDeviceLocalBuffer(data.data(), BYTE_SIZE(data), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto actual = gpu::average(buffer);

    ASSERT_NEAR(expected, actual, 0.00001);
}

TEST_F(GpuAlgorithms, averageWithExternalDescriptor){
    std::vector<float> data(1 << 20);
    auto rng = rngFunc<int>(0, 10, 1 << 20);
    std::generate(begin(data), end(data), [&]{ return float(rng()); });

    VulkanDescriptorSetLayout setLayout =
            device.descriptorSetLayoutBuilder()
                    .binding(0)
                    .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                    .descriptorCount(1)
                    .shaderStages(VK_SHADER_STAGE_COMPUTE_BIT)
                    .binding(1)
                    .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                    .descriptorCount(1)
                    .shaderStages(VK_SHADER_STAGE_COMPUTE_BIT)
                    .createLayout();
    VkDescriptorSet descriptorSet = descriptorPool.allocate( { setLayout }).front();

    VulkanBuffer input = device.createCpuVisibleBuffer(data.data(), BYTE_SIZE(data), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    VulkanBuffer output = device.createBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU, sizeof(float));

    auto writes = initializers::writeDescriptorSets<2>();

    writes[0].dstSet = descriptorSet;
    writes[0].dstBinding = 0;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[0].descriptorCount = 1;
    VkDescriptorBufferInfo inInfo{ input, 0, VK_WHOLE_SIZE};
    writes[0].pBufferInfo = &inInfo;

    writes[1].dstSet = descriptorSet;
    writes[1].dstBinding = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[1].descriptorCount = 1;
    VkDescriptorBufferInfo outInfo{ output, 0, VK_WHOLE_SIZE};
    writes[1].pBufferInfo = &outInfo;

    device.updateDescriptorSets(writes);

    device.computeCommandPool().oneTimeCommand([&](auto commandBuffer){
        gpu::average(commandBuffer, descriptorSet , input);
    });

    auto actual = *reinterpret_cast<float*>(output.map());
    fmt::print("actual: {}\n", actual);
    float expected = std::accumulate(begin(data), end(data), 0.0f)/data.size();
    output.unmap();
    ASSERT_NEAR(expected, actual, 0.001);

}

TEST_F(GpuAlgorithms, reduceAdd){
    std::vector<float> data(1 << 20);
    auto rng = rngFunc<int>(0, 10, 1 << 20);
    std::generate(begin(data), end(data), [&]{ return float(rng()); });

    VulkanDescriptorSetLayout setLayout =
        device.descriptorSetLayoutBuilder()
            .binding(0)
                .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                .descriptorCount(1)
                .shaderStages(VK_SHADER_STAGE_COMPUTE_BIT)
            .binding(1)
                .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                .descriptorCount(1)
                .shaderStages(VK_SHADER_STAGE_COMPUTE_BIT)
            .createLayout();
    VkDescriptorSet descriptorSet = descriptorPool.allocate( { setLayout }).front();
    
    VulkanBuffer input = device.createCpuVisibleBuffer(data.data(), BYTE_SIZE(data), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    VulkanBuffer output = device.createBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU, sizeof(float));
    
    auto writes = initializers::writeDescriptorSets<2>();
    
    writes[0].dstSet = descriptorSet;
    writes[0].dstBinding = 0;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[0].descriptorCount = 1;
    VkDescriptorBufferInfo inInfo{ input, 0, VK_WHOLE_SIZE};
    writes[0].pBufferInfo = &inInfo;

    writes[1].dstSet = descriptorSet;
    writes[1].dstBinding = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[1].descriptorCount = 1;
    VkDescriptorBufferInfo outInfo{ output, 0, VK_WHOLE_SIZE};
    writes[1].pBufferInfo = &outInfo;

    device.updateDescriptorSets(writes);

    device.computeCommandPool().oneTimeCommand([&](auto commandBuffer){
        gpu::reduce(commandBuffer, descriptorSet , input);
    });

    auto actual = *reinterpret_cast<float*>(output.map());
    fmt::print("actual: {}\n", actual);
    float expected = std::accumulate(begin(data), end(data), 0.0f);
    output.unmap();
    ASSERT_NEAR(expected, actual, 0.001);
    
}

TEST_F(GpuAlgorithms, mathOperations){
    std::vector<float> a{ 2.0f, 4.0f, 6.0f, 8.0f, 10.0f };
    std::vector<float> b{ 1.0f, 2.0f, 3.0f, 4.0f, 5.0f };
    std::vector<float> result(a.size());

    VulkanBuffer as = device.createCpuVisibleBuffer(a.data(), BYTE_SIZE(a), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    VulkanBuffer bs = device.createCpuVisibleBuffer(b.data(), BYTE_SIZE(b), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    VulkanBuffer cs = device.createCpuVisibleBuffer(result.data(), BYTE_SIZE(result), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    auto expect = [&](auto operation, const std::vector<float>& expected) {
        device.computeCommandPool().oneTimeCommand([&](auto commandBuffer){
            operation(commandBuffer, as.region(0), bs.region(0), cs.region(0));
        });

        auto actual = cs.span<float>();
        for(size_t i = 0; i < expected.size(); ++i) {
            ASSERT_FLOAT_EQ(expected[i], actual[i]);
        }
        cs.unmap();
    };

    expect(gpu::add, { 3.0f, 6.0f, 9.0f, 12.0f, 15.0f });
    expect(gpu::subtract, { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f });
    expect(gpu::multiply, { 2.0f, 8.0f, 18.0f, 32.0f, 50.0f });
    expect(gpu::divide, { 2.0f, 2.0f, 2.0f, 2.0f, 2.0f });
}

TEST_F(GpuAlgorithms, mathOperationsRespectBufferRegions){
    constexpr size_t start = 7;
    constexpr size_t count = 37;
    constexpr float sentinel = -1000.0f;

    std::vector<float> a(64, sentinel);
    std::vector<float> b(64, sentinel);
    std::vector<float> result(64, sentinel);

    for(size_t i = 0; i < count; ++i) {
        a[start + i] = static_cast<float>(i + 10);
        b[start + i] = static_cast<float>(i + 1);
    }

    VulkanBuffer as = device.createCpuVisibleBuffer(a.data(), BYTE_SIZE(a), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    VulkanBuffer bs = device.createCpuVisibleBuffer(b.data(), BYTE_SIZE(b), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    VulkanBuffer cs = device.createCpuVisibleBuffer(result.data(), BYTE_SIZE(result), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    const auto begin = start * sizeof(float);
    const auto end = (start + count) * sizeof(float);

    auto expect = [&](auto operation, auto expectedValue) {
        cs.copy(result.data(), BYTE_SIZE(result), 0);

        device.computeCommandPool().oneTimeCommand([&](auto commandBuffer){
            operation(commandBuffer, as.region(begin, end), bs.region(begin, end), cs.region(begin, end));
        });

        auto actual = cs.span<float>();
        for(size_t i = 0; i < actual.size(); ++i) {
            if(i < start || i >= start + count) {
                ASSERT_FLOAT_EQ(sentinel, actual[i]);
            } else {
                const auto index = i - start;
                ASSERT_FLOAT_EQ(expectedValue(a[i], b[i], index), actual[i]);
            }
        }
        cs.unmap();
    };

    expect(gpu::add, [](float x, float y, size_t) { return x + y; });
    expect(gpu::subtract, [](float x, float y, size_t) { return x - y; });
    expect(gpu::multiply, [](float x, float y, size_t) { return x * y; });
    expect(gpu::divide, [](float x, float y, size_t) { return x / y; });
}
