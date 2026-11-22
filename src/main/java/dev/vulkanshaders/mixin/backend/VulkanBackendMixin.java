package dev.vulkanshaders.mixin.backend;

import com.mojang.renderpearl.backend.vulkan.VulkanBackend;
import dev.vulkanshaders.backend.NativeCoreBridge;
import org.lwjgl.PointerBuffer;
import org.lwjgl.vulkan.VK12;
import org.lwjgl.vulkan.VkAllocationCallbacks;
import org.lwjgl.vulkan.VkDeviceCreateInfo;
import org.lwjgl.vulkan.VkPhysicalDevice;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Redirect;

/**
 * Allows the native Rust `vulkanshaders_core` library to merge optional Ray Tracing
 * (`VK_KHR_acceleration_structure`, `VK_KHR_ray_query`, `VK_KHR_ray_tracing_pipeline`),
 * `bufferDeviceAddress`, `descriptorIndexing`, and `VK_NV_low_latency2` extensions and
 * feature structs into `RenderPearl`'s `VkDeviceCreateInfo` when supported by the GPU.
 */
@Mixin(VulkanBackend.class)
public class VulkanBackendMixin {
    @Redirect(
        method = "createDevice(Lcom/mojang/renderpearl/backend/vulkan/init/FeatureSet;Lcom/mojang/renderpearl/backend/vulkan/VulkanPhysicalDevice;)Lorg/lwjgl/vulkan/VkDevice;",
        at = @At(
            value = "INVOKE",
            target = "Lorg/lwjgl/vulkan/VK12;vkCreateDevice(Lorg/lwjgl/vulkan/VkPhysicalDevice;Lorg/lwjgl/vulkan/VkDeviceCreateInfo;Lorg/lwjgl/vulkan/VkAllocationCallbacks;Lorg/lwjgl/PointerBuffer;)I"
        )
    )
    private static int vulkanshaders$createMergedDevice(
        VkPhysicalDevice physicalDevice,
        VkDeviceCreateInfo createInfo,
        VkAllocationCallbacks allocator,
        PointerBuffer device
    ) {
        if (!NativeCoreBridge.nativeAvailable) {
            return VK12.vkCreateDevice(physicalDevice, createInfo, allocator, device);
        }
        int rc = NativeCoreBridge.createMergedDevice(
            physicalDevice.address(),
            createInfo.address(),
            allocator == null ? 0L : allocator.address(),
            device.address()
        );
        if (rc != VK12.VK_SUCCESS) {
            return VK12.vkCreateDevice(physicalDevice, createInfo, allocator, device);
        }
        return rc;
    }
}
