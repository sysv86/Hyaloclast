package dev.vulkanshaders.mixin.backend;

import com.llamalad7.mixinextras.injector.wrapoperation.Operation;
import com.llamalad7.mixinextras.injector.wrapoperation.WrapOperation;
import com.mojang.renderpearl.backend.vulkan.VulkanInstance;
import dev.vulkanshaders.backend.NativeCoreBridge;
import org.lwjgl.PointerBuffer;
import org.lwjgl.vulkan.VK12;
import org.lwjgl.vulkan.VkAllocationCallbacks;
import org.lwjgl.vulkan.VkInstanceCreateInfo;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;

/**
 * Allows the native Rust {@code vulkanshaders_core} library to merge instance extensions
 * ({@code VK_KHR_get_physical_device_properties2}, {@code VK_EXT_swapchain_colorspace}) into
 * {@code RenderPearl}'s {@code VkInstanceCreateInfo} when creating the Vulkan instance.
 */
@Mixin(VulkanInstance.class)
public class VulkanInstanceMixin {
    @WrapOperation(
        method = "<init>",
        at = @At(
            value = "INVOKE",
            target = "Lorg/lwjgl/vulkan/VK12;vkCreateInstance(Lorg/lwjgl/vulkan/VkInstanceCreateInfo;Lorg/lwjgl/vulkan/VkAllocationCallbacks;Lorg/lwjgl/PointerBuffer;)I"
        )
    )
    private int vulkanshaders$createMergedInstance(
        VkInstanceCreateInfo createInfo,
        VkAllocationCallbacks allocator,
        PointerBuffer instance,
        Operation<Integer> original
    ) {
        if (!NativeCoreBridge.nativeAvailable) {
            return original.call(createInfo, allocator, instance);
        }
        int rc = NativeCoreBridge.createMergedInstance(
            createInfo.address(),
            allocator == null ? 0L : allocator.address(),
            instance.address()
        );
        if (rc != VK12.VK_SUCCESS) {
            return original.call(createInfo, allocator, instance);
        }
        return rc;
    }
}
