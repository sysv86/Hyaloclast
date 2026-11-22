package dev.vulkanshaders.mixin.backend;

import com.mojang.renderpearl.backend.vulkan.VulkanDevice;
import dev.vulkanshaders.backend.VulkanDeviceContext;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

/**
 * Drains the native Rust `RetireQueue` and shuts down `VulkanDeviceContext` cleanly
 * before `RenderPearl` destroys the `VkDevice`.
 */
@Mixin(VulkanDevice.class)
public class VulkanDeviceMixin {
    @Inject(method = "close", at = @At("HEAD"))
    private void vulkanshaders$onDeviceClosing(CallbackInfo ci) {
        VulkanDeviceContext.shutdown();
    }
}
