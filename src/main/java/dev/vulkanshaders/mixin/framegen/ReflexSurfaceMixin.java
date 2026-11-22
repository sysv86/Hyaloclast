package dev.vulkanshaders.mixin.framegen;

import com.mojang.renderpearl.api.device.GpuSurface;
import com.mojang.renderpearl.backend.vulkan.VulkanGpuSurface;
import dev.vulkanshaders.backend.NativeCoreBridge;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Shadow;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

/**
 * Notifies the native `FrameGenController`, `PresentThread`, and `VK_NV_low_latency2` (Reflex)
 * pacing state whenever `RenderPearl`'s `VkSwapchainKHR` is configured or destroyed.
 */
@Mixin(VulkanGpuSurface.class)
public class ReflexSurfaceMixin {
    @Shadow
    private long swapchain;

    @Inject(method = "configure", at = @At("HEAD"))
    private void vulkanshaders$beforeSwapchainConfigured(GpuSurface.Configuration configuration, CallbackInfo ci) {
        if (NativeCoreBridge.initialized) {
            NativeCoreBridge.waitPresentIdle();
        }
    }

    @Inject(method = "configure", at = @At("RETURN"))
    private void vulkanshaders$onSwapchainConfigured(GpuSurface.Configuration configuration, CallbackInfo ci) {
        if (NativeCoreBridge.initialized) {
            NativeCoreBridge.reflexSetSwapchain(this.swapchain);
        }
    }

    @Inject(method = "destroySwapchain", at = @At("HEAD"))
    private void vulkanshaders$onSwapchainDestroyed(CallbackInfo ci) {
        if (NativeCoreBridge.initialized) {
            NativeCoreBridge.waitPresentIdle();
            NativeCoreBridge.reflexSetSwapchain(0L);
        }
    }
}
