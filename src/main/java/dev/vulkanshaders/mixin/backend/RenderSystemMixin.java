package dev.vulkanshaders.mixin.backend;

import com.mojang.blaze3d.systems.RenderSystem;
import com.mojang.renderpearl.api.device.GpuDevice;
import dev.vulkanshaders.VulkanShaders;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

/**
 * Initializes Hyaloclast backend context once RenderPearl creates the GpuDevice.
 */
@Mixin(value = RenderSystem.class, priority = 1100)
public class RenderSystemMixin {
    @Inject(method = "initRenderer", at = @At("RETURN"))
    private static void vulkanshaders$onRendererInitialized(GpuDevice device, CallbackInfo ci) {
        VulkanShaders.onRendererReady(device);
    }
}
