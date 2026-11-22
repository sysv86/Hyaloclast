package dev.vulkanshaders.mixin.backend;

import com.mojang.renderpearl.api.GpuFormat;
import com.mojang.renderpearl.backend.vulkan.VulkanConst;
import com.mojang.renderpearl.backend.vulkan.VulkanGpuTexture;
import com.mojang.renderpearl.backend.vulkan.VulkanGpuTextureView;
import dev.vulkanshaders.texture.TextureRegistry;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.Redirect;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

/**
 * 1. Ensures render-attachment textures created by {@code RenderPearl} or {@code GbufferTargetManager}
 *    include {@code VK_IMAGE_USAGE_STORAGE_BIT (8)}, {@code VK_IMAGE_USAGE_TRANSFER_SRC_BIT (1)}, and
 *    {@code VK_IMAGE_USAGE_TRANSFER_DST_BIT (2)} so compute shaders in the native {@code RenderGraph}
 *    can bind them as {@code image2D} storage targets without violating Vulkan validation rules.
 * 2. Registers live {@link VulkanGpuTexture} and {@link VulkanGpuTextureView} handles into
 *    {@link TextureRegistry} for the bindless descriptor table.
 */
@Mixin(VulkanGpuTexture.class)
public class VulkanGpuTextureMixin {
    private static final int VK_IMAGE_USAGE_TRANSFER_SRC_BIT = 0x00000001;
    private static final int VK_IMAGE_USAGE_TRANSFER_DST_BIT = 0x00000002;
    private static final int VK_IMAGE_USAGE_STORAGE_BIT = 0x00000008;

    @Redirect(
        method = "<init>",
        at = @At(
            value = "INVOKE",
            target = "Lcom/mojang/renderpearl/backend/vulkan/VulkanConst;textureUsageToVk(ILcom/mojang/renderpearl/api/GpuFormat;)I"
        )
    )
    private int vulkanshaders$enableStorageAndTransferUsageOnRenderAttachments(int usage, GpuFormat format) {
        int vkUsage = VulkanConst.textureUsageToVk(usage, format);
        // Never add VK_IMAGE_USAGE_STORAGE_BIT to SRGB formats (forbidden by Vulkan spec unless extended).
        boolean isSrgb = format != null && format.name().contains("SRGB");
        if ((usage & 8) != 0 && format != null && format.hasColorAspect()) {
            vkUsage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
            if (!isSrgb) {
                vkUsage |= VK_IMAGE_USAGE_STORAGE_BIT;
            }
        }
        return vkUsage;
    }

    @Inject(method = "<init>", at = @At("RETURN"))
    private void vulkanshaders$onTextureCreated(CallbackInfo ci) {
        TextureRegistry.register((VulkanGpuTexture) (Object) this);
    }

    @Inject(method = "close", at = @At("HEAD"))
    private void vulkanshaders$onTextureClosed(CallbackInfo ci) {
        TextureRegistry.unregister((VulkanGpuTexture) (Object) this);
    }

    @Mixin(VulkanGpuTextureView.class)
    public static class TextureViewMixin {
        @Inject(method = "<init>", at = @At("RETURN"))
        private void vulkanshaders$onTextureViewCreated(CallbackInfo ci) {
            TextureRegistry.onTextureViewCreated((VulkanGpuTextureView) (Object) this);
        }
    }
}
