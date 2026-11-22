package dev.vulkanshaders.mixin.world;

import net.minecraft.client.renderer.CloudRenderer;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

import dev.vulkanshaders.VulkanShaders;

/**
 * The Hyaloclast pack renders its own analytic/volumetric cloud layer inside the lighting
 * compute pass. Mojang's flat extruded-cell cloud geometry would draw on top of it as opaque
 * white g-buffer geometry, so the vanilla cloud renderer is skipped entirely while a shader
 * pack is active.
 */
@Mixin(CloudRenderer.class)
public class CloudRendererMixin {

    @Inject(method = "render(Lnet/minecraft/client/CloudStatus;Lcom/mojang/renderpearl/api/commands/RenderPass;)V", at = @At("HEAD"), cancellable = true)
    private void vulkanshaders$skipVanillaClouds(CallbackInfo ci) {
        if (VulkanShaders.shaderPackActive) {
            ci.cancel();
        }
    }

    @Inject(method = "renderOit(Lnet/minecraft/client/CloudStatus;Lnet/minecraft/client/renderer/oit/OitStage;Lcom/mojang/renderpearl/api/textures/GpuTextureView;Lnet/minecraft/client/renderer/oit/OitRenderPassProvider$Parameters;)V", at = @At("HEAD"), cancellable = true)
    private void vulkanshaders$skipVanillaCloudsOit(CallbackInfo ci) {
        if (VulkanShaders.shaderPackActive) {
            ci.cancel();
        }
    }
}
