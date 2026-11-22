package dev.vulkanshaders.mixin.compat;

import com.llamalad7.mixinextras.injector.wrapoperation.Operation;
import com.llamalad7.mixinextras.injector.wrapoperation.WrapOperation;
import dev.vulkanshaders.VulkanShaders;
import dev.vulkanshaders.bridge.DistantHorizonsBridge;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Pseudo;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Coerce;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

/**
 * Clean-room {@code @Pseudo} hooks suppressing Distant Horizons' unshaded post-processors
 * (SSAO, Fog, Far Fade, TAA) during Hyaloclast deferred shading so
 * {@code dhColorTextureWrapper} stays pure unfogged albedo for deferred shading.
 */
public final class DhPostProcessSuppressMixins {
    private DhPostProcessSuppressMixins() {}

    @Pseudo
    @Mixin(targets = "com.seibel.distanthorizons.core.render.renderer.CloudRenderHandler", remap = false)
    public static abstract class CloudMixin {
        @WrapOperation(
            method = "preRender",
            at = @At(value = "INVOKE",
                target = "Lcom/seibel/distanthorizons/api/interfaces/render/IDhApiRenderableBoxGroup;setActive(Z)V"),
            require = 0
        )
        private void vulkanshaders$setDhCloudVisibility(@Coerce Object cloudGroup, boolean active, Operation<Void> next) {
            next.call(cloudGroup, active && !VulkanShaders.shaderPackActive);
        }
    }

    @Pseudo
    @Mixin(targets = "com.seibel.distanthorizons.common.render.blaze.postProcessing.BlazeDhSsaoRenderer", remap = false)
    public static abstract class SsaoMixin {
        @Inject(
            method = {"render(Lcom/seibel/distanthorizons/core/render/RenderParams;)V", "render(Lcom/seibel/distanthorizons/api/methods/events/sharedParameterObjects/DhApiRenderParam;)V"},
            at = @At("HEAD"),
            cancellable = true,
            require = 0
        )
        private void vulkanshaders$suppressDhSsao(@Coerce Object apiRenderParams, CallbackInfo ci) {
            if (DistantHorizonsBridge.shouldSuppressDhRasterComposite()) {
                ci.cancel();
            }
        }
    }

    @Pseudo
    @Mixin(targets = "com.seibel.distanthorizons.common.render.blaze.postProcessing.BlazeDhFogRenderer", remap = false)
    public static abstract class FogMixin {
        @Inject(
            method = {"render(Lcom/seibel/distanthorizons/core/render/RenderParams;Lcom/seibel/distanthorizons/api/methods/events/sharedParameterObjects/DhApiFogRenderParam;)V", "render(Lcom/seibel/distanthorizons/api/methods/events/sharedParameterObjects/DhApiRenderParam;Lcom/seibel/distanthorizons/api/methods/events/sharedParameterObjects/DhApiFogRenderParam;)V"},
            at = @At("HEAD"),
            cancellable = true,
            require = 0
        )
        private void vulkanshaders$suppressDhFog(
            @Coerce Object apiRenderParams,
            @Coerce Object fogRenderParams,
            CallbackInfo ci
        ) {
            if (DistantHorizonsBridge.shouldSuppressDhRasterComposite()) {
                ci.cancel();
            }
        }
    }

    @Pseudo
    @Mixin(targets = "com.seibel.distanthorizons.common.render.blaze.postProcessing.BlazeDhFarFadeRenderer", remap = false)
    public static abstract class FarFadeMixin {
        @Inject(
            method = {"render(Lcom/seibel/distanthorizons/core/render/RenderParams;)V", "render(Lcom/seibel/distanthorizons/api/methods/events/sharedParameterObjects/DhApiRenderParam;)V"},
            at = @At("HEAD"),
            cancellable = true,
            require = 0
        )
        private void vulkanshaders$suppressDhFarFade(@Coerce Object renderParams, CallbackInfo ci) {
            if (DistantHorizonsBridge.shouldSuppressDhRasterComposite()) {
                ci.cancel();
            }
        }
    }

    @Pseudo
    @Mixin(targets = "com.seibel.distanthorizons.common.render.blaze.postProcessing.BlazeDhTaaRenderer", remap = false)
    public static abstract class TaaMixin {
        @Inject(
            method = {"render(Lcom/seibel/distanthorizons/core/render/RenderParams;)V", "render(Lcom/seibel/distanthorizons/api/methods/events/sharedParameterObjects/DhApiRenderParam;)V"},
            at = @At("HEAD"),
            cancellable = true,
            require = 0
        )
        private void vulkanshaders$suppressDhTaa(@Coerce Object apiRenderParams, CallbackInfo ci) {
            if (DistantHorizonsBridge.shouldSuppressDhRasterComposite()) {
                ci.cancel();
            }
        }
    }
}
