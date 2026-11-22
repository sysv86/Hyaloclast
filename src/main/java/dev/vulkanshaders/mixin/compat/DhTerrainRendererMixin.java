package dev.vulkanshaders.mixin.compat;

import com.llamalad7.mixinextras.injector.wrapoperation.Operation;
import com.llamalad7.mixinextras.injector.wrapoperation.WrapOperation;
import com.mojang.renderpearl.api.pipeline.RenderPipeline;
import dev.vulkanshaders.VulkanShaders;
import dev.vulkanshaders.bridge.DistantHorizonsBridge;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Pseudo;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Coerce;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

/**
 * Clean-room {@code @Pseudo} hook on Distant Horizons 3.x {@code BlazeDhTerrainRenderer} that:
 * 1. Captures per-frame DH matrices, clip planes, texture handles, and container metadata for
 *    LOD BLAS registration and deferred DH shading.
 * 2. Cancels DH raster terrain rendering on Tier 3 Path Tracing (where LODs are ray-traced in TLAS).
 * 3. Redirects Tiers 0-2 DH terrain passes into Hyaloclast's 3-MRT DH G-Buffer pipelines.
 */
@Pseudo
@Mixin(targets = "com.seibel.distanthorizons.common.render.blaze.BlazeDhTerrainRenderer", remap = false)
public abstract class DhTerrainRendererMixin {
    private static final String LEGACY_RENDER = "render(Lcom/seibel/distanthorizons/core/render/RenderParams;ZLcom/seibel/distanthorizons/core/util/objects/SortedArraySet;Lcom/seibel/distanthorizons/core/wrapperInterfaces/minecraft/IProfilerWrapper;)V";
    private static final String API_RENDER = "render(Lcom/seibel/distanthorizons/api/methods/events/sharedParameterObjects/DhApiRenderParam;Lcom/seibel/distanthorizons/api/enums/rendering/EDhApiTerrainRenderPass;Lcom/seibel/distanthorizons/api/objects/util/IDhApiReadOnlyList;)V";

    @Inject(
        method = API_RENDER,
        at = @At("HEAD"),
        cancellable = true,
        require = 0
    )
    private void vulkanshaders$onDhTerrainRender(
        @Coerce Object apiRenderEventParam,
        @Coerce Object renderPass,
        @Coerce Object bufferContainers,
        CallbackInfo ci
    ) {
        DistantHorizonsBridge.onBeforeDhTerrainRender(apiRenderEventParam, renderPass, bufferContainers);
        if (DistantHorizonsBridge.shouldCancelDhTerrainRaster()) {
            ci.cancel();
        }
    }

    @Inject(method = LEGACY_RENDER, at = @At("HEAD"), cancellable = true, require = 0)
    private void vulkanshaders$onDhTerrainRender(
        @Coerce Object renderParams,
        boolean transparentPass,
        @Coerce Object bufferContainers,
        @Coerce Object profiler,
        CallbackInfo ci
    ) {
        DistantHorizonsBridge.onBeforeDhTerrainRender(renderParams, transparentPass, bufferContainers);
        if (DistantHorizonsBridge.shouldCancelDhTerrainRaster()) {
            ci.cancel();
        }
    }

    @WrapOperation(
        method = {LEGACY_RENDER, API_RENDER},
        at = @At(
            value = "INVOKE",
            target = "Lcom/seibel/distanthorizons/common/render/blaze/wrappers/RenderPassWrapper;setPipeline(Lcom/mojang/renderpearl/api/pipeline/RenderPipeline;)V"
        ),
        require = 0
    )
    private void vulkanshaders$redirectDhTerrainPipeline(
        @Coerce Object renderPassWrapper,
        RenderPipeline originalPipeline,
        Operation<Void> original
    ) {
        if (VulkanShaders.shaderPackActive && !DistantHorizonsBridge.shouldCancelDhTerrainRaster()) {
            RenderPipeline target = DistantHorizonsBridge.prepareDhGbufferPass(renderPassWrapper, originalPipeline);
            original.call(renderPassWrapper, target);
            return;
        }
        original.call(renderPassWrapper, originalPipeline);
    }
}
