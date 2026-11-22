package dev.vulkanshaders.mixin.world;

import com.mojang.renderpearl.api.pipeline.RenderPipeline;
import dev.vulkanshaders.pipeline.GbufferPipelineRegistry;
import net.minecraft.client.renderer.RenderPipelines;
import org.spongepowered.asm.mixin.Final;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Shadow;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

/**
 * Registers Hyaloclast's 40-byte (`IGNIS_TERRAIN_40B`) G-Buffer region terrain pipelines.
 */
@Mixin(value = RenderPipelines.class, priority = 1100)
public abstract class GbufferRenderPipelinesMixin {
    @Shadow
    @Final
    private static RenderPipeline.Snippet LIT_BLOCKS_SNIPPET;

    @Shadow
    @Final
    private static RenderPipeline WIREFRAME_MULTIDRAW;

    @Shadow
    @Final
    private static RenderPipeline.Snippet ENTITY_SNIPPET;

    @Shadow
    @Final
    private static RenderPipeline.Snippet PARTICLE_SNIPPET;

    @Shadow
    private static RenderPipeline registerOptional(RenderPipeline pipeline) {
        throw new AssertionError();
    }

    @Inject(method = "<clinit>", at = @At("TAIL"))
    private static void vulkanshaders$registerGbufferRegionPipelines(CallbackInfo ci) {
        GbufferPipelineRegistry.registerAll(
            LIT_BLOCKS_SNIPPET,
            ENTITY_SNIPPET,
            PARTICLE_SNIPPET,
            WIREFRAME_MULTIDRAW,
            GbufferRenderPipelinesMixin::registerOptional
        );
    }
}
