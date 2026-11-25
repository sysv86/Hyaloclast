package dev.vulkanshaders.mixin.world;

import com.llamalad7.mixinextras.injector.wrapoperation.Operation;
import com.llamalad7.mixinextras.injector.wrapoperation.WrapOperation;
import com.mojang.blaze3d.pipeline.RenderTarget;
import com.mojang.blaze3d.resource.GraphicsResourceAllocator;
import com.mojang.blaze3d.vertex.PoseStack;
import com.mojang.renderpearl.api.buffers.GpuBufferSlice;
import com.mojang.renderpearl.api.commands.RenderPass;
import com.mojang.renderpearl.api.textures.GpuTexture;
import dev.vulkanshaders.VulkanShaders;
import dev.vulkanshaders.backend.NativeCoreBridge;
import dev.vulkanshaders.bridge.VulkanShadersTurboExtension;
import dev.vulkanshaders.config.LightingMode;
import dev.vulkanshaders.pipeline.GbufferTargetManager;
import com.mojang.blaze3d.framegraph.FrameGraphBuilder;
import net.minecraft.client.Minecraft;
import net.minecraft.client.renderer.chunk.ChunkSectionsToRender;
import net.minecraft.client.renderer.GameRenderer;
import net.minecraft.client.renderer.LevelRenderer;
import net.minecraft.client.renderer.SectionOcclusionGraph;
import net.minecraft.client.renderer.feature.FeatureRenderDispatcher;
import net.minecraft.client.renderer.SubmitNodeStorage;
import net.minecraft.client.renderer.entity.state.AvatarRenderState;
import net.minecraft.client.renderer.state.level.CameraRenderState;
import net.minecraft.client.renderer.state.level.ChunkLoadingRenderState;
import net.minecraft.client.renderer.state.level.LevelRenderState;
import org.joml.Matrix4f;
import org.joml.Vector4f;
import org.spongepowered.asm.mixin.Final;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Shadow;
import org.spongepowered.asm.mixin.Unique;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

/**
 * Hooks {@link LevelRenderer} to:
 * 1. Set the projection parameters used for Tier 3 overlay depth.
 * 2. Stage 4-MRT G-Buffer color attachment views ({@code mainColor}, {@code gNormalRoughness},
 *    {@code gMaterialPbr}, {@code gVelocity}) right before the main world {@code RenderPass} opens.
 * 3. In Tier 3 Full Path Tracing ({@link LightingMode#PATH}), skip vanilla world
 *    rasterization passes ({@code addSkyPass}, {@code executeSolid}, {@code executeClassicTransparency},
 *    {@code executeOit}) while keeping {@code SectionOcclusionGraph}, chunk compilation/uploads,
 *    and depth-reprojected gizmo/overlay passes active.
 */
@Mixin(value = LevelRenderer.class, priority = 1100)
public abstract class LevelRendererMixin {
    @Shadow
    @Final
    private GameRenderer gameRenderer;

    @Shadow
    @Final
    private LevelRenderState levelRenderState;

    @Unique
    private final SubmitNodeStorage vulkanshaders$firstPersonSubmits = new SubmitNodeStorage();

    @Shadow
    public abstract SectionOcclusionGraph sectionOcclusionGraph();

    @WrapOperation(
        method = "render",
        at = @At(
            value = "INVOKE",
            target = "Lnet/minecraft/client/renderer/feature/FeatureRenderDispatcher;prepareFrame(Lnet/minecraft/client/renderer/SubmitNodeStorage;)Lnet/minecraft/client/renderer/feature/FeatureRenderDispatcher$PreparedFrame;"
        )
    )
    private FeatureRenderDispatcher.PreparedFrame vulkanshaders$prepareWorldFeatures(
        FeatureRenderDispatcher dispatcher,
        SubmitNodeStorage submits,
        Operation<FeatureRenderDispatcher.PreparedFrame> original
    ) {
        VulkanShadersTurboExtension.capturingWorldFeatures =
            VulkanShadersTurboExtension.INSTANCE.shouldCaptureWorldFeatures() && NativeCoreBridge.initialized;
        try {
            if (VulkanShadersTurboExtension.capturingWorldFeatures) {
                this.vulkanshaders$captureFirstPersonAvatar(dispatcher);
            }
            return original.call(dispatcher, submits);
        } finally {
            VulkanShadersTurboExtension.capturingWorldFeatures = false;
        }
    }

    @Unique
    private void vulkanshaders$captureFirstPersonAvatar(FeatureRenderDispatcher dispatcher) {
        CameraRenderState camera = this.levelRenderState.cameraRenderState;
        AvatarRenderState avatar = this.levelRenderState.playerRenderState.avatarRenderState;
        Minecraft client = Minecraft.getInstance();
        if (!camera.isFirstPerson || camera.entityRenderState.isSleeping
            || client.getCameraEntity() != client.player || avatar == null || avatar.isSpectator) {
            return;
        }

        var renderer = (LevelRenderer) (Object) this;
        renderer.entityRenderDispatcher().submit(
            avatar, camera, avatar.x - camera.pos.x(), avatar.y - camera.pos.y(), avatar.z - camera.pos.z(),
            new PoseStack(), this.vulkanshaders$firstPersonSubmits
        );
        VulkanShadersTurboExtension.capturingFirstPersonAvatar = true;
        try {
            // Preparation feeds the RT mesh capture. Closing the frame skips all raster draws.
            dispatcher.prepareFrame(this.vulkanshaders$firstPersonSubmits).close();
        } finally {
            VulkanShadersTurboExtension.capturingFirstPersonAvatar = false;
        }
    }

    @Inject(method = "render", at = @At("HEAD"))
    private void vulkanshaders$beginWorldFrame(
        GraphicsResourceAllocator resourceAllocator,
        boolean renderOutline,
        CameraRenderState cameraState,
        GpuBufferSlice terrainFog,
        Vector4f fogColor,
        boolean shouldRenderSky,
        boolean consistentDepthRequired,
        CallbackInfo ci
    ) {
        if (!VulkanShaders.shaderPackActive) {
            return;
        }

        RenderTarget mainTarget = this.gameRenderer.mainRenderTarget();
        if (VulkanShaders.INSTANCE.getConfig().getLightingMode() == LightingMode.PATH) {
            this.vulkanshaders$keepOcclusionGraphFed();
            if (VulkanShaders.INSTANCE.getConfig().getEnableOverlayPass() && NativeCoreBridge.initialized && mainTarget != null) {
                GpuTexture depthTex = mainTarget.getDepthTexture();
                Matrix4f proj = cameraState.projectionMatrix;
                if (depthTex != null && proj != null) {
                    int w = depthTex.getWidth(0);
                    int h = depthTex.getHeight(0);
                    NativeCoreBridge.setOverlayDepthParams(
                        proj.m22(),
                        proj.m32(),
                        proj.m23(),
                        proj.m33(),
                        w,
                        h,
                        w,
                        h
                    );
                }
            }
        }
    }

    @Inject(method = "prepareTranslucents", at = @At("RETURN"))
    private void vulkanshaders$stageMrtColorViewsBeforeMainPass(CallbackInfo ci) {
        if (!VulkanShaders.shaderPackActive) {
            return;
        }
        if (VulkanShaders.INSTANCE.getConfig().getLightingMode() != LightingMode.PATH) {
            GbufferTargetManager.prepareTerrainMrtPass();
        }
    }

    @Inject(method = "addSkyPass", at = @At("HEAD"), cancellable = true)
    private void vulkanshaders$skipSkyPassInTier3(
        FrameGraphBuilder frame,
        CameraRenderState cameraState,
        GpuBufferSlice skyFog,
        CallbackInfo ci
    ) {
        if (VulkanShaders.shaderPackActive
            && VulkanShaders.INSTANCE.getConfig().getLightingMode() == LightingMode.PATH) {
            ci.cancel();
        }
    }

    @Inject(method = "executeSolid", at = @At("HEAD"), cancellable = true)
    private void vulkanshaders$skipSolidRasterInTier3(
        ChunkSectionsToRender chunkSectionsToRender,
        FeatureRenderDispatcher.PreparedFrame featureFrame,
        RenderPass renderPass,
        CallbackInfo ci
    ) {
        if (VulkanShaders.shaderPackActive
            && VulkanShaders.INSTANCE.getConfig().getLightingMode() == LightingMode.PATH) {
            ci.cancel();
        }
    }

    @Inject(method = "executeClassicTransparency", at = @At("HEAD"), cancellable = true)
    private void vulkanshaders$skipClassicTransparencyInTier3(
        ChunkSectionsToRender chunkSectionsToRender,
        FeatureRenderDispatcher.PreparedFrame featureFrame,
        RenderPass renderPass,
        CallbackInfo ci
    ) {
        if (VulkanShaders.shaderPackActive
            && VulkanShaders.INSTANCE.getConfig().getLightingMode() == LightingMode.PATH) {
            ci.cancel();
        }
    }

    @Inject(method = "executeOit", at = @At("HEAD"), cancellable = true)
    private void vulkanshaders$skipOitInTier3(
        ChunkSectionsToRender chunkSectionsToRender,
        FeatureRenderDispatcher.PreparedFrame featureFrame,
        CallbackInfo ci
    ) {
        if (VulkanShaders.shaderPackActive
            && VulkanShaders.INSTANCE.getConfig().getLightingMode() == LightingMode.PATH) {
            ci.cancel();
        }
    }

    @Unique
    private void vulkanshaders$keepOcclusionGraphFed() {
        if (this.levelRenderState == null) {
            return;
        }
        ChunkLoadingRenderState chunkLoading = this.levelRenderState.chunkLoadingRenderState;
        if (chunkLoading == null) {
            return;
        }
        SectionOcclusionGraph graph = this.sectionOcclusionGraph();
        if (graph == null) {
            return;
        }
        if (chunkLoading.addedLoadedChunks != null && chunkLoading.removedLoadedChunks != null) {
            graph.updateLoadedChunks(chunkLoading.addedLoadedChunks, chunkLoading.removedLoadedChunks);
        }
        if (chunkLoading.addedEmptySections != null && chunkLoading.removedEmptySections != null) {
            graph.updateEmptySections(chunkLoading.addedEmptySections, chunkLoading.removedEmptySections);
        }
    }
}
