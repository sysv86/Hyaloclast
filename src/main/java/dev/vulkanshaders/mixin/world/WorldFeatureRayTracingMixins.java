package dev.vulkanshaders.mixin.world;

import com.llamalad7.mixinextras.injector.wrapoperation.Operation;
import com.llamalad7.mixinextras.injector.wrapoperation.WrapOperation;
import com.llamalad7.mixinextras.sugar.Local;
import com.mojang.blaze3d.systems.RenderSystem;
import com.mojang.blaze3d.vertex.DefaultVertexFormat;
import com.mojang.blaze3d.vertex.MeshData;
import com.mojang.blaze3d.vertex.VertexSorting;
import com.mojang.renderpearl.api.commands.RenderPass;
import com.mojang.renderpearl.api.pipeline.CompiledRenderPipeline;
import com.mojang.renderpearl.api.pipeline.PrimitiveTopology;
import com.mojang.renderpearl.api.pipeline.RenderPipeline;
import com.mojang.renderpearl.api.textures.GpuTextureView;
import com.mojang.renderpearl.api.vertex.VertexFormat;
import dev.vulkanshaders.VulkanShaders;
import dev.vulkanshaders.backend.NativeCoreBridge;
import dev.vulkanshaders.bridge.VulkanShadersTurboExtension;
import dev.vulkanshaders.config.RenderTier;
import dev.vulkanshaders.pipeline.GbufferPipelineRegistry;
import dev.vulkanshaders.pipeline.GbufferTargetManager;
import dev.vulkanshaders.pipeline.RayTracedWorldDraw;
import dev.vulkanshaders.texture.TextureRegistry;
import net.minecraft.client.renderer.RenderPipelines;
import net.minecraft.client.renderer.StagedVertexBuffer;
import net.minecraft.client.renderer.rendertype.PreparedRenderType;
import net.minecraft.client.renderer.rendertype.RenderType;
import org.lwjgl.system.MemoryUtil;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Unique;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

public final class WorldFeatureRayTracingMixins {
    @Mixin(PreparedRenderType.class)
    public static class PreparedRenderTypeMixin {
        @WrapOperation(
            method = "draw",
            at = @At(
                value = "INVOKE",
                target = "Lcom/mojang/renderpearl/api/commands/RenderPass;setPipeline(Lcom/mojang/renderpearl/api/pipeline/CompiledRenderPipeline;)V"
            )
        )
        private void vulkanshaders$bindWorldFeaturePipeline(
            RenderPass pass,
            CompiledRenderPipeline vanilla,
            Operation<Void> original,
            @Local(argsOnly = true) RenderPipeline pipeline
        ) {
            if (GbufferTargetManager.hasWorldGbuffer(pass)) {
                RenderPipeline replacement = GbufferPipelineRegistry.resolveWorldPipeline(pipeline);
                if (replacement != pipeline
                    && pipeline.getPrimitiveTopology() == replacement.getPrimitiveTopology()
                    && pipeline.getVertexFormatBinding(0).equals(replacement.getVertexFormatBinding(0))) {
                    original.call(pass, RenderSystem.getCompiledPipeline(replacement));
                    GpuTextureView sampler0View = null;
                    PreparedRenderType self = (PreparedRenderType) (Object) this;
                    for (PreparedRenderType.Texture texture : self.textures()) {
                        if ("Sampler0".equals(texture.name())) {
                            sampler0View = texture.textureView();
                            break;
                        }
                    }
                    GbufferTargetManager.onWorldFeaturePipelineBound(pass, sampler0View);
                    return;
                }
            }
            original.call(pass, vanilla);
        }
    }

    @Mixin(targets = "net.minecraft.client.renderer.feature.RenderTypeFeatureRenderer$Group")
    public static class GroupMixin {
        @WrapOperation(
            method = "getOrAddDraw",
            at = @At(
                value = "INVOKE",
                target = "Lnet/minecraft/client/renderer/StagedVertexBuffer;appendDraw(Lcom/mojang/renderpearl/api/vertex/VertexFormat;Lcom/mojang/renderpearl/api/pipeline/PrimitiveTopology;Lcom/mojang/blaze3d/vertex/VertexSorting;)Lnet/minecraft/client/renderer/StagedVertexBuffer$Draw;"
            )
        )
        private StagedVertexBuffer.Draw vulkanshaders$bindRayTracingMaterial(
            StagedVertexBuffer buffer,
            VertexFormat format,
            PrimitiveTopology topology,
            VertexSorting sorting,
            Operation<StagedVertexBuffer.Draw> original,
            @Local(argsOnly = true) RenderType renderType,
            @Local PreparedRenderType prepared
        ) {
            StagedVertexBuffer.Draw draw = original.call(buffer, format, topology, sorting);
            if (!VulkanShadersTurboExtension.capturingWorldFeatures
                || topology != PrimitiveTopology.QUADS
                || (format != DefaultVertexFormat.ENTITY && format != DefaultVertexFormat.ENTITY_GLINT_SPECIAL
                    && format != DefaultVertexFormat.BLOCK)
                || renderType.isOutline()
                || prepared.pipeline() == RenderPipelines.ENTITY_SHADOW
                || prepared.pipeline() == RenderPipelines.GLINT) {
                return draw;
            }

            boolean crumbling = prepared.pipeline() == RenderPipelines.CRUMBLING;
            if (crumbling && VulkanShaders.INSTANCE.getConfig().getRenderTier() != RenderTier.TIER_3_PATH_TRACING) {
                return draw;
            }
            for (PreparedRenderType.Texture texture : prepared.textures()) {
                if (texture.name().equals("Sampler0")) {
                    int slot = TextureRegistry.bindlessSlot(texture.textureView(), texture.sampler());
                    ((RayTracedWorldDraw) draw).vulkanshaders$setRayTracingMaterial(slot, crumbling);
                    break;
                }
            }
            return draw;
        }
    }

    @Mixin(StagedVertexBuffer.Draw.class)
    public static class DrawMixin implements RayTracedWorldDraw {
        @Unique
        private int vulkanshaders$textureSlot = -1;

        @Unique
        private boolean vulkanshaders$crumbling;

        @Override
        public void vulkanshaders$setRayTracingMaterial(int slot, boolean crumbling) {
            this.vulkanshaders$textureSlot = slot;
            this.vulkanshaders$crumbling = crumbling;
        }

        @Inject(method = "append", at = @At("TAIL"))
        private void vulkanshaders$captureTexturedMesh(MeshData mesh, CallbackInfo ci) {
            if (this.vulkanshaders$textureSlot < 0) {
                return;
            }
            MeshData.DrawState state = mesh.drawState();
            NativeCoreBridge.submitEntityVertices(
                MemoryUtil.memAddress(mesh.vertexBuffer()),
                state.vertexCount(),
                state.format().getVertexSize(),
                this.vulkanshaders$textureSlot,
                this.vulkanshaders$crumbling,
                VulkanShadersTurboExtension.capturingFirstPersonAvatar
            );
        }
    }
}
