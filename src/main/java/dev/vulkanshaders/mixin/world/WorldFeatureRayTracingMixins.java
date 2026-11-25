package dev.vulkanshaders.mixin.world;

import com.llamalad7.mixinextras.injector.wrapoperation.Operation;
import com.llamalad7.mixinextras.injector.wrapoperation.WrapOperation;
import com.llamalad7.mixinextras.sugar.Local;
import com.mojang.blaze3d.systems.RenderSystem;
import com.mojang.blaze3d.vertex.DefaultVertexFormat;
import com.mojang.blaze3d.vertex.MeshData;
import com.mojang.blaze3d.vertex.PoseStack;
import com.mojang.blaze3d.vertex.VertexConsumer;
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
import net.minecraft.client.Minecraft;
import net.minecraft.client.model.Model;
import net.minecraft.client.renderer.RenderPipelines;
import net.minecraft.client.renderer.StagedVertexBuffer;
import net.minecraft.client.renderer.entity.state.AvatarRenderState;
import net.minecraft.client.renderer.feature.ModelFeatureRenderer;
import net.minecraft.client.renderer.rendertype.PreparedRenderType;
import net.minecraft.client.renderer.rendertype.RenderType;
import org.lwjgl.system.MemoryUtil;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Unique;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

import java.util.ArrayList;
import java.util.List;

public final class WorldFeatureRayTracingMixins {
    private static final int ENTITY_CATEGORY_PLAYER = 1;
    // EntityCategory::Player defaults (CAST_SHADOW | RECEIVE_GI | ALPHA_TESTED) plus NO_SHADOW.
    private static final int ENTITY_FLAGS_PLAYER_NO_SHADOW = 0x07 | 0x20;

    private static final List<int[]> localPlayerVertexRanges = new ArrayList<>();
    private static int modelMeshVertexCursor = 0;

    /** Counts vertices written by one model submit without changing the underlying mesh. */
    private static final class CountingVertexConsumer implements VertexConsumer {
        private final VertexConsumer delegate;
        private int written;

        private CountingVertexConsumer(VertexConsumer delegate) {
            this.delegate = delegate;
        }

        @Override
        public VertexConsumer addVertex(float x, float y, float z) {
            this.written++;
            this.delegate.addVertex(x, y, z);
            return this;
        }

        @Override
        public VertexConsumer setColor(int a, int r, int g, int b) {
            this.delegate.setColor(a, r, g, b);
            return this;
        }

        @Override
        public VertexConsumer setColor(int color) {
            this.delegate.setColor(color);
            return this;
        }

        @Override
        public VertexConsumer setUv(float u, float v) {
            this.delegate.setUv(u, v);
            return this;
        }

        @Override
        public VertexConsumer setUv1(int u, int v) {
            this.delegate.setUv1(u, v);
            return this;
        }

        @Override
        public VertexConsumer setUv2(int u, int v) {
            this.delegate.setUv2(u, v);
            return this;
        }

        @Override
        public VertexConsumer setUv3(float u, float v) {
            this.delegate.setUv3(u, v);
            return this;
        }

        @Override
        public VertexConsumer setNormal(float x, float y, float z) {
            this.delegate.setNormal(x, y, z);
            return this;
        }

        @Override
        public VertexConsumer setLineWidth(float width) {
            this.delegate.setLineWidth(width);
            return this;
        }
    }

    private static boolean isLocalPlayerThirdPersonBody(ModelFeatureRenderer.Submit submit) {
        if (submit == null || !(submit.state() instanceof AvatarRenderState avatar)) {
            return false;
        }
        Minecraft client = Minecraft.getInstance();
        if (client.player == null || avatar.id != client.player.getId()
            || client.player.isSpectator() || client.getCameraEntity() != client.player
            || client.options.getCameraType().isFirstPerson()
            || VulkanShadersTurboExtension.capturingFirstPersonAvatar) {
            return false;
        }
        return true;
    }

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

    @Mixin({ StagedVertexBuffer.Draw.class, ModelFeatureRenderer.class })
    public static class DrawMixin implements RayTracedWorldDraw {
        @WrapOperation(
            method = "prepareModel",
            at = @At(
                value = "INVOKE",
                target = "Lnet/minecraft/client/model/Model;renderToBuffer(Lcom/mojang/blaze3d/vertex/PoseStack;Lcom/mojang/blaze3d/vertex/VertexConsumer;III)V"
            ),
            require = 0
        )
        private void vulkanshaders$recordLocalPlayerVertices(
            Model<?> model,
            PoseStack pose,
            VertexConsumer consumer,
            int light,
            int overlay,
            int color,
            Operation<Void> original,
            @Local(argsOnly = true) ModelFeatureRenderer.Submit submit
        ) {
            boolean isLocalPlayer = isLocalPlayerThirdPersonBody(submit);
            int before = modelMeshVertexCursor;
            CountingVertexConsumer counting = new CountingVertexConsumer(consumer);
            original.call(model, pose, counting, light, overlay, color);
            int written = counting.written;
            if (isLocalPlayer && written > 0) {
                localPlayerVertexRanges.add(new int[] { before, before + written });
            }
            modelMeshVertexCursor += written;
        }

        @Unique
        private int vulkanshaders$textureSlot = -1;

        @Unique
        private boolean vulkanshaders$crumbling;

        @Override
        public void vulkanshaders$setRayTracingMaterial(int slot, boolean crumbling) {
            this.vulkanshaders$textureSlot = slot;
            this.vulkanshaders$crumbling = crumbling;
        }

        @Unique
        private void vulkanshaders$submitLegacy(long basePtr, int firstVertex, int vertexCount, int stride) {
            if (vertexCount <= 0) {
                return;
            }
            NativeCoreBridge.submitEntityVertices(
                basePtr + (long) firstVertex * stride,
                vertexCount,
                stride,
                this.vulkanshaders$textureSlot,
                this.vulkanshaders$crumbling,
                VulkanShadersTurboExtension.capturingFirstPersonAvatar
            );
        }

        @Inject(method = "append", at = @At("TAIL"), require = 0)
        private void vulkanshaders$captureTexturedMesh(MeshData mesh, CallbackInfo ci) {
            if (this.vulkanshaders$textureSlot < 0) {
                localPlayerVertexRanges.clear();
                modelMeshVertexCursor = 0;
                return;
            }
            MeshData.DrawState state = mesh.drawState();
            int stride = state.format().getVertexSize();
            int vertexCount = state.vertexCount();
            long basePtr = MemoryUtil.memAddress(mesh.vertexBuffer());
            boolean rangesValid = modelMeshVertexCursor == vertexCount && !localPlayerVertexRanges.isEmpty();
            modelMeshVertexCursor = 0;
            if (!rangesValid) {
                localPlayerVertexRanges.clear();
                vulkanshaders$submitLegacy(basePtr, 0, vertexCount, stride);
                return;
            }

            int cursor = 0;
            for (int[] range : localPlayerVertexRanges) {
                int start = range[0];
                int end = Math.min(range[1], vertexCount);
                if (start < cursor || end <= start) {
                    continue;
                }
                if (start > cursor) {
                    vulkanshaders$submitLegacy(basePtr, cursor, start - cursor, stride);
                }
                int localCount = end - start;
                if (start % 4 == 0 && localCount % 4 == 0 && localCount >= 4
                    && !this.vulkanshaders$crumbling
                    && !VulkanShadersTurboExtension.capturingFirstPersonAvatar) {
                    NativeCoreBridge.submitEntityVerticesCategorized(
                        basePtr + (long) start * stride,
                        localCount,
                        stride,
                        true,
                        this.vulkanshaders$textureSlot,
                        0,
                        0,
                        0,
                        ENTITY_CATEGORY_PLAYER,
                        ENTITY_FLAGS_PLAYER_NO_SHADOW
                    );
                } else {
                    vulkanshaders$submitLegacy(basePtr, start, localCount, stride);
                }
                cursor = end;
            }
            if (cursor < vertexCount) {
                vulkanshaders$submitLegacy(basePtr, cursor, vertexCount - cursor, stride);
            }
            localPlayerVertexRanges.clear();
        }
    }
}
