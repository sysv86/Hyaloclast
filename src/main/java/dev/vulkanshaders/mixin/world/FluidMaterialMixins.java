package dev.vulkanshaders.mixin.world;

import com.mojang.blaze3d.vertex.BufferBuilder;
import com.mojang.blaze3d.vertex.VertexConsumer;
import com.mojang.renderpearl.api.vertex.VertexFormat;
import dev.vulkanshaders.bridge.StandaloneRenderPearlBridge;
import dev.vulkanshaders.bridge.VulkanShadersTurboExtension;
import net.minecraft.client.renderer.block.FluidRenderer;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.gen.Accessor;
import org.spongepowered.asm.mixin.gen.Invoker;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

public final class FluidMaterialMixins {
    @Mixin(BufferBuilder.class)
    public interface BufferBuilderAccessor {
        @Accessor("format")
        VertexFormat vulkanshaders$getFormat();

        @Invoker("beginVertex")
        long vulkanshaders$beginVertex();
    }

    @Mixin(value = FluidRenderer.class, priority = 1200)
    public static class FluidRendererMixin {
        @Inject(method = "addFace", at = @At("HEAD"), cancellable = true)
        private void vulkanshaders$writeFluidMaterial(
            VertexConsumer consumer,
            float x0, float y0, float z0, float u0, float v0,
            float x1, float y1, float z1, float u1, float v1,
            float x2, float y2, float z2, float u2, float v2,
            float x3, float y3, float z3, float u3, float v3,
            int color, int light, boolean addBackFace,
            CallbackInfo ci
        ) {
            if (!(consumer instanceof BufferBuilderAccessor builder)
                || builder.vulkanshaders$getFormat() != VulkanShadersTurboExtension.IGNIS_TERRAIN_40B) {
                return;
            }

            // Reserve the complete face in Minecraft's builder so its vertex count and pointer stay valid.
            int count = addBackFace ? 8 : 4;
            int stride = builder.vulkanshaders$getFormat().getVertexSize();
            for (int i = 1; i < count; ++i) {
                builder.vulkanshaders$beginVertex();
            }
            long base = builder.vulkanshaders$beginVertex() - (long) (count - 1) * stride;
            boolean wroteViaTurbo = false;
            if (StandaloneRenderPearlBridge.vulkanTurboLoaded) {
                try {
                    dev.vulkanturbo.core.ChunkPipelineOptimizer.writeFluidFaceExtended(
                        base, stride,
                        x0, y0, z0, u0, v0, x1, y1, z1, u1, v1,
                        x2, y2, z2, u2, v2, x3, y3, z3, u3, v3,
                        color, light, addBackFace
                    );
                    wroteViaTurbo = true;
                } catch (NoClassDefFoundError ignored) {
                    // Fall back to standalone vertex write below.
                }
            }
            if (!wroteViaTurbo) {
                // Standalone fallback when VulkanTurbo is not installed
                float[] xs = {x0, x1, x2, x3};
                float[] ys = {y0, y1, y2, y3};
                float[] zs = {z0, z1, z2, z3};
                float[] us = {u0, u1, u2, u3};
                float[] vsCoords = {v0, v1, v2, v3};
                for (int i = 0; i < 4; ++i) {
                    long ptr = base + (long) i * stride;
                    org.lwjgl.system.MemoryUtil.memPutFloat(ptr, xs[i]);
                    org.lwjgl.system.MemoryUtil.memPutFloat(ptr + 4L, ys[i]);
                    org.lwjgl.system.MemoryUtil.memPutFloat(ptr + 8L, zs[i]);
                    org.lwjgl.system.MemoryUtil.memPutInt(ptr + 12L, color);
                    org.lwjgl.system.MemoryUtil.memPutFloat(ptr + 16L, us[i]);
                    org.lwjgl.system.MemoryUtil.memPutFloat(ptr + 20L, vsCoords[i]);
                    org.lwjgl.system.MemoryUtil.memPutInt(ptr + 24L, light);
                }
                if (addBackFace) {
                    int[] rev = {0, 3, 2, 1};
                    for (int i = 0; i < 4; ++i) {
                        int k = rev[i];
                        long ptr = base + (long) (4 + i) * stride;
                        org.lwjgl.system.MemoryUtil.memPutFloat(ptr, xs[k]);
                        org.lwjgl.system.MemoryUtil.memPutFloat(ptr + 4L, ys[k]);
                        org.lwjgl.system.MemoryUtil.memPutFloat(ptr + 8L, zs[k]);
                        org.lwjgl.system.MemoryUtil.memPutInt(ptr + 12L, color);
                        org.lwjgl.system.MemoryUtil.memPutFloat(ptr + 16L, us[k]);
                        org.lwjgl.system.MemoryUtil.memPutFloat(ptr + 20L, vsCoords[k]);
                        org.lwjgl.system.MemoryUtil.memPutInt(ptr + 24L, light);
                    }
                }
            }
            try {
                VulkanShadersTurboExtension.INSTANCE.populateExtendedFluidFace(
                    base, stride,
                    x0, y0, z0, u0, v0, x1, y1, z1, u1, v1,
                    x2, y2, z2, u2, v2, x3, y3, z3, u3, v3,
                    addBackFace
                );
            } catch (NoClassDefFoundError ignored) {
                // Standalone guard if VulkanTurbo extension interface cannot be loaded.
            }
            ci.cancel();
        }
    }
}
