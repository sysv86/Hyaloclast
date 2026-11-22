package dev.vulkanshaders.mixin.compat;

import dev.vulkanshaders.bridge.DistantHorizonsBridge;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Pseudo;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

import java.nio.ByteBuffer;

/**
 * Clean-room {@code @Pseudo} hook on Distant Horizons' {@code BlazeVertexBufferWrapper} to capture
 * CPU-side 24-byte LOD vertex uploads and buffer closes for Vulkan LOD BLAS registration.
 */
@Pseudo
@Mixin(targets = "com.seibel.distanthorizons.common.render.blaze.wrappers.buffer.BlazeVertexBufferWrapper", remap = false)
public abstract class DhVertexBufferWrapperMixin {
    @Inject(method = "uploadVertexBuffer(Ljava/nio/ByteBuffer;I)V", at = @At("HEAD"), require = 0)
    private void vulkanshaders$onDhVertexBufferUpload(ByteBuffer vertexBuffer, int vertexCount, CallbackInfo ci) {
        DistantHorizonsBridge.onDhVertexBufferUpload(this, vertexBuffer, vertexCount);
    }

    @Inject(method = "close()V", at = @At("HEAD"), require = 0)
    private void vulkanshaders$onDhVertexBufferClose(CallbackInfo ci) {
        DistantHorizonsBridge.onDhVertexBufferClose(this);
    }
}
