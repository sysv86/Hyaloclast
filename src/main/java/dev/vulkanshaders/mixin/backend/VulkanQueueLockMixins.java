package dev.vulkanshaders.mixin.backend;

import com.llamalad7.mixinextras.injector.wrapmethod.WrapMethod;
import com.llamalad7.mixinextras.injector.wrapoperation.Operation;
import com.mojang.renderpearl.backend.vulkan.VulkanCommandEncoder;
import com.mojang.renderpearl.backend.vulkan.VulkanQueue;
import com.mojang.renderpearl.frontend.FrontendGpuSurface;
import dev.vulkanshaders.backend.NativeCoreBridge;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Shadow;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

/**
 * Serializes all `RenderPearl` queue submissions, `waitIdle`, timeline semaphore waits, and
 * swapchain acquire/present operations with the native Rust `ReentrantMutex`
 * (`gpu::lock_queue` / `unlock_queue`), tracks real GPU timeline submission & completion values
 * for `RetireQueue`, and emits `VK_NV_low_latency2` `PRESENT_START` / `PRESENT_END` markers around
 * swapchain presentation.
 */
public final class VulkanQueueLockMixins {
    private VulkanQueueLockMixins() {
    }

    @Mixin(VulkanQueue.Submission.class)
    public static class SubmissionMixin {
        @WrapMethod(method = "close")
        private void vulkanshaders$lockedSubmissionClose(Operation<Void> original) {
            if (!NativeCoreBridge.initialized) {
                original.call();
                return;
            }
            if (!NativeCoreBridge.lockQueue()) {
                original.call();
                return;
            }
            try {
                NativeCoreBridge.reflexMarker(NativeCoreBridge.REFLEX_MARKER_RENDERSUBMIT_START);
                original.call();
            } finally {
                NativeCoreBridge.reflexMarker(NativeCoreBridge.REFLEX_MARKER_RENDERSUBMIT_END);
                NativeCoreBridge.unlockQueue();
            }
        }
    }

    @Mixin(VulkanQueue.class)
    public static class QueueMixin {
        @WrapMethod(method = "waitIdle")
        private void vulkanshaders$lockedWaitIdle(Operation<Void> original) {
            if (!NativeCoreBridge.initialized) {
                original.call();
                return;
            }
            if (!NativeCoreBridge.lockQueue()) {
                original.call();
                return;
            }
            try {
                original.call();
            } finally {
                NativeCoreBridge.unlockQueue();
            }
        }
    }

    @Mixin(VulkanCommandEncoder.class)
    public static class EncoderWaitMixin {
        @Shadow
        private long currentSubmitIndex;

        @Inject(method = "submit", at = @At("RETURN"))
        private void vulkanshaders$recordRealSubmit(CallbackInfo ci) {
            if (NativeCoreBridge.initialized && this.currentSubmitIndex > 0L) {
                // RenderPearl increments the index after signaling the submitted value.
                NativeCoreBridge.recordSubmission(this.currentSubmitIndex - 1L);
            }
        }

        @WrapMethod(method = "awaitSubmitCompletion")
        private boolean vulkanshaders$lockedAwaitSubmitCompletion(
            long submitIndex,
            long timeoutNS,
            Operation<Boolean> original
        ) {
            // Do NOT hold QUEUE_MUTEX while blocking in vkWaitSemaphores/awaitSubmitCompletion —
            // doing so stalls PresentThread and background uploads. Notify RetireQueue of completed timeline value.
            boolean done = original.call(submitIndex, timeoutNS);
            if (done && NativeCoreBridge.initialized && submitIndex > 0L) {
                NativeCoreBridge.notifyTimelineCompleted(submitIndex);
            }
            return done;
        }
    }

    @Mixin(FrontendGpuSurface.class)
    public static class SurfaceMixin {
        @WrapMethod(method = "present")
        private void vulkanshaders$lockedPresent(Operation<Void> original) {
            if (!NativeCoreBridge.initialized) {
                original.call();
                return;
            }
            // The frame generation worker also needs the queue lock to finish presenting.
            NativeCoreBridge.waitPresentIdle();
            if (!NativeCoreBridge.lockQueue()) {
                original.call();
                return;
            }
            try {
                NativeCoreBridge.reflexMarker(NativeCoreBridge.REFLEX_MARKER_PRESENT_START);
                original.call();
            } finally {
                NativeCoreBridge.reflexMarker(NativeCoreBridge.REFLEX_MARKER_PRESENT_END);
                NativeCoreBridge.unlockQueue();
            }
        }

        @WrapMethod(method = "acquireNextTexture")
        private void vulkanshaders$lockedAcquireNextTexture(
            Operation<Void> original
        ) {
            if (!NativeCoreBridge.initialized) {
                original.call();
                return;
            }
            NativeCoreBridge.waitPresentIdle();
            if (!NativeCoreBridge.lockQueue()) {
                original.call();
                return;
            }
            try {
                original.call();
            } finally {
                NativeCoreBridge.unlockQueue();
            }
        }
    }
}
