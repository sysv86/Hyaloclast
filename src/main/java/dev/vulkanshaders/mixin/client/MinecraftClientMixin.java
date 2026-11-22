package dev.vulkanshaders.mixin.client;

import dev.vulkanshaders.VulkanShaders;
import dev.vulkanshaders.backend.NativeCoreBridge;
import dev.vulkanshaders.diagnostics.HyaloclastErrorLog;
import net.minecraft.client.Minecraft;
import net.minecraft.client.gui.screens.Screen;
import net.minecraft.client.multiplayer.ClientLevel;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Unique;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfoReturnable;

import java.util.concurrent.CompletableFuture;

/**
 * 1. Invokes NVIDIA Reflex `vkLatencySleepNV` + `SIMULATION_START` + `INPUT_SAMPLE` markers at
 *    the head of `Minecraft.runTick` before input polling and world simulation begin.
 * 2. Emits `SIMULATION_END` and flushes the once-per-session Hyaloclast status banner at the
 *    tail of `Minecraft.tick`.
 * 3. Resets per-world shader graph, temporal history, G-buffer targets, and section slices
 *    when disconnecting / quitting to the title screen.
 * 4. Marks clean client shutdown on `stop` / `close` so normal exits do not trigger crash dumps.
 * 5. Propagates display resize (`F11` / window resize) and resource pack reload (`F3+T`)
 *    lifecycle events to Hyaloclast and the native Vulkan core.
 */
@Mixin(Minecraft.class)
public class MinecraftClientMixin {
    @Inject(method = "runTick", at = @At("HEAD"))
    private void vulkanshaders$onBeginFrameSleepAndInputSample(boolean renderLevel, CallbackInfo ci) {
        if (NativeCoreBridge.initialized && VulkanShaders.shaderPackActive) {
            NativeCoreBridge.reflexSleepAndSampleInput();
        }
    }

    @Inject(method = "tick", at = @At("TAIL"))
    private void vulkanshaders$onEndSimulationAndFlushChatNotice(CallbackInfo ci) {
        if (NativeCoreBridge.initialized && VulkanShaders.shaderPackActive) {
            NativeCoreBridge.reflexMarker(NativeCoreBridge.REFLEX_MARKER_SIMULATION_END);
        }
        VulkanShaders.onClientTick((Minecraft) (Object) this);
    }

    @Inject(method = "disconnect(Lnet/minecraft/client/gui/screens/Screen;ZZ)V", at = @At("HEAD"))
    private void vulkanshaders$onDisconnectHead(Screen nextScreen, boolean keepResourcePacks, boolean stopSounds, CallbackInfo ci) {
        VulkanShaders.onWorldUnloading();
    }

    @Inject(method = "clearClientLevel(Lnet/minecraft/client/gui/screens/Screen;)V", at = @At("HEAD"))
    private void vulkanshaders$onClearClientLevelHead(Screen nextScreen, CallbackInfo ci) {
        VulkanShaders.onWorldUnloading();
    }

    @Inject(method = "updateLevelInEngines(Lnet/minecraft/client/multiplayer/ClientLevel;Z)V", at = @At("RETURN"))
    private void vulkanshaders$onUpdateLevelInEngines(ClientLevel level, boolean stopSounds, CallbackInfo ci) {
        if (level == null) {
            VulkanShaders.onWorldUnloaded();
        }
    }

    @Inject(method = "stop", at = @At("HEAD"), require = 0)
    private void vulkanshaders$onStopHead(CallbackInfo ci) {
        HyaloclastErrorLog.markCleanShutdown();
    }

    @Inject(method = "close", at = @At("HEAD"), require = 0)
    private void vulkanshaders$onCloseHead(CallbackInfo ci) {
        HyaloclastErrorLog.markCleanShutdown();
    }

    @Inject(method = "resizeDisplay", at = @At("TAIL"), require = 0)
    private void vulkanshaders$onResizeDisplayTail(CallbackInfo ci) {
        VulkanShaders.onDisplayResized();
    }

    @Inject(method = "reloadResourcePacks(Z)Ljava/util/concurrent/CompletableFuture;", at = @At("RETURN"), require = 0)
    private void vulkanshaders$onReloadResourcePacksFlagReturn(
        boolean force,
        CallbackInfoReturnable<CompletableFuture<Void>> cir
    ) {
        vulkanshaders$scheduleResourceReloadHook(cir.getReturnValue());
    }

    @Inject(method = "reloadResourcePacks()Ljava/util/concurrent/CompletableFuture;", at = @At("RETURN"), require = 0)
    private void vulkanshaders$onReloadResourcePacksReturn(
        CallbackInfoReturnable<CompletableFuture<Void>> cir
    ) {
        vulkanshaders$scheduleResourceReloadHook(cir.getReturnValue());
    }

    @Unique
    private void vulkanshaders$scheduleResourceReloadHook(CompletableFuture<Void> future) {
        if (future != null) {
            Minecraft self = (Minecraft) (Object) this;
            future.whenComplete((ignored, err) -> {
                if (err == null) {
                    self.execute(() -> VulkanShaders.onResourcePacksReloaded());
                }
            });
        } else {
            VulkanShaders.onResourcePacksReloaded();
        }
    }
}
