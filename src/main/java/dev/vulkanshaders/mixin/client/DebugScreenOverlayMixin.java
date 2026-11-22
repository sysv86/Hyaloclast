package dev.vulkanshaders.mixin.client;

import dev.vulkanshaders.VulkanShaders;
import dev.vulkanshaders.diagnostics.HyaloclastDebugHud;
import java.util.List;
import net.minecraft.client.gui.GuiGraphicsExtractor;
import net.minecraft.client.gui.components.DebugScreenOverlay;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Unique;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfoReturnable;

/**
 * Appends Hyaloclast's compact F3 debug HUD telemetry lines (active pack, render tier,
 * upscaler/frame-gen mode, BLAS/TLAS counts, VRAM usage/budget, and optional per-pass GPU
 * timestamps) to the left-side debug overlay when {@code config.showF3DebugInfo} is enabled.
 *
 * <p>Uses {@code require = 0} on all injections to stay resilient across MC 26.3 snapshot changes.
 */
@Mixin(DebugScreenOverlay.class)
public class DebugScreenOverlayMixin {
    @Unique
    private boolean vulkanshaders$appendedInGameInfo;

    @Inject(method = "getGameInformation", at = @At("RETURN"), require = 0)
    private void vulkanshaders$appendGameInformation(CallbackInfoReturnable<List<String>> cir) {
        this.vulkanshaders$appendedInGameInfo = false;
        if (!VulkanShaders.INSTANCE.getConfig().getShowF3DebugInfo()) {
            return;
        }
        List<String> lines = cir.getReturnValue();
        if (lines == null) {
            return;
        }
        List<String> hyaloclastLines = HyaloclastDebugHud.buildDebugLines();
        if (!hyaloclastLines.isEmpty()) {
            lines.addAll(hyaloclastLines);
            this.vulkanshaders$appendedInGameInfo = true;
        }
    }

    @Inject(method = "renderLines", at = @At("HEAD"), require = 0)
    private void vulkanshaders$appendLeftRenderLines(
        GuiGraphicsExtractor guiGraphics,
        List<String> lines,
        boolean leftSide,
        CallbackInfo ci
    ) {
        if (!leftSide) {
            return;
        }
        boolean alreadyAppended = this.vulkanshaders$appendedInGameInfo;
        this.vulkanshaders$appendedInGameInfo = false;
        if (alreadyAppended || lines == null || !VulkanShaders.INSTANCE.getConfig().getShowF3DebugInfo()) {
            return;
        }
        for (String line : lines) {
            if (line != null && line.startsWith("[Hyaloclast]")) {
                return;
            }
        }
        List<String> hyaloclastLines = HyaloclastDebugHud.buildDebugLines();
        if (!hyaloclastLines.isEmpty()) {
            lines.addAll(hyaloclastLines);
        }
    }
}
