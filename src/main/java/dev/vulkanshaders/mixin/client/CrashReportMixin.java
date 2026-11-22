package dev.vulkanshaders.mixin.client;

import dev.vulkanshaders.bridge.StandaloneRenderPearlBridge;
import dev.vulkanshaders.diagnostics.HyaloclastErrorLog;
import java.nio.file.Path;
import java.util.List;
import net.minecraft.CrashReport;
import net.minecraft.ReportType;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfoReturnable;

/**
 * Injects Hyaloclast diagnostic details into vanilla Minecraft {@link CrashReport} output and
 * mirrors crash logs into {@code <gameDir>/hyaloclast/crash-logs/} when Pyroclast is not present.
 * When Pyroclast is loaded, Pyroclast's {@code CrashReportMixin} collects Hyaloclast's diagnostic
 * block and writes the single unified crash log instead.
 */
@Mixin(CrashReport.class)
public class CrashReportMixin {
    @Inject(method = "getDetails(Ljava/lang/StringBuilder;)V", at = @At("TAIL"), require = 0)
    private void vulkanshaders$appendHyaloclastDiagnostics(StringBuilder details, CallbackInfo ci) {
        if (StandaloneRenderPearlBridge.vulkanTurboLoaded) {
            return;
        }
        try {
            if (details.indexOf("-- Hyaloclast (VulkanShaders) Diagnostics --") >= 0
                || details.indexOf("-- Hyaloclast Diagnostics --") >= 0) {
                return;
            }
            String section = HyaloclastErrorLog.buildCrashDiagnosticSection();
            if (section != null && !section.isEmpty()) {
                details.append("\n\n").append(section).append('\n');
            }
        } catch (Throwable ignored) {
            // Never interfere with vanilla crash report formatting.
        }
    }

    @Inject(
        method = "saveToFile(Ljava/nio/file/Path;Lnet/minecraft/ReportType;Ljava/util/List;)Z",
        at = @At("HEAD"),
        require = 0
    )
    private void vulkanshaders$dumpHyaloclastCrashLog(
        Path path,
        ReportType type,
        List<String> extraComments,
        CallbackInfoReturnable<Boolean> cir
    ) {
        if (StandaloneRenderPearlBridge.vulkanTurboLoaded) {
            return;
        }
        try {
            CrashReport self = (CrashReport) (Object) this;
            String title = self.getTitle();
            Throwable cause = self.getException();
            String reportDetails = null;
            try {
                reportDetails = self.getDetails();
            } catch (Throwable ignored) {
            }
            String reason = (title != null && !title.isEmpty())
                ? "Minecraft CrashReport: " + title
                : "Minecraft CrashReport";
            HyaloclastErrorLog.dumpCrashLog(reason, reportDetails, cause);
        } catch (Throwable ignored) {
            // Never interfere with vanilla crash report saving.
        }
    }
}
