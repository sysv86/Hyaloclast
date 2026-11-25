package dev.vulkanshaders.diagnostics

import dev.vulkanshaders.VulkanShaders
import dev.vulkanshaders.backend.NativeCoreBridge
import dev.vulkanshaders.config.FrameGenMode
import dev.vulkanshaders.config.LightingMode
import dev.vulkanshaders.config.UpscalerMode
import dev.vulkanshaders.config.VulkanShadersConfig
import dev.vulkanshaders.pack.ShaderPackManager
import dev.vulkanshaders.pipeline.SectionSliceRegistry
import java.util.Locale

/**
 * Compact F3 debug HUD line builder for Hyaloclast.
 *
 * Displays active shader pack, Render Tier, active upscaler / frame generation mode,
 * BLAS / TLAS instance counts, VRAM usage vs. budget, and an optional per-pass GPU timestamp
 * breakdown when [VulkanShadersConfig.enableGpuProfiler] is enabled.
 */
object HyaloclastDebugHud {
    data class PassTiming(
        val name: String,
        val durationMs: Double,
    )

    data class DebugSnapshot(
        val shadersEnabled: Boolean,
        val activePackName: String?,
        val renderMode: LightingMode,
        val upscalerMode: UpscalerMode,
        val upscalerScalePercent: Int,
        val frameGenMode: FrameGenMode,
        val blasCount: Int,
        val tlasInstanceCount: Int,
        val vramUsedMb: Long,
        val vramBudgetMb: Long,
        val gpuName: String = "",
        val passTimings: List<PassTiming> = emptyList(),
    )

    @Volatile
    private var nativeBlasCount: Int = -1

    @Volatile
    private var nativeTlasInstanceCount: Int = -1

    @Volatile
    private var nativeVramUsedMb: Long = -1L

    @Volatile
    private var nativeVramBudgetMb: Long = -1L

    @Volatile
    private var nativePassTimings: List<PassTiming> = emptyList()

    /**
     * Updates live native RT acceleration structure, VRAM budget, and GPU timestamp metrics
     * pushed from `NativeCoreBridge` in Wave 2.
     */
    @JvmStatic
    @JvmOverloads
    fun updateNativeMetrics(
        blasCount: Int,
        tlasInstanceCount: Int,
        vramUsedMb: Long,
        vramBudgetMb: Long,
        passTimings: List<PassTiming> = emptyList(),
    ) {
        nativeBlasCount = blasCount.coerceAtLeast(0)
        nativeTlasInstanceCount = tlasInstanceCount.coerceAtLeast(0)
        nativeVramUsedMb = vramUsedMb.coerceAtLeast(0L)
        nativeVramBudgetMb = vramBudgetMb.coerceAtLeast(0L)
        nativePassTimings = passTimings
    }

    @JvmStatic
    fun clearNativeMetrics() {
        nativeBlasCount = -1
        nativeTlasInstanceCount = -1
        nativeVramUsedMb = -1L
        nativeVramBudgetMb = -1L
        nativePassTimings = emptyList()
    }

    /**
     * Captures a consistent [DebugSnapshot] from live Hyaloclast runtime state.
     */
    @JvmStatic
    fun captureSnapshot(config: VulkanShadersConfig = VulkanShaders.config): DebugSnapshot {
        val pack = ShaderPackManager.activePack.takeIf { VulkanShaders.shaderPackActive }
        val profile = runCatching { NativeCoreBridge.getHardwareProfile() }
            .getOrElse { NativeCoreBridge.HardwareProfile() }
        val fallbackSlices = runCatching { SectionSliceRegistry.activeCount() }.getOrDefault(0)
        val blas = if (nativeBlasCount >= 0) nativeBlasCount else fallbackSlices
        val tlas = if (nativeTlasInstanceCount >= 0) nativeTlasInstanceCount else fallbackSlices
        val budgetMb = when {
            nativeVramBudgetMb > 0L -> nativeVramBudgetMb
            profile.vramMb > 0L -> profile.vramMb
            else -> 0L
        }
        val usedMb = if (nativeVramUsedMb >= 0L) nativeVramUsedMb else 0L
        val scalePct = (config.effectiveUpscalerQuality.renderScale * 100f).toInt()

        return DebugSnapshot(
            shadersEnabled = VulkanShaders.shaderPackActive,
            activePackName = pack?.manifest?.name,
            renderMode = config.renderTier,
            upscalerMode = config.upscalerMode,
            upscalerScalePercent = scalePct,
            frameGenMode = config.frameGenMode,
            blasCount = blas,
            tlasInstanceCount = tlas,
            vramUsedMb = usedMb,
            vramBudgetMb = budgetMb,
            gpuName = profile.gpuName,
            passTimings = if (config.enableGpuProfiler) nativePassTimings else emptyList(),
        )
    }

    /**
     * Builds the compact F3 debug HUD lines for Hyaloclast.
     * Returns an empty list when [VulkanShadersConfig.showF3DebugInfo] is disabled.
     */
    @JvmStatic
    @JvmOverloads
    fun buildDebugLines(
        snapshot: DebugSnapshot = captureSnapshot(),
        config: VulkanShadersConfig = VulkanShaders.config,
    ): List<String> {
        if (!config.showF3DebugInfo) return emptyList()

        val lines = ArrayList<String>(4)
        val packText = if (snapshot.shadersEnabled && !snapshot.activePackName.isNullOrBlank()) {
            snapshot.activePackName
        } else {
            "Off"
        }
        val tierLabel = formatTierShort(snapshot.renderMode)
        val upscalerLabel = formatUpscaler(snapshot.upscalerMode, snapshot.upscalerScalePercent)
        val fgLabel = formatFrameGen(snapshot.frameGenMode)

        lines.add(
            "[Hyaloclast] Pack: $packText | $tierLabel | Upscaler: $upscalerLabel | FG: $fgLabel",
        )

        val vramText = when {
            snapshot.vramBudgetMb > 0L && snapshot.vramUsedMb > 0L ->
                "${snapshot.vramUsedMb} / ${snapshot.vramBudgetMb} MB"
            snapshot.vramBudgetMb > 0L ->
                "— / ${snapshot.vramBudgetMb} MB"
            else -> "N/A"
        }
        lines.add(
            "[Hyaloclast] BLAS/TLAS: ${snapshot.blasCount}/${snapshot.tlasInstanceCount} | VRAM: $vramText",
        )

        if (config.enableGpuProfiler) {
            if (snapshot.passTimings.isEmpty()) {
                lines.add("[Hyaloclast GPU] Profiler enabled (awaiting timestamps)")
            } else {
                val totalMs = snapshot.passTimings.sumOf { it.durationMs }
                val breakdown = snapshot.passTimings.joinToString(", ") { pass ->
                    "${pass.name}: ${formatMs(pass.durationMs)}"
                }
                lines.add("[Hyaloclast GPU] Total: ${formatMs(totalMs)} ms ($breakdown)")
            }
        }
        return lines
    }

    private fun formatTierShort(tier: LightingMode): String = when (tier) {
        LightingMode.CLASSIC -> "Tier 0 (Raster)"
        LightingMode.COMPUTE -> "Tier 1 (Compute GI)"
        LightingMode.HYBRID -> "Tier 2 (Ray Query)"
        LightingMode.PATH -> "Tier 3 (Path Tracing)"
    }

    private fun formatUpscaler(mode: UpscalerMode, scalePercent: Int): String = when (mode) {
        UpscalerMode.NONE -> "Off"
        UpscalerMode.TSR -> "TSR ($scalePercent%)"
        UpscalerMode.FSR3 -> "FSR 3 ($scalePercent%)"
        UpscalerMode.DLSS -> "DLSS ($scalePercent%)"
        UpscalerMode.XESS -> "XeSS ($scalePercent%)"
    }

    private fun formatFrameGen(mode: FrameGenMode): String = when (mode) {
        FrameGenMode.NONE -> "Off"
        FrameGenMode.OPTICAL_FLOW -> "Optical Flow"
        FrameGenMode.FSR3_FG -> "FSR 3 FG"
        FrameGenMode.DLSS_FG -> "DLSS FG"
    }

    private fun formatMs(ms: Double): String = String.format(Locale.ROOT, "%.2f", ms)
}
