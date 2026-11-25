package dev.vulkanshaders.config

import dev.vulkanshaders.VulkanShaders
import dev.vulkanshaders.backend.NativeCoreBridge
import dev.vulkanshaders.pack.PackPresetDefinition
import dev.vulkanshaders.pack.PackSettingDefinition
import dev.vulkanshaders.pack.SettingType
import dev.vulkanshaders.pack.ShaderPackManager
import dev.vulkanshaders.pack.ShaderPackManifest
import dev.vulkanshaders.state.WorldStateCollector
import net.minecraft.client.Minecraft
import kotlin.math.abs

/**
 * Pure, testable hardware auto-calibration engine for Hyaloclast.
 *
 * Calibration NEVER runs automatically on startup; it is invoked strictly on explicit user action
 * from the Shader Graphics screen.
 */
object HardwareAutoCalibrator {
    const val VENDOR_NVIDIA: Int = 0x10DE
    const val VENDOR_AMD: Int = 0x1002
    const val VENDOR_INTEL: Int = 0x8086

    private const val REFERENCE_1080P_PIXELS: Double = 1920.0 * 1080.0
    private const val HEAVY_FRAME_DT_MS: Double = 22.0

    data class HardwareProbe(
        val gpuName: String = "",
        val vendorId: Int = 0,
        val arch: String = "unknown",
        val tierClass: String = "low",
        val vramMb: Long = 0L,
        val rtPipeline: Boolean = false,
        val rayQuery: Boolean = false,
        val serActive: Boolean = false,
        val dlssSr: Boolean = false,
        val dlssRr: Boolean = false,
        val dlssFg: Boolean = false,
        val fsr3: Boolean = false,
        val xess: Boolean = false,
        val nrd: Boolean = false,
        val displayWidth: Int = 1920,
        val displayHeight: Int = 1080,
        val refreshRateHz: Int = 60,
        val cpuCores: Int = 4,
        val maxJvmHeapMb: Long = 4096L,
        val lastFrameDtMs: Double = 0.0,
        val averageGpuFrameTimeMs: Double? = null,
        val vramBudgetMb: Long = 0L,
    )

    data class CalibrationResult(
        val config: VulkanShadersConfig,
        val recommendedPresetId: String,
        val summary: String,
        val appliedPresetLabel: String? = null,
        val requiresRecompile: Boolean = false,
    )

    /**
     * Probes the active Vulkan device, vendor SDKs, display geometry/refresh rate, and JVM runtime.
     */
    fun probeCurrentHardware(): HardwareProbe {
        val profile = NativeCoreBridge.getHardwareProfile()
        val vramBudget = NativeCoreBridge.getVramBudgetSnapshot()
        val caps = profile.capabilitiesMask

        var width = 1920
        var height = 1080
        var refreshHz = 60
        runCatching {
            val win = Minecraft.getInstance().window
            if (win != null) {
                if (win.width > 0) width = win.width
                if (win.height > 0) height = win.height
                val probedHz = runCatching {
                    val m = win.javaClass.methods.firstOrNull {
                        (it.name == "getRefreshRate" || it.name == "refreshRate") && it.parameterCount == 0
                    }
                    (m?.invoke(win) as? Number)?.toInt() ?: 60
                }.getOrDefault(60)
                if (probedHz in 30..540) {
                    refreshHz = probedHz
                }
            }
        }

        val cpuCores = Runtime.getRuntime().availableProcessors().coerceAtLeast(1)
        val maxHeapMb = (Runtime.getRuntime().maxMemory() / (1024L * 1024L)).coerceAtLeast(512L)

        return HardwareProbe(
            gpuName = profile.gpuName,
            vendorId = profile.vendorId,
            arch = profile.gpuArch,
            tierClass = profile.gpuTierClass,
            vramMb = profile.vramMb,
            rtPipeline = (caps and NativeCoreBridge.CAP_RAY_TRACING_PIPELINE) != 0 &&
                (caps and NativeCoreBridge.CAP_RAY_TRACING_AS) != 0,
            rayQuery = (caps and NativeCoreBridge.CAP_RAY_QUERY) != 0,
            serActive = profile.serActive ||
                (caps and NativeCoreBridge.CAP_SER_NV) != 0 ||
                (caps and NativeCoreBridge.CAP_SER_EXT) != 0,
            dlssSr = (caps and NativeCoreBridge.CAP_DLSS_SR) != 0,
            dlssRr = (caps and NativeCoreBridge.CAP_DLSS_RR) != 0,
            dlssFg = (caps and NativeCoreBridge.CAP_DLSS_FG) != 0,
            fsr3 = (caps and NativeCoreBridge.CAP_FSR3) != 0,
            xess = (caps and NativeCoreBridge.CAP_XESS) != 0,
            nrd = (caps and NativeCoreBridge.CAP_NRD) != 0,
            displayWidth = width,
            displayHeight = height,
            refreshRateHz = refreshHz,
            cpuCores = cpuCores,
            maxJvmHeapMb = maxHeapMb,
            lastFrameDtMs = profile.lastFrameDtMs,
            averageGpuFrameTimeMs = NativeCoreBridge.getAverageGpuFrameTimeMs(),
            vramBudgetMb = vramBudget?.budgetMb ?: 0L,
        )
    }

    /**
     * Probes current hardware and computes a calibrated configuration for the active pack.
     */
    fun calibrateCurrent(
        currentConfig: VulkanShadersConfig = VulkanShaders.config,
        manifest: ShaderPackManifest? = ShaderPackManager.activePack?.manifest,
    ): CalibrationResult = calibrate(probeCurrentHardware(), currentConfig, manifest)

    /**
     * Pure calibration function computing an optimal [VulkanShadersConfig] and recommended pack
     * preset ID from a [HardwareProbe].
     */
    fun calibrate(
        probe: HardwareProbe,
        currentConfig: VulkanShadersConfig,
        manifest: ShaderPackManifest? = null,
    ): CalibrationResult {
        val tierClass = probe.tierClass.trim().lowercase()
        val width = probe.displayWidth.coerceAtLeast(640)
        val height = probe.displayHeight.coerceAtLeast(360)
        val pixelLoad = (width.toDouble() * height.toDouble()) / REFERENCE_1080P_PIXELS
        val is4kOrHigher = pixelLoad >= 3.5
        val isAbove1440p = pixelLoad > 1.85
        val usableVramMb = when {
            probe.vramMb > 0L && probe.vramBudgetMb > 0L -> minOf(probe.vramMb, probe.vramBudgetMb)
            probe.vramBudgetMb > 0L -> probe.vramBudgetMb
            else -> probe.vramMb
        }

        val bestUpscaler = selectBestUpscaler(probe)
        val bestRtDenoiser = selectBestRtDenoiser(probe)
        val isNvidia = probe.vendorId == VENDOR_NVIDIA ||
            probe.arch.startsWith("nvidia", ignoreCase = true) ||
            probe.dlssSr

        var chosenTier: LightingMode
        var presetId: String
        var upscaler = UpscalerMode.NONE
        var upscalerQuality = UpscalerQuality.NATIVE_AA
        var denoiser: DenoiserMode
        var frameGen: FrameGenMode
        var shadowRes: Int
        var shadowDist: Int

        when {
            // Enthusiast + RT Pipeline (e.g., RTX 4080/4090/5080/5090, >=16 GB VRAM)
            tierClass == "enthusiast" && probe.rtPipeline && usableVramMb >= 12_288L -> {
                chosenTier = LightingMode.PATH
                upscaler = bestUpscaler
                denoiser = bestRtDenoiser
                val hasDlssUpscale = probe.dlssSr || probe.dlssRr
                presetId = if (is4kOrHigher && !hasDlssUpscale) "high" else "ultra"
                upscalerQuality = if (is4kOrHigher && !hasDlssUpscale) {
                    UpscalerQuality.BALANCED
                } else {
                    UpscalerQuality.QUALITY
                }
                frameGen = if (probe.dlssFg && (probe.refreshRateHz >= 120 || is4kOrHigher)) {
                    FrameGenMode.DLSS_FG
                } else {
                    FrameGenMode.NONE
                }
                shadowRes = 4096
                shadowDist = 32
            }

            // High + RT Pipeline (e.g., RTX 4070/5070/3080/3090, 12 GB+)
            tierClass == "high" && probe.rtPipeline && usableVramMb >= 8_192L -> {
                val nonNvidiaNoHardwareDenoiser = !isNvidia && !probe.nrd && !probe.dlssRr
                if (nonNvidiaNoHardwareDenoiser && isAbove1440p) {
                    chosenTier = LightingMode.HYBRID
                    presetId = "ultra"
                    denoiser = DenoiserMode.SVGF
                } else {
                    chosenTier = LightingMode.PATH
                    presetId = "high"
                    upscaler = bestUpscaler
                    upscalerQuality = if (is4kOrHigher) {
                        UpscalerQuality.BALANCED
                    } else {
                        UpscalerQuality.QUALITY
                    }
                    denoiser = bestRtDenoiser
                }
                frameGen = if (probe.dlssFg && (probe.refreshRateHz >= 120 || is4kOrHigher)) {
                    FrameGenMode.DLSS_FG
                } else {
                    FrameGenMode.NONE
                }
                shadowRes = 4096
                shadowDist = 24
            }

            // Mid (or High/Enthusiast without RT Pipeline) + RayQuery (e.g., RTX 2060/3060/4060, RX 6700/7600, 8 GB)
            probe.rayQuery && tierClass != "low" && usableVramMb >= 4_096L -> {
                chosenTier = LightingMode.HYBRID
                presetId = if (is4kOrHigher) "low" else "medium"
                denoiser = DenoiserMode.SVGF
                frameGen = FrameGenMode.NONE
                shadowRes = 2048
                shadowDist = 16
            }

            // Mid without RT or Low discrete with >= 4 GB VRAM (6-8 GB class)
            tierClass == "mid" || (tierClass == "low" && usableVramMb >= 4096L) -> {
                chosenTier = LightingMode.COMPUTE
                presetId = if (tierClass == "mid" && !is4kOrHigher) "high" else if (usableVramMb in 1L..6143L || isAbove1440p) "medium" else "high"
                denoiser = DenoiserMode.SVGF
                frameGen = FrameGenMode.NONE
                shadowRes = 2048
                shadowDist = 12
            }

            // Integrated / Very Low VRAM (< 4 GB)
            else -> {
                chosenTier = LightingMode.CLASSIC
                presetId = "low"
                denoiser = DenoiserMode.NONE
                frameGen = FrameGenMode.NONE
                shadowRes = 1024
                shadowDist = 8
            }
        }

        // Clamp tier to what the active shader pack declares in capabilities.tiers.
        if (manifest != null && manifest.supportedTiers.isNotEmpty() && chosenTier.tierIndex !in manifest.supportedTiers) {
            val fallbackIdx = manifest.supportedTiers
                .filter { it <= chosenTier.tierIndex }
                .maxOrNull()
                ?: manifest.supportedTiers.min()
            chosenTier = LightingMode.entries.firstOrNull { it.tierIndex == fallbackIdx } ?: chosenTier
            when (chosenTier) {
                LightingMode.CLASSIC -> denoiser = DenoiserMode.NONE
                LightingMode.COMPUTE -> if (denoiser == DenoiserMode.DLSS_RR || denoiser == DenoiserMode.NRD) {
                    denoiser = DenoiserMode.SVGF
                }
                else -> {}
            }
        }

        // Incorporate live frame time telemetry when available: if the current tier is already
        // exceeding 22 ms/frame (~45 FPS), step upscaler quality down one notch (or preset down).
        val measuredFrameMs = probe.averageGpuFrameTimeMs?.takeIf { it > 0.0 } ?: probe.lastFrameDtMs
        if (measuredFrameMs > HEAVY_FRAME_DT_MS && chosenTier == currentConfig.renderTier) {
            if (chosenTier != LightingMode.PATH) {
                presetId = stepDownPreset(presetId)
            } else if (upscaler == UpscalerMode.NONE) {
                upscaler = bestUpscaler
                upscalerQuality = UpscalerQuality.QUALITY
            } else {
                val steppedQuality = stepDownUpscalerQuality(upscalerQuality)
                if (steppedQuality != upscalerQuality) {
                    upscalerQuality = steppedQuality
                } else {
                    presetId = stepDownPreset(presetId)
                }
            }
        }

        if (chosenTier == LightingMode.CLASSIC && !isAbove1440p) {
            upscaler = UpscalerMode.NONE
            upscalerQuality = UpscalerQuality.NATIVE_AA
        } else if (chosenTier == LightingMode.COMPUTE && (isAbove1440p || usableVramMb in 1L..6143L)) {
            if (upscaler == UpscalerMode.NONE && bestUpscaler != UpscalerMode.NONE) {
                upscaler = bestUpscaler
                upscalerQuality = UpscalerQuality.QUALITY
            }
        }

        // Ensure shadow resolution stays within hardware bounds.
        val maxShadowRes = if (tierClass == "enthusiast" || tierClass == "high") 8192 else 4096
        shadowRes = shadowRes.coerceIn(512, maxShadowRes)

        var updatedConfig = currentConfig.copy(
            renderTier = chosenTier,
            shadowResolution = shadowRes,
            shadowDistanceChunks = shadowDist,
            upscalerMode = upscaler,
            upscalerQuality = upscalerQuality,
            denoiserMode = denoiser,
            frameGenMode = frameGen,
            enableOverlayPass = if (chosenTier == LightingMode.PATH) true else currentConfig.enableOverlayPass,
            autoCalibrated = true,
        )

        var appliedPresetLabel: String? = null
        var packSettingsRequireRecompile = false

        if (manifest != null) {
            val matchedPreset = findMatchingPreset(manifest, chosenTier.tierIndex, presetId)
            if (matchedPreset != null) {
                appliedPresetLabel = matchedPreset.label
                val tierSettings = manifest.settings.filter { it.appliesToTier(chosenTier.tierIndex) }
                val oldOverrides = currentConfig.activePackOverrides(manifest.id, chosenTier)
                val newOverrides = LinkedHashMap<String, String>()

                for (setting in tierSettings) {
                    val presetRaw = matchedPreset.values[setting.key] ?: continue
                    if (valuesDiffer(setting, presetRaw, setting.defaultValue)) {
                        newOverrides[setting.key] = presetRaw
                    }
                }

                packSettingsRequireRecompile = tierSettings.any { setting ->
                    if (setting.runtime) return@any false
                    val oldVal = oldOverrides[setting.key] ?: setting.defaultValue
                    val newVal = newOverrides[setting.key] ?: setting.defaultValue
                    valuesDiffer(setting, oldVal, newVal)
                }

                updatedConfig = updatedConfig.withPackOverrides(manifest.id, chosenTier, newOverrides)
            }
        }

        val requiresRecompile = currentConfig.renderTier != chosenTier || packSettingsRequireRecompile
        val summary = buildSummary(
            tier = chosenTier,
            presetLabel = appliedPresetLabel ?: presetId.replaceFirstChar { it.uppercaseChar() },
            upscaler = upscaler,
            upscalerQuality = upscalerQuality,
            denoiser = denoiser,
            frameGen = frameGen,
        )

        return CalibrationResult(
            config = updatedConfig,
            recommendedPresetId = presetId,
            summary = summary,
            appliedPresetLabel = appliedPresetLabel,
            requiresRecompile = requiresRecompile,
        )
    }

    private fun selectBestUpscaler(probe: HardwareProbe): UpscalerMode = when (probe.vendorId) {
        VENDOR_NVIDIA -> when {
            probe.dlssSr -> UpscalerMode.DLSS
            probe.fsr3 -> UpscalerMode.FSR3
            probe.xess -> UpscalerMode.XESS
            else -> UpscalerMode.TSR
        }
        VENDOR_AMD -> when {
            probe.fsr3 -> UpscalerMode.FSR3
            probe.xess -> UpscalerMode.XESS
            else -> UpscalerMode.TSR
        }
        VENDOR_INTEL -> when {
            probe.xess -> UpscalerMode.XESS
            probe.fsr3 -> UpscalerMode.FSR3
            else -> UpscalerMode.TSR
        }
        else -> when {
            probe.dlssSr -> UpscalerMode.DLSS
            probe.fsr3 -> UpscalerMode.FSR3
            probe.xess -> UpscalerMode.XESS
            else -> UpscalerMode.TSR
        }
    }

    private fun selectBestRtDenoiser(probe: HardwareProbe): DenoiserMode = when {
        probe.dlssRr -> DenoiserMode.DLSS_RR
        probe.nrd -> DenoiserMode.NRD
        else -> DenoiserMode.SVGF
    }

    private fun stepDownUpscalerQuality(current: UpscalerQuality): UpscalerQuality = when (current) {
        UpscalerQuality.NATIVE_AA -> UpscalerQuality.QUALITY
        UpscalerQuality.QUALITY -> UpscalerQuality.BALANCED
        UpscalerQuality.BALANCED -> UpscalerQuality.PERFORMANCE
        UpscalerQuality.PERFORMANCE -> UpscalerQuality.PERFORMANCE
    }

    private fun stepDownPreset(current: String): String = when (current.lowercase()) {
        "ultra" -> "high"
        "high" -> "medium"
        "medium" -> "low"
        else -> "low"
    }

    private fun findMatchingPreset(
        manifest: ShaderPackManifest,
        tierIndex: Int,
        recommendedId: String,
    ): PackPresetDefinition? {
        val eligible = manifest.presets.filter { it.appliesToTier(tierIndex) }
        if (eligible.isEmpty()) return null
        eligible.firstOrNull { it.id.equals(recommendedId, ignoreCase = true) }?.let { return it }
        // Fallback chain if the pack uses a subset of standard preset IDs.
        val fallbackOrder = when (recommendedId.lowercase()) {
            "ultra" -> listOf("high", "medium", "low")
            "high" -> listOf("medium", "ultra", "low")
            "medium" -> listOf("low", "high", "ultra")
            else -> listOf("medium", "high", "ultra")
        }
        for (candidate in fallbackOrder) {
            eligible.firstOrNull { it.id.equals(candidate, ignoreCase = true) }?.let { return it }
        }
        return eligible.first()
    }

    private fun valuesDiffer(setting: PackSettingDefinition, current: String, defaultVal: String): Boolean =
        when (setting.type) {
            SettingType.BOOL -> {
                val c = current.equals("true", ignoreCase = true) || current == "1"
                val d = defaultVal.equals("true", ignoreCase = true) || defaultVal == "1"
                c != d
            }
            SettingType.INT -> {
                val c = current.toIntOrNull()
                val d = defaultVal.toIntOrNull()
                if (c != null && d != null) c != d else current != defaultVal
            }
            SettingType.FLOAT -> {
                val c = current.toDoubleOrNull()
                val d = defaultVal.toDoubleOrNull()
                if (c != null && d != null) abs(c - d) > 1e-5 else current != defaultVal
            }
            SettingType.ENUM -> !current.equals(defaultVal, ignoreCase = true)
            SettingType.COLOR_RGB -> {
                val c = WorldStateCollector.parseColorRgb(current)
                val d = WorldStateCollector.parseColorRgb(defaultVal)
                if (c != null && d != null) c != d else !current.equals(defaultVal, ignoreCase = true)
            }
        }

    private fun buildSummary(
        tier: LightingMode,
        presetLabel: String,
        upscaler: UpscalerMode,
        upscalerQuality: UpscalerQuality,
        denoiser: DenoiserMode,
        frameGen: FrameGenMode,
    ): String {
        val tierLabel = when (tier) {
            LightingMode.CLASSIC -> "Tier 0"
            LightingMode.COMPUTE -> "Tier 1"
            LightingMode.HYBRID -> "Tier 2"
            LightingMode.PATH -> "Tier 3"
        }
        val parts = mutableListOf(tierLabel, presetLabel)
        if (upscaler != UpscalerMode.NONE) {
            parts.add("${upscaler.name} ${upscalerQuality.name.lowercase().replace('_', ' ').replaceFirstChar { it.uppercaseChar() }}")
        }
        if (denoiser != DenoiserMode.NONE) {
            parts.add(denoiser.name)
        }
        if (frameGen != FrameGenMode.NONE) {
            parts.add(frameGen.name)
        }
        return parts.joinToString(" • ")
    }
}
