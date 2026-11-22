package dev.vulkanshaders.config

enum class RenderTier(val tierIndex: Int, val translationKey: String) {
    TIER_0_RASTER(0, "vulkanshaders.screen.tier.tier_0_raster"),
    TIER_1_COMPUTE_GI(1, "vulkanshaders.screen.tier.tier_1_compute_gi"),
    TIER_2_RAY_QUERY(2, "vulkanshaders.screen.tier.tier_2_ray_query"),
    TIER_3_PATH_TRACING(3, "vulkanshaders.screen.tier.tier_3_path_tracing"),
}

enum class UpscalerMode {
    NONE,
    TSR,
    FSR3,
    DLSS,
    XESS,
}

enum class UpscalerQuality(val renderScale: Float, val displayName: String, val translationKey: String) {
    NATIVE_AA(1.00f, "Native (100%)", "vulkanshaders.screen.upscaler_quality.native_aa"),
    QUALITY(0.67f, "Quality (67%)", "vulkanshaders.screen.upscaler_quality.quality"),
    BALANCED(0.58f, "Balanced (58%)", "vulkanshaders.screen.upscaler_quality.balanced"),
    PERFORMANCE(0.50f, "Performance (50%)", "vulkanshaders.screen.upscaler_quality.performance"),
}

enum class DenoiserMode {
    NONE,
    SVGF,
    NRD,
    DLSS_RR,
}

enum class FrameGenMode {
    NONE,
    OPTICAL_FLOW,
    FSR3_FG,
    DLSS_FG,
}

data class VulkanShadersConfig(
    val enabled: Boolean = true,
    val selectedPackId: String = "hyaloclast-sunlit-vanilla",
    val renderTier: RenderTier = RenderTier.TIER_1_COMPUTE_GI,
    val shadowResolution: Int = 2048,
    val shadowDistanceChunks: Int = 12,
    val upscalerMode: UpscalerMode = UpscalerMode.NONE,
    val upscalerQuality: UpscalerQuality = UpscalerQuality.QUALITY,
    val denoiserMode: DenoiserMode = DenoiserMode.SVGF,
    val frameGenMode: FrameGenMode = FrameGenMode.NONE,
    val hdrEnabled: Boolean = false,
    val hdrPeakBrightnessNits: Float = 1000f,
    val hdrPaperWhiteNits: Float = 200f,
    val enableOverlayPass: Boolean = true,
    val showF3DebugInfo: Boolean = true,
    val enableGpuProfiler: Boolean = false,
    val packOverrides: Map<String, Map<String, String>> = emptyMap(),
    val packOverridesByTier: Map<String, Map<Int, Map<String, String>>> = emptyMap(),
    val autoCalibrated: Boolean = false,
) {
    val effectiveUpscalerQuality: UpscalerQuality
        get() = if (renderTier == RenderTier.TIER_3_PATH_TRACING) upscalerQuality else UpscalerQuality.NATIVE_AA

    fun activePackOverrides(packId: String, tier: RenderTier = renderTier): Map<String, String> {
        val normalizedId = packId.trim()
        if (normalizedId.isEmpty()) return emptyMap()
        val byTier = packOverridesByTier[normalizedId]
        if (byTier != null) {
            return byTier[tier.tierIndex].orEmpty()
        }
        return packOverrides[normalizedId].orEmpty()
    }

    fun withPackOverrides(
        packId: String,
        tier: RenderTier,
        overrides: Map<String, String>,
    ): VulkanShadersConfig {
        val normalizedId = packId.trim()
        if (normalizedId.isEmpty()) return this
        val updatedAll = LinkedHashMap<String, Map<Int, Map<String, String>>>(packOverridesByTier)
        val perPack = LinkedHashMap<Int, Map<String, String>>(updatedAll[normalizedId].orEmpty())
        val cleaned = LinkedHashMap<String, String>()
        for ((key, value) in overrides) {
            if (key.isNotBlank()) {
                cleaned[key] = value
            }
        }
        if (cleaned.isEmpty()) {
            perPack.remove(tier.tierIndex)
        } else {
            perPack[tier.tierIndex] = cleaned
        }
        if (perPack.isEmpty()) {
            updatedAll.remove(normalizedId)
        } else {
            updatedAll[normalizedId] = perPack
        }
        val cleanedLegacy = if (normalizedId in packOverrides) {
            packOverrides - normalizedId
        } else {
            packOverrides
        }
        return copy(
            packOverrides = cleanedLegacy,
            packOverridesByTier = updatedAll,
        )
    }

    /**
     * Clears all stored option overrides for [packId] across every render tier.
     */
    fun resetAllPackOverrides(packId: String): VulkanShadersConfig {
        val normalizedId = packId.trim()
        if (normalizedId.isEmpty()) return this
        val updatedAll = LinkedHashMap<String, Map<Int, Map<String, String>>>(packOverridesByTier)
        updatedAll.remove(normalizedId)
        val cleanedLegacy = if (normalizedId in packOverrides) {
            packOverrides - normalizedId
        } else {
            packOverrides
        }
        return copy(
            packOverrides = cleanedLegacy,
            packOverridesByTier = updatedAll,
        )
    }
}
