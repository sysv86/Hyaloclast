package dev.vulkanshaders.config

import dev.vulkanshaders.pack.PackPresetDefinition
import dev.vulkanshaders.pack.PackSettingDefinition
import dev.vulkanshaders.pack.SettingType
import dev.vulkanshaders.pack.ShaderPackManifest
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertFalse
import org.junit.jupiter.api.Assertions.assertNotEquals
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test

/** Unit tests for [HardwareAutoCalibrator] across GPU tiers, resolutions, and live frame timing. */
class HardwareAutoCalibratorTest {
    private fun sampleManifest(): ShaderPackManifest = ShaderPackManifest(
        id = "test-pack",
        name = "Test Pack",
        version = "1.0.0",
        author = "Test",
        description = "Test manifest",
        supportedTiers = listOf(0, 1, 2, 3),
        settings = listOf(
            PackSettingDefinition(
                key = "sun_intensity",
                label = "Sun Intensity",
                type = SettingType.FLOAT,
                defaultValue = "2.8",
                min = 0.5,
                max = 5.0,
                runtime = true,
            ),
            PackSettingDefinition(
                key = "pt_bounces",
                label = "PT Bounces",
                type = SettingType.INT,
                defaultValue = "2",
                min = 1.0,
                max = 4.0,
                runtime = false,
                minTier = 3,
                maxTier = 3,
            ),
            PackSettingDefinition(
                key = "foliage_wind",
                label = "Foliage Wind",
                type = SettingType.BOOL,
                defaultValue = "true",
                runtime = true,
            ),
        ),
        presets = listOf(
            PackPresetDefinition(
                id = "low",
                label = "Low",
                order = 0,
                values = mapOf(
                    "sun_intensity" to "2.6",
                    "pt_bounces" to "1",
                    "foliage_wind" to "false",
                ),
            ),
            PackPresetDefinition(
                id = "medium",
                label = "Medium",
                order = 1,
                values = mapOf(
                    "sun_intensity" to "2.8",
                    "pt_bounces" to "1",
                    "foliage_wind" to "true",
                ),
            ),
            PackPresetDefinition(
                id = "high",
                label = "High",
                order = 2,
                values = mapOf(
                    "sun_intensity" to "2.8",
                    "pt_bounces" to "2",
                    "foliage_wind" to "true",
                ),
            ),
            PackPresetDefinition(
                id = "ultra",
                label = "Ultra",
                order = 3,
                values = mapOf(
                    "sun_intensity" to "3.0",
                    "pt_bounces" to "4",
                    "foliage_wind" to "true",
                ),
            ),
        ),
        resources = emptyList(),
        hooks = emptyMap(),
    )

    @Test
    fun `default config has autoCalibrated set to false`() {
        val defaultConfig = VulkanShadersConfig()
        assertFalse(defaultConfig.autoCalibrated, "Auto-calibration must never be marked active by default")
    }

    @Test
    fun `rtx 5090 enthusiast with SER and 144Hz selects Tier 3 Ultra DLSS-RR and DLSS-FG`() {
        val probe = HardwareAutoCalibrator.HardwareProbe(
            gpuName = "NVIDIA GeForce RTX 5090",
            vendorId = HardwareAutoCalibrator.VENDOR_NVIDIA,
            arch = "nvidia_blackwell",
            tierClass = "enthusiast",
            vramMb = 32768L,
            rtPipeline = true,
            rayQuery = true,
            serActive = true,
            dlssSr = true,
            dlssRr = true,
            dlssFg = true,
            nrd = true,
            displayWidth = 2560,
            displayHeight = 1440,
            refreshRateHz = 144,
            cpuCores = 16,
            maxJvmHeapMb = 8192L,
        )
        val manifest = sampleManifest()
        val result = HardwareAutoCalibrator.calibrate(probe, VulkanShadersConfig(), manifest)

        assertEquals(LightingMode.PATH, result.config.renderTier)
        assertEquals("ultra", result.recommendedPresetId)
        assertEquals(UpscalerMode.DLSS, result.config.upscalerMode)
        assertEquals(UpscalerQuality.QUALITY, result.config.upscalerQuality)
        assertEquals(DenoiserMode.DLSS_RR, result.config.denoiserMode)
        assertEquals(FrameGenMode.DLSS_FG, result.config.frameGenMode)
        assertEquals(4096, result.config.shadowResolution)
        assertEquals(32, result.config.shadowDistanceChunks)
        assertTrue(result.config.autoCalibrated)

        // Only values differing from defaults should be persisted in pack overrides
        val overrides = result.config.activePackOverrides(manifest.id, LightingMode.PATH)
        assertEquals("3.0", overrides["sun_intensity"])
        assertEquals("4", overrides["pt_bounces"])
        assertFalse("foliage_wind" in overrides, "Default-matching values must not be stored as overrides")
        assertTrue(result.requiresRecompile)
    }

    @Test
    fun `rtx 4090 enthusiast at 4K without DLSS drops preset to high and uses FSR3 Balanced`() {
        val probe = HardwareAutoCalibrator.HardwareProbe(
            gpuName = "NVIDIA GeForce RTX 4090",
            vendorId = HardwareAutoCalibrator.VENDOR_NVIDIA,
            arch = "nvidia_ada",
            tierClass = "enthusiast",
            vramMb = 24576L,
            rtPipeline = true,
            rayQuery = true,
            serActive = true,
            dlssSr = false,
            dlssRr = false,
            dlssFg = false,
            fsr3 = true,
            nrd = true,
            displayWidth = 3840,
            displayHeight = 2160,
            refreshRateHz = 60,
        )
        val result = HardwareAutoCalibrator.calibrate(probe, VulkanShadersConfig(), sampleManifest())

        assertEquals(LightingMode.PATH, result.config.renderTier)
        assertEquals("high", result.recommendedPresetId)
        assertEquals(UpscalerMode.FSR3, result.config.upscalerMode)
        assertEquals(UpscalerQuality.BALANCED, result.config.upscalerQuality)
        assertEquals(DenoiserMode.NRD, result.config.denoiserMode)
        assertEquals(FrameGenMode.NONE, result.config.frameGenMode)
        assertTrue(result.config.autoCalibrated)
    }

    @Test
    fun `rtx 4070 high tier selects Tier 3 High with DLSS Quality and DLSS-RR`() {
        val probe = HardwareAutoCalibrator.HardwareProbe(
            gpuName = "NVIDIA GeForce RTX 4070",
            vendorId = HardwareAutoCalibrator.VENDOR_NVIDIA,
            arch = "nvidia_ada",
            tierClass = "high",
            vramMb = 12288L,
            rtPipeline = true,
            rayQuery = true,
            serActive = true,
            dlssSr = true,
            dlssRr = true,
            dlssFg = true,
            displayWidth = 2560,
            displayHeight = 1440,
            refreshRateHz = 60,
        )
        val result = HardwareAutoCalibrator.calibrate(probe, VulkanShadersConfig(), sampleManifest())

        assertEquals(LightingMode.PATH, result.config.renderTier)
        assertEquals("high", result.recommendedPresetId)
        assertEquals(UpscalerMode.DLSS, result.config.upscalerMode)
        assertEquals(UpscalerQuality.QUALITY, result.config.upscalerQuality)
        assertEquals(DenoiserMode.DLSS_RR, result.config.denoiserMode)
        assertEquals(FrameGenMode.NONE, result.config.frameGenMode)
        assertTrue(result.config.autoCalibrated)
    }

    @Test
    fun `rtx 3060 mid RT selects Tier 2 RayQuery Medium with DLSS Quality and SVGF`() {
        val probe = HardwareAutoCalibrator.HardwareProbe(
            gpuName = "NVIDIA GeForce RTX 3060",
            vendorId = HardwareAutoCalibrator.VENDOR_NVIDIA,
            arch = "nvidia_ampere",
            tierClass = "mid",
            vramMb = 8192L,
            rtPipeline = true,
            rayQuery = true,
            dlssSr = true,
            displayWidth = 1920,
            displayHeight = 1080,
            refreshRateHz = 60,
        )
        val manifest = sampleManifest()
        val result = HardwareAutoCalibrator.calibrate(probe, VulkanShadersConfig(), manifest)

        assertEquals(LightingMode.HYBRID, result.config.renderTier)
        assertEquals("medium", result.recommendedPresetId)
        assertEquals(UpscalerMode.NONE, result.config.upscalerMode)
        assertEquals(UpscalerQuality.NATIVE_AA, result.config.upscalerQuality)
        assertEquals(DenoiserMode.SVGF, result.config.denoiserMode)
        assertEquals(2048, result.config.shadowResolution)
        assertEquals(16, result.config.shadowDistanceChunks)
        assertTrue(result.config.autoCalibrated)

        // Tier-3-only setting `pt_bounces` must not be written for Tier 2
        val overrides = result.config.activePackOverrides(manifest.id, LightingMode.HYBRID)
        assertFalse("pt_bounces" in overrides, "Settings outside active tier range must not be stored")
    }

    @Test
    fun `gtx 1660 or rx 6600 without RT selects Tier 1 Compute GI`() {
        val gtx1660 = HardwareAutoCalibrator.HardwareProbe(
            gpuName = "NVIDIA GeForce GTX 1660",
            vendorId = HardwareAutoCalibrator.VENDOR_NVIDIA,
            arch = "nvidia_turing",
            tierClass = "mid",
            vramMb = 6144L,
            rtPipeline = false,
            rayQuery = false,
            displayWidth = 1920,
            displayHeight = 1080,
            refreshRateHz = 60,
        )
        val resGtx = HardwareAutoCalibrator.calibrate(gtx1660, VulkanShadersConfig(), sampleManifest())
        assertEquals(LightingMode.COMPUTE, resGtx.config.renderTier)
        assertEquals("medium", resGtx.recommendedPresetId)
        assertEquals(2048, resGtx.config.shadowResolution)
        assertEquals(12, resGtx.config.shadowDistanceChunks)
        assertTrue(resGtx.config.autoCalibrated)

        val rx6600 = HardwareAutoCalibrator.HardwareProbe(
            gpuName = "AMD Radeon RX 6600",
            vendorId = HardwareAutoCalibrator.VENDOR_AMD,
            arch = "amd_rdna2",
            tierClass = "mid",
            vramMb = 8192L,
            rtPipeline = false,
            rayQuery = false,
            fsr3 = true,
            displayWidth = 2560,
            displayHeight = 1440,
            refreshRateHz = 144,
        )
        val resRx = HardwareAutoCalibrator.calibrate(rx6600, VulkanShadersConfig(), sampleManifest())
        assertEquals(LightingMode.COMPUTE, resRx.config.renderTier)
        assertEquals("medium", resRx.recommendedPresetId)
        assertEquals(UpscalerMode.NONE, resRx.config.upscalerMode)
    }

    @Test
    fun `integrated low VRAM GPU selects Tier 0 Raster Low`() {
        val igpu = HardwareAutoCalibrator.HardwareProbe(
            gpuName = "Intel UHD Graphics 770",
            vendorId = HardwareAutoCalibrator.VENDOR_INTEL,
            arch = "unknown",
            tierClass = "low",
            vramMb = 2048L,
            rtPipeline = false,
            rayQuery = false,
            displayWidth = 1920,
            displayHeight = 1080,
            refreshRateHz = 60,
        )
        val result = HardwareAutoCalibrator.calibrate(igpu, VulkanShadersConfig(), sampleManifest())

        assertEquals(LightingMode.CLASSIC, result.config.renderTier)
        assertEquals("low", result.recommendedPresetId)
        assertEquals(DenoiserMode.NONE, result.config.denoiserMode)
        assertEquals(FrameGenMode.NONE, result.config.frameGenMode)
        assertEquals(1024, result.config.shadowResolution)
        assertEquals(8, result.config.shadowDistanceChunks)
        assertTrue(result.config.autoCalibrated)
    }

    @Test
    fun `4K load scaling and heavy frame time nudge quality down`() {
        val probe4k = HardwareAutoCalibrator.HardwareProbe(
            gpuName = "NVIDIA GeForce RTX 3060",
            vendorId = HardwareAutoCalibrator.VENDOR_NVIDIA,
            arch = "nvidia_ampere",
            tierClass = "mid",
            vramMb = 8192L,
            rtPipeline = true,
            rayQuery = true,
            dlssSr = true,
            displayWidth = 3840,
            displayHeight = 2160,
            refreshRateHz = 60,
        )
        val res4k = HardwareAutoCalibrator.calibrate(probe4k, VulkanShadersConfig(), sampleManifest())
        assertEquals(LightingMode.HYBRID, res4k.config.renderTier)
        assertEquals("low", res4k.recommendedPresetId)
        assertEquals(UpscalerQuality.NATIVE_AA, res4k.config.upscalerQuality)

        // Heavy live frame time (> 22ms) at the active tier nudges upscaler quality down one step
        val probeHeavy = HardwareAutoCalibrator.HardwareProbe(
            gpuName = "NVIDIA GeForce RTX 4070",
            vendorId = HardwareAutoCalibrator.VENDOR_NVIDIA,
            arch = "nvidia_ada",
            tierClass = "high",
            vramMb = 12288L,
            rtPipeline = true,
            rayQuery = true,
            dlssSr = true,
            dlssRr = true,
            displayWidth = 2560,
            displayHeight = 1440,
            refreshRateHz = 60,
            lastFrameDtMs = 28.5,
        )
        val baseCfg = VulkanShadersConfig(renderTier = LightingMode.PATH)
        val resHeavy = HardwareAutoCalibrator.calibrate(probeHeavy, baseCfg, sampleManifest())
        assertEquals(LightingMode.PATH, resHeavy.config.renderTier)
        assertNotEquals(UpscalerQuality.QUALITY, resHeavy.config.upscalerQuality)
        assertEquals(UpscalerQuality.BALANCED, resHeavy.config.upscalerQuality)
    }
}
