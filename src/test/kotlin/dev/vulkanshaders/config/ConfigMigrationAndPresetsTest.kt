package dev.vulkanshaders.config

import com.google.gson.GsonBuilder
import dev.vulkanshaders.pack.ShaderPackManifest
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test

class ConfigMigrationAndPresetsTest {

    private val gson = GsonBuilder().setPrettyPrinting().create()

    @Test
    fun `round-trips VulkanShadersConfig through Gson JSON without losing per-tier overrides`() {
        val original = VulkanShadersConfig(
            enabled = true,
            selectedPackId = "hyaloclast-obsidian-showcase",
            renderTier = LightingMode.PATH,
            shadowResolution = 4096,
            shadowDistanceChunks = 24,
            upscalerMode = UpscalerMode.DLSS,
            upscalerQuality = UpscalerQuality.QUALITY,
            denoiserMode = DenoiserMode.DLSS_RR,
            frameGenMode = FrameGenMode.DLSS_FG,
            hdrEnabled = true,
            hdrPeakBrightnessNits = 1200f,
            hdrPaperWhiteNits = 250f,
            enableOverlayPass = true,
            enableSer = false,
            packOverridesByTier = mapOf(
                "hyaloclast-obsidian-showcase" to mapOf(
                    LightingMode.HYBRID.tierIndex to mapOf("sun_intensity" to "3.1"),
                    LightingMode.PATH.tierIndex to mapOf("sun_intensity" to "3.6", "pt_max_bounces" to "3"),
                ),
            ),
            autoCalibrated = true,
        )

        val json = gson.toJson(original)
        val deserialized = ConfigStore.sanitize(gson.fromJson(json, VulkanShadersConfig::class.java))

        assertEquals(original, deserialized)
        assertEquals(UpscalerQuality.QUALITY, deserialized.effectiveUpscalerQuality)
        assertEquals(
            mapOf("sun_intensity" to "3.6", "pt_max_bounces" to "3"),
            deserialized.activePackOverrides("hyaloclast-obsidian-showcase", LightingMode.PATH),
        )
    }

    @Test
    fun `sanitizes legacy pack IDs, out-of-range numeric bounds, and non-finite HDR values`() {
        val malformed = VulkanShadersConfig(
            selectedPackId = "hyaloclast-default",
            renderTier = LightingMode.COMPUTE,
            shadowResolution = 128,
            shadowDistanceChunks = 256,
            hdrPeakBrightnessNits = Float.NaN,
            hdrPaperWhiteNits = 9999f,
            packOverrides = mapOf(
                "hyaloclast-sunlit-vanilla" to mapOf("sun_intensity" to "2.9", "   " to "ignored"),
            ),
        )

        val sanitized = ConfigStore.sanitize(malformed)
        val defaults = VulkanShadersConfig()

        assertEquals(defaults.selectedPackId, sanitized.selectedPackId)
        assertEquals(512, sanitized.shadowResolution)
        assertEquals(64, sanitized.shadowDistanceChunks)
        assertEquals(defaults.hdrPeakBrightnessNits, sanitized.hdrPeakBrightnessNits)
        assertEquals(1000f, sanitized.hdrPaperWhiteNits)
        assertTrue(sanitized.packOverrides.isEmpty())
        assertEquals(
            mapOf("sun_intensity" to "2.9"),
            sanitized.activePackOverrides("hyaloclast-sunlit-vanilla", LightingMode.COMPUTE),
        )
        assertEquals(UpscalerQuality.NATIVE_AA, sanitized.effectiveUpscalerQuality)
    }

    @Test
    fun `applies per-pack preset overrides independently across tiers and resets cleanly`() {
        val toml = """
            [pack]
            id = "preset-test-pack"
            name = "Preset Test Pack"

            [capabilities]
            tiers = [0, 1, 2, 3]

            [settings.sun_intensity]
            label = "Sun Intensity"
            type = "float"
            default = 2.5
            min = 0.5
            max = 8.0
            runtime = true

            [settings.enable_wind]
            label = "Foliage Wind"
            type = "bool"
            default = true
            runtime = false

            [presets.low]
            label = "Low"
            order = 0
            values = { sun_intensity = 2.0, enable_wind = false }

            [presets.ultra]
            label = "Ultra"
            order = 1
            values = { sun_intensity = 3.8, enable_wind = true }
        """.trimIndent()

        val manifest = ShaderPackManifest.fromToml(toml, "preset-test-pack")
        val lowPreset = manifest.presets.first { it.id == "low" }
        val ultraPreset = manifest.presets.first { it.id == "ultra" }

        var cfg = VulkanShadersConfig(selectedPackId = "preset-test-pack")
        cfg = cfg.withPackOverrides("preset-test-pack", LightingMode.CLASSIC, lowPreset.values)
        cfg = cfg.withPackOverrides("preset-test-pack", LightingMode.PATH, ultraPreset.values)

        assertEquals(
            mapOf("sun_intensity" to "2.0", "enable_wind" to "false"),
            cfg.activePackOverrides("preset-test-pack", LightingMode.CLASSIC),
        )
        assertEquals(
            mapOf("sun_intensity" to "3.8", "enable_wind" to "true"),
            cfg.activePackOverrides("preset-test-pack", LightingMode.PATH),
        )
        assertTrue(cfg.activePackOverrides("preset-test-pack", LightingMode.COMPUTE).isEmpty())

        // Resetting Tier 0 overrides must preserve Tier 3 overrides
        val resetTier0 = cfg.withPackOverrides("preset-test-pack", LightingMode.CLASSIC, emptyMap())
        assertTrue(resetTier0.activePackOverrides("preset-test-pack", LightingMode.CLASSIC).isEmpty())
        assertEquals(
            mapOf("sun_intensity" to "3.8", "enable_wind" to "true"),
            resetTier0.activePackOverrides("preset-test-pack", LightingMode.PATH),
        )
    }
}
