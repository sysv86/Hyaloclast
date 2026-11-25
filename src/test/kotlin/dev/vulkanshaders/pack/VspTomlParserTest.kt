package dev.vulkanshaders.pack

import dev.vulkanshaders.config.ConfigStore
import dev.vulkanshaders.config.RenderTier
import dev.vulkanshaders.config.VulkanShadersConfig
import dev.vulkanshaders.state.WorldStateCollector
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertFalse
import org.junit.jupiter.api.Assertions.assertNotNull
import org.junit.jupiter.api.Assertions.assertThrows
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test
import java.lang.Float.intBitsToFloat

class VspTomlParserTest {
    @Test
    fun `parses hyaloclast-default pack manifest, resolves extends inheritance, and computes specialization overrides`() {
        val baseToml = """
            [pack]
            id = "hyaloclast-default"
            name = "Hyaloclast Default"
            version = "0.2.0"
            author = "VulkanShaders"

            [capabilities]
            tiers = [0, 1, 2, 3]

            [hooks]
            surface = "hooks/surface.vsl"
            sky = "hooks/sky.vsl"

            [settings.sun_intensity]
            label = "Sun Intensity"
            type = "float"
            default = 3.2
            min = 0.5
            max = 8.0
            runtime = true

            [settings.enable_wind]
            label = "Foliage Wind"
            type = "bool"
            default = true
            runtime = false

            [settings.zenith_tint]
            label = "Zenith Tint"
            type = "color_rgb"
            default = "#FF8040"
            runtime = false
        """.trimIndent()

        val childToml = """
            [pack]
            id = "hyaloclast-cinema"
            name = "Hyaloclast Cinema"
            version = "0.2.0"
            author = "VulkanShaders"
            extends = "hyaloclast-default"

            [capabilities]
            tiers = [0, 1, 2, 3]

            [hooks]
            post_grade = "hooks/post_grade.vsl"

            [settings.film_grain]
            label = "Film Grain"
            type = "float"
            default = 0.15
            runtime = true
        """.trimIndent()

        val baseManifest = ShaderPackManifest.fromToml(baseToml, "fallback")
        val childRaw = ShaderPackManifest.fromToml(childToml, "fallback-child")
        val merged = childRaw.inheritFrom(baseManifest)

        assertEquals("hyaloclast-cinema", merged.id)
        assertEquals("hyaloclast-default", merged.extendsPackId)
        assertEquals(listOf(0, 1, 2, 3), merged.supportedTiers)
        assertEquals("hooks/surface.vsl", merged.hooks["surface"])
        assertEquals("hooks/sky.vsl", merged.hooks["sky"])
        assertEquals("hooks/post_grade.vsl", merged.hooks["post_grade"])
        assertEquals(4, merged.settings.size)
        assertEquals(SettingType.FLOAT, merged.settings[0].type)
        assertTrue(merged.settings[0].runtime)
        assertEquals(listOf("film_grain", "sun_intensity"), merged.runtimeSettings.map { it.key })

        val specDefault = merged.buildSpecializationOverrides(emptyMap())
        assertEquals(2, specDefault.size)
        assertEquals(100 to 1, specDefault[0])
        assertEquals(0xFF8040, WorldStateCollector.unpackFloatAsRgb(intBitsToFloat(specDefault[1].second)))

        val specOverridden = merged.buildSpecializationOverrides(
            mapOf("enable_wind" to "false", "zenith_tint" to "#336699"),
        )
        assertEquals(100 to 0, specOverridden[0])
        assertEquals(0x336699, WorldStateCollector.unpackFloatAsRgb(intBitsToFloat(specOverridden[1].second)))
    }

    @Test
    fun `parses VSP Graph v2 pipeline targets buffers passes and all 16 hook stages`() {
        val v2Toml = """
            [pack]
            id = "vsp-v2-test"
            name = "VSP v2 Test"
            version = "0.3.0"

            [capabilities]
            tiers = [0, 1, 2, 3]

            [pipeline]
            mode = "extend"
            gbuffer_MRT_count = 5

            [targets.volumetric_fog]
            format = "rgba16f"
            scale = 0.5
            temporal = true
            gbuffer_attachment = 4

            [buffers.restir_grid]
            size_bytes = 65536
            persistent = true

            [[passes]]
            id = "volumetric_fog_pass"
            kind = "compute"
            shader = "shaders/compute/volumetric_fog.comp"
            min_tier = 0
            max_tier = 2
            replaces = "volumetric_fog_pass"
            reads = ["mc_main_depth", "shadow_csm"]
            writes = ["volumetric_fog"]
            scale = 0.5
            bindings = { b1 = "mc_main_depth", b4 = "volumetric_fog" }

            [hooks]
            gbuffer_pack = "hooks/gbuffer_pack.vsl"
            restir_reservoir = "hooks/restir.vsl"
            volumetric_media = "hooks/volumetric.vsl"
        """.trimIndent()

        val manifest = ShaderPackManifest.fromToml(v2Toml, "vsp-v2-test")
        assertEquals(PipelineGraphMode.EXTEND, manifest.pipeline.mode)
        assertEquals(5, manifest.pipeline.gbufferMrtCount)
        assertEquals(1, manifest.targets.size)
        assertEquals("volumetric_fog", manifest.targets[0].name)
        assertEquals(0.5f, manifest.targets[0].scale)
        assertTrue(manifest.targets[0].temporal)
        assertEquals(1, manifest.buffers.size)
        assertEquals(1, manifest.passes.size)
        assertEquals("volumetric_fog_pass", manifest.passes[0].id)
        assertEquals("mc_main_depth", manifest.passes[0].bindings["b1"])
        assertEquals(16, ShaderPackManifest.VALID_HOOK_STAGES.size)
    }

    @Test
    fun `rejects invalid hook stage or reversed min max bounds`() {
        val invalidToml = """
            [pack]
            id = "broken-pack"
            name = "Broken Pack"

            [capabilities]
            tiers = [0]

            [hooks]
            nonexistent_stage = "hooks/foo.vsl"
        """.trimIndent()

        assertThrows(IllegalArgumentException::class.java) {
            ShaderPackManifest.fromToml(invalidToml, "broken-pack")
        }
    }

    @Test
    fun `parses presets inline and subtable, sorts by order, merges inheritance, and checks tier ranges`() {
        val parentToml = """
            [pack]
            id = "parent-presets"
            name = "Parent Presets"

            [capabilities]
            tiers = [0, 1, 2, 3]

            [settings.sun_intensity]
            label = "Sun Intensity"
            type = "float"
            default = 2.8
            min = 0.5
            max = 8.0
            runtime = true

            [settings.gi_bounce_strength]
            label = "GI Bounce"
            type = "float"
            default = 1.0
            min = 0.2
            max = 2.0
            runtime = true
            min_tier = 1
            max_tier = 3

            [settings.pt_max_bounces]
            label = "PT Max Bounces"
            type = "int"
            default = 2
            min = 1
            max = 4
            runtime = false
            min_tier = 3
            max_tier = 3

            [settings.enable_wind]
            label = "Foliage Wind"
            type = "bool"
            default = true
            runtime = false

            [presets.high]
            label = "High"
            description = "Default quality"
            order = 2
            values = { sun_intensity = 2.8, pt_max_bounces = 2, enable_wind = true }

            [presets.low]
            label = "Low"
            description = "Fast raster"
            order = 0
            min_tier = 0
            max_tier = 2

            [presets.low.values]
            sun_intensity = 2.2
            enable_wind = false
        """.trimIndent()

        val childToml = """
            [pack]
            id = "child-presets"
            name = "Child Presets"
            extends = "parent-presets"

            [capabilities]
            tiers = [0, 1, 2, 3]

            [settings.sun_intensity]
            label = "Sun Intensity"
            type = "float"
            default = 2.8
            min = 0.5
            max = 8.0
            runtime = true

            [presets.high]
            label = "High Cinema"
            order = 2
            values = { sun_intensity = 3.1 }

            [presets.ultra]
            label = "Ultra"
            order = 3
            min_tier = 2
            max_tier = 3
            values = { sun_intensity = 3.5, gi_bounce_strength = 1.4 }
        """.trimIndent()

        val parent = ShaderPackManifest.fromToml(parentToml, "parent")
        assertEquals(listOf("low", "high"), parent.presets.map { it.id })
        assertEquals("2.2", parent.presets[0].values["sun_intensity"])
        assertEquals("false", parent.presets[0].values["enable_wind"])
        assertTrue(parent.presets[0].appliesToTier(0))
        assertFalse(parent.presets[0].appliesToTier(3))

        val giSetting = parent.settings.first { it.key == "gi_bounce_strength" }
        assertFalse(giSetting.appliesToTier(0))
        assertTrue(giSetting.appliesToTier(1))
        assertTrue(giSetting.appliesToTier(3))

        val ptSetting = parent.settings.first { it.key == "pt_max_bounces" }
        assertFalse(ptSetting.appliesToTier(2))
        assertTrue(ptSetting.appliesToTier(3))

        val child = ShaderPackManifest.fromToml(childToml, "child")
        val merged = child.inheritFrom(parent)
        assertEquals(listOf("low", "high", "ultra"), merged.presets.map { it.id })
        assertEquals("High Cinema", merged.presets[1].label)
        assertEquals("3.1", merged.presets[1].values["sun_intensity"])
        assertEquals("1.4", merged.presets[2].values["gi_bounce_strength"])
    }

    @Test
    fun `scopes compile-time specialization overrides to active tier while keeping constant IDs invariant`() {
        val toml = """
            [pack]
            id = "spec-tier-test"
            name = "Spec Tier Test"

            [capabilities]
            tiers = [0, 1, 2, 3]

            [settings.enable_wind]
            label = "Foliage Wind"
            type = "bool"
            default = true
            runtime = false

            [settings.pt_max_bounces]
            label = "PT Max Bounces"
            type = "int"
            default = 2
            min = 1
            max = 4
            runtime = false
            min_tier = 3
            max_tier = 3
        """.trimIndent()

        val manifest = ShaderPackManifest.fromToml(toml, "spec-tier-test")
        val overrides = mapOf("enable_wind" to "false", "pt_max_bounces" to "4")

        val tier0Specs = manifest.buildSpecializationOverrides(overrides, activeTierIndex = 0)
        assertEquals(listOf(100 to 0, 101 to 2), tier0Specs)

        val tier3Specs = manifest.buildSpecializationOverrides(overrides, activeTierIndex = 3)
        assertEquals(listOf(100 to 0, 101 to 4), tier3Specs)
    }

    @Test
    fun `rejects invalid preset declarations and invalid setting tier ranges`() {
        val reservedCustom = """
            [pack]
            id = "bad-custom"
            name = "Bad Custom"
            [capabilities]
            tiers = [0]
            [presets.Custom]
            label = "Custom"
        """.trimIndent()
        assertThrows(IllegalArgumentException::class.java) {
            ShaderPackManifest.fromToml(reservedCustom, "bad-custom")
        }

        val unknownKey = """
            [pack]
            id = "bad-unknown"
            name = "Bad Unknown"
            [capabilities]
            tiers = [0]
            [presets.low]
            label = "Low"
            values = { does_not_exist = 1 }
        """.trimIndent()
        assertThrows(IllegalArgumentException::class.java) {
            ShaderPackManifest.fromToml(unknownKey, "bad-unknown")
        }

        val outOfRange = """
            [pack]
            id = "bad-range"
            name = "Bad Range"
            [capabilities]
            tiers = [0]
            [settings.pt_max_bounces]
            label = "PT Bounces"
            type = "int"
            default = 2
            min = 1
            max = 4
            runtime = false
            [presets.ultra]
            label = "Ultra"
            values = { pt_max_bounces = 9 }
        """.trimIndent()
        assertThrows(IllegalArgumentException::class.java) {
            ShaderPackManifest.fromToml(outOfRange, "bad-range")
        }

        val badSettingTier = """
            [pack]
            id = "bad-tier"
            name = "Bad Tier"
            [capabilities]
            tiers = [0]
            [settings.gi_bounce_strength]
            label = "GI"
            type = "float"
            default = 1.0
            min_tier = 3
            max_tier = 1
        """.trimIndent()
        assertThrows(IllegalArgumentException::class.java) {
            ShaderPackManifest.fromToml(badSettingTier, "bad-tier")
        }
    }

    @Test
    fun `migrates legacy packOverrides into active tier and isolates overrides per tier`() {
        val legacy = VulkanShadersConfig(
            renderTier = RenderTier.TIER_2_RAY_QUERY,
            packOverrides = mapOf("hyaloclast-default" to mapOf("sun_intensity" to "3.4")),
        )
        val sanitized = ConfigStore.sanitize(legacy)
        assertTrue(sanitized.packOverrides.isEmpty())
        assertEquals(
            mapOf("sun_intensity" to "3.4"),
            sanitized.activePackOverrides("hyaloclast-default", RenderTier.TIER_2_RAY_QUERY),
        )
        assertTrue(sanitized.activePackOverrides("hyaloclast-default", RenderTier.TIER_0_RASTER).isEmpty())

        val updated = sanitized.withPackOverrides(
            "hyaloclast-default",
            RenderTier.TIER_0_RASTER,
            mapOf("fog_density" to "0.0015"),
        )
        assertEquals(
            mapOf("fog_density" to "0.0015"),
            updated.activePackOverrides("hyaloclast-default", RenderTier.TIER_0_RASTER),
        )
        assertEquals(
            mapOf("sun_intensity" to "3.4"),
            updated.activePackOverrides("hyaloclast-default", RenderTier.TIER_2_RAY_QUERY),
        )
    }

    @Test
    fun `migrates retired legacy built-in selectedPackId values to hyaloclast-sunlit-vanilla`() {
        for (legacyId in listOf(
            "hyaloclast-default",
            "hyaloclast-vanilla-reverie",
            "hyaloclast-golden-solstice",
            "hyaloclast-nordic-borealis",
            "hyaloclast-cinder-mist",
        )) {
            val sanitized = ConfigStore.sanitize(VulkanShadersConfig(selectedPackId = legacyId))
            assertEquals("hyaloclast-sunlit-vanilla", sanitized.selectedPackId)
        }
        val custom = ConfigStore.sanitize(VulkanShadersConfig(selectedPackId = "my-custom-pack"))
        assertEquals("my-custom-pack", custom.selectedPackId)
    }

    @Test
    fun `consolidated built-in packs parse cleanly, inherit properly, and provide all hooks and presets`() {
        val cl = ShaderPackManager::class.java.classLoader
        val baseId = "hyaloclast-sunlit-vanilla"
        val showcaseId = "hyaloclast-obsidian-showcase"

        val baseToml = cl.getResourceAsStream("builtin-packs/$baseId.vsp/pack.toml")
            ?.bufferedReader()
            ?.use { it.readText() }
        assertNotNull(baseToml, "$baseId pack.toml must exist on classpath")
        val baseManifest = ShaderPackManifest.fromToml(baseToml!!, baseId)
        val expectedRuntimeKeys = baseManifest.runtimeSettings.map { it.key }
        val expectedSettingKeys = baseManifest.settings.map { it.key }.toSet()

        assertEquals(
            listOf(baseId, "relight-vanilla", showcaseId),
            ShaderPackManager.BUILTIN_PACK_IDS,
        )
        assertEquals(listOf(0, 1, 2, 3), baseManifest.supportedTiers)
        assertEquals(listOf("low", "medium", "high", "ultra"), baseManifest.presets.map { it.id })
        assertTrue(baseManifest.resources.any { it.name == "blue_noise" })

        val allHooks = listOf(
            "vertex_displace", "surface", "direct_light", "block_light",
            "water_waves", "water_surface", "sky", "clouds", "fog", "post_grade",
        )
        for (stage in allHooks) {
            val relPath = baseManifest.hooks[stage]
            assertNotNull(relPath, "Base pack '$baseId' must declare hook '$stage'")
            val hookSource = cl.getResourceAsStream("builtin-packs/$baseId.vsp/$relPath")
                ?.bufferedReader()
                ?.use { it.readText() }
            assertNotNull(hookSource, "Missing hook file '$relPath' in '$baseId'")
            assertTrue(hookSource!!.contains("@hook(stage = \"$stage\""))
        }

        val showcaseToml = cl.getResourceAsStream("builtin-packs/$showcaseId.vsp/pack.toml")
            ?.bufferedReader()
            ?.use { it.readText() }
        assertNotNull(showcaseToml, "Missing pack.toml for built-in pack '$showcaseId'")

        val childRaw = ShaderPackManifest.fromToml(showcaseToml!!, showcaseId)
        assertEquals(showcaseId, childRaw.id)
        assertEquals(baseId, childRaw.extendsPackId)
        assertEquals(listOf(0, 1, 2, 3), childRaw.supportedTiers)

        for (stage in allHooks) {
            val relPath = childRaw.hooks[stage]
            assertNotNull(relPath, "Built-in pack '$showcaseId' must override hook '$stage'")
            val hookSource = cl.getResourceAsStream("builtin-packs/$showcaseId.vsp/$relPath")
                ?.bufferedReader()
                ?.use { it.readText() }
            assertNotNull(hookSource, "Missing hook file '$relPath' in built-in pack '$showcaseId'")
            assertTrue(
                hookSource!!.contains("@hook(stage = \"$stage\""),
                "Hook '$relPath' in '$showcaseId' must declare @hook(stage = \"$stage\")",
            )
        }

        val merged = childRaw.inheritFrom(baseManifest)
        assertEquals(baseManifest.hooks.keys, merged.hooks.keys)
        assertEquals(expectedSettingKeys, merged.settings.map { it.key }.toSet())
        assertEquals(expectedRuntimeKeys, merged.runtimeSettings.map { it.key })
        assertEquals(listOf("low", "medium", "high", "ultra"), merged.presets.map { it.id })
        assertTrue(merged.resources.any { it.name == "blue_noise" })
    }
}
