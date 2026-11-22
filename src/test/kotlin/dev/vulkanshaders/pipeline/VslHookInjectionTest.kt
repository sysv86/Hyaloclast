package dev.vulkanshaders.pipeline

import com.mojang.renderpearl.api.pipeline.ShaderType
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertFalse
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.charset.StandardCharsets

/**
 * Verifies that the *body* and helper functions of a pack's `.vsl` hook are injected into the
 * raster G-Buffer GLSL, all pack parameters and specialization constants are emitted, comments
 * containing braces `{`/`}` do not break lexer extraction, colliding helper names across hooks
 * are isolated, and `GbufferTargetManager.packVsMotionBuffer` writes exact `std140` offsets.
 */
class VslHookInjectionTest {
    private fun bundled(path: String): String =
        javaClass.getResourceAsStream(path)!!.use { String(it.readAllBytes(), StandardCharsets.UTF_8) }

    private fun shader(name: String): String = bundled("/assets/vulkanshaders/shaders/core/$name")

    private fun hook(name: String): String = bundled("/builtin-packs/hyaloclast-sunlit-vanilla.vsp/hooks/$name")

    @Test
    fun `vertex_displace hook body is injected and called`() {
        val result = GbufferPipelineRegistry.applyPackHooks(
            source = shader("gbuffer_terrain_region.vsh"),
            stem = "gbuffer_terrain_region",
            type = ShaderType.VERTEX,
            hooks = mapOf("vertex_displace" to hook("vertex_displace.vsl")),
        )
        assertTrue(result.contains("float phase = vsWorld.gameTimeSec * windSpeed + localPos.x * 0.68"), "hook body missing")
        assertTrue(result.contains("localPos.x += gust;"), "hook deformation missing")
        assertTrue(result.contains("localPos.z += gust * 0.62;"), "hook deformation missing")
        assertTrue(result.contains("void vsHookVertexDisplace(inout vec3 localPos, VsVertexInput v)"), "adapter missing")
        assertTrue(result.contains("vsHookVertexDisplace(localPos, VsVertexInput(vertexHookFlags));"), "hook call missing")
        assertFalse(result.contains("// @hook vertex_displace"), "marker must be replaced by real code")
        assertTrue(result.indexOf("#version 330") < result.indexOf("void vsHookVertexDisplace"), "#version must stay first")
    }

    @Test
    fun `water_waves hook body drives the wave parameters and preserves helper functions`() {
        val customHook = """
            #include <vulkanshaders/v1/common.vsl>
            float customWaveScale(float s) { return s * 1.25; }
            @hook(stage = "water_waves")
            void custom_waves(inout float strength, inout float speed, inout float scale) {
                float packWaveStrength = VS_PARAM_WATER_WAVE_STRENGTH;
                strength = packWaveStrength > 0.0 ? packWaveStrength : 0.35;
                scale = customWaveScale(scale);
            }
        """.trimIndent()
        val result = GbufferPipelineRegistry.applyPackHooks(
            source = shader("gbuffer_terrain_region.vsh"),
            stem = "gbuffer_terrain_region",
            type = ShaderType.VERTEX,
            hooks = mapOf("water_waves" to customHook),
            params = mapOf("water_wave_strength" to 0.55f, "sun_intensity" to 3.2f),
            specDefines = mapOf("enable_wind" to "false"),
        )
        assertTrue(result.contains("float customWaveScale(float s) { return s * 1.25; }"), "helper function missing")
        assertTrue(result.contains("float packWaveStrength = VS_PARAM_WATER_WAVE_STRENGTH;"), "hook body missing")
        assertTrue(result.contains("#define VS_PARAM_WATER_WAVE_STRENGTH 0.55f"), "pack parameter not propagated")
        assertTrue(result.contains("#define VS_PARAM_SUN_INTENSITY 3.2f"), "custom pack parameter not propagated")
        assertTrue(result.contains("#define VS_SPEC_ENABLE_WIND false"), "spec constant override not propagated")
        assertTrue(result.contains("vsHookWaterWaves(vsWaveStrength, vsWaveSpeed, vsWaveScale);"), "hook call missing")
        assertTrue(result.contains("#define VS_HOOK_WATER_WAVES_BOUND 1"))
        assertFalse(result.contains("// @hook water_waves"))
    }

    @Test
    fun `glsl lexer ignores braces inside comments and isolates colliding helpers across hooks`() {
        val displaceHook = """
            #include <vulkanshaders/v1/common.vsl>
            // comment with unbalanced braces: { { }
            float sharedHash(float x) { return fract(sin(x) * 43758.5453); }
            // @hook(stage = "vertex_displace")
            void my_displace(inout vec3 localPos, in VsVertexInput v) {
                /* { block comment with brace */
                localPos.y += sharedHash(localPos.x) * 0.01;
            }
        """.trimIndent()
        val wavesHook = """
            #include <vulkanshaders/v1/common.vsl>
            float sharedHash(float x) { return fract(cos(x) * 12345.6789); }
            // @hook(stage = "water_waves")
            void my_waves(inout float strength, inout float speed, inout float scale) {
                strength += sharedHash(scale) * 0.05;
            }
        """.trimIndent()

        val result = GbufferPipelineRegistry.applyPackHooks(
            source = shader("gbuffer_terrain_region.vsh"),
            stem = "gbuffer_terrain_region",
            type = ShaderType.VERTEX,
            hooks = mapOf("vertex_displace" to displaceHook, "water_waves" to wavesHook),
        )
        assertTrue(result.contains("vs_ns_vertex_displace_sharedHash"), "colliding helper in vertex_displace must be isolated")
        assertTrue(result.contains("vs_ns_water_waves_sharedHash"), "colliding helper in water_waves must be isolated")
    }

    @Test
    fun `surface hook body is injected into terrain and entity fragment shaders`() {
        val terrainHook = hook("surface.vsl")
        for (name in listOf("gbuffer_terrain_region.fsh", "gbuffer_entity.fsh")) {
            val result = GbufferPipelineRegistry.applyPackHooks(
                source = shader(name),
                stem = name.removeSuffix(".fsh"),
                type = ShaderType.FRAGMENT,
                hooks = mapOf("surface" to terrainHook),
            )
            assertTrue(result.contains("s.roughness = clamp(s.roughness, 0.72, 0.88);"), "$name: hook body missing")
            assertTrue(result.contains("s.albedo *= 0.84;"), "$name: hook body missing")
            assertTrue(result.contains("s.f0 = min(s.f0, 0.038);"), "$name: hook body missing")
            assertTrue(result.contains("void vsHookSurface(inout VsSurface s, uint flags, uint bounce)"), "$name: adapter missing")
            assertTrue(result.contains("vsHookSurface(vsHookSurf, materialFlags, 0u);"), "$name: hook call missing")
            assertTrue(result.contains("roughness = clamp(roughness, 0.02, 1.0);"), "$name: roughness clamp missing")
            assertFalse(result.contains("// @hook surface"), "$name: marker must be replaced")
        }
    }

    @Test
    fun `no hook supplied leaves shader unchanged`() {
        val original = shader("gbuffer_terrain_region.vsh")
        val result = GbufferPipelineRegistry.applyPackHooks(
            source = original,
            stem = "gbuffer_terrain_region",
            type = ShaderType.VERTEX,
            hooks = emptyMap(),
        )
        assertEquals(original, result)
    }

    @Test
    fun `packVsMotionBuffer writes exact std140 offsets matching VsMotion uniform block`() {
        val ubo = ByteBuffer.allocateDirect(512).order(ByteOrder.nativeOrder())
        // 64..127: prevViewProj diagonal = 2.0f
        ubo.putFloat(64, 2.0f)
        // 272..287: cameraDelta = (1.5, -2.5, 3.5), gameTimeSec = 42.0
        ubo.putFloat(272, 1.5f)
        ubo.putFloat(276, -2.5f)
        ubo.putFloat(280, 3.5f)
        ubo.putFloat(284, 42.0f)
        // 288..303: sunDir = (0.0, 1.0, 0.0), rain = 0.75
        ubo.putFloat(288, 0.0f)
        ubo.putFloat(292, 1.0f)
        ubo.putFloat(296, 0.0f)
        ubo.putFloat(300, 0.75f)
        // 304..319: thunder = 0.25, eyeInFluid = 1, depthFar = 256.0, heldLight = 0.9
        ubo.putFloat(304, 0.25f)
        ubo.putInt(308, 1)
        ubo.putFloat(312, 256.0f)
        ubo.putFloat(316, 0.9f)

        val dst = ByteBuffer.allocateDirect(128).order(ByteOrder.nativeOrder())
        GbufferTargetManager.packVsMotionBuffer(
            ubo = ubo,
            dst = dst,
            hasPbr = true,
            hasWindHook = true,
            hasWavesHook = false,
        )

        assertEquals(2.0f, dst.getFloat(0))
        assertEquals(1.5f, dst.getFloat(64))
        assertEquals(-2.5f, dst.getFloat(68))
        assertEquals(3.5f, dst.getFloat(72))
        assertEquals(42.0f, dst.getFloat(76))
        assertEquals(1.0f, dst.getFloat(80))
        assertEquals(1.0f, dst.getFloat(84))
        assertEquals(0.0f, dst.getFloat(88))
        assertEquals(0.0f, dst.getFloat(96))
        assertEquals(1.0f, dst.getFloat(100))
        assertEquals(0.0f, dst.getFloat(104))
        assertEquals(0.75f, dst.getFloat(108))
        assertEquals(0.9f, dst.getFloat(112))
        assertEquals(0.25f, dst.getFloat(116))
        assertEquals(1.0f, dst.getFloat(120))
        assertEquals(256.0f, dst.getFloat(124))
    }
}
