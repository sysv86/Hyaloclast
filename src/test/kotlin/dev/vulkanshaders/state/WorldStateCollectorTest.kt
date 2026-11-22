package dev.vulkanshaders.state

import dev.vulkanshaders.pack.PackSettingDefinition
import dev.vulkanshaders.pack.SettingType
import net.minecraft.world.phys.Vec3
import org.joml.Matrix4f
import org.joml.Vector4f
import org.junit.jupiter.api.AfterEach
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertFalse
import org.junit.jupiter.api.Assertions.assertNotEquals
import org.junit.jupiter.api.Assertions.assertNull
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.PI

class WorldStateCollectorTest {

    @BeforeEach
    @AfterEach
    fun resetCollectorState() {
        WorldStateCollector.resetHistory()
        WorldStateCollector.cameraPosX = 0.0
        WorldStateCollector.cameraPosY = 0.0
        WorldStateCollector.cameraPosZ = 0.0
        WorldStateCollector.prevCameraPosX = 0.0
        WorldStateCollector.prevCameraPosY = 0.0
        WorldStateCollector.prevCameraPosZ = 0.0
    }

    @Test
    fun `parses hex and comma-separated COLOR_RGB formats accurately`() {
        assertEquals(0xFF8040, WorldStateCollector.parseColorRgb("#FF8040"))
        assertEquals(0x00AAFF, WorldStateCollector.parseColorRgb(" #00aaff "))
        assertEquals(0xFF8844, WorldStateCollector.parseColorRgb("#F84"))
        assertEquals(0x123456, WorldStateCollector.parseColorRgb("0x123456"))
        assertEquals(0xABCDEF, WorldStateCollector.parseColorRgb("abcdef"))
        assertEquals(0xFF8040, WorldStateCollector.parseColorRgb("255, 128, 64"))
        assertEquals(0x00FF7F, WorldStateCollector.parseColorRgb("0,255,127"))
        assertEquals(0xFF8040, WorldStateCollector.parseColorRgb("1.0, 0.502, 0.251"))
        assertEquals(0x000000, WorldStateCollector.parseColorRgb("0.0, 0.0, 0.0"))

        assertNull(WorldStateCollector.parseColorRgb(""))
        assertNull(WorldStateCollector.parseColorRgb("#12345"))
        assertNull(WorldStateCollector.parseColorRgb("#GGGGGG"))
        assertNull(WorldStateCollector.parseColorRgb("255, 128"))
        assertNull(WorldStateCollector.parseColorRgb("not_a_color"))
    }

    @Test
    fun `packs and unpacks 24-bit COLOR_RGB into normalized float losslessly`() {
        val samples = listOf(
            0x000000,
            0x000001,
            0x000100,
            0x010000,
            0x7FFFFF,
            0x800000,
            0xFF8040,
            0x123456,
            0xABCDEF,
            0xFFFFFE,
            0xFFFFFF,
        )
        for (rgb in samples) {
            val packed = WorldStateCollector.packRgbAsFloat(rgb)
            assertTrue(packed in 0.0f..1.0f, "Packed RGB float must be normalized [0, 1] for 0x${rgb.toString(16)}")
            val unpacked = WorldStateCollector.unpackFloatAsRgb(packed)
            assertEquals(rgb, unpacked, "Round-trip failed for 0x${rgb.toString(16)}")
        }

        assertEquals("#000000", WorldStateCollector.formatColorRgb(0))
        assertEquals("#FF8040", WorldStateCollector.formatColorRgb(0xFF8040))
        assertEquals(Triple(255, 128, 64), WorldStateCollector.unpackColorRgb(0xFF8040))
    }

    @Test
    fun `settingAsFloat packs COLOR_RGB override and falls back to default on invalid override`() {
        val def = PackSettingDefinition(
            key = "sky_tint",
            label = "Sky Tint",
            type = SettingType.COLOR_RGB,
            defaultValue = "#336699",
            runtime = true,
        )

        val defaultPacked = WorldStateCollector.settingAsFloat(def, null)
        assertEquals(0x336699, WorldStateCollector.unpackFloatAsRgb(defaultPacked))

        val overridePacked = WorldStateCollector.settingAsFloat(def, "#FF4020")
        assertEquals(0xFF4020, WorldStateCollector.unpackFloatAsRgb(overridePacked))

        val fallbackPacked = WorldStateCollector.settingAsFloat(def, "invalid-color")
        assertEquals(0x336699, WorldStateCollector.unpackFloatAsRgb(fallbackPacked))

        val badDef = def.copy(defaultValue = "invalid")
        assertEquals(0.0f, WorldStateCollector.settingAsFloat(badDef, "also-invalid"))
    }

    @Test
    fun `halton sequence produces expected base-2 and base-3 values`() {
        assertEquals(0.5f, WorldStateCollector.halton(1, 2), 1e-6f)
        assertEquals(0.25f, WorldStateCollector.halton(2, 2), 1e-6f)
        assertEquals(0.75f, WorldStateCollector.halton(3, 2), 1e-6f)
        assertEquals(0.125f, WorldStateCollector.halton(4, 2), 1e-6f)

        assertEquals(1.0f / 3.0f, WorldStateCollector.halton(1, 3), 1e-6f)
        assertEquals(2.0f / 3.0f, WorldStateCollector.halton(2, 3), 1e-6f)
        assertEquals(1.0f / 9.0f, WorldStateCollector.halton(3, 3), 1e-6f)
        assertEquals(4.0f / 9.0f, WorldStateCollector.halton(4, 3), 1e-6f)
    }

    @Test
    fun `updateJitter centers Halton(2, 3) sequence and zeros when temporal is inactive`() {
        WorldStateCollector.frameIndex = 0
        WorldStateCollector.updateJitter(1920, 1080, temporalActive = false)
        assertEquals(0.0f, WorldStateCollector.jitterX)
        assertEquals(0.0f, WorldStateCollector.jitterY)

        val seen = HashSet<Pair<Float, Float>>()
        for (frame in 0 until WorldStateCollector.JITTER_PHASE_COUNT) {
            WorldStateCollector.frameIndex = frame
            WorldStateCollector.updateJitter(1920, 1080, temporalActive = true)
            assertTrue(WorldStateCollector.jitterX in -0.5f..0.5f)
            assertTrue(WorldStateCollector.jitterY in -0.5f..0.5f)
            seen.add(WorldStateCollector.jitterX to WorldStateCollector.jitterY)
        }
        assertEquals(WorldStateCollector.JITTER_PHASE_COUNT, seen.size, "Expected 16 distinct Halton(2,3) samples")

        // Sequence wraps cleanly after JITTER_PHASE_COUNT frames
        WorldStateCollector.frameIndex = 0
        WorldStateCollector.updateJitter(1920, 1080, temporalActive = true)
        val x0 = WorldStateCollector.jitterX
        val y0 = WorldStateCollector.jitterY

        WorldStateCollector.frameIndex = WorldStateCollector.JITTER_PHASE_COUNT
        WorldStateCollector.updateJitter(1920, 1080, temporalActive = true)
        assertEquals(x0, WorldStateCollector.jitterX, 1e-6f)
        assertEquals(y0, WorldStateCollector.jitterY, 1e-6f)
    }

    @Test
    fun `camera bob reconstructs the same terrain point the rasterizer projected`() {
        val projection = Matrix4f().perspective(1.2f, 16.0f / 9.0f, 0.05f, 512.0f)
        val rotation = Matrix4f().rotateX(0.23f).rotateY(-0.41f)
        val bob = Matrix4f().translate(0.08f, -0.12f, 0.0f).rotateZ(0.04f)
        val rasterProjection = Matrix4f(projection).mul(bob)
        val view = WorldStateCollector.effectiveWorldView(projection, rasterProjection, rotation)

        val point = Vector4f(3.0f, -1.0f, -14.0f, 1.0f)
        val clip = Vector4f(point).mul(rotation).mul(rasterProjection)
        val reconstructed = Vector4f(clip).mul(Matrix4f(projection).invert()).mul(Matrix4f(view).invert())

        assertTrue(point.equals(reconstructed, 1e-3f), "Raster and ray passes must agree on the surface position")
        assertNotEquals(0.0f, Matrix4f(view).invert().m31(), "The ray origin must include the camera bob")
    }

    @Test
    fun `applyProjectionJitter shifts NDC by subpixel offset while WorldUBO matrices stay unjittered`() {
        val width = 1600
        val height = 900
        val proj = Matrix4f().perspective(Math.toRadians(70.0).toFloat(), width.toFloat() / height.toFloat(), 0.05f, 512.0f)
        val view = Matrix4f().lookAt(0.0f, 64.0f, 0.0f, 0.0f, 64.0f, -10.0f, 0.0f, 1.0f, 0.0f)

        WorldStateCollector.frameIndex = 1
        WorldStateCollector.updateJitter(width, height, temporalActive = true)
        assertNotEquals(0.0f, WorldStateCollector.jitterX)
        assertNotEquals(0.0f, WorldStateCollector.jitterY)

        val jitteredProj = WorldStateCollector.applyProjectionJitter(proj, width, height)
        val viewPoint = Vector4f(0.0f, 0.0f, -10.0f, 1.0f)
        val clipUnjittered = Vector4f(viewPoint).mul(proj)
        val clipJittered = Vector4f(viewPoint).mul(jitteredProj)

        val ndcUnjitteredX = clipUnjittered.x / clipUnjittered.w
        val ndcUnjitteredY = clipUnjittered.y / clipUnjittered.w
        val ndcJitteredX = clipJittered.x / clipJittered.w
        val ndcJitteredY = clipJittered.y / clipJittered.w

        assertEquals(2.0f * WorldStateCollector.jitterX / width.toFloat(), ndcJitteredX - ndcUnjitteredX, 1e-5f)
        assertEquals(2.0f * WorldStateCollector.jitterY / height.toFloat(), ndcJitteredY - ndcUnjitteredY, 1e-5f)
        assertEquals(clipUnjittered.z, clipJittered.z, 1e-5f)
        assertEquals(clipUnjittered.w, clipJittered.w, 1e-5f)

        // Verify WorldUBO curViewProj / prevViewProj remain unjittered across frames for consistent motion vectors
        WorldStateCollector.updateMatrices(view, proj, isDiscontinuousFrame = true)
        WorldStateCollector.packStd140(0f, 1f, 0f, 0f, 0f, 0f, 0, 512f, 0f, 0f)
        val buf = WorldStateCollector.uniformBufferReadOnly()
        val expectedViewProj = Matrix4f(proj).mul(view)
        val curMat = Matrix4f().set(0, buf)
        val prevMat = Matrix4f().set(64, buf)
        assertEquals(expectedViewProj, curMat)
        assertEquals(expectedViewProj, prevMat)

        // Frame 2 with different jitter still writes identical unjittered curViewProj / prevViewProj for static camera
        WorldStateCollector.frameIndex = 2
        WorldStateCollector.updateJitter(width, height, temporalActive = true)
        WorldStateCollector.updateMatrices(view, proj, isDiscontinuousFrame = false)
        WorldStateCollector.packStd140(0f, 1f, 0f, 0f, 0f, 0.016f, 0, 512f, 0f, 0f)
        val buf2 = WorldStateCollector.uniformBufferReadOnly()
        assertEquals(expectedViewProj, Matrix4f().set(0, buf2))
        assertEquals(expectedViewProj, Matrix4f().set(64, buf2))
    }

    @Test
    fun `packEmissiveLightDesc matches Rust WorldEmissiveLightDesc 48-byte repr(C) layout`() {
        val buf = ByteBuffer.allocate(WorldStateCollector.EMISSIVE_LIGHT_DESC_BYTES).order(ByteOrder.nativeOrder())
        WorldStateCollector.packEmissiveLightDesc(
            buffer = buf,
            index = 0,
            worldX = 12.5f,
            worldY = 65.5f,
            worldZ = -42.5f,
            radius = 16.0f,
            r = 1.0f,
            g = 0.68f,
            b = 0.34f,
            intensity = 2.4f,
            normalX = 0.0f,
            normalY = 1.0f,
            normalZ = 0.0f,
            flags = 2,
        )
        assertEquals(12.5f, buf.getFloat(0), 1e-6f)
        assertEquals(65.5f, buf.getFloat(4), 1e-6f)
        assertEquals(-42.5f, buf.getFloat(8), 1e-6f)
        assertEquals(16.0f, buf.getFloat(12), 1e-6f)
        assertEquals(1.0f, buf.getFloat(16), 1e-6f)
        assertEquals(0.68f, buf.getFloat(20), 1e-6f)
        assertEquals(0.34f, buf.getFloat(24), 1e-6f)
        assertEquals(2.4f, buf.getFloat(28), 1e-6f)
        assertEquals(0.0f, buf.getFloat(32), 1e-6f)
        assertEquals(1.0f, buf.getFloat(36), 1e-6f)
        assertEquals(0.0f, buf.getFloat(40), 1e-6f)
        assertEquals(2, buf.getInt(44))
    }

    @Test
    fun `sunAngleFromOverworldClock tracks noon, sunset, midnight, sunrise, and day wrap`() {
        val noon = WorldStateCollector.sunAngleFromOverworldClock(6000L)
        val noonDir = WorldStateCollector.sunDirectionForDimension(0, noon)
        assertEquals(0.0f, noon, 1e-4f)
        assertEquals(0.0f, noonDir.x, 1e-4f)
        assertEquals(1.0f, noonDir.y, 1e-4f)

        val midnight = WorldStateCollector.sunAngleFromOverworldClock(18000L)
        val midnightDir = WorldStateCollector.sunDirectionForDimension(0, midnight)
        assertEquals(PI.toFloat(), midnight, 1e-4f)
        assertEquals(0.0f, midnightDir.x, 1e-4f)
        assertEquals(-1.0f, midnightDir.y, 1e-4f)

        val sunriseDir = WorldStateCollector.sunDirectionForDimension(0, WorldStateCollector.sunAngleFromOverworldClock(0L))
        assertTrue(sunriseDir.x > 0.9f, "Sunrise sun must rise in the east (+X), got ${sunriseDir.x}")

        val sunsetDir = WorldStateCollector.sunDirectionForDimension(0, WorldStateCollector.sunAngleFromOverworldClock(12000L))
        assertTrue(sunsetDir.x < -0.9f, "Sunset sun must set in the west (-X), got ${sunsetDir.x}")

        // Multi-day wrap and negative clock values stay consistent with 24000-tick cycle
        assertEquals(midnight, WorldStateCollector.sunAngleFromOverworldClock(18000L + 24000L * 5L), 1e-4f)
        assertEquals(midnight, WorldStateCollector.sunAngleFromOverworldClock(-6000L), 1e-4f)

        // Sub-tick partialTick advances smoothly
        val t0 = WorldStateCollector.sunAngleFromOverworldClock(6000L, 0.0f)
        val tHalf = WorldStateCollector.sunAngleFromOverworldClock(6000L, 0.5f)
        assertTrue(tHalf > t0, "Positive partialTick must advance the sun angle")
    }

    @Test
    fun `resolveOverworldSunAngle prevents Tier 3 daytime lock when SkyRenderState is stale or zero`() {
        // Tier 3 (preferClock = true) ignores stale daytime sunAngle=0.25f and tracks midnight from clock
        val tier3Midnight = WorldStateCollector.resolveOverworldSunAngle(
            reportedSunAngle = 0.25f,
            overworldClockTime = 18000L,
            partialTick = 0.0f,
            preferClock = true,
        )
        val tier3Dir = WorldStateCollector.sunDirectionForDimension(0, tier3Midnight)
        assertEquals(-1.0f, tier3Dir.y, 1e-4f, "Tier 3 must point sun downward at midnight")

        // Tier 1/2 (preferClock = false) falls back to clock when SkyRenderState.sunAngle is unpopulated 0.0f off-noon
        val fallbackMidnight = WorldStateCollector.resolveOverworldSunAngle(
            reportedSunAngle = 0.0f,
            overworldClockTime = 18000L,
            partialTick = 0.0f,
            preferClock = false,
        )
        assertEquals(PI.toFloat(), fallbackMidnight, 1e-4f)

        // Tier 1/2 preserves explicit non-zero SkyRenderState.sunAngle (e.g., custom datapack timeline)
        val customAngle = 1.23f
        assertEquals(
            customAngle,
            WorldStateCollector.resolveOverworldSunAngle(
                reportedSunAngle = customAngle,
                overworldClockTime = 18000L,
                partialTick = 0.0f,
                preferClock = false,
            ),
            1e-6f,
        )

        // Non-finite reported angle falls back cleanly to clock
        val nanFallback = WorldStateCollector.resolveOverworldSunAngle(
            reportedSunAngle = Float.NaN,
            overworldClockTime = 18000L,
            partialTick = 0.0f,
            preferClock = false,
        )
        assertEquals(PI.toFloat(), nanFallback, 1e-4f)
    }

    @Test
    fun `packHeldItemLight preserves sky bits and never downgrades ambient block light`() {
        val sky15Block0 = 15 shl 20
        val litFull = WorldStateCollector.packHeldItemLight(sky15Block0, 1.0f)
        assertEquals((15 shl 20) or (15 shl 4), litFull)

        val sky8Block12 = (8 shl 20) or (12 shl 4)
        // Dim enchanted item (level 0.35 -> block 5) must not reduce ambient block light 12
        assertEquals(sky8Block12, WorldStateCollector.packHeldItemLight(sky8Block12, 0.35f))
        // Torch (14/15 -> block 14) raises block light to 14 while preserving sky 8
        assertEquals((8 shl 20) or (14 shl 4), WorldStateCollector.packHeldItemLight(sky8Block12, 14.0f / 15.0f))

        // Zero, negative, and NaN levels leave packedLight untouched
        assertEquals(sky15Block0, WorldStateCollector.packHeldItemLight(sky15Block0, 0.0f))
        assertEquals(sky15Block0, WorldStateCollector.packHeldItemLight(sky15Block0, -0.5f))
        assertEquals(sky15Block0, WorldStateCollector.packHeldItemLight(sky15Block0, Float.NaN))
    }

    @Test
    fun `heldLightRadius and heldLightIntensity scale with emission level and perspective`() {
        val dimRadius = WorldStateCollector.heldLightRadius(0.35f)
        val torchRadius = WorldStateCollector.heldLightRadius(14.0f / 15.0f)
        val lavaRadius = WorldStateCollector.heldLightRadius(1.0f)
        assertTrue(dimRadius < torchRadius && torchRadius < lavaRadius)
        assertEquals(16.0f, lavaRadius, 1e-4f)

        val dimIntensity = WorldStateCollector.heldLightIntensity(0.35f, firstPerson = true)
        val torchFirst = WorldStateCollector.heldLightIntensity(14.0f / 15.0f, firstPerson = true)
        val torchThird = WorldStateCollector.heldLightIntensity(14.0f / 15.0f, firstPerson = false)
        assertTrue(dimIntensity < torchFirst * 0.2f, "Dim enchanted gear should have quadratic falloff vs torch")
        assertTrue(torchFirst > torchThird, "First-person held intensity should exceed third-person intensity")
        assertEquals(0.0f, WorldStateCollector.heldLightIntensity(0.0f, firstPerson = true), 1e-6f)
    }

    @Test
    fun `isOpenFlameBlockName extinguishes open flames underwater without extinguishing bioluminescent blocks`() {
        assertTrue(WorldStateCollector.isOpenFlameBlockName("block.minecraft.torch"))
        assertTrue(WorldStateCollector.isOpenFlameBlockName("block.minecraft.soul_torch"))
        assertTrue(WorldStateCollector.isOpenFlameBlockName("block.minecraft.copper_torch"))
        assertTrue(WorldStateCollector.isOpenFlameBlockName("block.minecraft.campfire"))
        assertTrue(WorldStateCollector.isOpenFlameBlockName("block.minecraft.candle"))
        assertTrue(WorldStateCollector.isOpenFlameBlockName("block.minecraft.fire"))

        assertFalse(WorldStateCollector.isOpenFlameBlockName("block.minecraft.redstone_torch"))
        assertFalse(WorldStateCollector.isOpenFlameBlockName("block.minecraft.torchflower"))
        assertFalse(WorldStateCollector.isOpenFlameBlockName("block.minecraft.firefly_bush"))
        assertFalse(WorldStateCollector.isOpenFlameBlockName("block.minecraft.sea_lantern"))
        assertFalse(WorldStateCollector.isOpenFlameBlockName("block.minecraft.glowstone"))
    }

    @Test
    fun `packStd140 writes sunDir, heldLight, and heldLightPosition at exact ABI offsets`() {
        WorldStateCollector.packStd140(
            sunDirX = 0.0f,
            sunDirY = -1.0f,
            sunDirZ = 0.0f,
            rain = 0.25f,
            thunder = 0.1f,
            gameTimeSec = 42.5f,
            eyeInFluid = 0,
            depthFar = 256.0f,
            heldLightMain = 0.4f,
            heldLightOff = 0.9333f,
            dimensionId = 0,
            heldPosition = Vec3(0.18, -0.18, -0.24),
            heldLightKind = 3,
        )
        val buf = WorldStateCollector.uniformBufferReadOnly()
        assertEquals(0.0f, buf.getFloat(288), 1e-6f)
        assertEquals(-1.0f, buf.getFloat(292), 1e-6f)
        assertEquals(0.0f, buf.getFloat(296), 1e-6f)
        assertEquals(0.9333f, buf.getFloat(316), 1e-5f)
        assertEquals(0.18f, buf.getFloat(512), 1e-5f)
        assertEquals(-0.18f, buf.getFloat(516), 1e-5f)
        assertEquals(-0.24f, buf.getFloat(520), 1e-5f)
        assertEquals(3.0f, buf.getFloat(524), 1e-5f)
    }
}
