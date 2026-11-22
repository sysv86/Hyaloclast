package dev.vulkanshaders.gui

import dev.vulkanshaders.backend.NativeCoreBridge
import dev.vulkanshaders.config.DenoiserMode
import dev.vulkanshaders.config.FrameGenMode
import dev.vulkanshaders.config.RenderTier
import dev.vulkanshaders.config.UpscalerMode
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertFalse
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test

/** Verifies honest capability gating in the shader-pack settings screen. */
class ShaderPacksCapabilityTest {
    @Test
    fun `tiers 0 and 1 always available, RT tiers require device capabilities`() {
        assertTrue(ShaderGraphicsScreen.tierAvailable(RenderTier.TIER_0_RASTER, 0))
        assertTrue(ShaderGraphicsScreen.tierAvailable(RenderTier.TIER_1_COMPUTE_GI, 0))
        assertFalse(ShaderGraphicsScreen.tierAvailable(RenderTier.TIER_2_RAY_QUERY, 0))
        assertFalse(ShaderGraphicsScreen.tierAvailable(RenderTier.TIER_3_PATH_TRACING, NativeCoreBridge.CAP_RAY_QUERY))
        val allRtCaps = NativeCoreBridge.CAP_RAY_QUERY or
            NativeCoreBridge.CAP_RAY_TRACING_AS or
            NativeCoreBridge.CAP_RAY_TRACING_PIPELINE
        assertTrue(
            ShaderGraphicsScreen.tierAvailable(
                RenderTier.TIER_3_PATH_TRACING,
                allRtCaps,
                macOsMoltenVk = false,
            ),
        )
        assertFalse(
            ShaderGraphicsScreen.tierAvailable(
                RenderTier.TIER_2_RAY_QUERY,
                allRtCaps,
                macOsMoltenVk = true,
            ),
        )
        assertFalse(
            ShaderGraphicsScreen.tierAvailable(
                RenderTier.TIER_3_PATH_TRACING,
                allRtCaps,
                macOsMoltenVk = true,
            ),
        )
    }

    @Test
    fun `vendor upscalers and denoisers disabled when their library bit is absent`() {
        assertTrue(ShaderGraphicsScreen.upscalerAvailable(UpscalerMode.TSR, 0))
        assertFalse(ShaderGraphicsScreen.upscalerAvailable(UpscalerMode.XESS, 0))
        assertFalse(ShaderGraphicsScreen.upscalerAvailable(UpscalerMode.FSR3, 0))
        assertTrue(ShaderGraphicsScreen.upscalerAvailable(UpscalerMode.FSR3, NativeCoreBridge.CAP_FSR3))
        assertTrue(ShaderGraphicsScreen.upscalerAvailable(UpscalerMode.XESS, NativeCoreBridge.CAP_XESS))
        assertFalse(ShaderGraphicsScreen.denoiserAvailable(DenoiserMode.NRD, 0))
        assertTrue(ShaderGraphicsScreen.denoiserAvailable(DenoiserMode.NRD, NativeCoreBridge.CAP_NRD))
        assertFalse(ShaderGraphicsScreen.frameGenAvailable(FrameGenMode.DLSS_FG, 0))
        assertTrue(ShaderGraphicsScreen.frameGenAvailable(FrameGenMode.DLSS_FG, NativeCoreBridge.CAP_DLSS_FG))
    }

    @Test
    fun `cycling skips unavailable modes`() {
        val caps = 0
        // Tier 1 -> Tier 0 -> Tier 1 (RT tiers skipped).
        assertEquals(RenderTier.TIER_0_RASTER, nextTier(RenderTier.TIER_1_COMPUTE_GI, caps))
        // TSR -> NONE -> TSR (vendor modes skipped).
        assertEquals(UpscalerMode.NONE, nextUpscaler(UpscalerMode.TSR, caps))
    }

    private fun nextTier(current: RenderTier, caps: Int): RenderTier {
        val enabled = RenderTier.entries.filter { ShaderGraphicsScreen.tierAvailable(it, caps) }
        return enabled[(enabled.indexOf(current) + 1) % enabled.size]
    }

    private fun nextUpscaler(current: UpscalerMode, caps: Int): UpscalerMode {
        val enabled = UpscalerMode.entries.filter { ShaderGraphicsScreen.upscalerAvailable(it, caps) }
        return enabled[(enabled.indexOf(current) + 1) % enabled.size]
    }
}
