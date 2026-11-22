package dev.vulkanshaders.texture

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertFalse
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test

class TextureRegistryTest {
    @Test
    fun `filters transient animation frames and scratch textures`() {
        assertTrue(TextureRegistry.shouldSkipTracking("minecraft:textures/atlas/blocks.png animation frame 12"))
        assertTrue(TextureRegistry.shouldSkipTracking("atlas_scratch_upload"))
        assertFalse(TextureRegistry.shouldSkipTracking("minecraft:textures/atlas/blocks.png"))
    }

    @Test
    fun `classifies PBR texture roles by suffix`() {
        assertEquals(TextureRegistry.PbrRole.NORMAL, TextureRegistry.classifyRole("blocks_n.png"))
        assertEquals(TextureRegistry.PbrRole.SPECULAR, TextureRegistry.classifyRole("blocks_s.png"))
        assertEquals(TextureRegistry.PbrRole.ALBEDO, TextureRegistry.classifyRole("textures/atlas/blocks.png"))
    }
}
