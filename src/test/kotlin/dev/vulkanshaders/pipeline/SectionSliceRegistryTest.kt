package dev.vulkanshaders.pipeline

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertNotEquals
import org.junit.jupiter.api.Test

class SectionSliceRegistryTest {
    @Test
    fun `computes RegionOrigin TLAS anchor matching VulkanTurbo 4x4x4 region shift`() {
        for (secX in -9..9) {
            for (secY in -4..8) {
                for (secZ in -9..9) {
                    val packed = ((secX.toLong() and 0x3FFFFFL) shl 42) or
                        ((secZ.toLong() and 0x3FFFFFL) shl 20) or
                        (secY.toLong() and 0xFFFFFL)

                    val recShifted = SectionSliceRegistry.SliceRecord(
                        allocationId = 1L,
                        vkBuffer = 100L,
                        byteOffset = 280L,
                        byteSize = 1600L,
                        layerOrdinal = 0,
                        sectionPosLong = packed,
                        vertexStride = 40,
                        vertexCount = 40,
                        isRegionShifted = true,
                    )

                    val (ox, oy, oz) = recShifted.sectionOriginBlocks()
                    val (rx, ry, rz) = recShifted.tlasWorldAnchorBlocks()
                    val bakedShiftX = (secX and 3) * 16
                    val bakedShiftY = (secY and 3) * 16
                    val bakedShiftZ = (secZ and 3) * 16

                    assertEquals(secX * 16, ox)
                    assertEquals(secY * 16, oy)
                    assertEquals(secZ * 16, oz)
                    assertEquals(ox, rx + bakedShiftX)
                    assertEquals(oy, ry + bakedShiftY)
                    assertEquals(oz, rz + bakedShiftZ)
                }
            }
        }
    }

    @Test
    fun `resolveAllocationId assigns unique 64-bit IDs without identityHashCode collisions`() {
        SectionSliceRegistry.clear()
        val keyA = Any()
        val keyB = Any()
        val idA = SectionSliceRegistry.resolveAllocationId(keyA)
        val idB = SectionSliceRegistry.resolveAllocationId(keyB)
        assertNotEquals(idA, idB)
        assertEquals(idA, SectionSliceRegistry.resolveAllocationId(keyA))
    }
}
