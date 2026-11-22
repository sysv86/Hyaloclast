package dev.vulkanshaders.pipeline

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test

/** Verifies the 7-bit material-ID bit layout (`0..127`) shared with `Normal.w`, `gbuffer_terrain_region.vsh/.fsh`, and `InstanceMetaGpu`. */
class MaterialPackingTest {
    @Test
    fun `material flags pack 4-bit id into bits 3 through 6 within 7-bit SNORM range`() {
        assertEquals(SectionMaterialPacking.ID_STONE shl 3, SectionMaterialPacking.pack(1, 0))
        assertEquals((4 shl 3) or 1, SectionMaterialPacking.pack(SectionMaterialPacking.ID_FOLIAGE, SectionMaterialPacking.FLAG_FOLIAGE_WIND))
        assertEquals((8 shl 3) or 2, SectionMaterialPacking.pack(SectionMaterialPacking.ID_WATER, SectionMaterialPacking.FLAG_FLUID))
        assertEquals((9 shl 3) or 6, SectionMaterialPacking.pack(SectionMaterialPacking.ID_LAVA, SectionMaterialPacking.FLAG_FLUID or SectionMaterialPacking.FLAG_EMISSIVE))
        assertEquals((10 shl 3) or 4, SectionMaterialPacking.pack(SectionMaterialPacking.ID_EMISSIVE, SectionMaterialPacking.FLAG_EMISSIVE))
        assertEquals(11 shl 3, SectionMaterialPacking.pack(11, 0))
        assertEquals(0, SectionMaterialPacking.pack(11, 0) and 0x07)
        for (id in 0..15) {
            for (flags in 0..7) {
                val packed = SectionMaterialPacking.pack(id, flags)
                assertTrue(packed in 0..127, "Packed material mask must fit in 7-bit unsigned SNORM range (0..127)")
                assertEquals(id, (packed ushr 3) and 0x0F)
                assertEquals(flags, packed and 0x07)
            }
        }
    }

    @Test
    fun `extended material IDs 12 through 15 preserve existing IDs and support FLAG_EMISSIVE composition`() {
        assertEquals(7, SectionMaterialPacking.ID_GLASS)
        assertEquals(8, SectionMaterialPacking.ID_WATER)
        assertEquals(9, SectionMaterialPacking.ID_LAVA)
        assertEquals(10, SectionMaterialPacking.ID_EMISSIVE)
        assertEquals(11, SectionMaterialPacking.ID_WOOL)

        assertEquals(12, SectionMaterialPacking.ID_GEM_CRYSTAL)
        assertEquals(13, SectionMaterialPacking.ID_SNOW_ORGANIC)
        assertEquals(14, SectionMaterialPacking.ID_POLISHED_CERAMIC)
        assertEquals(15, SectionMaterialPacking.ID_SCULK_MAGICAL)

        val extendedIds = listOf(
            SectionMaterialPacking.ID_GEM_CRYSTAL,
            SectionMaterialPacking.ID_SNOW_ORGANIC,
            SectionMaterialPacking.ID_POLISHED_CERAMIC,
            SectionMaterialPacking.ID_SCULK_MAGICAL,
        )
        for (id in extendedIds) {
            val basePacked = SectionMaterialPacking.pack(id, 0)
            assertTrue(basePacked in 0..127)
            assertEquals(id, (basePacked ushr 3) and 0x0F)
            assertEquals(0, basePacked and 0x07)

            val emissivePacked = SectionMaterialPacking.pack(id, SectionMaterialPacking.FLAG_EMISSIVE)
            assertTrue(emissivePacked in 0..127)
            assertEquals(id, (emissivePacked ushr 3) and 0x0F)
            assertEquals(SectionMaterialPacking.FLAG_EMISSIVE, emissivePacked and 0x07)
            assertEquals(0, emissivePacked and SectionMaterialPacking.FLAG_FLUID)
        }
    }

    @Test
    fun `untagged section faces use the solid material default`() {
        assertEquals(
            SectionMaterialPacking.pack(SectionMaterialPacking.ID_STONE, 0),
            SectionMaterialPacking.DEFAULT_MATERIAL,
        )
    }
}
