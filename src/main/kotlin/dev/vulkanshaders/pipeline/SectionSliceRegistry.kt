package dev.vulkanshaders.pipeline

import com.mojang.renderpearl.api.buffers.GpuBuffer
import com.mojang.renderpearl.backend.vulkan.VulkanGpuBuffer
import dev.vulkanshaders.backend.NativeCoreBridge
import dev.vulkanshaders.bridge.VulkanShadersTurboExtension
import dev.vulkanshaders.texture.TextureRegistry
import net.minecraft.client.renderer.chunk.ChunkSectionLayer
import java.util.Collections
import java.util.WeakHashMap
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicLong

/**
 * 4-bit material IDs (`0..15`, bits `3..6`) and 3 bit flags (`bits 0..2`) packed into the 7-bit
 * unsigned mask (`0..127`) carried by `RGBA8_SNORM` `Normal.w`, section `InstanceMetaGpu` records,
 * and `gbuffer_terrain_region.vsh/.fsh`.
 */
object SectionMaterialPacking {
    const val FLAG_FOLIAGE_WIND: Int = 1 shl 0
    const val FLAG_FLUID: Int = 1 shl 1
    const val FLAG_EMISSIVE: Int = 1 shl 2

    const val ID_STONE: Int = 1
    const val ID_DIRT: Int = 2
    const val ID_WOOD: Int = 3
    const val ID_FOLIAGE: Int = 4
    const val ID_SAND: Int = 5
    const val ID_METAL: Int = 6
    const val ID_GLASS: Int = 7
    const val ID_WATER: Int = 8
    const val ID_LAVA: Int = 9
    const val ID_EMISSIVE: Int = 10
    const val ID_WOOL: Int = 11
    const val ID_GEM_CRYSTAL: Int = 12
    const val ID_SNOW_ORGANIC: Int = 13
    const val ID_POLISHED_CERAMIC: Int = 14
    const val ID_SCULK_MAGICAL: Int = 15
    const val DEFAULT_MATERIAL: Int = ID_STONE shl 3

    fun pack(id: Int, flags: Int): Int =
        ((flags and 0x07) or ((id and 0x0F) shl 3)) and 0x7F
}

/**
 * Tracks live terrain section vertex slices uploaded to `UberGpuBuffer` heaps
 * so both raster passes and Tier 2/3 zero-copy BLAS builders share exact `VkBuffer` ranges
 * and `RegionOrigin` (`(sectionOrigin & ~63)`) world-space anchors.
 *
 * Assigns a collision-free monotonic 64-bit `allocationId` per live allocation object via a
 * weak identity table instead of `System.identityHashCode(allocationKey)`.
 */
object SectionSliceRegistry {
    data class SliceKey(val allocationId: Long, val layerOrdinal: Int)

    data class SliceRecord(
        val allocationId: Long,
        val vkBuffer: Long,
        val byteOffset: Long,
        val byteSize: Long,
        val layerOrdinal: Int,
        val sectionPosLong: Long,
        val vertexStride: Int,
        val vertexCount: Int,
        val isRegionShifted: Boolean,
    ) {
        /** Block-space section origin `(secX * 16, secY * 16, secZ * 16)`. */
        fun sectionOriginBlocks(): Triple<Int, Int, Int> {
            val sx = (sectionPosLong shr 42).toInt()
            val sy = ((sectionPosLong shl 44) shr 44).toInt()
            val sz = ((sectionPosLong shl 22) shr 42).toInt()
            return Triple(sx shl 4, sy shl 4, sz shl 4)
        }

        /**
         * World-space anchor for TLAS instance placement:
         * `RegionOrigin = sectionOrigin & ~63` when `isRegionShifted == true`
         * (because `(sectionPos & 3) * 16` is already baked into `UberGpuBuffer` vertex positions).
         */
        fun tlasWorldAnchorBlocks(): Triple<Int, Int, Int> {
            val (ox, oy, oz) = sectionOriginBlocks()
            return if (isRegionShifted) {
                Triple(ox and -64, oy and -64, oz and -64)
            } else {
                Triple(ox, oy, oz)
            }
        }
    }

    private val nextAllocationId = AtomicLong(1L shl 62)
    private val allocationIds = Collections.synchronizedMap(WeakHashMap<Any, Long>())
    private val activeSlices = ConcurrentHashMap<SliceKey, SliceRecord>(1024)

    internal fun resolveAllocationId(allocationKey: Any): Long {
        if (allocationKey is Long && allocationKey != 0L) {
            return allocationKey
        }
        return synchronized(allocationIds) {
            allocationIds.getOrPut(allocationKey) { nextAllocationId.getAndIncrement() }
        }
    }

    @JvmStatic
    fun onUploaded(
        allocationKey: Any,
        layerOrdinal: Int,
        gpuBuffer: GpuBuffer,
        byteOffset: Long,
        byteSize: Long,
    ) {
        onUploadedDetailed(
            allocationKey = allocationKey,
            layerOrdinal = layerOrdinal,
            sectionPosLong = 0L,
            gpuBuffer = gpuBuffer,
            byteOffset = byteOffset,
            byteSize = byteSize,
            vertexStride = 40,
            vertexCount = (byteSize / 40L).toInt(),
            isRegionShifted = true,
        )
    }

    @JvmStatic
    fun onUploadedDetailed(
        allocationKey: Any,
        layerOrdinal: Int,
        sectionPosLong: Long,
        gpuBuffer: GpuBuffer,
        byteOffset: Long,
        byteSize: Long,
        vertexStride: Int,
        vertexCount: Int,
        isRegionShifted: Boolean,
    ) {
        val vkBuffer = (gpuBuffer as? VulkanGpuBuffer)?.vkBuffer() ?: 0L
        val allocId = resolveAllocationId(allocationKey)
        val key = SliceKey(allocId, layerOrdinal)
        val stride = vertexStride.coerceAtLeast(20)
        val count = if (vertexCount > 0) vertexCount else (byteSize / stride.toLong()).toInt()

        val record = SliceRecord(
            allocationId = allocId,
            vkBuffer = vkBuffer,
            byteOffset = byteOffset,
            byteSize = byteSize,
            layerOrdinal = layerOrdinal,
            sectionPosLong = sectionPosLong,
            vertexStride = stride,
            vertexCount = count,
            isRegionShifted = isRegionShifted,
        )
        activeSlices[key] = record

        if (vkBuffer != 0L) {
            val slots = TextureRegistry.pbrSlots()
            NativeCoreBridge.registerSectionSliceWithPbr(
                keyId = allocId,
                layerOrdinal = layerOrdinal,
                sectionPosPacked = sectionPosLong,
                vkBuffer = vkBuffer,
                byteOffset = byteOffset,
                byteSize = byteSize,
                vertexStride = stride,
                vertexCount = count,
                isRegionShifted = isRegionShifted,
                albedoSlot = slots[0],
                normalSlot = slots[1],
                specularSlot = slots[2],
                materialFlags = SectionMaterialPacking.DEFAULT_MATERIAL or (1 shl 8) or (
                    if ((layerOrdinal == ChunkSectionLayer.CUTOUT.ordinal ||
                            layerOrdinal == ChunkSectionLayer.TRANSLUCENT.ordinal) &&
                        VulkanShadersTurboExtension.hasSectionFoliage(sectionPosLong)
                    ) {
                        SectionMaterialPacking.FLAG_FOLIAGE_WIND
                    } else {
                        0
                    }
                ),
            )
        }
    }

    @JvmStatic
    fun onFreed(allocationKey: Any, layerOrdinal: Int) {
        val allocId = synchronized(allocationIds) { allocationIds[allocationKey] }
            ?: (allocationKey as? Long)
            ?: return
        val removed = activeSlices.remove(SliceKey(allocId, layerOrdinal))
        if (removed != null && layerOrdinal == ChunkSectionLayer.CUTOUT.ordinal) {
            VulkanShadersTurboExtension.clearSectionFoliageFlag(removed.sectionPosLong)
        }
        // Always release the native slice. Restricting this to Kotlin records that still exist
        // leaked BLAS/TLAS state when the weak registry entry had already been dropped.
        NativeCoreBridge.unregisterSectionSlice(allocId, layerOrdinal)
    }

    @JvmStatic
    fun activeCount(): Int = activeSlices.size

    @JvmStatic
    fun getRecord(allocationKey: Any, layerOrdinal: Int): SliceRecord? {
        val allocId = resolveAllocationId(allocationKey)
        return activeSlices[SliceKey(allocId, layerOrdinal)]
    }

    @JvmStatic
    fun clear() {
        NativeCoreBridge.clearSectionSlices()
        VulkanShadersTurboExtension.clearSectionFoliageFlags()
        activeSlices.clear()
        synchronized(allocationIds) {
            allocationIds.clear()
        }
    }
}
