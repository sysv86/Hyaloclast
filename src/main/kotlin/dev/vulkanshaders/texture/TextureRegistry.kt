package dev.vulkanshaders.texture

import com.mojang.renderpearl.api.textures.GpuSampler
import com.mojang.renderpearl.api.textures.GpuTexture
import com.mojang.renderpearl.api.textures.GpuTextureView
import com.mojang.renderpearl.backend.vulkan.VulkanConst
import com.mojang.renderpearl.backend.vulkan.VulkanGpuSampler
import com.mojang.renderpearl.backend.vulkan.VulkanGpuTexture
import com.mojang.renderpearl.backend.vulkan.VulkanGpuTextureView
import dev.vulkanshaders.VulkanShaders
import dev.vulkanshaders.backend.NativeCoreBridge
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Tracks live `VulkanGpuTexture` and `VulkanGpuTextureView` instances, extracting real
 * `VkImageView` (`VulkanGpuTextureView.vkImageView()`) and `VkSampler` (`VulkanGpuSampler.vkSampler()`)
 * handles for the native Rust `BindlessTextureTable` while filtering transient animation-frame
 * scratch uploads, and maintains per-atlas LabPBR (`_n` / `_s`) companion entries.
 */
object TextureRegistry {
    enum class PbrRole {
        ALBEDO,
        NORMAL,
        SPECULAR,
        OTHER,
    }

    data class TrackedTextureInfo(
        val vkImage: Long,
        val vkImageView: Long = 0L,
        val vkSampler: Long = 0L,
        val vkFormat: Int = 37,
        val label: String,
        val width: Int,
        val height: Int,
        val mipLevels: Int,
        val role: PbrRole,
        val bindlessSlot: Int = -1,
    )

    data class CompanionPbrEntry(
        val albedoVkImage: Long,
        val albedoSlot: Int,
        val normalTexture: GpuTexture?,
        val normalView: GpuTextureView?,
        val normalSampler: GpuSampler?,
        val normalSlot: Int,
        val specularTexture: GpuTexture?,
        val specularView: GpuTextureView?,
        val specularSampler: GpuSampler?,
        val specularSlot: Int,
        val isFlatFallback: Boolean = false,
    ) {
        fun closeResources() {
            runCatching { normalView?.close() }
            runCatching { normalTexture?.close() }
            runCatching { normalSampler?.close() }
            runCatching { specularView?.close() }
            runCatching { specularTexture?.close() }
            runCatching { specularSampler?.close() }
        }
    }

    private val textures = ConcurrentHashMap<Long, TrackedTextureInfo>(256)
    private val companionsByAlbedoImage = ConcurrentHashMap<Long, CompanionPbrEntry>(32)
    private val companionAtlasBuilders = ConcurrentHashMap<String, Runnable>(16)
    private val builtCompanionAtlasKeys: MutableSet<String> = ConcurrentHashMap.newKeySet()
    private val skipScratchUploads = AtomicBoolean(false)

    /** Bindless slots of the blocks atlas and its LabPBR companions (shared by every section slice). */
    @JvmField @Volatile var albedoAtlasBindlessSlot: Int = -1
    @JvmField @Volatile var normalAtlasBindlessSlot: Int = -1
    @JvmField @Volatile var specularAtlasBindlessSlot: Int = -1
    @JvmField @Volatile var blocksAtlasVkImage: Long = 0L

    @JvmField @Volatile var normalAtlasTexture: GpuTexture? = null
    @JvmField @Volatile var normalAtlasView: GpuTextureView? = null
    @JvmField @Volatile var normalAtlasSampler: GpuSampler? = null
    @JvmField @Volatile var specularAtlasTexture: GpuTexture? = null
    @JvmField @Volatile var specularAtlasView: GpuTextureView? = null
    @JvmField @Volatile var specularAtlasSampler: GpuSampler? = null

    @Volatile
    private var companionAtlasBuilder: Runnable? = null

    @JvmStatic
    fun setCompanionAtlasBuilder(builder: Runnable?) {
        companionAtlasBuilder = builder
        if (builder != null) {
            companionAtlasBuilders["minecraft:textures/atlas/blocks.png"] = builder
        } else {
            companionAtlasBuilders.remove("minecraft:textures/atlas/blocks.png")
        }
        builtCompanionAtlasKeys.remove("minecraft:textures/atlas/blocks.png")
    }

    @JvmStatic
    fun registerCompanionAtlasBuilder(atlasKey: String, builder: Runnable?) {
        if (builder != null) {
            companionAtlasBuilders[atlasKey] = builder
        } else {
            companionAtlasBuilders.remove(atlasKey)
        }
        builtCompanionAtlasKeys.remove(atlasKey)
    }

    /**
     * Ensures LabPBR `_n` / `_s` companion atlases (or 1x1 flat fallbacks for the blocks atlas)
     * are built when a shader pack is enabled after initial `TextureAtlas.upload` has completed.
     */
    @JvmStatic
    fun ensureCompanionAtlasesBuilt() {
        if (!VulkanShaders.shaderPackActive) return
        if (companionAtlasBuilders.isNotEmpty()) {
            for ((atlasKey, builder) in companionAtlasBuilders) {
                if (atlasKey in builtCompanionAtlasKeys) continue
                runCatching {
                    builder.run()
                    builtCompanionAtlasKeys.add(atlasKey)
                }
            }
        } else if (normalAtlasView == null || specularAtlasView == null) {
            runCatching { companionAtlasBuilder?.run() }
        }
    }

    /** Registers a LabPBR companion atlas pair for a specific albedo atlas `VkImage`. */
    @JvmStatic
    fun registerCompanionPair(
        albedoVkImage: Long,
        albedoSlot: Int,
        normalTexture: GpuTexture?,
        normalView: GpuTextureView?,
        normalSampler: GpuSampler?,
        normalSlot: Int,
        specularTexture: GpuTexture?,
        specularView: GpuTextureView?,
        specularSampler: GpuSampler?,
        specularSlot: Int,
        isBlocksAtlas: Boolean,
        isFlatFallback: Boolean = false,
    ) {
        val entry = CompanionPbrEntry(
            albedoVkImage = albedoVkImage,
            albedoSlot = albedoSlot,
            normalTexture = normalTexture,
            normalView = normalView,
            normalSampler = normalSampler,
            normalSlot = normalSlot,
            specularTexture = specularTexture,
            specularView = specularView,
            specularSampler = specularSampler,
            specularSlot = specularSlot,
            isFlatFallback = isFlatFallback,
        )
        if (albedoVkImage != 0L) {
            companionsByAlbedoImage.put(albedoVkImage, entry)?.closeResources()
        }
        if (isBlocksAtlas) {
            builtCompanionAtlasKeys.add("minecraft:textures/atlas/blocks.png")
            if (albedoVkImage != 0L) {
                blocksAtlasVkImage = albedoVkImage
            }
            normalAtlasTexture = normalTexture
            normalAtlasView = normalView
            normalAtlasSampler = normalSampler
            normalAtlasBindlessSlot = normalSlot
            specularAtlasTexture = specularTexture
            specularAtlasView = specularView
            specularAtlasSampler = specularSampler
            specularAtlasBindlessSlot = specularSlot
        }
        if (!isFlatFallback && albedoSlot >= 0 && (normalSlot >= 0 || specularSlot >= 0)) {
            NativeCoreBridge.linkPbrCompanions(albedoSlot, normalSlot, specularSlot)
        }
    }

    @JvmStatic
    fun getCompanionForAlbedo(albedoVkImage: Long): CompanionPbrEntry? {
        if (albedoVkImage == 0L) return null
        return companionsByAlbedoImage[albedoVkImage]
    }

    @JvmStatic
    fun getBindlessSlotForImage(vkImage: Long): Int {
        if (vkImage == 0L) return -1
        return textures[vkImage]?.bindlessSlot ?: -1
    }

    @JvmStatic
    @JvmOverloads
    fun closeCompanionForAlbedo(albedoVkImage: Long, isBlocksAtlas: Boolean = false, atlasKey: String? = null) {
        val removed = if (albedoVkImage != 0L) {
            companionsByAlbedoImage.remove(albedoVkImage)
        } else {
            null
        }
        removed?.closeResources()
        when {
            atlasKey != null -> builtCompanionAtlasKeys.remove(atlasKey)
            isBlocksAtlas -> builtCompanionAtlasKeys.remove("minecraft:textures/atlas/blocks.png")
            removed != null -> builtCompanionAtlasKeys.clear()
        }
        if (isBlocksAtlas) {
            normalAtlasTexture = null
            normalAtlasView = null
            normalAtlasSampler = null
            specularAtlasTexture = null
            specularAtlasView = null
            specularAtlasSampler = null
            normalAtlasBindlessSlot = -1
            specularAtlasBindlessSlot = -1
        }
    }

    /** Registers a LabPBR companion atlas (identical sprite UVs to the blocks albedo atlas). */
    @JvmStatic
    fun setCompanionAtlas(
        role: PbrRole,
        texture: GpuTexture,
        view: GpuTextureView,
        sampler: GpuSampler,
        slot: Int,
    ) {
        when (role) {
            PbrRole.NORMAL -> {
                normalAtlasTexture = texture
                normalAtlasView = view
                normalAtlasSampler = sampler
                normalAtlasBindlessSlot = slot
            }
            PbrRole.SPECULAR -> {
                specularAtlasTexture = texture
                specularAtlasView = view
                specularAtlasSampler = sampler
                specularAtlasBindlessSlot = slot
            }
            else -> Unit
        }
    }

    /** Returns `(albedo, normal, specular)` bindless slots for `registerSectionSliceWithPbr`. */
    @JvmStatic
    fun pbrSlots(): IntArray = intArrayOf(albedoAtlasBindlessSlot, normalAtlasBindlessSlot, specularAtlasBindlessSlot)

    @JvmStatic
    fun hasLabPbrCompanions(): Boolean = normalAtlasBindlessSlot >= 0 || specularAtlasBindlessSlot >= 0

    @JvmStatic
    fun setSkipScratchUploads(skip: Boolean) {
        skipScratchUploads.set(skip)
    }

    @JvmStatic
    fun isSkippingScratchUploads(): Boolean = skipScratchUploads.get()

    @JvmStatic
    fun register(texture: VulkanGpuTexture) {
        if (skipScratchUploads.get() || !isSampledTexture(texture)) return
        val label = texture.label ?: ""
        if (shouldSkipTracking(label)) return

        val vkImage = texture.vkImage()
        if (vkImage == 0L) return

        val role = classifyRole(label)
        val width = texture.getWidth(0)
        val height = texture.getHeight(0)
        val mipLevels = texture.mipLevels
        val vkFormat = runCatching { VulkanConst.toVk(texture.format) }.getOrDefault(37)

        val slot = NativeCoreBridge.registerTextureWithSampler(
            vkImage = vkImage,
            vkImageView = 0L,
            vkSampler = 0L,
            vkFormat = vkFormat,
            width = width,
            height = height,
            mipLevels = mipLevels,
            pbrRoleOrdinal = role.ordinal,
        )
        val info = TrackedTextureInfo(
            vkImage = vkImage,
            vkImageView = 0L,
            vkSampler = 0L,
            vkFormat = vkFormat,
            label = label,
            width = width,
            height = height,
            mipLevels = mipLevels,
            role = role,
            bindlessSlot = slot,
        )
        textures[vkImage] = info
    }

    @JvmStatic
    fun onTextureViewCreated(view: VulkanGpuTextureView) {
        if (skipScratchUploads.get() || view.baseMipLevel() != 0) return
        val texture = view.texture()
        if (!isSampledTexture(texture)) return
        val label = texture.label ?: ""
        if (shouldSkipTracking(label)) return

        val vkImage = texture.vkImage()
        val vkImageView = view.vkImageView()
        if (vkImage == 0L || vkImageView == 0L) return

        val existing = textures[vkImage]
        val role = existing?.role ?: classifyRole(label)
        val width = texture.getWidth(0)
        val height = texture.getHeight(0)
        val mipLevels = texture.mipLevels
        val vkFormat = existing?.vkFormat ?: runCatching { VulkanConst.toVk(texture.format) }.getOrDefault(37)
        val vkSampler = existing?.vkSampler ?: 0L

        val slot = NativeCoreBridge.registerTextureWithSampler(
            vkImage = vkImage,
            vkImageView = vkImageView,
            vkSampler = vkSampler,
            vkFormat = vkFormat,
            width = width,
            height = height,
            mipLevels = mipLevels,
            pbrRoleOrdinal = role.ordinal,
        )
        textures[vkImage] = TrackedTextureInfo(
            vkImage = vkImage,
            vkImageView = vkImageView,
            vkSampler = vkSampler,
            vkFormat = vkFormat,
            label = label,
            width = width,
            height = height,
            mipLevels = mipLevels,
            role = role,
            bindlessSlot = slot,
        )
    }

    @JvmStatic
    fun onTextureViewAndSamplerBound(
        texture: GpuTexture?,
        textureView: GpuTextureView?,
        sampler: GpuSampler?,
        isBlocksAtlas: Boolean,
    ) {
        val vkTex = texture as? VulkanGpuTexture ?: return
        if (!isSampledTexture(vkTex)) return
        val vkImage = vkTex.vkImage()
        if (vkImage == 0L) return

        val label = vkTex.label ?: ""
        if (shouldSkipTracking(label)) return

        val vkImageView = (textureView as? VulkanGpuTextureView)?.vkImageView() ?: 0L
        val vkSampler = (sampler as? VulkanGpuSampler)?.vkSampler() ?: 0L
        val vkFormat = runCatching { VulkanConst.toVk(vkTex.format) }.getOrDefault(37)
        val role = classifyRole(label)
        val width = vkTex.getWidth(0)
        val height = vkTex.getHeight(0)
        val mipLevels = vkTex.mipLevels

        val slot = NativeCoreBridge.registerTextureWithSampler(
            vkImage = vkImage,
            vkImageView = vkImageView,
            vkSampler = vkSampler,
            vkFormat = vkFormat,
            width = width,
            height = height,
            mipLevels = mipLevels,
            pbrRoleOrdinal = role.ordinal,
        )
        textures[vkImage] = TrackedTextureInfo(
            vkImage = vkImage,
            vkImageView = vkImageView,
            vkSampler = vkSampler,
            vkFormat = vkFormat,
            label = label,
            width = width,
            height = height,
            mipLevels = mipLevels,
            role = role,
            bindlessSlot = slot,
        )
        if (isBlocksAtlas) {
            albedoAtlasBindlessSlot = slot
            blocksAtlasVkImage = vkImage
        }
    }

    @JvmStatic
    fun bindlessSlot(view: GpuTextureView, sampler: GpuSampler): Int {
        val vkView = view as? VulkanGpuTextureView ?: return -1
        val vkSampler = sampler as? VulkanGpuSampler ?: return -1
        val texture = vkView.texture()
        val vkImage = texture.vkImage()
        val current = textures[vkImage]
        if (current != null && current.bindlessSlot >= 0 &&
            current.vkImageView == vkView.vkImageView() && current.vkSampler == vkSampler.vkSampler()
        ) {
            return current.bindlessSlot
        }
        onTextureViewAndSamplerBound(texture, view, sampler, isBlocksAtlas = false)
        return textures[vkImage]?.bindlessSlot ?: -1
    }

    @JvmStatic
    fun closeCompanionAtlases() {
        builtCompanionAtlasKeys.clear()
        val closedImages = HashSet<GpuTexture>()
        val iter = companionsByAlbedoImage.entries.iterator()
        while (iter.hasNext()) {
            val entry = iter.next().value
            iter.remove()
            entry.normalTexture?.let { closedImages.add(it) }
            entry.specularTexture?.let { closedImages.add(it) }
            entry.closeResources()
        }
        normalAtlasTexture?.takeIf { it !in closedImages }?.let {
            runCatching { normalAtlasView?.close() }
            runCatching { it.close() }
            runCatching { normalAtlasSampler?.close() }
        }
        specularAtlasTexture?.takeIf { it !in closedImages }?.let {
            runCatching { specularAtlasView?.close() }
            runCatching { it.close() }
            runCatching { specularAtlasSampler?.close() }
        }
        normalAtlasTexture = null
        normalAtlasView = null
        normalAtlasSampler = null
        specularAtlasTexture = null
        specularAtlasView = null
        specularAtlasSampler = null
        normalAtlasBindlessSlot = -1
        specularAtlasBindlessSlot = -1
    }

    @JvmStatic
    fun unregister(texture: VulkanGpuTexture) {
        val vkImage = texture.vkImage()
        if (vkImage == 0L) return
        companionsByAlbedoImage.remove(vkImage)?.closeResources()
        val removed = textures.remove(vkImage)
        if (removed != null) {
            if (removed.bindlessSlot == albedoAtlasBindlessSlot) albedoAtlasBindlessSlot = -1
            if (removed.bindlessSlot == normalAtlasBindlessSlot) normalAtlasBindlessSlot = -1
            if (removed.bindlessSlot == specularAtlasBindlessSlot) specularAtlasBindlessSlot = -1
            if (vkImage == blocksAtlasVkImage) blocksAtlasVkImage = 0L
            NativeCoreBridge.unregisterTexture(vkImage)
        }
    }

    internal fun shouldSkipTracking(label: String): Boolean {
        if (label.contains(" animation frame ")) return true
        if (label.contains("staging") || label.contains("scratch")) return true
        return false
    }

    private fun isSampledTexture(texture: GpuTexture): Boolean =
        (texture.usage() and GpuTexture.USAGE_TEXTURE_BINDING) != 0

    internal fun classifyRole(label: String): PbrRole {
        val lower = label.lowercase()
        return when {
            lower.endsWith("_n") || lower.contains("_n.") || lower.contains("normal atlas") -> PbrRole.NORMAL
            lower.endsWith("_s") || lower.contains("_s.") || lower.contains("specular atlas") -> PbrRole.SPECULAR
            lower.contains("atlas") || lower.contains("blocks") || lower.contains("entity") -> PbrRole.ALBEDO
            else -> PbrRole.OTHER
        }
    }

    @JvmStatic
    fun count(): Int = textures.size

    @JvmStatic
    fun snapshot(): Collection<TrackedTextureInfo> = textures.values
}
