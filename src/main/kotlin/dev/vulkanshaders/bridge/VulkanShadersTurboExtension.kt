package dev.vulkanshaders.bridge

import com.mojang.blaze3d.vertex.QuadInstance
import com.mojang.renderpearl.api.GpuFormat
import com.mojang.renderpearl.api.buffers.GpuBuffer
import com.mojang.renderpearl.api.commands.RenderPass
import com.mojang.renderpearl.api.pipeline.RenderPipeline
import com.mojang.renderpearl.api.vertex.VertexFormat
import com.mojang.renderpearl.backend.vulkan.VulkanGpuBuffer
import com.mojang.renderpearl.backend.vulkan.VulkanGpuTexture
import com.mojang.renderpearl.backend.vulkan.VulkanGpuTextureView
import dev.vulkanturbo.api.shader.SwapchainConfigOverride
import dev.vulkanturbo.api.shader.TurboShadersBridge
import dev.vulkanturbo.api.shader.TurboShadersExtension
import dev.vulkanturbo.core.TerrainRegionPipelines
import dev.vulkanshaders.VulkanShaders
import dev.vulkanshaders.backend.NativeCoreBridge
import dev.vulkanshaders.config.FrameGenMode
import dev.vulkanshaders.config.LightingMode
import dev.vulkanshaders.backend.VulkanDeviceContext
import dev.vulkanshaders.diagnostics.HyaloclastErrorLog
import dev.vulkanturbo.gui.VulkanErrorLogScreen
import dev.vulkanshaders.pipeline.GbufferPipelineRegistry
import dev.vulkanshaders.pipeline.GbufferTargetManager
import dev.vulkanshaders.pipeline.SectionMaterialPacking
import dev.vulkanshaders.pipeline.SectionSliceRegistry
import dev.vulkanshaders.state.WorldStateCollector
import net.minecraft.client.Minecraft
import net.minecraft.client.Options
import net.minecraft.client.gui.screens.Screen
import net.minecraft.client.renderer.chunk.ChunkSectionLayer
import net.minecraft.client.renderer.texture.TextureAtlas
import net.minecraft.client.resources.model.geometry.BakedQuad
import net.minecraft.core.BlockPos
import net.minecraft.tags.BlockTags
import net.minecraft.tags.FluidTags
import net.minecraft.world.level.block.state.BlockState
import net.minecraft.world.level.material.FluidState
import org.lwjgl.system.MemoryUtil
import org.lwjgl.vulkan.VK10.VK_IMAGE_LAYOUT_GENERAL
import org.lwjgl.vulkan.VK10.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
import org.lwjgl.vulkan.VK10.VK_IMAGE_LAYOUT_UNDEFINED
import org.lwjgl.vulkan.VkCommandBuffer
import org.lwjgl.vulkan.VkImageBlit
import java.lang.Float.floatToFloat16

private const val DEFERRED_TIER_2_ACTIVATION_FRAMES: Int = 3

/**
 * `TurboShadersExtension` SPI implementation connecting **VulkanShaders (Hyaloclast / IGNIS)**
 * to **VulkanTurbo (Pyroclast / TEPHRA)**.
 *
 * Architectural flow:
 * 1. Uses a 40-byte terrain vertex layout (`IGNIS_TERRAIN_40B`) whose first 12 bytes remain
 *    `RGB32_FLOAT` local position so the same `UberGpuBuffer` slices feed both rasterization
 *    and Tier 2/3 hardware BLAS builds without vertex duplication.
 * 2. Executes `recordFrameRtx` (BLAS/TLAS + wind + SHaRC) and `executeRenderGraph` (RayQuery /
 *    PathTracing + SVGF + TSR + Tonemapping + `depth_blit`) in `onAfterWorldRenderBeforeGui(...)`
 *    **immediately after `LevelRenderer.renderLevel()` and BEFORE `Gui.render()`** while
 *    `mainRenderTarget` still contains pure HUD-less world color and un-cleared world depth.
 * 3. Executes post-GUI passes (`HdrCompose` and `FrameGenInterpolate`) in `notifyAfterGuiRender(...)`
 *    immediately after `GuiRenderer.render()` draws the HUD into `mainColorTexture`, preserving
 *    pure HUD-less history in `RES_PREV_HUDLESS_COLOR` and crisp HUD in `RES_UI_COLOR`.
 */
object VulkanShadersTurboExtension : TurboShadersExtension {
    override val engineId: String = VulkanShaders.MOD_ID
    override val displayName: String = VulkanShaders.PUBLIC_NAME

    override fun buildCrashDiagnosticSection(): String = HyaloclastErrorLog.buildHyaloclastDiagnosticBlock()

    override fun drainDiagnosticLogs() = NativeCoreBridge.drainNativeErrorsToLog()

    override fun createLogScreen(parent: Screen?, options: Options): Screen = VulkanErrorLogScreen(parent, options)

    override fun needsAfterGuiPass(): Boolean {
        if (!NativeCoreBridge.initialized) return false
        if (hdrSwapchainActive) return true
        val cfg = VulkanShaders.config
        return VulkanShaders.shaderPackActive &&
            (cfg.hdrEnabled || cfg.frameGenMode != FrameGenMode.NONE)
    }

    /** Syncs logical device RT capabilities from NativeCoreBridge into Pyroclast's bridge flags once per device init. */
    fun syncCapabilitiesToBridge() {
        if (capabilitiesSynced) return
        val vkDev = VulkanDeviceContext.vkDevice
        if (vkDev == 0L || !StandaloneRenderPearlBridge.vulkanTurboLoaded) return
        val capsMask = NativeCoreBridge.getCapabilitiesMask(vkDev)
        runCatching {
            TurboShadersBridge.accelStructSupported =
                VulkanDeviceContext.supportsAccelerationStructure ||
                ((capsMask and NativeCoreBridge.CAP_RAY_TRACING_AS) != 0)
            TurboShadersBridge.rayTracingPipelineSupported =
                VulkanDeviceContext.supportsRayTracingPipeline ||
                ((capsMask and NativeCoreBridge.CAP_RAY_TRACING_PIPELINE) != 0)
            capabilitiesSynced = true
        }
    }

    /** Resets cached per-device bridge sync flags on device shutdown. */
    fun onDeviceShutdown() {
        capabilitiesSynced = false
        lastDhBoundActive = false
    }

    private const val MATERIAL_FLAG_FOLIAGE_WIND: Int = 1 shl 0
    private const val MATERIAL_FLAG_FLUID: Int = 1 shl 1
    private const val MATERIAL_FLAG_EMISSIVE: Int = 1 shl 2
    private const val MATERIAL_ID_SHIFT: Int = 3
    private const val MATERIAL_ID_MASK: Int = 0x78

    // 4-bit material IDs (bits 3..6 of the 7-bit SNORM mask), shared with gbuffer_terrain_region.vsh/.fsh.
    private const val MID_STONE = 1
    private const val MID_DIRT = 2
    private const val MID_WOOD = 3
    private const val MID_FOLIAGE = 4
    private const val MID_SAND = 5
    private const val MID_METAL = 6
    private const val MID_GLASS = 7
    private const val MID_WATER = 8
    private const val MID_LAVA = 9
    private const val MID_EMISSIVE = 10
    private const val MID_WOOL = SectionMaterialPacking.ID_WOOL
    private const val MID_GEM_CRYSTAL = SectionMaterialPacking.ID_GEM_CRYSTAL
    private const val MID_SNOW_ORGANIC = SectionMaterialPacking.ID_SNOW_ORGANIC
    private const val MID_POLISHED_CERAMIC = SectionMaterialPacking.ID_POLISHED_CERAMIC
    private const val MID_SCULK_MAGICAL = SectionMaterialPacking.ID_SCULK_MAGICAL

    private val renderGraphDisabled: Boolean = System.getenv("VS_DISABLE_RENDER_GRAPH") == "1"

    @Volatile
    private var capabilitiesSynced: Boolean = false

    @Volatile
    private var lastDhBoundActive: Boolean = false

    @Volatile
    private var loggedFirstGraphExec: Boolean = false

    @Volatile
    private var loggedHdrUnsupported: Boolean = false

    private const val VK_FORMAT_A2B10G10R10_UNORM_PACK32: Int = 64
    private const val VK_FORMAT_R16G16B16A16_SFLOAT: Int = 97
    private const val VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT: Int = 1000104002
    private const val VK_COLOR_SPACE_HDR10_ST2084_EXT: Int = 1000104008
    private const val VK_IMAGE_USAGE_STORAGE_BIT: Int = 0x00000008

    @Volatile
    private var hdrSwapchainActive: Boolean = false
    private val sectionFoliageFlags = java.util.concurrent.ConcurrentHashMap<Long, Boolean>(512)

    /**
     * 40-byte terrain vertex format:
     * - `0..11`  (12B): `Position` (`RGB32_FLOAT`)
     * - `12..15` (4B):  `Color` (`RGBA8_UNORM`)
     * - `16..23` (8B):  `UV0` (`RG32_FLOAT`)
     * - `24..27` (4B):  `UV2` (`RG16_SINT`)
     * - `28..31` (4B):  `Normal` (`RGBA8_SNORM` — xyz normal, w packed 7-bit material mask)
     * - `32..35` (4B):  `Tangent` (`RGBA8_SNORM` — xyz tangent, w handedness)
     * - `36..39` (4B):  `MidUV` (`RG16_FLOAT` — half2 sprite center)
     */
    @JvmField
    val IGNIS_TERRAIN_40B: VertexFormat = VertexFormat.builder(0)
        .addAttribute("Position", GpuFormat.RGB32_FLOAT)
        .addAttribute("Color", GpuFormat.RGBA8_UNORM)
        .addAttribute("UV0", GpuFormat.RG32_FLOAT)
        .addAttribute("UV2", GpuFormat.RG16_SINT)
        .addAttribute("Normal", GpuFormat.RGBA8_SNORM)
        .addAttribute("Tangent", GpuFormat.RGBA8_SNORM)
        .addAttribute("MidUV", GpuFormat.RG16_FLOAT)
        .build()

    private class TesselationContext {
        var blockMaterialFlags: Int = 0
        var fluidMaterialFlags: Int = 0
        var currentSectionPosLong: Long = Long.MIN_VALUE
        var sectionHasFoliage: Boolean = false
    }

    fun hasSectionFoliage(sectionPosLong: Long): Boolean =
        sectionFoliageFlags[sectionPosLong] == true

    fun clearSectionFoliageFlag(sectionPosLong: Long) {
        sectionFoliageFlags.remove(sectionPosLong)
    }

    fun clearSectionFoliageFlags() {
        sectionFoliageFlags.clear()
    }

    private val threadContext = ThreadLocal.withInitial { TesselationContext() }
    private val blockBaseMaterialCache = java.util.concurrent.ConcurrentHashMap<net.minecraft.world.level.block.Block, Int>(256)

    @Volatile
    private var worldGraphExecutedThisFrame: Boolean = false

    @Volatile
    private var deferredTier2Frames: Int = 0

    @JvmField
    var capturingWorldFeatures: Boolean = false

    @JvmField
    var capturingFirstPersonAvatar: Boolean = false

    override fun isShaderPackActive(): Boolean = VulkanShaders.shaderPackActive

    fun shouldRenderPathTracedWeather(): Boolean =
        worldGraphExecutedThisFrame && shouldSkipWorldRasterization() && VulkanShaders.config.enableOverlayPass

    fun shouldCaptureWorldFeatures(): Boolean {
        if (!VulkanShaders.shaderPackActive) return false
        val tier = VulkanShaders.config.renderTier
        return tier == LightingMode.HYBRID || tier == LightingMode.PATH
    }

    override fun shouldSkipWorldRasterization(): Boolean =
        VulkanShaders.shaderPackActive &&
            VulkanShaders.config.renderTier == LightingMode.PATH

    override fun shouldSuppressEntityBlobShadows(): Boolean {
        if (!VulkanShaders.shaderPackActive) return false
        return VulkanShaders.config.renderTier == LightingMode.HYBRID
    }

    override fun resolveTerrainVertexFormat(layer: ChunkSectionLayer, defaultFormat: VertexFormat): VertexFormat =
        if (usesExtendedBlockVertexFormat()) IGNIS_TERRAIN_40B else defaultFormat

    override fun acceptsRegionMultiDrawOverride(
        layer: ChunkSectionLayer,
        multidrawOverride: RenderPipeline?,
    ): Boolean = VulkanShaders.shaderPackActive && multidrawOverride == null

    override fun resolveRegionTerrainPipeline(
        layer: ChunkSectionLayer,
        multidrawOverride: RenderPipeline?,
        defaultPipeline: RenderPipeline,
    ): RenderPipeline = GbufferPipelineRegistry.resolveTerrainPipeline(layer, multidrawOverride, defaultPipeline)

    override fun resolveWorldPipeline(defaultPipeline: RenderPipeline): RenderPipeline =
        GbufferPipelineRegistry.resolveWorldPipeline(defaultPipeline)

    override fun onBeginRegionTerrainPass(renderPass: RenderPass, layer: ChunkSectionLayer) {
        GbufferTargetManager.onBeginTerrainPass(renderPass, layer)
    }

    override fun onAfterRegionPipelineBound(renderPass: RenderPass, layer: ChunkSectionLayer) {
        GbufferTargetManager.onTerrainPipelineBound(renderPass)
    }

    override fun onBeginNonTerrainWorldPass(renderPass: RenderPass) {
        GbufferTargetManager.onBeginWorldPass(renderPass)
    }

    override fun getCustomDescriptorSetLayouts(vkDevice: Long): LongArray {
        if (!VulkanShaders.shaderPackActive) return LongArray(0)
        return NativeCoreBridge.getCustomDescriptorSetLayouts()
    }

    override fun onBindCustomDescriptorSetsWithFirstSet(
        vkCommandBuffer: Long,
        vkPipelineLayout: Long,
        pipelineBindPoint: Int,
        passKind: Int,
        firstSet: Int,
    ) {
        if (!VulkanShaders.shaderPackActive) return
        NativeCoreBridge.bindCustomDescriptorSets(vkCommandBuffer, vkPipelineLayout, pipelineBindPoint, firstSet = firstSet)
    }

    override fun onBindCustomDescriptorSets(
        vkCommandBuffer: Long,
        vkPipelineLayout: Long,
        pipelineBindPoint: Int,
        passKind: Int,
    ) {
        if (!VulkanShaders.shaderPackActive) return
        val firstSet = TurboShadersBridge.getFirstSetForPipelineLayout(vkPipelineLayout)
        NativeCoreBridge.bindCustomDescriptorSets(vkCommandBuffer, vkPipelineLayout, pipelineBindPoint, firstSet = firstSet)
    }

    override fun usesExtendedBlockVertexFormat(): Boolean =
        VulkanShaders.shaderPackActive &&
            !GbufferPipelineRegistry.isLegacyRasterRequested() &&
            (GbufferPipelineRegistry.isReady() || TerrainRegionPipelines.solidExtendedPipeline != null)

    override fun onBeginBlockTesselation(blockState: BlockState, pos: BlockPos) {
        val ctx = threadContext.get()
        ctx.currentSectionPosLong = net.minecraft.core.SectionPos.asLong(pos.x shr 4, pos.y shr 4, pos.z shr 4)
        val id = classifyBlockMaterial(blockState)
        var flags = 0
        if (id == MID_FOLIAGE) {
            flags = flags or MATERIAL_FLAG_FOLIAGE_WIND
            ctx.sectionHasFoliage = true
        }
        val emission = blockState.lightEmission
        if (emission > 0) {
            flags = flags or MATERIAL_FLAG_EMISSIVE
        }
        ctx.blockMaterialFlags = packMaterial(id, flags)
        ctx.fluidMaterialFlags = 0
    }

    override fun onBeginFluidTesselation(fluidState: FluidState, blockState: BlockState, pos: BlockPos) {
        threadContext.get().fluidMaterialFlags = if (fluidState.`is`(FluidTags.LAVA)) {
            packMaterial(MID_LAVA, MATERIAL_FLAG_FLUID or MATERIAL_FLAG_EMISSIVE)
        } else {
            packMaterial(MID_WATER, MATERIAL_FLAG_FLUID)
        }
    }

    /** Packs the 4-bit material ID (`0..15`) into bits 3..6 alongside the three bit flags (`0..127`). */
    internal fun packMaterial(id: Int, flags: Int): Int = SectionMaterialPacking.pack(id, flags)

    /** Classifies a block into the 15 material IDs shared with `gbuffer_terrain_region.fsh`. */
    internal fun classifyBlockMaterial(state: BlockState): Int {
        val block = state.block
        val cachedBase = blockBaseMaterialCache[block] ?: run {
            val blockPath = try {
                net.minecraft.core.registries.BuiltInRegistries.BLOCK.getKey(block).path
            } catch (_: Throwable) {
                ""
            }
            val isOreOrRawOrOxidized = blockPath.contains("ore") ||
                blockPath.contains("raw_") ||
                blockPath.contains("oxidized") ||
                blockPath.contains("weathered")
            val isBambooWood = blockPath.startsWith("bamboo_") && blockPath != "bamboo_sapling"
            val computed = when {
                blockPath.contains("sculk") || blockPath.contains("enchanting_table") ||
                    blockPath.contains("end_portal") || blockPath.contains("ender_chest") ||
                    blockPath.contains("respawn_anchor") || blockPath.contains("conduit") ||
                    blockPath.contains("nether_portal") || blockPath.contains("dragon_egg") ||
                    blockPath.contains("trial_spawner") || blockPath.contains("vault") ||
                    blockPath.contains("spawner") || blockPath.contains("beacon") -> MID_SCULK_MAGICAL
                state.`is`(BlockTags.LEAVES) || state.`is`(BlockTags.SAPLINGS) ||
                    state.`is`(BlockTags.FLOWERS) || state.`is`(BlockTags.SMALL_FLOWERS) ||
                    state.`is`(BlockTags.CROPS) ||
                    (!isBambooWood && (
                        blockPath == "bamboo" || blockPath == "bamboo_sapling" ||
                            blockPath.contains("vine") || blockPath.contains("fern") ||
                            blockPath.contains("seagrass") || blockPath.contains("kelp") ||
                            blockPath.contains("short_grass") || blockPath.contains("tall_grass") ||
                            blockPath.contains("moss") || blockPath.contains("azalea") ||
                            blockPath.contains("dripleaf") || blockPath.contains("lily_pad") ||
                            blockPath.contains("sugar_cane") || blockPath.contains("cactus") ||
                            blockPath.contains("sweet_berry") || blockPath.contains("glow_lichen") ||
                            blockPath.contains("hanging_roots") || blockPath.contains("spore_blossom") ||
                            blockPath.contains("mangrove_roots") || blockPath.contains("bush") ||
                            blockPath.contains("pitcher") || blockPath.contains("torchflower") ||
                            blockPath.contains("leaf_litter")
                        )) -> MID_FOLIAGE
                blockPath.contains("glowstone") || blockPath.contains("torch") ||
                    blockPath.contains("lantern") || blockPath.contains("shroomlight") ||
                    blockPath.contains("froglight") || blockPath.contains("redstone_lamp") ||
                    blockPath.contains("magma") ||
                    state.`is`(BlockTags.FIRE) || state.`is`(BlockTags.CAMPFIRES) -> MID_EMISSIVE
                state.`is`(BlockTags.WOOL) || state.`is`(BlockTags.WOOL_CARPETS) ||
                    blockPath.endsWith("_bed") || blockPath.contains("banner") -> MID_WOOL
                blockPath.contains("amethyst") || blockPath.contains("diamond_block") ||
                    blockPath.contains("emerald_block") || blockPath.contains("lapis_block") ||
                    blockPath.contains("redstone_block") || blockPath.contains("prismarine") ||
                    blockPath.contains("obsidian") || blockPath.contains("calcite") ||
                    blockPath.contains("crystal") || blockPath.contains("quartz_block") -> MID_GEM_CRYSTAL
                !isOreOrRawOrOxidized && (
                    state.`is`(BlockTags.ANVIL) || blockPath.contains("iron") ||
                        blockPath.contains("gold") || blockPath.contains("copper") ||
                        blockPath.contains("netherite") || blockPath.contains("chain") ||
                        blockPath.contains("cauldron") || blockPath.contains("hopper") ||
                        blockPath.contains("bell") || blockPath.contains("lightning_rod") ||
                        blockPath.contains("rail") || blockPath.contains("iron_bars") ||
                        blockPath.contains("smithing_table") || blockPath.contains("blast_furnace") ||
                        blockPath.contains("crafter") || blockPath.contains("heavy_core")
                    ) -> MID_METAL
                state.`is`(BlockTags.ICE) || blockPath.contains("glass") ||
                    blockPath.contains("packed_ice") || blockPath.contains("blue_ice") ||
                    blockPath.contains("frosted_ice") -> MID_GLASS
                blockPath.contains("snow") || blockPath.contains("slime") ||
                    blockPath.contains("honey") || blockPath.contains("mushroom") ||
                    blockPath.contains("mycelium") || blockPath.contains("sponge") ||
                    blockPath.contains("coral") || blockPath.contains("sea_pickle") ||
                    blockPath.contains("hay_block") || blockPath.contains("melon") ||
                    blockPath.contains("pumpkin") || blockPath.contains("cake") ||
                    blockPath.contains("cobweb") || blockPath.contains("bone_block") ||
                    blockPath.contains("wart_block") || blockPath.contains("scaffolding") ||
                    blockPath.contains("frogspawn") -> MID_SNOW_ORGANIC
                state.`is`(BlockTags.SAND) || blockPath.contains("gravel") ||
                    blockPath.contains("sandstone") || blockPath.contains("soul_sand") ||
                    blockPath.contains("concrete_powder") -> MID_SAND
                state.`is`(BlockTags.LOGS) || state.`is`(BlockTags.PLANKS) ||
                    state.`is`(BlockTags.WOODEN_SLABS) || state.`is`(BlockTags.WOODEN_STAIRS) ||
                    state.`is`(BlockTags.WOODEN_FENCES) || state.`is`(BlockTags.FENCE_GATES) ||
                    state.`is`(BlockTags.WOODEN_DOORS) || state.`is`(BlockTags.WOODEN_TRAPDOORS) ||
                    state.`is`(BlockTags.WOODEN_PRESSURE_PLATES) || state.`is`(BlockTags.WOODEN_BUTTONS) ||
                    state.`is`(BlockTags.SIGNS) || blockPath.contains("_sign") || isBambooWood ||
                    blockPath.contains("barrel") || blockPath.contains("bookshelf") ||
                    blockPath.contains("chest") || blockPath.contains("crafting_table") ||
                    blockPath.contains("lectern") || blockPath.contains("loom") ||
                    blockPath.contains("composter") || blockPath.contains("beehive") ||
                    blockPath.contains("bee_nest") || blockPath.contains("jukebox") ||
                    blockPath.contains("note_block") || blockPath.contains("ladder") ||
                    blockPath.contains("cartography_table") || blockPath.contains("fletching_table") ||
                    blockPath.contains("_wood") || blockPath.contains("_stem") ||
                    blockPath.contains("_hyphae") -> MID_WOOD
                state.`is`(BlockTags.DIRT) || state.`is`(BlockTags.GRASS_BLOCKS) ||
                    (blockPath.contains("mud") && !blockPath.contains("brick")) ||
                    blockPath.contains("clay") || blockPath.contains("podzol") ||
                    blockPath.contains("farmland") || blockPath.contains("dirt_path") ||
                    blockPath.contains("soul_soil") -> MID_DIRT
                blockPath.contains("terracotta") || blockPath.contains("glazed") ||
                    blockPath.contains("concrete") || blockPath.contains("polished_") ||
                    blockPath.contains("smooth_") || blockPath.contains("brick") ||
                    blockPath.contains("purpur") || blockPath.contains("tile") ||
                    blockPath.contains("chiseled_") || blockPath.contains("cut_") ||
                    blockPath.contains("decorated_pot") || blockPath.contains("flower_pot") ||
                    blockPath.contains("quartz") -> MID_POLISHED_CERAMIC
                else -> MID_STONE
            }
            blockBaseMaterialCache.putIfAbsent(block, computed)
            computed
        }
        val preserveBaseOnEmissive = cachedBase == MID_FOLIAGE ||
            cachedBase == MID_METAL ||
            cachedBase == MID_GEM_CRYSTAL ||
            cachedBase == MID_SCULK_MAGICAL ||
            cachedBase == MID_GLASS
        return if (state.lightEmission > 0 && !preserveBaseOnEmissive) MID_EMISSIVE else cachedBase
    }

    override fun onEndSectionTesselation() {
        val ctx = threadContext.get()
        if (ctx.currentSectionPosLong != Long.MIN_VALUE) {
            if (ctx.sectionHasFoliage) {
                sectionFoliageFlags[ctx.currentSectionPosLong] = true
            } else {
                sectionFoliageFlags.remove(ctx.currentSectionPosLong)
            }
        }
        ctx.blockMaterialFlags = 0
        ctx.fluidMaterialFlags = 0
        ctx.currentSectionPosLong = Long.MIN_VALUE
        ctx.sectionHasFoliage = false
    }

    override fun populateExtendedBlockQuad(
        basePtr: Long,
        vertexStride: Int,
        x: Float,
        y: Float,
        z: Float,
        quad: BakedQuad,
        instance: QuadInstance,
    ) {
        if (vertexStride < 40) return

        val v1 = basePtr + vertexStride
        val v2 = v1 + vertexStride
        val v3 = v2 + vertexStride

        val x0 = MemoryUtil.memGetFloat(basePtr)
        val y0 = MemoryUtil.memGetFloat(basePtr + 4L)
        val z0 = MemoryUtil.memGetFloat(basePtr + 8L)
        val u0 = MemoryUtil.memGetFloat(basePtr + 16L)
        val w0 = MemoryUtil.memGetFloat(basePtr + 20L)

        val x1 = MemoryUtil.memGetFloat(v1)
        val y1 = MemoryUtil.memGetFloat(v1 + 4L)
        val z1 = MemoryUtil.memGetFloat(v1 + 8L)
        val u1 = MemoryUtil.memGetFloat(v1 + 16L)
        val w1 = MemoryUtil.memGetFloat(v1 + 20L)

        val x2 = MemoryUtil.memGetFloat(v2)
        val y2 = MemoryUtil.memGetFloat(v2 + 4L)
        val z2 = MemoryUtil.memGetFloat(v2 + 8L)
        val u2 = MemoryUtil.memGetFloat(v2 + 16L)
        val w2 = MemoryUtil.memGetFloat(v2 + 20L)

        val u3 = MemoryUtil.memGetFloat(v3 + 16L)
        val w3 = MemoryUtil.memGetFloat(v3 + 20L)

        val ctx = threadContext.get()
        val packedNormal = TurboShadersBridge.computePackedFaceNormal(
            x0, y0, z0,
            x1, y1, z1,
            x2, y2, z2,
            ctx.blockMaterialFlags,
        )
        val packedTangent = TurboShadersBridge.computePackedTangent(
            x0, y0, z0, u0, w0,
            x1, y1, z1, u1, w1,
            x2, y2, z2, u2, w2,
        )
        val packedMidUv = packHalf2((u0 + u1 + u2 + u3) * 0.25f, (w0 + w1 + w2 + w3) * 0.25f)

        writeExtendedAttributes(basePtr, packedNormal, packedTangent, packedMidUv)
        writeExtendedAttributes(v1, packedNormal, packedTangent, packedMidUv)
        writeExtendedAttributes(v2, packedNormal, packedTangent, packedMidUv)
        writeExtendedAttributes(v3, packedNormal, packedTangent, packedMidUv)
    }

    override fun populateExtendedFluidFace(
        basePtr: Long,
        vertexStride: Int,
        x0: Float, y0: Float, z0: Float, u0: Float, v0: Float,
        x1: Float, y1: Float, z1: Float, u1: Float, v1: Float,
        x2: Float, y2: Float, z2: Float, u2: Float, v2: Float,
        x3: Float, y3: Float, z3: Float, u3: Float, v3: Float,
        addBackFace: Boolean,
    ) {
        if (vertexStride < 40) return

        val ctx = threadContext.get()
        val fluidMask = if (ctx.fluidMaterialFlags != 0) {
            ctx.fluidMaterialFlags
        } else {
            packMaterial(MID_WATER, MATERIAL_FLAG_FLUID)
        }
        val packedNormal = computePackedFaceNormalSafe(
            x0, y0, z0,
            x1, y1, z1,
            x2, y2, z2,
            fluidMask,
        )
        val packedTangent = computePackedTangentSafe(
            x0, y0, z0, u0, v0,
            x1, y1, z1, u1, v1,
            x2, y2, z2, u2, v2,
        )
        val packedMidUv = packHalf2((u0 + u1 + u2 + u3) * 0.25f, (v0 + v1 + v2 + v3) * 0.25f)

        for (i in 0 until 4) {
            writeExtendedAttributes(basePtr + i.toLong() * vertexStride, packedNormal, packedTangent, packedMidUv)
        }
        if (addBackFace) {
            // Invert normal for back-facing quad vertices 4..7 (preserve material mask in upper byte)
            val packedBackNormal = computePackedFaceNormalSafe(
                x0, y0, z0,
                x2, y2, z2,
                x1, y1, z1,
                fluidMask,
            )
            for (i in 4 until 8) {
                writeExtendedAttributes(basePtr + i.toLong() * vertexStride, packedBackNormal, packedTangent, packedMidUv)
            }
        }
    }

    override fun onSectionSliceUploaded(
        allocationKey: Any,
        layerOrdinal: Int,
        gpuBuffer: GpuBuffer,
        byteOffset: Long,
        byteSize: Long,
    ) {
        SectionSliceRegistry.onUploaded(allocationKey, layerOrdinal, gpuBuffer, byteOffset, byteSize)
    }

    override fun onSectionSliceUploadedDetailed(
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
        SectionSliceRegistry.onUploadedDetailed(
            allocationKey = allocationKey,
            layerOrdinal = layerOrdinal,
            sectionPosLong = sectionPosLong,
            gpuBuffer = gpuBuffer,
            byteOffset = byteOffset,
            byteSize = byteSize,
            vertexStride = vertexStride,
            vertexCount = vertexCount,
            isRegionShifted = isRegionShifted,
        )
    }

    override fun onSectionSliceFreed(allocationKey: Any, layerOrdinal: Int) {
        SectionSliceRegistry.onFreed(allocationKey, layerOrdinal)
    }

    /**
     * Invoked from `GameRendererMixin` right after `renderLevel()` / `applyPostEffects()` and
     * BEFORE `clearDepthTexture` and `GuiRenderer.render()`.
     *
     * Records BLAS/TLAS updates and executes the pre-GUI lighting/denoising/upscaling/tonemapping
     * `RenderGraph` directly into the provided transient `cmd` command buffer.
     */
    override fun onAfterWorldRenderBeforeGui(
        cmd: VkCommandBuffer,
        mainColorTexture: VulkanGpuTexture,
        mainColorView: VulkanGpuTextureView,
        mainDepthTexture: VulkanGpuTexture,
        mainDepthView: VulkanGpuTextureView,
        width: Int,
        height: Int,
    ): Boolean {
        worldGraphExecutedThisFrame = false
        if (!VulkanShaders.shaderPackActive || !NativeCoreBridge.initialized || WorldStateCollector.frameIndex == 0) return false
        if (renderGraphDisabled) return false

        GbufferTargetManager.syncNativeGraphBindings(
            colorTex = mainColorTexture,
            colorView = mainColorView,
            depthTex = mainDepthTexture,
            depthView = mainDepthView,
        )

        val cfg = VulkanShaders.config
        if (cfg.renderTier == LightingMode.HYBRID || cfg.renderTier == LightingMode.PATH) {
            NativeCoreBridge.recordFrameRtx(
                cmdAddress = cmd.address(),
                cameraX = WorldStateCollector.cameraPosX,
                cameraY = WorldStateCollector.cameraPosY,
                cameraZ = WorldStateCollector.cameraPosZ,
                prevCameraX = WorldStateCollector.prevCameraPosX,
                prevCameraY = WorldStateCollector.prevCameraPosY,
                prevCameraZ = WorldStateCollector.prevCameraPosZ,
                frameIndex = WorldStateCollector.frameIndex,
                maxBlasBuilds = when (NativeCoreBridge.getGpuTierClass()) {
                    "enthusiast" -> 96
                    "high" -> 64
                    "mid" -> 24
                    else -> 16
                },
            )
        }

        val atlas = Minecraft.getInstance().textureManager.getTexture(TextureAtlas.LOCATION_BLOCKS).texture as? VulkanGpuTexture
        val atlasLayout = atlas?.let(TurboShadersBridge::getTrackedImageLayout)
        val transitionAtlas = atlas != null && atlasLayout != null && atlasLayout != VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
        if (transitionAtlas) {
            TurboShadersBridge.transitionImageLayout(cmd, atlas, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
        }

        val gbufNormal = GbufferTargetManager.normalRoughnessVulkanTexture
        val gbufMaterial = GbufferTargetManager.materialPbrVulkanTexture
        val gbufVelocity = GbufferTargetManager.velocityVulkanTexture

        // Resolve Distant Horizons 3-MRT G-Buffer targets when DH rendered in Tiers 0-2.
        var dhColorTex: VulkanGpuTexture? = null
        var dhNormalTex: VulkanGpuTexture? = null
        var dhMatTex: VulkanGpuTexture? = null
        var dhDepthTex: VulkanGpuTexture? = null
        var prevDhColorLayout = VK_IMAGE_LAYOUT_UNDEFINED
        var prevDhNormalLayout = VK_IMAGE_LAYOUT_UNDEFINED
        var prevDhMatLayout = VK_IMAGE_LAYOUT_UNDEFINED
        var prevDhDepthLayout = VK_IMAGE_LAYOUT_UNDEFINED
        var dhBoundActive = false

        if (DistantHorizonsBridge.dhRenderedThisFrame && cfg.renderTier != LightingMode.PATH) {
            val cTex = DistantHorizonsBridge.dhColorTexture?.takeIf { !it.isClosed }
            val cView = DistantHorizonsBridge.dhColorTextureView
            val dTex = DistantHorizonsBridge.dhDepthTexture?.takeIf { !it.isClosed }
            val dView = DistantHorizonsBridge.dhDepthTextureView
            val nTex = GbufferTargetManager.dhNormalRoughnessVulkanTexture
            val nView = GbufferTargetManager.dhNormalRoughnessView as? VulkanGpuTextureView
            val mTex = GbufferTargetManager.dhMaterialPbrVulkanTexture
            val mView = GbufferTargetManager.dhMaterialPbrView as? VulkanGpuTextureView

            if (cTex != null && cView != null && dTex != null && dView != null &&
                nTex != null && nView != null && mTex != null && mView != null
            ) {
                dhColorTex = cTex
                dhNormalTex = nTex
                dhMatTex = mTex
                dhDepthTex = dTex
                prevDhColorLayout = TurboShadersBridge.getTrackedImageLayout(cTex)
                prevDhNormalLayout = TurboShadersBridge.getTrackedImageLayout(nTex)
                prevDhMatLayout = TurboShadersBridge.getTrackedImageLayout(mTex)
                prevDhDepthLayout = TurboShadersBridge.getTrackedImageLayout(dTex)

                val dhNear = DistantHorizonsBridge.dhNearClip.coerceAtLeast(16.0f)
                val fadeStart = dhNear * 1.15f
                val fadeEnd = dhNear * 1.65f
                val dhFar = DistantHorizonsBridge.dhFarClip.coerceAtLeast(fadeEnd + 64.0f)

                NativeCoreBridge.bindDhTargets(
                    colorImg = cTex.vkImage(),
                    colorView = cView.vkImageView(),
                    normalImg = nTex.vkImage(),
                    normalView = nView.vkImageView(),
                    materialImg = mTex.vkImage(),
                    materialView = mView.vkImageView(),
                    depthImg = dTex.vkImage(),
                    depthView = dView.vkImageView(),
                    width = cTex.getWidth(0),
                    height = cTex.getHeight(0),
                    invViewProj = DistantHorizonsBridge.dhInvViewProjMatrix,
                    fadeStart = fadeStart,
                    fadeEnd = fadeEnd,
                    dhFarClip = dhFar,
                    active = true,
                )
                dhBoundActive = true
                lastDhBoundActive = true
            }
        }
        if (!dhBoundActive && lastDhBoundActive) {
            NativeCoreBridge.bindDhTargets(
                colorImg = 0L,
                colorView = 0L,
                normalImg = 0L,
                normalView = 0L,
                materialImg = 0L,
                materialView = 0L,
                depthImg = 0L,
                depthView = 0L,
                width = 0,
                height = 0,
                invViewProj = null,
                fadeStart = 0.0f,
                fadeEnd = 0.0f,
                dhFarClip = 0.0f,
                active = false,
            )
            lastDhBoundActive = false
        }

        // Batch all G-Buffer + DH layout transitions to GENERAL into a single pipeline barrier.
        val toGeneral: Array<VulkanGpuTexture?> = if (dhBoundActive) {
            arrayOf(
                mainColorTexture,
                mainDepthTexture,
                gbufNormal,
                gbufMaterial,
                gbufVelocity,
                dhColorTex,
                dhNormalTex,
                dhMatTex,
                dhDepthTex,
            )
        } else {
            arrayOf(mainColorTexture, mainDepthTexture, gbufNormal, gbufMaterial, gbufVelocity)
        }
        @Suppress("UNCHECKED_CAST")
        TurboShadersBridge.transitionImageLayoutBatch(cmd, toGeneral as Array<Any?>, VK_IMAGE_LAYOUT_GENERAL)

        val steps = try {
            NativeCoreBridge.executeRenderGraph(cmd.address(), width, height)
        } finally {
            TurboShadersBridge.notifyExternalImageLayout(mainColorTexture, VK_IMAGE_LAYOUT_GENERAL)
            TurboShadersBridge.notifyExternalImageLayout(mainDepthTexture, VK_IMAGE_LAYOUT_GENERAL)
            gbufNormal?.let { TurboShadersBridge.notifyExternalImageLayout(it, VK_IMAGE_LAYOUT_GENERAL) }
            gbufMaterial?.let { TurboShadersBridge.notifyExternalImageLayout(it, VK_IMAGE_LAYOUT_GENERAL) }
            gbufVelocity?.let { TurboShadersBridge.notifyExternalImageLayout(it, VK_IMAGE_LAYOUT_GENERAL) }
            if (dhBoundActive) {
                val cTex = dhColorTex!!
                val nTex = dhNormalTex!!
                val mTex = dhMatTex!!
                val dTex = dhDepthTex!!
                TurboShadersBridge.notifyExternalImageLayout(cTex, VK_IMAGE_LAYOUT_GENERAL)
                TurboShadersBridge.notifyExternalImageLayout(nTex, VK_IMAGE_LAYOUT_GENERAL)
                TurboShadersBridge.notifyExternalImageLayout(mTex, VK_IMAGE_LAYOUT_GENERAL)
                TurboShadersBridge.notifyExternalImageLayout(dTex, VK_IMAGE_LAYOUT_GENERAL)
                restoreDhLayoutsBatched(
                    cmd = cmd,
                    t0 = cTex, l0 = prevDhColorLayout,
                    t1 = nTex, l1 = prevDhNormalLayout,
                    t2 = mTex, l2 = prevDhMatLayout,
                    t3 = dTex, l3 = prevDhDepthLayout,
                )
            }
            if (transitionAtlas && atlasLayout != VK_IMAGE_LAYOUT_UNDEFINED) {
                TurboShadersBridge.transitionImageLayout(cmd, atlas, atlasLayout)
            }
        }
        if (!loggedFirstGraphExec) {
            loggedFirstGraphExec = true
            VulkanShaders.log.info(
                "First world frame RenderGraph executed: steps={}, size={}x{}, tier={}",
                steps,
                width,
                height,
                cfg.renderTier,
            )
        }

        worldGraphExecutedThisFrame = steps > 0
        if (VulkanShaders.hasPendingDeferredTier2()) {
            if (worldGraphExecutedThisFrame) {
                val frames = ++deferredTier2Frames
                if (frames >= DEFERRED_TIER_2_ACTIVATION_FRAMES) {
                    deferredTier2Frames = 0
                    // Rebuild on the client task queue so it lands between frames, not mid-render.
                    Minecraft.getInstance().execute {
                        VulkanShaders.applyDeferredTier2(frames)
                    }
                }
            }
        } else {
            deferredTier2Frames = 0
        }
        return worldGraphExecutedThisFrame
    }

    /**
     * Invoked from `GameRendererMixin` immediately AFTER `GuiRenderer.render()` has composited the
     * HUD into [uiColorTexture] (`mainRenderTarget.colorTexture`). Executes `HdrCompose` and/or
     * `FrameGenInterpolate` with the real rendered GUI (`RES_UI_COLOR`) and pure HUD-less world
     * color (`RES_POST_OUTPUT`), and updates `RES_PREV_HUDLESS_COLOR` for next frame's history.
     */
    override fun notifyAfterGuiRender(
        cmd: VkCommandBuffer,
        uiColorTexture: VulkanGpuTexture,
        uiColorView: VulkanGpuTextureView,
        width: Int,
        height: Int,
    ): Boolean {
        if (!NativeCoreBridge.initialized) {
            return false
        }
        val mainMenuHdrUiOnly = hdrSwapchainActive && !worldGraphExecutedThisFrame
        if (!mainMenuHdrUiOnly && (!VulkanShaders.shaderPackActive || !worldGraphExecutedThisFrame)) {
            return false
        }
        val executed = NativeCoreBridge.executeAfterGui(
            cmdAddress = cmd.address(),
            uiImage = uiColorTexture.vkImage(),
            uiView = uiColorView.vkImageView(),
            width = width,
            height = height,
        )
        return executed > 0
    }

    override fun onConfigureSwapchain(
        surfaceHandle: Long,
        requestedWidth: Int,
        requestedHeight: Int,
        defaultFormat: Int,
        defaultColorSpace: Int,
        defaultUsage: Int,
        defaultPresentMode: Int,
    ): SwapchainConfigOverride? {
        if (!VulkanShaders.config.hdrEnabled) return null
        if (!NativeCoreBridge.isHdrSwapchainSupported(surfaceHandle)) {
            if (!loggedHdrUnsupported) {
                loggedHdrUnsupported = true
                VulkanShaders.log.warn(
                    "HDR output requested but surface 0x{} does not support EXTENDED_SRGB_LINEAR_EXT; keeping SDR swapchain.",
                    java.lang.Long.toHexString(surfaceHandle),
                )
            }
            return null
        }
        return SwapchainConfigOverride(
            imageFormat = VK_FORMAT_R16G16B16A16_SFLOAT,
            colorSpace = VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT,
            extraUsageFlags = VK_IMAGE_USAGE_STORAGE_BIT,
            presentMode = defaultPresentMode,
        )
    }

    override fun onSwapchainCreated(
        swapchainHandle: Long,
        images: LongArray,
        imageFormat: Int,
        colorSpace: Int,
        width: Int,
        height: Int,
    ) {
        hdrSwapchainActive = (imageFormat == VK_FORMAT_R16G16B16A16_SFLOAT && colorSpace == VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT) ||
            (imageFormat == VK_FORMAT_A2B10G10R10_UNORM_PACK32 && colorSpace == VK_COLOR_SPACE_HDR10_ST2084_EXT)
        if (NativeCoreBridge.initialized) {
            NativeCoreBridge.onSwapchainCreated(imageFormat, colorSpace, width, height)
            NativeCoreBridge.reflexSetSwapchain(swapchainHandle)
        }
    }

    override fun interceptAcquireNextImage(
        swapchainHandle: Long,
        timeoutNs: Long,
        semaphore: Long,
        fence: Long,
    ): Int {
        if (!NativeCoreBridge.initialized || VulkanShaders.config.frameGenMode == FrameGenMode.NONE) {
            return -1
        }
        return NativeCoreBridge.interceptAcquireNextImage(swapchainHandle, timeoutNs, semaphore, fence)
    }

    override fun interceptQueuePresent(
        queueHandle: Long,
        swapchainHandle: Long,
        imageIndex: Int,
        waitSemaphore: Long,
    ): Boolean {
        if (!NativeCoreBridge.initialized || VulkanShaders.config.frameGenMode == FrameGenMode.NONE) {
            return false
        }
        return NativeCoreBridge.interceptQueuePresent(queueHandle, swapchainHandle, imageIndex, waitSemaphore)
    }

    /** Batches DH image layout restores by shared target layout to minimize pipeline barriers. */
    private fun restoreDhLayoutsBatched(
        cmd: VkCommandBuffer,
        t0: VulkanGpuTexture, l0: Int,
        t1: VulkanGpuTexture, l1: Int,
        t2: VulkanGpuTexture, l2: Int,
        t3: VulkanGpuTexture, l3: Int,
    ) {
        val textures = arrayOf(t0, t1, t2, t3)
        val layouts = intArrayOf(l0, l1, l2, l3)
        var doneMask = 0
        for (i in 0 until 4) {
            if ((doneMask and (1 shl i)) != 0) continue
            val target = layouts[i]
            if (target == VK_IMAGE_LAYOUT_UNDEFINED || target == VK_IMAGE_LAYOUT_GENERAL) {
                doneMask = doneMask or (1 shl i)
                continue
            }
            var count = 0
            for (j in i until 4) {
                if (layouts[j] == target) count++
            }
            if (count == 1) {
                doneMask = doneMask or (1 shl i)
                TurboShadersBridge.transitionImageLayout(cmd, textures[i], target)
            } else {
                val batch = ArrayList<VulkanGpuTexture>(count)
                for (j in i until 4) {
                    if (layouts[j] == target) {
                        doneMask = doneMask or (1 shl j)
                        batch.add(textures[j])
                    }
                }
                @Suppress("UNCHECKED_CAST")
                TurboShadersBridge.transitionImageLayoutBatch(cmd, batch.toTypedArray() as Array<Any?>, target)
            }
        }
    }

    fun resetPerWorldState() {
        worldGraphExecutedThisFrame = false
        loggedFirstGraphExec = false
        lastDhBoundActive = false
        sectionFoliageFlags.clear()
        NativeCoreBridge.onWorldUnloaded()
    }

    override fun interceptSurfaceBlit(
        cmd: VkCommandBuffer,
        srcTexture: VulkanGpuTexture?,
        srcImage: Long,
        srcLayout: Int,
        dstImage: Long,
        dstLayout: Int,
        regions: VkImageBlit.Buffer,
        filter: Int,
    ): Boolean {
        val cfg = VulkanShaders.config
        if (!NativeCoreBridge.initialized) return false

        val executedWorldGraph = worldGraphExecutedThisFrame
        worldGraphExecutedThisFrame = false

        // In HDR mode, still intercept blit in the main menu so sRGB UI is converted cleanly
        // into the linear scRGB / HDR10 swapchain instead of being blown out.
        if (!executedWorldGraph && !hdrSwapchainActive) return false

        if (cfg.frameGenMode == FrameGenMode.NONE && !cfg.hdrEnabled && !hdrSwapchainActive) return false

        val win = Minecraft.getInstance().window
        val width = srcTexture?.getWidth(0) ?: win?.width?.coerceAtLeast(1) ?: 1280
        val height = srcTexture?.getHeight(0) ?: win?.height?.coerceAtLeast(1) ?: 720

        return NativeCoreBridge.interceptSurfaceBlit(
            cmdAddress = cmd.address(),
            srcImage = srcImage,
            srcLayout = srcLayout,
            dstImage = dstImage,
            dstLayout = dstLayout,
            width = width,
            height = height,
        )
    }

    private fun writeExtendedAttributes(ptr: Long, packedNormal: Int, packedTangent: Int, packedMidUv: Int) {
        MemoryUtil.memPutInt(ptr + 28L, packedNormal)
        MemoryUtil.memPutInt(ptr + 32L, packedTangent)
        MemoryUtil.memPutInt(ptr + 36L, packedMidUv)
    }

    private fun packSnorm8(v: Float): Int =
        kotlin.math.round(v.coerceIn(-1.0f, 1.0f) * 127.0f).toInt() and 0xFF

    private fun computePackedFaceNormalSafe(
        x0: Float, y0: Float, z0: Float,
        x1: Float, y1: Float, z1: Float,
        x2: Float, y2: Float, z2: Float,
        materialMask: Int,
    ): Int {
        if (StandaloneRenderPearlBridge.vulkanTurboLoaded) {
            runCatching {
                return TurboShadersBridge.computePackedFaceNormal(x0, y0, z0, x1, y1, z1, x2, y2, z2, materialMask)
            }
        }
        val ax = x1 - x0
        val ay = y1 - y0
        val az = z1 - z0
        val bx = x2 - x0
        val by = y2 - y0
        val bz = z2 - z0
        var nx = ay * bz - az * by
        var ny = az * bx - ax * bz
        var nz = ax * by - ay * bx
        val len = kotlin.math.sqrt(nx * nx + ny * ny + nz * nz)
        if (len > 1e-6f) {
            val inv = 1.0f / len
            nx *= inv
            ny *= inv
            nz *= inv
        } else {
            nx = 0.0f
            ny = 1.0f
            nz = 0.0f
        }
        return packSnorm8(nx) or (packSnorm8(ny) shl 8) or (packSnorm8(nz) shl 16) or ((materialMask and 0x7F) shl 24)
    }

    private fun computePackedTangentSafe(
        x0: Float, y0: Float, z0: Float, u0: Float, v0: Float,
        x1: Float, y1: Float, z1: Float, u1: Float, v1: Float,
        x2: Float, y2: Float, z2: Float, u2: Float, v2: Float,
    ): Int {
        if (StandaloneRenderPearlBridge.vulkanTurboLoaded) {
            runCatching {
                return TurboShadersBridge.computePackedTangent(x0, y0, z0, u0, v0, x1, y1, z1, u1, v1, x2, y2, z2, u2, v2)
            }
        }
        val dx1 = x1 - x0
        val dy1 = y1 - y0
        val dz1 = z1 - z0
        val dx2 = x2 - x0
        val dy2 = y2 - y0
        val dz2 = z2 - z0
        val du1 = u1 - u0
        val dv1 = v1 - v0
        val du2 = u2 - u0
        val dv2 = v2 - v0
        val det = du1 * dv2 - du2 * dv1
        var tx = 1.0f
        var ty = 0.0f
        var tz = 0.0f
        if (kotlin.math.abs(det) > 1e-8f) {
            val invDet = 1.0f / det
            tx = (dv2 * dx1 - dv1 * dx2) * invDet
            ty = (dv2 * dy1 - dv1 * dy2) * invDet
            tz = (dv2 * dz1 - dv1 * dz2) * invDet
            val len = kotlin.math.sqrt(tx * tx + ty * ty + tz * tz)
            if (len > 1e-6f) {
                val inv = 1.0f / len
                tx *= inv
                ty *= inv
                tz *= inv
            }
        }
        return packSnorm8(tx) or (packSnorm8(ty) shl 8) or (packSnorm8(tz) shl 16) or (packSnorm8(1.0f) shl 24)
    }

    private fun packHalf2(u: Float, v: Float): Int {
        val hu = floatToFloat16(u).toInt() and 0xFFFF
        val hv = floatToFloat16(v).toInt() and 0xFFFF
        return hu or (hv shl 16)
    }
}
