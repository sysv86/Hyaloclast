package dev.vulkanshaders.state

import dev.vulkanshaders.VulkanShaders
import dev.vulkanshaders.backend.NativeCoreBridge
import dev.vulkanshaders.bridge.DistantHorizonsBridge
import dev.vulkanshaders.config.RenderTier
import dev.vulkanshaders.config.UpscalerMode
import dev.vulkanshaders.pack.PackSettingDefinition
import dev.vulkanshaders.pack.SettingType
import dev.vulkanshaders.pack.ShaderPackManager
import it.unimi.dsi.fastutil.longs.Long2ObjectOpenHashMap
import net.minecraft.client.Minecraft
import net.minecraft.client.renderer.state.level.CameraRenderState
import net.minecraft.core.BlockPos
import net.minecraft.core.SectionPos
import net.minecraft.world.item.BlockItem
import net.minecraft.world.item.Item
import net.minecraft.world.item.ItemStack
import net.minecraft.world.item.Items
import net.minecraft.world.entity.Entity
import net.minecraft.world.entity.EntityTypes
import net.minecraft.world.entity.HumanoidArm
import net.minecraft.world.entity.monster.Creeper
import net.minecraft.world.entity.item.ItemEntity
import net.minecraft.world.entity.player.Player
import net.minecraft.world.level.block.Block
import net.minecraft.world.level.block.state.properties.BlockStateProperties
import net.minecraft.world.level.chunk.LevelChunkSection
import net.minecraft.world.phys.Vec3
import org.joml.Matrix4f
import org.joml.Matrix4fc
import org.joml.Vector3f
import org.joml.Vector4fc
import org.lwjgl.system.MemoryUtil
import java.lang.ref.WeakReference
import java.nio.ByteBuffer
import java.util.IdentityHashMap
import java.util.Locale
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.floor
import kotlin.math.pow
import kotlin.math.roundToInt
import kotlin.math.sin

/**
 * Collects per-frame world, camera, sky, dimension, biome atmosphere, held-light, and active pack
 * parameter state into a `std140`-aligned off-heap uniform buffer matching
 * `common.vsl` and Rust `vulkanshaders-core::modules::WorldUboData` byte-for-byte.
 *
 * Underwater open-flame held-light extinguishing policy adapted from NVIDIA RTX Remix for Minecraft
 * (`HeldLightPolicy.java`, Apache License 2.0).
 */
object WorldStateCollector {
    // 528-byte v1 WorldUBO payload + 8-byte VSL packParams BDA + 8-byte std140 padding.
    const val WORLD_UBO_BYTES: Int = 544
    const val MAX_PACK_PARAM_FLOATS: Int = 48
    const val MAX_RGB24: Int = 0xFFFFFF
    const val JITTER_PHASE_COUNT: Int = 16
    private const val TELEPORT_RESET_DIST_SQ: Double = 64.0 * 64.0
    private const val EMISSIVE_SCAN_INTERVAL_TICKS: Long = 10L
    private const val EMISSIVE_RESCAN_MOVE_DIST_SQ: Int = 9
    private const val LIGHT_SCAN_RADIUS: Int = 24

    // Bit flags packed into WorldEmissiveLightDesc.flags (matching restir.vsl & lights.rs)
    private const val HELD_LIGHT_FLAG: Int = 1
    private const val THIN_EMITTER_FLAG: Int = 2
    private const val FLICKER_FLAG: Int = 4
    private const val OFFHAND_HELD_FLAG: Int = 8
    private const val STANDING_TORCH_FLAG: Int = 16

    // Reserved tail float slots in WorldUBO.packParams[10..11] (indices 40..47) for biome/dimension telemetry
    private const val BIOME_TELEMETRY_START_FLOAT: Int = 40

    const val MAX_EMISSIVE_LIGHTS: Int = 512
    const val MAX_EMISSIVE_LIGHTS_LOW_TIER: Int = 64
    private const val MAX_DYNAMIC_ENTITY_LIGHTS: Int = 48
    private const val MAX_BLOCK_LIGHTS: Int = MAX_EMISSIVE_LIGHTS - 2 // Reserve slots for dual-wield held lights
    private const val MAX_CACHED_SECTIONS: Int = 4096
    const val EMISSIVE_LIGHT_DESC_BYTES: Int = 48
    private val uboBuffer: ByteBuffer = MemoryUtil.memCalloc(WORLD_UBO_BYTES)
    private val emissiveLightBuffer: ByteBuffer = MemoryUtil.memCalloc(MAX_EMISSIVE_LIGHTS * EMISSIVE_LIGHT_DESC_BYTES)
    private val blockLightProfileCache = IdentityHashMap<Block, BlockLightProfile>(128)

    private class SectionLightCache(
        val sectionRef: WeakReference<LevelChunkSection>,
        val count: Int,
        val worldX: FloatArray,
        val worldY: FloatArray,
        val worldZ: FloatArray,
        val radius: FloatArray,
        val r: FloatArray,
        val g: FloatArray,
        val b: FloatArray,
        val intensity: FloatArray,
        val flags: IntArray,
    )

    private val EMPTY_FLOAT_ARRAY = FloatArray(0)
    private val EMPTY_INT_ARRAY = IntArray(0)

    private fun emptySectionCache(section: LevelChunkSection): SectionLightCache = SectionLightCache(
        sectionRef = WeakReference(section),
        count = 0,
        worldX = EMPTY_FLOAT_ARRAY,
        worldY = EMPTY_FLOAT_ARRAY,
        worldZ = EMPTY_FLOAT_ARRAY,
        radius = EMPTY_FLOAT_ARRAY,
        r = EMPTY_FLOAT_ARRAY,
        g = EMPTY_FLOAT_ARRAY,
        b = EMPTY_FLOAT_ARRAY,
        intensity = EMPTY_FLOAT_ARRAY,
        flags = EMPTY_INT_ARRAY,
    )

    private val sectionLightCacheLock = Any()
    private val sectionLightCache = Long2ObjectOpenHashMap<SectionLightCache>(256)

    // Precomputed section offsets within LIGHT_SCAN_RADIUS sorted by Euclidean distSq ascending (dcx, dsy, dcz).
    private val SORTED_SECTION_OFFSETS: IntArray = run {
        val maxSecOff = (LIGHT_SCAN_RADIUS + 15) shr 4
        val list = ArrayList<IntArray>()
        for (dcx in -maxSecOff..maxSecOff) {
            for (dsy in -maxSecOff..maxSecOff) {
                for (dcz in -maxSecOff..maxSecOff) {
                    list.add(intArrayOf(dcx, dsy, dcz, dcx * dcx + dsy * dsy + dcz * dcz))
                }
            }
        }
        list.sortBy { it[3] }
        val flat = IntArray(list.size * 3)
        for (i in list.indices) {
            val e = list[i]
            flat[i * 3] = e[0]
            flat[i * 3 + 1] = e[1]
            flat[i * 3 + 2] = e[2]
        }
        flat
    }

    // Zero-GC fixed max-heap (ordered by distSq descending at root 0) to retain the top-511 closest block lights.
    private val heapDistSq = FloatArray(MAX_BLOCK_LIGHTS)
    private val heapX = FloatArray(MAX_BLOCK_LIGHTS)
    private val heapY = FloatArray(MAX_BLOCK_LIGHTS)
    private val heapZ = FloatArray(MAX_BLOCK_LIGHTS)
    private val heapRadius = FloatArray(MAX_BLOCK_LIGHTS)
    private val heapR = FloatArray(MAX_BLOCK_LIGHTS)
    private val heapG = FloatArray(MAX_BLOCK_LIGHTS)
    private val heapB = FloatArray(MAX_BLOCK_LIGHTS)
    private val heapIntensity = FloatArray(MAX_BLOCK_LIGHTS)
    private val heapFlags = IntArray(MAX_BLOCK_LIGHTS)

    // Scratch arrays for building a single SectionLightCache (reused on render thread, zero per-scan heap churn).
    private var secBuildX = FloatArray(64)
    private var secBuildY = FloatArray(64)
    private var secBuildZ = FloatArray(64)
    private var secBuildRadius = FloatArray(64)
    private var secBuildR = FloatArray(64)
    private var secBuildG = FloatArray(64)
    private var secBuildB = FloatArray(64)
    private var secBuildIntensity = FloatArray(64)
    private var secBuildFlags = IntArray(64)

    private data class BlockLightProfile(
        val rLin: Float,
        val gLin: Float,
        val bLin: Float,
        val yOffset: Float,
        val flags: Int,
        val intensityScale: Float,
    )

    private data class BiomeAtmosphereProfile(
        val temp: Float,
        val humidity: Float,
        val cloudScale: Float,
        val fogR: Float,
        val fogG: Float,
        val fogB: Float,
        val densityScale: Float,
    )

    private class LinearRgb(rgb: Int) {
        val r: Float = ((rgb shr 16) and 0xFF).toFloat().div(255.0f).pow(2.2f)
        val g: Float = ((rgb shr 8) and 0xFF).toFloat().div(255.0f).pow(2.2f)
        val b: Float = (rgb and 0xFF).toFloat().div(255.0f).pow(2.2f)
    }

    private val SRGB_TO_LINEAR_LUT: FloatArray = FloatArray(256) { (it / 255.0f).pow(2.2f) }

    // Pre-linearized constant RGB colors for dynamic entity lights (avoids Math.pow per entity per frame).
    private val COLOR_ENTITY_FIRE = LinearRgb(0xFF7A22)
    private val COLOR_ENTITY_BLAZE = LinearRgb(0xFF9C38)
    private val COLOR_ENTITY_GLOW_SQUID = LinearRgb(0x4CE8E0)
    private val COLOR_ENTITY_ALLAY = LinearRgb(0x66D8FF)
    private val COLOR_ENTITY_MAGMA_CUBE = LinearRgb(0xFF6B1E)
    private val COLOR_ENTITY_WARDEN = LinearRgb(0x1AC8D8)
    private val COLOR_ENTITY_CHARGED_CREEPER = LinearRgb(0x66B8FF)
    private val COLOR_ENTITY_SHULKER_BULLET = LinearRgb(0xE8D8FF)
    private val COLOR_ENTITY_SPECTRAL_ARROW = LinearRgb(0xFFE066)
    private val COLOR_ENTITY_FIREBALL = LinearRgb(0xFF7A24)
    private val COLOR_ENTITY_SMALL_FIREBALL = LinearRgb(0xFF8830)
    private val COLOR_ENTITY_DRAGON_FIREBALL = LinearRgb(0xD85CFF)
    private val COLOR_ENTITY_TNT = LinearRgb(0xFFAA44)
    private val COLOR_ENTITY_END_CRYSTAL = LinearRgb(0xE47CFF)
    private val COLOR_ENTITY_LIGHTNING = LinearRgb(0xC8E4FF)
    private val COLOR_ENTITY_EXPERIENCE_ORB = LinearRgb(0x9CFF58)

    private val prevViewProj = Matrix4f()
    private val rasterInvView = Matrix4f()
    private val curViewProj = Matrix4f()
    private val invProj = Matrix4f()
    private val invView = Matrix4f()

    // Reusable scratch objects for beginFrame() to eliminate per-frame allocations.
    private val scratchEffectiveView = Matrix4f()
    private val scratchSampledProj = Matrix4f()
    private val scratchSunDir = Vector3f()
    private val scratchHeldVec = Vector3f()
    private val scratchCamBlockPos = BlockPos.MutableBlockPos()

    // Cached resolved runtime pack settings (slots 0..39), invalidated only on pack or override changes.
    private val cachedPackSettings = FloatArray(BIOME_TELEMETRY_START_FLOAT)
    private var cachedPackConfigRef: dev.vulkanshaders.config.VulkanShadersConfig? = null
    private var cachedPackRef: dev.vulkanshaders.pack.LoadedShaderPack? = null
    private var cachedPackId: String = ""
    private var cachedPackTierIndex: Int = -1
    private var cachedPackOverridesHash: Int = 0
    private var cachedPackSettingsValid: Boolean = false

    // Cached biome atmosphere profile keyed by (camBlockX, camBlockY, camBlockZ, dimensionKey).
    private var cachedBiomeBlockX: Int = Int.MIN_VALUE
    private var cachedBiomeBlockY: Int = Int.MIN_VALUE
    private var cachedBiomeBlockZ: Int = Int.MIN_VALUE
    private var cachedBiomeDimKey: String = ""
    private var cachedBiomeProfile: BiomeAtmosphereProfile? = null

    @JvmField var frameIndex: Int = 0
    @JvmField @Volatile var cameraPosX: Double = 0.0
    @JvmField @Volatile var cameraPosY: Double = 0.0
    @JvmField @Volatile var cameraPosZ: Double = 0.0
    @JvmField var prevCameraPosX: Double = 0.0
    @JvmField var prevCameraPosY: Double = 0.0
    @JvmField var prevCameraPosZ: Double = 0.0
    @JvmField var jitterX: Float = 0.0f
    @JvmField var jitterY: Float = 0.0f
    @JvmField var prevJitterX: Float = 0.0f
    @JvmField var prevJitterY: Float = 0.0f

    private var lastDimensionKey: String = ""
    private var lastSkyClockTime: Long? = null
    private var lastSkyClockChangeGameTime: Long = Long.MIN_VALUE
    private var lastEmissiveScanX: Int = Int.MIN_VALUE
    private var lastEmissiveScanY: Int = Int.MIN_VALUE
    private var lastEmissiveScanZ: Int = Int.MIN_VALUE
    private var lastEmissiveScanBucket: Long = Long.MIN_VALUE
    private var cachedBlockLightCount: Int = 0
    private var smoothMainHeldBlend: Double = 0.0
    private var smoothOffHeldBlend: Double = 0.0
    private var smoothHeldInitialized: Boolean = false
    private val heldBlockProbe = BlockPos.MutableBlockPos()
    @Volatile private var nearbyLightingDirty: Boolean = false

    // Smoothly damped biome atmosphere state (avoids abrupt biome-border pops)
    private var smoothBiomeInitialized: Boolean = false
    private var smoothBiomeTemp: Float = 0.5f
    private var smoothBiomeHumidity: Float = 0.5f
    private var smoothBiomeCloudScale: Float = 1.0f
    private var smoothBiomeFogR: Float = 0.46f
    private var smoothBiomeFogG: Float = 0.64f
    private var smoothBiomeFogB: Float = 0.90f
    private var smoothBiomeDensity: Float = 1.0f

    @JvmStatic
    fun resetHistory() {
        frameIndex = 0
        lastDimensionKey = ""
        lastSkyClockTime = null
        lastSkyClockChangeGameTime = Long.MIN_VALUE
        lastEmissiveScanX = Int.MIN_VALUE
        lastEmissiveScanY = Int.MIN_VALUE
        lastEmissiveScanZ = Int.MIN_VALUE
        lastEmissiveScanBucket = Long.MIN_VALUE
        cachedBlockLightCount = 0
        smoothMainHeldBlend = 0.0
        smoothOffHeldBlend = 0.0
        smoothHeldInitialized = false
        nearbyLightingDirty = false
        synchronized(sectionLightCacheLock) {
            sectionLightCache.clear()
        }
        smoothBiomeInitialized = false
        cachedBiomeProfile = null
        cachedBiomeBlockX = Int.MIN_VALUE
        cachedBiomeBlockY = Int.MIN_VALUE
        cachedBiomeBlockZ = Int.MIN_VALUE
        cachedBiomeDimKey = ""
        cachedPackSettingsValid = false
        cachedPackConfigRef = null
        cachedPackRef = null
        jitterX = 0.0f
        jitterY = 0.0f
        prevJitterX = 0.0f
        prevJitterY = 0.0f
        NativeCoreBridge.setJitter(0.0f, 0.0f)
        NativeCoreBridge.markHistoryReset()
        NativeCoreBridge.resetTemporalHistory()
    }

    @JvmStatic
    fun onBlockLightingChanged(pos: BlockPos) {
        val secKey = SectionPos.asLong(pos.x shr 4, pos.y shr 4, pos.z shr 4)
        synchronized(sectionLightCacheLock) {
            sectionLightCache.remove(secKey)
        }

        if (abs(pos.x - cameraPosX) <= LIGHT_SCAN_RADIUS + 2 &&
            abs(pos.y - cameraPosY) <= LIGHT_SCAN_RADIUS + 2 &&
            abs(pos.z - cameraPosZ) <= LIGHT_SCAN_RADIUS + 2
        ) nearbyLightingDirty = true
    }

    @JvmStatic
    @JvmOverloads
    fun beginFrame(
        cameraState: CameraRenderState,
        rasterProjection: Matrix4f,
        sunAngle: Float,
        vanillaFogColor: Vector4fc? = null,
    ): Matrix4f {
        val mc = Minecraft.getInstance()
        val level = mc.level ?: return rasterProjection

        val dimKey = level.dimension().identifier().toString()
        val dimensionChanged = lastDimensionKey.isNotEmpty() && lastDimensionKey != dimKey
        lastDimensionKey = dimKey

        prevCameraPosX = cameraPosX
        prevCameraPosY = cameraPosY
        prevCameraPosZ = cameraPosZ

        val camPos: Vec3 = cameraState.pos
        cameraPosX = camPos.x
        cameraPosY = camPos.y
        cameraPosZ = camPos.z
        NativeCoreBridge.setCameraPosition(cameraPosX, cameraPosY, cameraPosZ)

        val camBlockX = floor(cameraPosX).toInt()
        val camBlockY = floor(cameraPosY).toInt()
        val camBlockZ = floor(cameraPosZ).toInt()
        DistantHorizonsBridge.onBeginFrame(level.gameTime, camBlockX, camBlockY, camBlockZ)

        val dx = cameraPosX - prevCameraPosX
        val dy = cameraPosY - prevCameraPosY
        val dz = cameraPosZ - prevCameraPosZ
        val teleported = (dx * dx + dy * dy + dz * dz) > TELEPORT_RESET_DIST_SQ
        val isDiscontinuousFrame = frameIndex == 0 || dimensionChanged || teleported

        if (isDiscontinuousFrame) {
            prevCameraPosX = cameraPosX
            prevCameraPosY = cameraPosY
            prevCameraPosZ = cameraPosZ
            if (dimensionChanged || teleported) {
                smoothBiomeInitialized = false
                if (dimensionChanged) {
                    synchronized(sectionLightCacheLock) {
                        sectionLightCache.clear()
                    }
                }
                NativeCoreBridge.resetTemporalHistory()
            }
        }

        val mainTarget = mc.gameRenderer.mainRenderTarget()
        val renderWidth = mainTarget.width
        val renderHeight = mainTarget.height
        val temporalActive = VulkanShaders.config.upscalerMode != UpscalerMode.NONE &&
            VulkanShaders.config.renderTier != RenderTier.TIER_3_PATH_TRACING
        updateJitter(renderWidth, renderHeight, temporalActive)
        NativeCoreBridge.setJitter(jitterX, jitterY)

        val projection = cameraState.projectionMatrix
        val rotation = cameraState.viewRotationMatrix
        val effectiveView = effectiveWorldViewInto(projection, rasterProjection, rotation, scratchEffectiveView)
        rasterInvView.set(rotation).invert()
        val sampledProjection = applyProjectionJitterInto(projection, renderWidth, renderHeight, scratchSampledProj)
        updateMatrices(effectiveView, projection, isDiscontinuousFrame, sampledProjection)

        val dimType = level.dimensionType()
        val dimensionId = when {
            dimType.hasCeiling() -> 1
            !dimType.hasSkyLight() -> 2
            else -> 0
        }

        val skyClockTime = if (dimensionId == 0) level.overworldClockTime else null
        val previousSkyClockTime = lastSkyClockTime
        val skyClockJumped = !dimensionChanged && skyClockTime != null && previousSkyClockTime != null &&
            abs(skyClockTime - previousSkyClockTime) > 120L
        if (skyClockTime != null && skyClockTime != previousSkyClockTime) {
            lastSkyClockChangeGameTime = level.gameTime
        }
        lastSkyClockTime = skyClockTime

        val partialTick = mc.deltaTracker.getGameTimeDeltaPartialTick(false)
        val skyClockAdvancing = skyClockTime != null &&
            lastSkyClockChangeGameTime != Long.MIN_VALUE &&
            (level.gameTime - lastSkyClockChangeGameTime) in 0L..2L
        val effectiveSunAngle = if (dimensionId == 0) {
            resolveOverworldSunAngle(
                reportedSunAngle = sunAngle,
                overworldClockTime = skyClockTime,
                partialTick = if (skyClockAdvancing) partialTick else 0.0f,
                preferClock = VulkanShaders.config.renderTier == RenderTier.TIER_3_PATH_TRACING,
            )
        } else {
            sunAngle
        }
        val sunDir = sunDirectionForDimensionInto(dimensionId, effectiveSunAngle, scratchSunDir)

        // Sample biome atmosphere profile (cached per block & dimension) and smoothly damp across borders.
        val targetBiome = if (cachedBiomeProfile != null &&
            camBlockX == cachedBiomeBlockX &&
            camBlockY == cachedBiomeBlockY &&
            camBlockZ == cachedBiomeBlockZ &&
            dimKey == cachedBiomeDimKey
        ) {
            cachedBiomeProfile!!
        } else {
            val biomeName = runCatching {
                level.getBiome(scratchCamBlockPos.set(camBlockX, camBlockY, camBlockZ)).registeredName
            }.getOrDefault("")
            resolveBiomeAtmosphere(biomeName, dimensionId, vanillaFogColor).also {
                cachedBiomeBlockX = camBlockX
                cachedBiomeBlockY = camBlockY
                cachedBiomeBlockZ = camBlockZ
                cachedBiomeDimKey = dimKey
                cachedBiomeProfile = it
            }
        }
        updateSmoothBiome(targetBiome, isDiscontinuousFrame)

        val rainLevel = level.getRainLevel(1.0f)
        val thunderLevel = level.getThunderLevel(1.0f)
        val eyeInFluid = cameraState.fogType.ordinal
        val submergedInWater = eyeInFluid == 1

        // Held-light evaluation for BOTH hands simultaneously with unified underwater open-flame policy
        val player = mc.player
        val firstPerson = mc.options.cameraType.isFirstPerson
        val activePlayer = player?.takeIf { !it.isSpectator }
        val playerSubmergedInWater = isEntitySubmergedInWater(activePlayer, firstPerson, submergedInWater)
        val mainStack = activePlayer?.mainHandItem ?: ItemStack.EMPTY
        val offStack = activePlayer?.offhandItem ?: ItemStack.EMPTY
        val heldMain = effectiveHeldLightLevel(mainStack, playerSubmergedInWater)
        val heldOff = effectiveHeldLightLevel(offStack, playerSubmergedInWater)

        if (isDiscontinuousFrame) {
            smoothHeldInitialized = false
        }
        val mainPos = if (activePlayer != null && heldMain > 0.0f) {
            computeHeldHandPosition(
                level = level,
                player = activePlayer,
                partialTick = partialTick,
                firstPerson = firstPerson,
                isOffhand = false,
                snapBlend = !smoothHeldInitialized,
            )
        } else {
            smoothMainHeldBlend = 0.0
            Vec3.ZERO
        }
        val offPos = if (activePlayer != null && heldOff > 0.0f) {
            computeHeldHandPosition(
                level = level,
                player = activePlayer,
                partialTick = partialTick,
                firstPerson = firstPerson,
                isOffhand = true,
                snapBlend = !smoothHeldInitialized,
            )
        } else {
            smoothOffHeldBlend = 0.0
            Vec3.ZERO
        }
        smoothHeldInitialized = true

        val primaryOffhand = heldOff > heldMain && heldOff > 0.0f
        val primaryStack = when {
            maxOf(heldMain, heldOff) <= 0.0f -> ItemStack.EMPTY
            primaryOffhand -> offStack
            else -> mainStack
        }
        val primaryItem: Item? = if (primaryStack.isEmpty) null else primaryStack.item
        val primaryLightKind = if (primaryItem != null) resolveHeldLightKind(primaryItem, primaryStack) else 0
        val primaryHeldPosition = if (primaryOffhand) offPos else mainPos

        if (skyClockJumped && !isDiscontinuousFrame) NativeCoreBridge.resetTemporalHistory()
        val effectiveDepthFar = DistantHorizonsBridge.effectiveFarPlaneBlocks(cameraState.depthFar)

        if (DistantHorizonsBridge.distantHorizonsLoaded && NativeCoreBridge.initialized) {
            val vanillaRenderDistanceBlocks = (mc.options.renderDistance().get() * 16).toFloat()
            NativeCoreBridge.setLodCutoffDistance(maxOf(vanillaRenderDistanceBlocks - 16.0f, 32.0f))
        }

        packStd140(
            sunDirX = sunDir.x,
            sunDirY = sunDir.y,
            sunDirZ = sunDir.z,
            rain = rainLevel,
            thunder = thunderLevel,
            gameTimeSec = ((level.gameTime % 72000L).toFloat() + partialTick) / 20.0f,
            eyeInFluid = eyeInFluid,
            depthFar = effectiveDepthFar,
            heldLightMain = heldMain,
            heldLightOff = heldOff,
            discardHistory = isDiscontinuousFrame || skyClockJumped,
            dimensionId = dimensionId,
            heldPosition = primaryHeldPosition,
            heldLightKind = primaryLightKind,
        )
        NativeCoreBridge.updateWorldUbo(uniformBufferAddress(), WORLD_UBO_BYTES)
        collectAndSubmitEmissiveLights(
            level = level,
            localPlayer = player,
            partialTick = partialTick,
            camX = cameraPosX,
            camY = cameraPosY,
            camZ = cameraPosZ,
            firstPerson = firstPerson,
            heldMainLevel = heldMain,
            heldMainStack = mainStack,
            heldMainPos = mainPos,
            heldOffLevel = heldOff,
            heldOffStack = offStack,
            heldOffPos = offPos,
            forceScan = dimensionChanged || nearbyLightingDirty,
        )
        nearbyLightingDirty = false
        frameIndex++
        return applyProjectionJitter(rasterProjection, renderWidth, renderHeight)
    }

    private fun updateSmoothBiome(target: BiomeAtmosphereProfile, snap: Boolean) {
        if (snap || !smoothBiomeInitialized) {
            smoothBiomeTemp = target.temp
            smoothBiomeHumidity = target.humidity
            smoothBiomeCloudScale = target.cloudScale
            smoothBiomeFogR = target.fogR
            smoothBiomeFogG = target.fogG
            smoothBiomeFogB = target.fogB
            smoothBiomeDensity = target.densityScale
            smoothBiomeInitialized = true
            return
        }
        val alpha = 0.035f
        smoothBiomeTemp += (target.temp - smoothBiomeTemp) * alpha
        smoothBiomeHumidity += (target.humidity - smoothBiomeHumidity) * alpha
        smoothBiomeCloudScale += (target.cloudScale - smoothBiomeCloudScale) * alpha
        smoothBiomeFogR += (target.fogR - smoothBiomeFogR) * alpha
        smoothBiomeFogG += (target.fogG - smoothBiomeFogG) * alpha
        smoothBiomeFogB += (target.fogB - smoothBiomeFogB) * alpha
        smoothBiomeDensity += (target.densityScale - smoothBiomeDensity) * alpha
    }

    private fun resolveBiomeAtmosphere(
        biomeIdRaw: String,
        dimensionId: Int,
        vanillaFogColor: Vector4fc?,
    ): BiomeAtmosphereProfile {
        val b = biomeIdRaw.lowercase(Locale.ROOT)
        if (dimensionId == 1) {
            return when {
                b.contains("soul_sand_valley") -> BiomeAtmosphereProfile(0.25f, 0.25f, 0.0f, 0.06f, 0.20f, 0.26f, 1.35f)
                b.contains("warped_forest") -> BiomeAtmosphereProfile(0.45f, 0.70f, 0.0f, 0.05f, 0.22f, 0.20f, 1.25f)
                b.contains("crimson_forest") -> BiomeAtmosphereProfile(0.85f, 0.45f, 0.0f, 0.28f, 0.05f, 0.04f, 1.30f)
                b.contains("basalt_deltas") -> BiomeAtmosphereProfile(0.65f, 0.20f, 0.0f, 0.16f, 0.14f, 0.16f, 1.55f)
                else -> BiomeAtmosphereProfile(0.95f, 0.15f, 0.0f, 0.25f, 0.06f, 0.03f, 1.25f)
            }
        }
        if (dimensionId == 2) {
            return BiomeAtmosphereProfile(0.2f, 0.1f, 0.0f, 0.020f, 0.014f, 0.042f, 0.75f)
        }

        // Overworld biome atmosphere presets (linear fog tint + temperature/humidity/cloud/density)
        val preset = when {
            b.contains("desert") -> BiomeAtmosphereProfile(0.96f, 0.06f, 0.55f, 0.72f, 0.60f, 0.42f, 1.32f)
            b.contains("badlands") || b.contains("mesa") -> BiomeAtmosphereProfile(0.92f, 0.10f, 0.62f, 0.70f, 0.50f, 0.36f, 1.28f)
            b.contains("savanna") -> BiomeAtmosphereProfile(0.82f, 0.22f, 0.78f, 0.62f, 0.60f, 0.48f, 1.12f)
            b.contains("snow") || b.contains("ice") || b.contains("frozen") || b.contains("grove") || b.contains("jagged") ->
                BiomeAtmosphereProfile(0.08f, 0.45f, 0.92f, 0.46f, 0.62f, 0.88f, 0.92f)
            b.contains("jungle") || b.contains("bamboo") -> BiomeAtmosphereProfile(0.76f, 0.95f, 1.22f, 0.36f, 0.62f, 0.66f, 1.48f)
            b.contains("swamp") || b.contains("mangrove") -> BiomeAtmosphereProfile(0.62f, 0.92f, 1.18f, 0.36f, 0.44f, 0.38f, 1.68f)
            b.contains("pale_garden") -> BiomeAtmosphereProfile(0.32f, 0.72f, 1.25f, 0.36f, 0.40f, 0.46f, 1.62f)
            b.contains("taiga") || b.contains("spruce") || b.contains("windswept") ->
                BiomeAtmosphereProfile(0.26f, 0.62f, 1.05f, 0.40f, 0.56f, 0.78f, 1.12f)
            b.contains("cherry") || b.contains("meadow") -> BiomeAtmosphereProfile(0.46f, 0.52f, 0.92f, 0.54f, 0.62f, 0.88f, 0.95f)
            b.contains("deep_dark") -> BiomeAtmosphereProfile(0.18f, 0.40f, 0.85f, 0.08f, 0.14f, 0.20f, 1.50f)
            b.contains("lush_caves") -> BiomeAtmosphereProfile(0.55f, 0.88f, 1.00f, 0.32f, 0.54f, 0.52f, 1.30f)
            b.contains("ocean") || b.contains("beach") || b.contains("river") ->
                BiomeAtmosphereProfile(0.50f, 0.72f, 1.02f, 0.44f, 0.64f, 0.90f, 1.02f)
            else -> BiomeAtmosphereProfile(0.52f, 0.50f, 1.00f, 0.46f, 0.64f, 0.90f, 1.00f)
        }

        // Gently blend vanilla's biome fog color if provided (e.g. modded biomes)
        if (vanillaFogColor != null && vanillaFogColor.x() >= 0.0f && vanillaFogColor.y() >= 0.0f && vanillaFogColor.z() >= 0.0f) {
            val vR = srgbToLinear(vanillaFogColor.x().coerceIn(0.0f, 1.0f))
            val vG = srgbToLinear(vanillaFogColor.y().coerceIn(0.0f, 1.0f))
            val vB = srgbToLinear(vanillaFogColor.z().coerceIn(0.0f, 1.0f))
            val luma = vR * 0.2126f + vG * 0.7152f + vB * 0.0722f
            if (luma > 0.02f) {
                return preset.copy(
                    fogR = preset.fogR * 0.75f + vR * 0.25f,
                    fogG = preset.fogG * 0.75f + vG * 0.25f,
                    fogB = preset.fogB * 0.75f + vB * 0.25f,
                )
            }
        }
        return preset
    }

    internal fun sunAngleFromOverworldClock(clockTimeTicks: Long, partialTick: Float = 0.0f): Float {
        val dayTicks = 24000.0
        val raw = clockTimeTicks.toDouble() + partialTick.toDouble().coerceIn(0.0, 1.0)
        val wrapped = ((raw % dayTicks) + dayTicks) % dayTicks
        val d = wrapped / dayTicks - 0.25
        val frac = d - floor(d)
        val eased = 0.5 - cos(frac * Math.PI) * 0.5
        val celestial = frac + (eased - frac) / 3.0
        return (celestial * (2.0 * Math.PI)).toFloat()
    }

    internal fun resolveOverworldSunAngle(
        reportedSunAngle: Float,
        overworldClockTime: Long?,
        partialTick: Float,
        preferClock: Boolean,
    ): Float {
        if (overworldClockTime == null) {
            return if (reportedSunAngle.isFinite()) reportedSunAngle else 0.0f
        }
        val clockAngle = sunAngleFromOverworldClock(overworldClockTime, partialTick)
        if (preferClock || !reportedSunAngle.isFinite()) {
            return clockAngle
        }
        // Fallback when vanilla SkyRenderState.sunAngle stays at 0.0f off-noon.
        if (abs(reportedSunAngle) < 1e-5f && cos(clockAngle) < 0.98f) {
            return clockAngle
        }
        return reportedSunAngle
    }

    internal fun sunDirectionForDimension(dimensionId: Int, sunAngle: Float): Vector3f =
        sunDirectionForDimensionInto(dimensionId, sunAngle, Vector3f())

    private fun sunDirectionForDimensionInto(dimensionId: Int, sunAngle: Float, dst: Vector3f): Vector3f =
        when (dimensionId) {
            1 -> dst.set(0.0f, 1.0f, 0.0f)
            2 -> dst.set(0.28f, 0.88f, -0.38f).normalize()
            // Match the Overworld sky renderer: rotate world-up around X, then Y by -90 degrees.
            else -> dst.set(-sin(sunAngle), cos(sunAngle), 0.0f)
        }

    internal fun heldLightRadius(heldLevel: Float): Float =
        (heldLevel.coerceIn(0.0f, 1.0f) * 14.0f + 2.0f).coerceIn(3.5f, 16.0f)

    internal fun heldLightIntensity(heldLevel: Float, firstPerson: Boolean): Float {
        val level = heldLevel.coerceIn(0.0f, 1.0f)
        return level * level * if (firstPerson) 7.8f else 6.0f
    }

    internal fun isOpenFlameBlockName(descriptionIdLower: String): Boolean =
        (descriptionIdLower.contains("torch") &&
            !descriptionIdLower.contains("redstone") &&
            !descriptionIdLower.contains("torchflower"))
            || descriptionIdLower.contains("campfire")
            || descriptionIdLower.contains("candle")
            || (descriptionIdLower.contains("fire") && !descriptionIdLower.contains("firefly"))

    private fun isEntitySubmergedInWater(
        entity: Entity?,
        firstPersonLocal: Boolean = false,
        eyeSubmergedInWater: Boolean = false,
    ): Boolean {
        if (entity == null) return eyeSubmergedInWater
        return entity.isUnderWater || (firstPersonLocal && eyeSubmergedInWater)
    }

    private fun computeHeldHandPosition(
        level: net.minecraft.client.multiplayer.ClientLevel,
        player: Player,
        partialTick: Float,
        firstPerson: Boolean,
        isOffhand: Boolean,
        snapBlend: Boolean,
    ): Vec3 {
        val arm = if (isOffhand) player.mainArm.opposite else player.mainArm
        val side = if (arm == HumanoidArm.RIGHT) 1.0 else -1.0
        val candidateRel: Vec3
        val fallbackRel: Vec3
        if (firstPerson) {
            invView.transformPosition(scratchHeldVec.set((side * 0.18).toFloat(), -0.18f, -0.24f))
            candidateRel = Vec3(scratchHeldVec.x.toDouble(), scratchHeldVec.y.toDouble(), scratchHeldVec.z.toDouble())
            invView.transformPosition(scratchHeldVec.set(0.0f, -0.10f, -0.05f))
            fallbackRel = Vec3(scratchHeldVec.x.toDouble(), scratchHeldVec.y.toDouble(), scratchHeldVec.z.toDouble())
        } else {
            val playerPosition = player.getPosition(partialTick)
            val yaw = Math.toRadians(net.minecraft.util.Mth.rotLerp(partialTick, player.yBodyRotO, player.yBodyRot).toDouble())
            fallbackRel = Vec3(
                playerPosition.x - cameraPosX,
                playerPosition.y - cameraPosY + player.eyeHeight * 0.73,
                playerPosition.z - cameraPosZ,
            )
            candidateRel = Vec3(
                fallbackRel.x - cos(yaw) * side * 0.32 - sin(yaw) * 0.20,
                fallbackRel.y,
                fallbackRel.z - sin(yaw) * side * 0.32 + cos(yaw) * 0.20,
            )
        }
        return clampHeldPositionOutOfSolid(
            level = level,
            camX = cameraPosX,
            camY = cameraPosY,
            camZ = cameraPosZ,
            candidateRel = candidateRel,
            fallbackRel = fallbackRel,
            isOffhand = isOffhand,
            snap = snapBlend,
        )
    }

    private fun clampHeldPositionOutOfSolid(
        level: net.minecraft.client.multiplayer.ClientLevel,
        camX: Double,
        camY: Double,
        camZ: Double,
        candidateRel: Vec3,
        fallbackRel: Vec3,
        isOffhand: Boolean = false,
        snap: Boolean = false,
    ): Vec3 {
        heldBlockProbe.set(
            floor(camX + candidateRel.x).toInt(),
            floor(camY + candidateRel.y).toInt(),
            floor(camZ + candidateRel.z).toInt(),
        )
        val targetBlend = if (!level.getBlockState(heldBlockProbe).canOcclude()) {
            0.0
        } else {
            val midX = (candidateRel.x + fallbackRel.x) * 0.5
            val midY = (candidateRel.y + fallbackRel.y) * 0.5
            val midZ = (candidateRel.z + fallbackRel.z) * 0.5
            heldBlockProbe.set(
                floor(camX + midX).toInt(),
                floor(camY + midY).toInt(),
                floor(camZ + midZ).toInt(),
            )
            if (!level.getBlockState(heldBlockProbe).canOcclude()) 0.5 else 1.0
        }
        val prevBlend = if (isOffhand) smoothOffHeldBlend else smoothMainHeldBlend
        val blend = if (snap) targetBlend else prevBlend + (targetBlend - prevBlend) * 0.25
        if (isOffhand) smoothOffHeldBlend = blend else smoothMainHeldBlend = blend
        if (blend <= 1e-4) return candidateRel
        if (blend >= 1.0 - 1e-4) return fallbackRel
        return Vec3(
            candidateRel.x + (fallbackRel.x - candidateRel.x) * blend,
            candidateRel.y + (fallbackRel.y - candidateRel.y) * blend,
            candidateRel.z + (fallbackRel.z - candidateRel.z) * blend,
        )
    }

    private fun isOpenFlameItem(item: Item): Boolean {
        if (item == Items.BLAZE_ROD || item == Items.BLAZE_POWDER || item == Items.FIRE_CHARGE) return true
        if (item is BlockItem) {
            return isOpenFlameBlockName(item.block.descriptionId.lowercase(Locale.ROOT))
        }
        return false
    }

    private fun isFlickeringItem(item: Item?): Boolean {
        if (item == null) return false
        if (item == Items.LAVA_BUCKET || item == Items.BLAZE_ROD || item == Items.BLAZE_POWDER ||
            item == Items.FIRE_CHARGE || item == Items.MAGMA_CREAM
        ) return true
        if (item is BlockItem) {
            val name = item.block.descriptionId.lowercase(Locale.ROOT)
            return (name.contains("torch") && !name.contains("torchflower"))
                || name.contains("campfire") || name.contains("candle")
                || name.contains("lantern") || name.contains("fire") || name.contains("lava")
                || name.contains("furnace") || name.contains("smoker") || name.contains("trial_spawner")
        }
        return false
    }

    private fun effectiveHeldLightLevel(stack: ItemStack, submergedInWater: Boolean): Float {
        if (stack.isEmpty) return 0.0f
        if (submergedInWater && isOpenFlameItem(stack.item)) return 0.0f
        return itemLightLevel(stack)
    }

    internal fun packHeldItemLight(packedLight: Int, heldLightLevel: Float): Int {
        if (heldLightLevel <= 0.0f || !heldLightLevel.isFinite()) return packedLight
        val heldBlock = (heldLightLevel.coerceIn(0.0f, 1.0f) * 15.0f).roundToInt().coerceIn(0, 15) shl 4
        val block = maxOf(packedLight and 0xFFFF, heldBlock)
        return (packedLight and -0x10000) or block
    }

    @JvmStatic
    fun heldItemLight(stack: ItemStack, packedLight: Int): Int {
        if (!VulkanShaders.shaderPackActive) return packedLight
        val player = Minecraft.getInstance().player
        if (player != null && player.isSpectator) return packedLight
        val level = effectiveHeldLightLevel(stack, isEntitySubmergedInWater(player))
        return packHeldItemLight(packedLight, level)
    }

    private fun resolveHeldItemRgb(item: Item?, stack: ItemStack = ItemStack.EMPTY): Int = when (item) {
        Items.HEAVY_CORE -> 0x7A96C8
        Items.GLOW_BERRIES, Items.GLOWSTONE_DUST -> 0xFFBE58
        Items.REDSTONE -> 0xFF2A14
        is BlockItem -> {
            if (item.block.defaultBlockState().lightEmission > 0) {
                resolveBlockEmissiveRgb(item.block)
            } else if (stack.isEnchanted) {
                0xA86CFF
            } else {
                resolveBlockEmissiveRgb(item.block)
            }
        }
        Items.LAVA_BUCKET, Items.MAGMA_CREAM -> 0xFF6B1E
        Items.BLAZE_ROD, Items.BLAZE_POWDER -> 0xFF9C38
        Items.FIRE_CHARGE -> 0xFF7A24
        Items.GLOW_INK_SAC -> 0x5CE6E0
        Items.NETHER_STAR -> 0xE6ECFF
        Items.ECHO_SHARD -> 0x1CC8D8
        Items.RECOVERY_COMPASS -> 0x20D4E0
        Items.HEART_OF_THE_SEA -> 0x5AE8FF
        Items.PRISMARINE_CRYSTALS -> 0x7CF0E8
        Items.ENDER_EYE -> 0x46E0A0
        Items.ENDER_PEARL -> 0x2EC4A0
        Items.END_CRYSTAL -> 0xE47CFF
        Items.DRAGON_BREATH -> 0xD85CFF
        Items.AMETHYST_SHARD -> 0xB878FF
        Items.EXPERIENCE_BOTTLE -> 0x9CFF58
        Items.ENCHANTED_GOLDEN_APPLE -> 0xD488FF
        Items.TOTEM_OF_UNDYING -> 0xFFD858
        Items.SPECTRAL_ARROW -> 0xFFE066
        Items.TRIAL_KEY -> 0xFF9C48
        Items.OMINOUS_TRIAL_KEY -> 0x34D8C8
        Items.BREEZE_ROD -> 0x9EC8FF
        null -> 0
        else -> if (stack.isEnchanted) 0xA86CFF else 0xFF9A3C
    }

    internal fun resolveHeldLightKind(item: Item, stack: ItemStack): Int {
        when (item) {
            Items.HEAVY_CORE -> return 7
            Items.GLOW_BERRIES -> return 5
            Items.REDSTONE -> return 4
        }
        if (item is BlockItem) {
            if (item.block.defaultBlockState().lightEmission <= 0 && stack.isEnchanted) {
                return 9
            }
            val name = item.block.descriptionId.lowercase(Locale.ROOT)
            return when {
                name.contains("soul") -> 1
                name.contains("lava") || name.contains("magma") -> 2
                name.contains("sea_lantern") || name.contains("conduit") ||
                    name.contains("beacon") || name.contains("sea_pickle") -> 3
                name.contains("redstone") -> 4
                name.contains("glowstone") || name.contains("shroomlight") ||
                    name.contains("ochre_froglight") || name.contains("cave_vines") ||
                    name.contains("glow_berries") || name.contains("firefly_bush") -> 5
                name.contains("end_rod") || name.contains("amethyst") ||
                    name.contains("pearlescent_froglight") -> 6
                name.contains("sculk") || name.contains("vault") -> 7
                name.contains("copper_torch") || name.contains("copper_lantern") ||
                    name.contains("verdant_froglight") || name.contains("glow_lichen") -> 8
                name.contains("crying_obsidian") || name.contains("respawn_anchor") ||
                    name.contains("ender_chest") || name.contains("enchanting_table") ||
                    name.contains("dragon_egg") || name.contains("nether_portal") -> 9
                else -> 0
            }
        }
        return when (item) {
            Items.LAVA_BUCKET, Items.MAGMA_CREAM -> 2
            Items.GLOW_INK_SAC, Items.HEART_OF_THE_SEA, Items.PRISMARINE_CRYSTALS -> 3
            Items.REDSTONE -> 4
            Items.GLOW_BERRIES, Items.GLOWSTONE_DUST, Items.TOTEM_OF_UNDYING, Items.SPECTRAL_ARROW -> 5
            Items.NETHER_STAR, Items.AMETHYST_SHARD, Items.BREEZE_ROD -> 6
            Items.ECHO_SHARD, Items.RECOVERY_COMPASS, Items.OMINOUS_TRIAL_KEY, Items.HEAVY_CORE -> 7
            Items.ENDER_EYE, Items.ENDER_PEARL, Items.EXPERIENCE_BOTTLE -> 8
            Items.END_CRYSTAL, Items.DRAGON_BREATH, Items.ENCHANTED_GOLDEN_APPLE -> 9
            Items.BLAZE_ROD, Items.BLAZE_POWDER, Items.FIRE_CHARGE, Items.TRIAL_KEY -> 0
            else -> if (stack.isEnchanted) 9 else 0
        }
    }

    internal fun itemLightLevel(stack: ItemStack): Float {
        if (stack.isEmpty) return 0.0f
        val item = stack.item
        if (item is BlockItem) {
            val blockLight = item.block.defaultBlockState().lightEmission
            if (blockLight > 0) {
                return (blockLight.toFloat() / 15.0f).coerceIn(0.0f, 1.0f)
            }
        }
        val explicit = when (item) {
            Items.LAVA_BUCKET -> 1.0f
            Items.NETHER_STAR, Items.END_CRYSTAL -> 0.9f
            Items.HEART_OF_THE_SEA -> 0.85f
            Items.BLAZE_ROD, Items.BLAZE_POWDER -> 0.8f
            Items.FIRE_CHARGE, Items.ENCHANTED_GOLDEN_APPLE -> 0.75f
            Items.ENDER_EYE, Items.DRAGON_BREATH -> 0.7f
            Items.MAGMA_CREAM, Items.EXPERIENCE_BOTTLE, Items.TOTEM_OF_UNDYING, Items.OMINOUS_TRIAL_KEY -> 0.65f
            Items.GLOW_INK_SAC, Items.GLOW_BERRIES, Items.ECHO_SHARD, Items.GLOWSTONE_DUST, Items.BREEZE_ROD -> 0.6f
            Items.PRISMARINE_CRYSTALS, Items.SPECTRAL_ARROW, Items.TRIAL_KEY -> 0.55f
            Items.RECOVERY_COMPASS -> 0.5f
            Items.ENDER_PEARL, Items.AMETHYST_SHARD, Items.HEAVY_CORE -> 0.45f
            Items.REDSTONE -> 0.4f
            else -> 0.0f
        }
        if (explicit > 0.0f) return explicit
        return if (stack.isEnchanted) 0.35f else 0.0f
    }

    private fun srgbToLinear(c: Float): Float = c.coerceIn(0.0f, 1.0f).pow(2.2f)

    internal fun packStd140(
        sunDirX: Float,
        sunDirY: Float,
        sunDirZ: Float,
        rain: Float,
        thunder: Float,
        gameTimeSec: Float,
        eyeInFluid: Int,
        depthFar: Float,
        heldLightMain: Float,
        heldLightOff: Float,
        discardHistory: Boolean = false,
        dimensionId: Int = 0,
        heldPosition: Vec3 = Vec3.ZERO,
        heldLightKind: Int = 0,
    ) {
        uboBuffer.clear()
        curViewProj.get(0, uboBuffer)
        prevViewProj.get(64, uboBuffer)
        invProj.get(128, uboBuffer)
        invView.get(192, uboBuffer)

        uboBuffer.putFloat(256, (cameraPosX - floor(cameraPosX)).toFloat())
        uboBuffer.putFloat(260, (cameraPosY - floor(cameraPosY)).toFloat())
        uboBuffer.putFloat(264, (cameraPosZ - floor(cameraPosZ)).toFloat())
        uboBuffer.putInt(268, if (discardHistory) 0 else frameIndex)

        uboBuffer.putFloat(272, (cameraPosX - prevCameraPosX).toFloat())
        uboBuffer.putFloat(276, (cameraPosY - prevCameraPosY).toFloat())
        uboBuffer.putFloat(280, (cameraPosZ - prevCameraPosZ).toFloat())
        uboBuffer.putFloat(284, gameTimeSec)

        uboBuffer.putFloat(288, sunDirX)
        uboBuffer.putFloat(292, sunDirY)
        uboBuffer.putFloat(296, sunDirZ)
        uboBuffer.putFloat(300, rain)

        uboBuffer.putFloat(304, thunder)
        uboBuffer.putInt(308, eyeInFluid)
        uboBuffer.putFloat(312, depthFar)
        uboBuffer.putFloat(316, maxOf(heldLightMain, heldLightOff))

        // 320..479: user pack runtime settings (slots 0..39), cached until pack or overrides change.
        val cfg = VulkanShaders.config
        val pack = ShaderPackManager.activePack
        if (!cachedPackSettingsValid || cfg !== cachedPackConfigRef || pack !== cachedPackRef) {
            val tierIndex = cfg.renderTier.tierIndex
            val packId = pack?.manifest?.id ?: ""
            val overrides = if (pack != null) cfg.activePackOverrides(packId) else emptyMap()
            val overridesHash = overrides.hashCode()
            if (!cachedPackSettingsValid ||
                cachedPackId != packId ||
                cachedPackTierIndex != tierIndex ||
                cachedPackRef !== pack ||
                cachedPackOverridesHash != overridesHash
            ) {
                val runtimeSettings = pack?.manifest?.runtimeSettings.orEmpty()
                for (i in 0 until BIOME_TELEMETRY_START_FLOAT) {
                    cachedPackSettings[i] = if (i < runtimeSettings.size) {
                        val setting = runtimeSettings[i]
                        val rawOverride = if (setting.appliesToTier(tierIndex)) overrides[setting.key] else null
                        settingAsFloat(setting, rawOverride)
                    } else {
                        0.0f
                    }
                }
                cachedPackId = packId
                cachedPackTierIndex = tierIndex
                cachedPackOverridesHash = overridesHash
            }
            cachedPackConfigRef = cfg
            cachedPackRef = pack
            cachedPackSettingsValid = true
        }
        for (i in 0 until BIOME_TELEMETRY_START_FLOAT) {
            uboBuffer.putFloat(320 + i * 4, cachedPackSettings[i])
        }

        // 480..511: reserved packParams[10..11] biome & dimension atmosphere telemetry
        uboBuffer.putFloat(320 + 40 * 4, smoothBiomeTemp)
        uboBuffer.putFloat(320 + 41 * 4, smoothBiomeHumidity)
        uboBuffer.putFloat(320 + 42 * 4, dimensionId.toFloat())
        uboBuffer.putFloat(320 + 43 * 4, smoothBiomeCloudScale)
        uboBuffer.putFloat(320 + 44 * 4, smoothBiomeFogR)
        uboBuffer.putFloat(320 + 45 * 4, smoothBiomeFogG)
        uboBuffer.putFloat(320 + 46 * 4, smoothBiomeFogB)
        uboBuffer.putFloat(320 + 47 * 4, smoothBiomeDensity)
        uboBuffer.putFloat(512, heldPosition.x.toFloat())
        uboBuffer.putFloat(516, heldPosition.y.toFloat())
        uboBuffer.putFloat(520, heldPosition.z.toFloat())
        uboBuffer.putFloat(524, heldLightKind.coerceIn(0, 9).toFloat())

        // 528..535: VSL pack-parameter BDA. Native core overwrites this with the live
        // device address each frame; the placeholder keeps the 544-byte upload deterministic.
        uboBuffer.putLong(528, 0L)
    }

    internal fun updateMatrices(
        modelView: Matrix4fc?,
        unjitteredProjection: Matrix4fc?,
        isDiscontinuousFrame: Boolean,
        sampledProjection: Matrix4fc? = unjitteredProjection,
    ) {
        prevViewProj.set(curViewProj)
        if (modelView != null && unjitteredProjection != null) {
            curViewProj.set(unjitteredProjection).mul(modelView)
            invProj.set(sampledProjection ?: unjitteredProjection).invert()
            invView.set(modelView).invert()
            if (isDiscontinuousFrame) {
                prevViewProj.set(curViewProj)
            }
        }
    }

    // Vanilla encodes walking, hurt and portal transforms in the uploaded projection.
    // Keep these transforms in the view so rays also get the correct displaced origin.
    internal fun effectiveWorldView(
        projection: Matrix4fc, rasterProjection: Matrix4fc, rotation: Matrix4fc,
    ): Matrix4f = effectiveWorldViewInto(projection, rasterProjection, rotation, Matrix4f())

    private fun effectiveWorldViewInto(
        projection: Matrix4fc, rasterProjection: Matrix4fc, rotation: Matrix4fc, dst: Matrix4f,
    ): Matrix4f = dst.set(projection).invert().mul(rasterProjection).mul(rotation)

    internal fun writeInverseRasterView(dst: ByteBuffer, offset: Int) {
        rasterInvView.get(offset, dst)
    }

    @JvmStatic
    fun halton(index: Int, base: Int): Float {
        require(base >= 2) { "Halton base must be >= 2 (got $base)" }
        var f = 1.0f
        var r = 0.0f
        var i = index
        while (i > 0) {
            f /= base.toFloat()
            r += f * (i % base).toFloat()
            i /= base
        }
        return r
    }

    @JvmStatic
    fun updateJitter(renderWidth: Int, renderHeight: Int, temporalActive: Boolean) {
        prevJitterX = jitterX
        prevJitterY = jitterY
        if (!temporalActive || renderWidth <= 0 || renderHeight <= 0) {
            jitterX = 0.0f
            jitterY = 0.0f
            return
        }
        val sampleIndex = Math.floorMod(frameIndex, JITTER_PHASE_COUNT) + 1
        jitterX = halton(sampleIndex, 2) - 0.5f
        jitterY = halton(sampleIndex, 3) - 0.5f
    }

    @JvmStatic
    fun applyProjectionJitter(proj: Matrix4fc, width: Int, height: Int): Matrix4f =
        applyProjectionJitterInto(proj, width, height, Matrix4f())

    private fun applyProjectionJitterInto(proj: Matrix4fc, width: Int, height: Int, dst: Matrix4f): Matrix4f {
        dst.set(proj)
        if (width <= 0 || height <= 0 || (jitterX == 0.0f && jitterY == 0.0f)) {
            return dst
        }
        val ndcJitterX = (2.0f * jitterX) / width.toFloat()
        val ndcJitterY = (2.0f * jitterY) / height.toFloat()
        val m03 = dst.m03()
        val m13 = dst.m13()
        val m23 = dst.m23()
        val m33 = dst.m33()
        dst.m00(dst.m00() + ndcJitterX * m03)
        dst.m10(dst.m10() + ndcJitterX * m13)
        dst.m20(dst.m20() + ndcJitterX * m23)
        dst.m30(dst.m30() + ndcJitterX * m33)
        dst.m01(dst.m01() + ndcJitterY * m03)
        dst.m11(dst.m11() + ndcJitterY * m13)
        dst.m21(dst.m21() + ndcJitterY * m23)
        dst.m31(dst.m31() + ndcJitterY * m33)
        return dst
    }

    internal fun settingAsFloat(def: PackSettingDefinition, overrideRaw: String?): Float {
        val raw = overrideRaw ?: def.defaultValue
        return when (def.type) {
            SettingType.BOOL -> if (raw.equals("true", ignoreCase = true)) 1.0f else 0.0f
            SettingType.INT, SettingType.FLOAT -> raw.toFloatOrNull() ?: 0.0f
            SettingType.ENUM -> def.options.indexOf(raw).coerceAtLeast(0).toFloat()
            SettingType.COLOR_RGB -> {
                val rgb24 = parseColorRgb(raw) ?: parseColorRgb(def.defaultValue) ?: return 0.0f
                packRgbAsFloat(rgb24)
            }
        }
    }

    @JvmStatic
    fun parseColorRgb(raw: String): Int? {
        val s = raw.trim()
        if (s.isEmpty()) return null

        val hexBody = when {
            s.startsWith("#") -> s.substring(1)
            s.startsWith("0x", ignoreCase = true) -> s.substring(2)
            !s.contains(',') -> s
            else -> null
        }
        if (hexBody != null) {
            if (hexBody.length == 6) {
                return hexBody.toIntOrNull(16)?.and(MAX_RGB24)
            }
            if (hexBody.length == 3 && s.startsWith("#")) {
                val parsed = hexBody.toIntOrNull(16) ?: return null
                val r = ((parsed shr 8) and 0xF) * 17
                val g = ((parsed shr 4) and 0xF) * 17
                val b = (parsed and 0xF) * 17
                return (r shl 16) or (g shl 8) or b
            }
            return null
        }

        val parts = s.split(',').map { it.trim() }
        if (parts.size != 3 || parts.any { it.isEmpty() }) return null

        val hasDecimal = parts.any { it.contains('.') }
        val floats = parts.map { it.toFloatOrNull() ?: return null }
        if (floats.any { !it.isFinite() }) return null

        val normalizedInput = hasDecimal && floats.all { it in 0.0f..1.0f }
        val r: Int
        val g: Int
        val b: Int
        if (normalizedInput) {
            r = (floats[0] * 255.0f).roundToInt().coerceIn(0, 255)
            g = (floats[1] * 255.0f).roundToInt().coerceIn(0, 255)
            b = (floats[2] * 255.0f).roundToInt().coerceIn(0, 255)
        } else {
            r = floats[0].roundToInt().coerceIn(0, 255)
            g = floats[1].roundToInt().coerceIn(0, 255)
            b = floats[2].roundToInt().coerceIn(0, 255)
        }
        return (r shl 16) or (g shl 8) or b
    }

    @JvmStatic
    fun packRgbAsFloat(rgb24: Int): Float =
        (rgb24 and MAX_RGB24).toFloat() / MAX_RGB24.toFloat()

    @JvmStatic
    fun unpackFloatAsRgb(packed: Float): Int =
        if (!packed.isFinite()) 0 else (packed.coerceIn(0.0f, 1.0f) * MAX_RGB24.toFloat()).roundToInt() and MAX_RGB24

    fun unpackColorRgb(rgb24: Int): Triple<Int, Int, Int> {
        val masked = rgb24 and MAX_RGB24
        return Triple((masked shr 16) and 0xFF, (masked shr 8) and 0xFF, masked and 0xFF)
    }

    @JvmStatic
    fun formatColorRgb(rgb24: Int): String =
        String.format(Locale.ROOT, "#%06X", rgb24 and MAX_RGB24)

    @JvmStatic
    fun packEmissiveLightDesc(
        buffer: ByteBuffer,
        index: Int,
        worldX: Float,
        worldY: Float,
        worldZ: Float,
        radius: Float,
        r: Float,
        g: Float,
        b: Float,
        intensity: Float,
        normalX: Float = 0.0f,
        normalY: Float = 0.0f,
        normalZ: Float = 0.0f,
        flags: Int = 0,
    ) {
        val baseOff = index * EMISSIVE_LIGHT_DESC_BYTES
        buffer.putFloat(baseOff + 0, worldX)
        buffer.putFloat(baseOff + 4, worldY)
        buffer.putFloat(baseOff + 8, worldZ)
        buffer.putFloat(baseOff + 12, radius)
        buffer.putFloat(baseOff + 16, r)
        buffer.putFloat(baseOff + 20, g)
        buffer.putFloat(baseOff + 24, b)
        buffer.putFloat(baseOff + 28, intensity)
        buffer.putFloat(baseOff + 32, normalX)
        buffer.putFloat(baseOff + 36, normalY)
        buffer.putFloat(baseOff + 40, normalZ)
        buffer.putInt(baseOff + 44, flags)
    }

    private fun resolveBlockEmissiveRgb(block: Block): Int {
        val blockName = block.descriptionId.lowercase(Locale.ROOT)
        return when {
            blockName.contains("soul") -> 0x4BC8FF
            blockName.contains("calibrated_sculk_sensor") -> 0x28D4DC
            blockName.contains("sculk_catalyst") || blockName.contains("sculk_sensor") ||
                blockName.contains("sculk_shrieker") -> 0x16B8C8
            blockName.contains("vault") -> 0x2AD0C4
            blockName.contains("heavy_core") -> 0x7A96C8
            blockName.contains("copper_torch") || blockName.contains("copper_lantern") -> 0x52E69E
            blockName.contains("redstone") -> 0xFF2A14
            blockName.contains("sea_lantern") || blockName.contains("conduit") -> 0x9EF0FF
            blockName.contains("sea_pickle") -> 0x8EF2D8
            blockName.contains("glow_lichen") -> 0x98E4B8
            blockName.contains("pearlescent_froglight") -> 0xF4B4FF
            blockName.contains("verdant_froglight") -> 0xA8F296
            blockName.contains("ochre_froglight") -> 0xFFD88A
            blockName.contains("amethyst") || blockName.contains("crying_obsidian") ||
                blockName.contains("nether_portal") || blockName.contains("respawn_anchor") ||
                blockName.contains("ender_chest") || blockName.contains("enchanting_table") ||
                blockName.contains("dragon_egg") -> 0xB056FF
            blockName.contains("lava") || blockName.contains("magma") -> 0xFF6B1E
            blockName.contains("end_rod") || blockName.contains("beacon") -> 0xE6EEFF
            blockName.contains("firefly_bush") -> 0xE8F078
            blockName.contains("cave_vines") || blockName.contains("glow_berries") ||
                blockName.contains("glowstone") || blockName.contains("shroomlight") -> 0xFFBC58
            blockName.contains("trial_spawner") -> 0xFF8830
            blockName.contains("furnace") || blockName.contains("blast_furnace") ||
                blockName.contains("smoker") -> 0xFF8226
            blockName.contains("eyeblossom") -> 0xFF9028
            blockName.contains("jack_o_lantern") || blockName.contains("torchflower") -> 0xFFA236
            blockName.contains("copper_bulb") || blockName.contains("brewing_stand") -> 0xFF9E44
            else -> 0xFF9838
        }
    }

    private fun resolveBlockLightProfile(block: Block): BlockLightProfile {
        blockLightProfileCache[block]?.let { return it }
        val name = block.descriptionId.lowercase(Locale.ROOT)
        val rgb = resolveBlockEmissiveRgb(block)
        val rLin = srgbToLinear(((rgb shr 16) and 0xFF) / 255.0f)
        val gLin = srgbToLinear(((rgb shr 8) and 0xFF) / 255.0f)
        val bLin = srgbToLinear((rgb and 0xFF) / 255.0f)

        val isThin = name.contains("torch") || name.contains("candle") || name.contains("lantern")
            || name.contains("end_rod") || name.contains("fire") || name.contains("campfire")
            || name.contains("sea_pickle") || name.contains("glow_lichen") || name.contains("amethyst")
            || name.contains("cave_vines") || name.contains("glow_berries") || name.contains("eyeblossom")
            || name.contains("firefly_bush") || name.contains("brewing_stand") || name.contains("dragon_egg")
            || name.contains("torchflower")
        val hasFlicker = (name.contains("torch") && !name.contains("torchflower"))
            || name.contains("fire") || name.contains("campfire")
            || name.contains("candle") || name.contains("lava") || name.contains("lantern")
            || name.contains("furnace") || name.contains("blast_furnace") || name.contains("smoker")
            || name.contains("trial_spawner") || name.contains("firefly_bush")
        val wallTorch = block is net.minecraft.world.level.block.WallTorchBlock
        var flags = 0
        if (isThin) flags = flags or THIN_EMITTER_FLAG
        if (hasFlicker) flags = flags or FLICKER_FLAG
        if (name.contains("torch") && !name.contains("torchflower") && !wallTorch) flags = flags or STANDING_TORCH_FLAG

        val yOffset = when {
            wallTorch -> 0.84f
            name.contains("torchflower") -> 0.42f
            name.contains("torch") -> 0.75f
            name.contains("jack_o_lantern") -> 0.50f
            name.contains("lantern") -> 0.44f
            name.contains("campfire") -> 0.62f
            name.contains("candle") -> 0.45f
            name.contains("sea_pickle") -> 0.30f
            name.contains("eyeblossom") -> 0.38f
            name.contains("firefly_bush") -> 0.45f
            name.contains("sculk_sensor") || name.contains("calibrated_sculk_sensor") ||
                name.contains("sculk_shrieker") -> 0.45f
            name.contains("enchanting_table") -> 0.65f
            name.contains("brewing_stand") || name.contains("dragon_egg") -> 0.55f
            name.contains("ender_chest") -> 0.52f
            else -> 0.50f
        }
        val intensityScale = when {
            name.contains("redstone") -> 0.65f
            name.contains("brewing_stand") -> 0.62f
            name.contains("glow_lichen") -> 0.68f
            name.contains("dragon_egg") || name.contains("torchflower") -> 0.72f
            name.contains("sculk_catalyst") || name.contains("sculk_sensor") ||
                name.contains("calibrated_sculk_sensor") || name.contains("sculk_shrieker") -> 0.74f
            name.contains("enchanting_table") || name.contains("ender_chest") -> 0.78f
            name.contains("eyeblossom") || name.contains("firefly_bush") -> 0.80f
            name.contains("sea_pickle") || name.contains("cave_vines") || name.contains("glow_berries") -> 0.86f
            name.contains("trial_spawner") || name.contains("vault") || name.contains("respawn_anchor") -> 0.95f
            name.contains("jack_o_lantern") || name.contains("furnace") ||
                name.contains("blast_furnace") || name.contains("smoker") -> 0.98f
            name.contains("lava") -> 1.15f
            name.contains("campfire") -> 1.35f
            name.contains("torch") || name.contains("lantern") -> 1.05f
            else -> 0.92f
        }

        val profile = BlockLightProfile(rLin, gLin, bLin, yOffset, flags, intensityScale)
        blockLightProfileCache[block] = profile
        return profile
    }

    private fun packRgbLight(
        index: Int,
        worldX: Float,
        worldY: Float,
        worldZ: Float,
        radius: Float,
        rgb: Int,
        intensity: Float,
        flags: Int,
    ) {
        packEmissiveLightDesc(
            buffer = emissiveLightBuffer,
            index = index,
            worldX = worldX,
            worldY = worldY,
            worldZ = worldZ,
            radius = radius,
            r = SRGB_TO_LINEAR_LUT[(rgb shr 16) and 0xFF],
            g = SRGB_TO_LINEAR_LUT[(rgb shr 8) and 0xFF],
            b = SRGB_TO_LINEAR_LUT[rgb and 0xFF],
            intensity = intensity,
            flags = flags,
        )
    }

    private fun packLinearLight(
        index: Int,
        worldX: Float,
        worldY: Float,
        worldZ: Float,
        radius: Float,
        color: LinearRgb,
        intensity: Float,
        flags: Int,
    ) {
        packEmissiveLightDesc(
            buffer = emissiveLightBuffer,
            index = index,
            worldX = worldX,
            worldY = worldY,
            worldZ = worldZ,
            radius = radius,
            r = color.r,
            g = color.g,
            b = color.b,
            intensity = intensity,
            flags = flags,
        )
    }

    private fun collectAndSubmitEmissiveLights(
        level: net.minecraft.client.multiplayer.ClientLevel,
        localPlayer: Player?,
        partialTick: Float,
        camX: Double,
        camY: Double,
        camZ: Double,
        firstPerson: Boolean,
        heldMainLevel: Float,
        heldMainStack: ItemStack,
        heldMainPos: Vec3,
        heldOffLevel: Float,
        heldOffStack: ItemStack,
        heldOffPos: Vec3,
        forceScan: Boolean,
    ) {
        if (!NativeCoreBridge.initialized) return

        val tierIndex = VulkanShaders.config.renderTier.tierIndex
        val maxTotalLights = if (tierIndex >= 2) MAX_EMISSIVE_LIGHTS else MAX_EMISSIVE_LIGHTS_LOW_TIER

        val baseX = floor(camX).toInt()
        val baseY = floor(camY).toInt()
        val baseZ = floor(camZ).toInt()
        val scanBucket = level.gameTime / EMISSIVE_SCAN_INTERVAL_TICKS

        val moveDx = baseX - lastEmissiveScanX
        val moveDy = baseY - lastEmissiveScanY
        val moveDz = baseZ - lastEmissiveScanZ
        val movedDistSq = if (lastEmissiveScanX == Int.MIN_VALUE) {
            Long.MAX_VALUE
        } else {
            moveDx.toLong() * moveDx + moveDy.toLong() * moveDy + moveDz.toLong() * moveDz
        }
        val movedMoreThanOneBlock = movedDistSq > 1L
        val movedFarEnough = movedDistSq >= EMISSIVE_RESCAN_MOVE_DIST_SQ

        val rescan = forceScan || movedFarEnough || (movedMoreThanOneBlock && scanBucket != lastEmissiveScanBucket)
        if (rescan) {
            scanBlockLights(level, baseX, baseY, baseZ, scanBucket)
        }

        var writeIdx = 0

        // 1. Local player held lights (main hand and offhand simultaneously).
        if (heldMainLevel > 0.0f && writeIdx < maxTotalLights) {
            val rgb = resolveHeldItemRgb(heldMainStack.item, heldMainStack)
            var flags = HELD_LIGHT_FLAG or THIN_EMITTER_FLAG
            if (isFlickeringItem(heldMainStack.item)) flags = flags or FLICKER_FLAG
            packRgbLight(
                index = writeIdx++,
                worldX = heldMainPos.x.toFloat(),
                worldY = heldMainPos.y.toFloat(),
                worldZ = heldMainPos.z.toFloat(),
                radius = heldLightRadius(heldMainLevel),
                rgb = rgb,
                intensity = heldLightIntensity(heldMainLevel, firstPerson),
                flags = flags,
            )
        }
        if (heldOffLevel > 0.0f && writeIdx < maxTotalLights) {
            val rgb = resolveHeldItemRgb(heldOffStack.item, heldOffStack)
            var flags = HELD_LIGHT_FLAG or THIN_EMITTER_FLAG or OFFHAND_HELD_FLAG
            if (isFlickeringItem(heldOffStack.item)) flags = flags or FLICKER_FLAG
            packRgbLight(
                index = writeIdx++,
                worldX = heldOffPos.x.toFloat(),
                worldY = heldOffPos.y.toFloat(),
                worldZ = heldOffPos.z.toFloat(),
                radius = heldLightRadius(heldOffLevel),
                rgb = rgb,
                intensity = heldLightIntensity(heldOffLevel, firstPerson),
                flags = flags,
            )
        }

        // 2. Dynamic entity lights (dropped items, burning entities, glowing mobs, projectiles, other players).
        val entityLimit = minOf(maxTotalLights, writeIdx + MAX_DYNAMIC_ENTITY_LIGHTS)
        writeIdx = collectDynamicEntityLights(
            level = level,
            localPlayer = localPlayer,
            partialTick = partialTick,
            camX = camX,
            camY = camY,
            camZ = camZ,
            startIndex = writeIdx,
            maxIndex = entityLimit,
        )

        // 3. Append cached nearby block lights up to the tier cap.
        val blockCount = minOf(cachedBlockLightCount, maxTotalLights - writeIdx)
        for (i in 0 until blockCount) {
            packEmissiveLightDesc(
                buffer = emissiveLightBuffer,
                index = writeIdx++,
                worldX = heapX[i],
                worldY = heapY[i],
                worldZ = heapZ[i],
                radius = heapRadius[i],
                r = heapR[i],
                g = heapG[i],
                b = heapB[i],
                intensity = heapIntensity[i],
                normalX = 0.0f,
                normalY = 0.0f,
                normalZ = 0.0f,
                flags = heapFlags[i],
            )
        }

        NativeCoreBridge.submitEmissiveLights(
            MemoryUtil.memAddress(emissiveLightBuffer),
            writeIdx,
            camX,
            camY,
            camZ,
        )
    }

    private fun collectDynamicEntityLights(
        level: net.minecraft.client.multiplayer.ClientLevel,
        localPlayer: Player?,
        partialTick: Float,
        camX: Double,
        camY: Double,
        camZ: Double,
        startIndex: Int,
        maxIndex: Int,
    ): Int {
        if (startIndex >= maxIndex) return startIndex
        val maxDistSq = (LIGHT_SCAN_RADIUS + 8.0) * (LIGHT_SCAN_RADIUS + 8.0)
        var idx = startIndex

        for (entity in level.entitiesForRendering()) {
            if (idx >= maxIndex) break

            val type = entity.type
            val isLightning = type === EntityTypes.LIGHTNING_BOLT
            if (!isLightning) {
                val dx = entity.x - camX
                val dy = entity.y - camY
                val dz = entity.z - camZ
                if (dx * dx + dy * dy + dz * dz > maxDistSq) continue
            }

            val isBurning = entity.isOnFire && entity !== localPlayer

            // Dropped emissive items: filter before allocating interpolated Vec3.
            if (entity is ItemEntity) {
                val stack = entity.item
                val itemLevel = effectiveHeldLightLevel(stack, isEntitySubmergedInWater(entity))
                if (!isBurning && itemLevel <= 0.0f) continue
                val pos = entity.getPosition(partialTick)
                val wx = pos.x.toFloat()
                val wz = pos.z.toFloat()
                if (isBurning) {
                    packLinearLight(
                        index = idx++,
                        worldX = wx,
                        worldY = (pos.y + entity.bbHeight * 0.5).toFloat(),
                        worldZ = wz,
                        radius = 12.0f,
                        color = COLOR_ENTITY_FIRE,
                        intensity = 7.5f,
                        flags = THIN_EMITTER_FLAG or FLICKER_FLAG,
                    )
                    if (idx >= maxIndex) break
                }
                if (itemLevel > 0.0f) {
                    val rgb = resolveHeldItemRgb(stack.item, stack)
                    var flags = THIN_EMITTER_FLAG
                    if (isFlickeringItem(stack.item)) flags = flags or FLICKER_FLAG
                    packRgbLight(
                        index = idx++,
                        worldX = wx,
                        worldY = (pos.y + 0.25).toFloat(),
                        worldZ = wz,
                        radius = heldLightRadius(itemLevel) * 0.75f,
                        rgb = rgb,
                        intensity = itemLevel * itemLevel * 5.8f,
                        flags = flags,
                    )
                }
                continue
            }

            // Other players holding emissive items: filter before allocating interpolated Vec3.
            if (entity is Player) {
                if (entity === localPlayer || entity.isSpectator) continue
                val submerged = isEntitySubmergedInWater(entity)
                val otherMain = entity.mainHandItem
                val otherMainLvl = effectiveHeldLightLevel(otherMain, submerged)
                val otherOff = entity.offhandItem
                val otherOffLvl = effectiveHeldLightLevel(otherOff, submerged)
                if (!isBurning && otherMainLvl <= 0.0f && otherOffLvl <= 0.0f) continue
                val pos = entity.getPosition(partialTick)
                val wx = pos.x.toFloat()
                val wz = pos.z.toFloat()
                if (isBurning) {
                    packLinearLight(
                        index = idx++,
                        worldX = wx,
                        worldY = (pos.y + entity.bbHeight * 0.5).toFloat(),
                        worldZ = wz,
                        radius = 12.0f,
                        color = COLOR_ENTITY_FIRE,
                        intensity = 7.5f,
                        flags = THIN_EMITTER_FLAG or FLICKER_FLAG,
                    )
                    if (idx >= maxIndex) break
                }
                val handY = (pos.y + entity.eyeHeight * 0.73).toFloat()
                if (otherMainLvl > 0.0f && idx < maxIndex) {
                    val rgb = resolveHeldItemRgb(otherMain.item, otherMain)
                    var flags = THIN_EMITTER_FLAG
                    if (isFlickeringItem(otherMain.item)) flags = flags or FLICKER_FLAG
                    packRgbLight(
                        index = idx++,
                        worldX = wx,
                        worldY = handY,
                        worldZ = wz,
                        radius = heldLightRadius(otherMainLvl),
                        rgb = rgb,
                        intensity = heldLightIntensity(otherMainLvl, firstPerson = false),
                        flags = flags,
                    )
                }
                if (otherOffLvl > 0.0f && idx < maxIndex) {
                    val rgb = resolveHeldItemRgb(otherOff.item, otherOff)
                    var flags = THIN_EMITTER_FLAG or OFFHAND_HELD_FLAG
                    if (isFlickeringItem(otherOff.item)) flags = flags or FLICKER_FLAG
                    packRgbLight(
                        index = idx++,
                        worldX = wx,
                        worldY = handY,
                        worldZ = wz,
                        radius = heldLightRadius(otherOffLvl),
                        rgb = rgb,
                        intensity = heldLightIntensity(otherOffLvl, firstPerson = false),
                        flags = flags,
                    )
                }
                continue
            }

            // Filter non-emissive entity types before calling getPosition(partialTick).
            val isEmissiveMob = when (type) {
                EntityTypes.BLAZE,
                EntityTypes.GLOW_SQUID,
                EntityTypes.ALLAY,
                EntityTypes.MAGMA_CUBE,
                EntityTypes.WARDEN,
                EntityTypes.SHULKER_BULLET,
                EntityTypes.SPECTRAL_ARROW,
                EntityTypes.FIREBALL,
                EntityTypes.SMALL_FIREBALL,
                EntityTypes.DRAGON_FIREBALL,
                EntityTypes.TNT,
                EntityTypes.END_CRYSTAL,
                EntityTypes.LIGHTNING_BOLT,
                EntityTypes.EXPERIENCE_ORB -> true
                EntityTypes.CREEPER -> (entity as? Creeper)?.isPowered == true
                else -> false
            }
            if (!isBurning && !isEmissiveMob) continue

            val pos = entity.getPosition(partialTick)
            val midY = (pos.y + entity.bbHeight * 0.5).toFloat()
            val wx = pos.x.toFloat()
            val wz = pos.z.toFloat()

            if (isBurning) {
                packLinearLight(
                    index = idx++,
                    worldX = wx,
                    worldY = midY,
                    worldZ = wz,
                    radius = 12.0f,
                    color = COLOR_ENTITY_FIRE,
                    intensity = 7.5f,
                    flags = THIN_EMITTER_FLAG or FLICKER_FLAG,
                )
                if (idx >= maxIndex) break
            }

            if (isEmissiveMob) {
                when (type) {
                    EntityTypes.BLAZE -> packLinearLight(idx++, wx, midY, wz, 13.0f, COLOR_ENTITY_BLAZE, 8.5f, THIN_EMITTER_FLAG or FLICKER_FLAG)
                    EntityTypes.GLOW_SQUID -> packLinearLight(idx++, wx, midY, wz, 10.0f, COLOR_ENTITY_GLOW_SQUID, 5.5f, THIN_EMITTER_FLAG)
                    EntityTypes.ALLAY -> packLinearLight(idx++, wx, midY, wz, 9.0f, COLOR_ENTITY_ALLAY, 4.8f, THIN_EMITTER_FLAG)
                    EntityTypes.MAGMA_CUBE -> packLinearLight(idx++, wx, midY, wz, 10.0f, COLOR_ENTITY_MAGMA_CUBE, 6.2f, THIN_EMITTER_FLAG or FLICKER_FLAG)
                    EntityTypes.WARDEN -> packLinearLight(idx++, wx, midY, wz, 11.0f, COLOR_ENTITY_WARDEN, 5.8f, THIN_EMITTER_FLAG)
                    EntityTypes.CREEPER -> packLinearLight(idx++, wx, midY, wz, 11.0f, COLOR_ENTITY_CHARGED_CREEPER, 6.5f, THIN_EMITTER_FLAG or FLICKER_FLAG)
                    EntityTypes.SHULKER_BULLET -> packLinearLight(idx++, wx, midY, wz, 8.0f, COLOR_ENTITY_SHULKER_BULLET, 4.5f, THIN_EMITTER_FLAG)
                    EntityTypes.SPECTRAL_ARROW -> packLinearLight(idx++, wx, midY, wz, 9.0f, COLOR_ENTITY_SPECTRAL_ARROW, 5.0f, THIN_EMITTER_FLAG)
                    EntityTypes.FIREBALL -> packLinearLight(idx++, wx, midY, wz, 14.0f, COLOR_ENTITY_FIREBALL, 10.0f, THIN_EMITTER_FLAG or FLICKER_FLAG)
                    EntityTypes.SMALL_FIREBALL -> packLinearLight(idx++, wx, midY, wz, 11.0f, COLOR_ENTITY_SMALL_FIREBALL, 7.5f, THIN_EMITTER_FLAG or FLICKER_FLAG)
                    EntityTypes.DRAGON_FIREBALL -> packLinearLight(idx++, wx, midY, wz, 13.0f, COLOR_ENTITY_DRAGON_FIREBALL, 9.0f, THIN_EMITTER_FLAG or FLICKER_FLAG)
                    EntityTypes.TNT -> packLinearLight(idx++, wx, midY, wz, 10.0f, COLOR_ENTITY_TNT, 6.5f, THIN_EMITTER_FLAG or FLICKER_FLAG)
                    EntityTypes.END_CRYSTAL -> packLinearLight(idx++, wx, (pos.y + 1.0).toFloat(), wz, 14.0f, COLOR_ENTITY_END_CRYSTAL, 9.5f, THIN_EMITTER_FLAG or FLICKER_FLAG)
                    EntityTypes.LIGHTNING_BOLT -> packLinearLight(idx++, wx, (pos.y + 4.0).toFloat(), wz, 28.0f, COLOR_ENTITY_LIGHTNING, 22.0f, THIN_EMITTER_FLAG)
                    EntityTypes.EXPERIENCE_ORB -> packLinearLight(idx++, wx, midY, wz, 5.5f, COLOR_ENTITY_EXPERIENCE_ORB, 3.2f, THIN_EMITTER_FLAG)
                }
            }
        }
        return idx
    }

    private fun ensureSecBuildCapacity(needed: Int) {
        if (needed <= secBuildX.size) return
        val next = maxOf(needed, secBuildX.size * 2)
        secBuildX = secBuildX.copyOf(next)
        secBuildY = secBuildY.copyOf(next)
        secBuildZ = secBuildZ.copyOf(next)
        secBuildRadius = secBuildRadius.copyOf(next)
        secBuildR = secBuildR.copyOf(next)
        secBuildG = secBuildG.copyOf(next)
        secBuildB = secBuildB.copyOf(next)
        secBuildIntensity = secBuildIntensity.copyOf(next)
        secBuildFlags = secBuildFlags.copyOf(next)
    }

    private fun buildSectionLightCache(
        section: LevelChunkSection,
        secOriginX: Int,
        secOriginY: Int,
        secOriginZ: Int,
    ): SectionLightCache {
        if (!section.maybeHas { it.lightEmission > 0 }) return emptySectionCache(section)
        var n = 0
        for (ly in 0 until 16) {
            val wy = secOriginY + ly
            for (lz in 0 until 16) {
                val wz = secOriginZ + lz
                for (lx in 0 until 16) {
                    val state = section.getBlockState(lx, ly, lz)
                    val emission = state.lightEmission
                    if (emission > 0) {
                        ensureSecBuildCapacity(n + 1)
                        val wx = secOriginX + lx
                        val normEm = emission.toFloat() / 15.0f
                        val profile = resolveBlockLightProfile(state.block)
                        val wallTorchFacing = if (state.block is net.minecraft.world.level.block.WallTorchBlock) {
                            state.getValue(BlockStateProperties.HORIZONTAL_FACING)
                        } else null
                        secBuildX[n] = wx + 0.5f - (wallTorchFacing?.stepX ?: 0) * 0.23f
                        secBuildY[n] = wy + profile.yOffset
                        secBuildZ[n] = wz + 0.5f - (wallTorchFacing?.stepZ ?: 0) * 0.23f
                        secBuildRadius[n] = emission + 2.0f
                        secBuildR[n] = profile.rLin
                        secBuildG[n] = profile.gLin
                        secBuildB[n] = profile.bLin
                        secBuildIntensity[n] = normEm * normEm * 8.5f * profile.intensityScale
                        secBuildFlags[n] = profile.flags
                        n++
                    }
                }
            }
        }
        if (n == 0) return emptySectionCache(section)
        return SectionLightCache(
            sectionRef = WeakReference(section),
            count = n,
            worldX = secBuildX.copyOf(n),
            worldY = secBuildY.copyOf(n),
            worldZ = secBuildZ.copyOf(n),
            radius = secBuildRadius.copyOf(n),
            r = secBuildR.copyOf(n),
            g = secBuildG.copyOf(n),
            b = secBuildB.copyOf(n),
            intensity = secBuildIntensity.copyOf(n),
            flags = secBuildFlags.copyOf(n),
        )
    }

    private fun heapSwap(a: Int, b: Int) {
        val td = heapDistSq[a]; heapDistSq[a] = heapDistSq[b]; heapDistSq[b] = td
        val tx = heapX[a]; heapX[a] = heapX[b]; heapX[b] = tx
        val ty = heapY[a]; heapY[a] = heapY[b]; heapY[b] = ty
        val tz = heapZ[a]; heapZ[a] = heapZ[b]; heapZ[b] = tz
        val tr = heapRadius[a]; heapRadius[a] = heapRadius[b]; heapRadius[b] = tr
        val tR = heapR[a]; heapR[a] = heapR[b]; heapR[b] = tR
        val tG = heapG[a]; heapG[a] = heapG[b]; heapG[b] = tG
        val tB = heapB[a]; heapB[a] = heapB[b]; heapB[b] = tB
        val ti = heapIntensity[a]; heapIntensity[a] = heapIntensity[b]; heapIntensity[b] = ti
        val tf = heapFlags[a]; heapFlags[a] = heapFlags[b]; heapFlags[b] = tf
    }

    private fun heapSet(
        idx: Int,
        distSq: Float,
        x: Float,
        y: Float,
        z: Float,
        radius: Float,
        r: Float,
        g: Float,
        b: Float,
        intensity: Float,
        flags: Int,
    ) {
        heapDistSq[idx] = distSq
        heapX[idx] = x
        heapY[idx] = y
        heapZ[idx] = z
        heapRadius[idx] = radius
        heapR[idx] = r
        heapG[idx] = g
        heapB[idx] = b
        heapIntensity[idx] = intensity
        heapFlags[idx] = flags
    }

    private fun heapSiftUp(startIdx: Int) {
        var i = startIdx
        while (i > 0) {
            val parent = (i - 1) ushr 1
            if (heapDistSq[i] <= heapDistSq[parent]) break
            heapSwap(i, parent)
            i = parent
        }
    }

    private fun heapSiftDown(size: Int) {
        var i = 0
        while (true) {
            val left = (i shl 1) + 1
            if (left >= size) break
            val right = left + 1
            val largest = if (right < size && heapDistSq[right] > heapDistSq[left]) right else left
            if (heapDistSq[i] >= heapDistSq[largest]) break
            heapSwap(i, largest)
            i = largest
        }
    }

    private fun sectionMinDistSq(camX: Float, camY: Float, camZ: Float, secOriginX: Int, secOriginY: Int, secOriginZ: Int): Float {
        val dx = when {
            camX < secOriginX -> secOriginX - camX
            camX > secOriginX + 16f -> camX - (secOriginX + 16f)
            else -> 0f
        }
        val dy = when {
            camY < secOriginY -> secOriginY - camY
            camY > secOriginY + 16f -> camY - (secOriginY + 16f)
            else -> 0f
        }
        val dz = when {
            camZ < secOriginZ -> secOriginZ - camZ
            camZ > secOriginZ + 16f -> camZ - (secOriginZ + 16f)
            else -> 0f
        }
        return dx * dx + dy * dy + dz * dz
    }

    private fun scanBlockLights(
        level: net.minecraft.client.multiplayer.ClientLevel,
        baseX: Int,
        baseY: Int,
        baseZ: Int,
        scanBucket: Long,
    ) {
        lastEmissiveScanX = baseX
        lastEmissiveScanY = baseY
        lastEmissiveScanZ = baseZ
        lastEmissiveScanBucket = scanBucket

        val camSecX = baseX shr 4
        val camSecY = baseY shr 4
        val camSecZ = baseZ shr 4
        val camXf = cameraPosX.toFloat()
        val camYf = cameraPosY.toFloat()
        val camZf = cameraPosZ.toFloat()
        val radiusF = LIGHT_SCAN_RADIUS.toFloat()
        val radiusSq = radiusF * radiusF
        val minSecY = level.minY shr 4
        val maxSecY = (level.maxY - 1) shr 4

        val chunkSource = level.chunkSource
        var heapSize = 0

        synchronized(sectionLightCacheLock) {
            if (sectionLightCache.size > MAX_CACHED_SECTIONS) {
                sectionLightCache.clear()
            }
        }

        var offIdx = 0
        while (offIdx < SORTED_SECTION_OFFSETS.size) {
            val cx = camSecX + SORTED_SECTION_OFFSETS[offIdx]
            val sy = camSecY + SORTED_SECTION_OFFSETS[offIdx + 1]
            val cz = camSecZ + SORTED_SECTION_OFFSETS[offIdx + 2]
            offIdx += 3

            if (sy < minSecY || sy > maxSecY) continue

            val secOriginX = cx shl 4
            val secOriginY = sy shl 4
            val secOriginZ = cz shl 4
            val secMinD2 = sectionMinDistSq(camXf, camYf, camZf, secOriginX, secOriginY, secOriginZ)
            if (secMinD2 > radiusSq) continue
            if (heapSize == MAX_BLOCK_LIGHTS && secMinD2 >= heapDistSq[0]) continue

            val chunk = chunkSource.getChunkNow(cx, cz) ?: continue
            val secIdx = chunk.getSectionIndexFromSectionY(sy)
            val sections = chunk.sections
            if (secIdx < 0 || secIdx >= sections.size) continue
            val section = sections[secIdx]

            val secKey = SectionPos.asLong(cx, sy, cz)
            var cached: SectionLightCache? = synchronized(sectionLightCacheLock) {
                sectionLightCache.get(secKey)
            }
            if (cached == null || cached.sectionRef.get() !== section) {
                cached = buildSectionLightCache(section, secOriginX, secOriginY, secOriginZ)
                synchronized(sectionLightCacheLock) {
                    sectionLightCache.put(secKey, cached)
                }
            }

            val n = cached.count
            if (n == 0) continue

            val xs = cached.worldX
            val ys = cached.worldY
            val zs = cached.worldZ
            val rads = cached.radius
            val rs = cached.r
            val gs = cached.g
            val bs = cached.b
            val ints = cached.intensity
            val flgs = cached.flags

            for (i in 0 until n) {
                val wx = xs[i]
                val wy = ys[i]
                val wz = zs[i]
                val dx = wx - camXf
                val dy = wy - camYf
                val dz = wz - camZf
                val distSq = dx * dx + dy * dy + dz * dz
                if (distSq > radiusSq) continue

                if (heapSize < MAX_BLOCK_LIGHTS) {
                    heapSet(heapSize, distSq, wx, wy, wz, rads[i], rs[i], gs[i], bs[i], ints[i], flgs[i])
                    heapSiftUp(heapSize)
                    heapSize++
                } else if (distSq < heapDistSq[0]) {
                    heapSet(0, distSq, wx, wy, wz, rads[i], rs[i], gs[i], bs[i], ints[i], flgs[i])
                    heapSiftDown(MAX_BLOCK_LIGHTS)
                }
            }
        }

        cachedBlockLightCount = heapSize
    }

    fun uniformBufferAddress(): Long = MemoryUtil.memAddress(uboBuffer)

    fun uniformBufferReadOnly(): ByteBuffer = uboBuffer.asReadOnlyBuffer().order(uboBuffer.order())
}
