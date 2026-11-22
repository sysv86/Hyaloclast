package dev.vulkanshaders.bridge

import com.mojang.renderpearl.api.commands.RenderPass
import com.mojang.renderpearl.api.pipeline.RenderPipeline
import com.mojang.renderpearl.api.textures.GpuTexture
import com.mojang.renderpearl.api.textures.GpuTextureView
import com.mojang.renderpearl.backend.vulkan.VulkanGpuTexture
import com.mojang.renderpearl.backend.vulkan.VulkanGpuTextureView
import dev.vulkanshaders.VulkanShaders
import dev.vulkanshaders.backend.NativeCoreBridge
import dev.vulkanshaders.config.RenderTier
import dev.vulkanshaders.pipeline.GbufferPipelineRegistry
import dev.vulkanshaders.pipeline.GbufferTargetManager
import dev.vulkanshaders.pipeline.SectionMaterialPacking
import org.joml.Matrix4f
import java.lang.reflect.Field
import java.lang.reflect.Method
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.CopyOnWriteArrayList
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicLong

/**
 * Clean-room bridge for **Distant Horizons 3.x (MC 26.3 Blaze3D/RenderPearl)** LOD terrain
 * rendering, G-Buffer MRT integration, and ray-tracing BLAS ingestion.
 *
 * - Tiers 0-2: redirects `BlazeDhTerrainRenderer` into Hyaloclast's 3-MRT DH G-Buffer
 *   (`[dhColor, dhNormalRoughness, dhMaterialPbr]`), captures DH's inverse view-projection matrix,
 *   clip planes, and Vulkan texture handles for deferred shading and seamless horizon compositing.
 * - Tier 2 & 3: unpacks DH's 24-byte `POS_META_COLOR_MATERIAL_NORMAL_TEXTURE_OFFSET` vertices
 *   (with 16-byte fallback) into 28-byte `LodVertexGpu` structs and uploads them to the native
 *   Rust `LodBlasManager` via [NativeCoreBridge.uploadLodMeshCpu].
 */
object DistantHorizonsBridge {
    private const val DH_VERTEX_STRIDE_BYTES = 24
    private const val DH_LEGACY_VERTEX_STRIDE_BYTES = 16
    private const val GPU_LOD_VERTEX_STRIDE_BYTES = 28
    private const val DH_KEY_NAMESPACE = 0x44485F4CL shl 32 // "DH_L"
    private const val API_PROBE_BACKOFF_NS = 2_000_000_000L // 2 seconds (~40 ticks)

    // Material IDs and flag bits matching VulkanShadersTurboExtension and SectionMaterialPacking.
    private const val MATERIAL_FLAG_FOLIAGE_WIND = 1 shl 0
    private const val MATERIAL_FLAG_FLUID = 1 shl 1
    private const val MATERIAL_FLAG_EMISSIVE = 1 shl 2
    private const val MID_STONE = 1
    private const val MID_DIRT = 2
    private const val MID_WOOD = 3
    private const val MID_FOLIAGE = 4
    private const val MID_SAND = 5
    private const val MID_METAL = 6
    private const val MID_WATER = 8
    private const val MID_LAVA = 9
    private const val MID_EMISSIVE = 10
    private const val WATER_MATERIAL_FLAGS = (MID_WATER shl 3) or MATERIAL_FLAG_FLUID // 0x42
    private const val LAVA_MATERIAL_FLAGS = (MID_LAVA shl 3) or MATERIAL_FLAG_FLUID or MATERIAL_FLAG_EMISSIVE // 0x4E
    private const val FOLIAGE_MATERIAL_FLAGS = (MID_FOLIAGE shl 3) or MATERIAL_FLAG_FOLIAGE_WIND // 0x21
    private const val EMISSIVE_MATERIAL_FLAGS = (MID_EMISSIVE shl 3) or MATERIAL_FLAG_EMISSIVE // 0x54

    // Distant Horizons EDhApiBlockMaterial ordinal/index values.
    private const val DH_MAT_LEAVES = 1
    private const val DH_MAT_STONE = 2
    private const val DH_MAT_WOOD = 3
    private const val DH_MAT_METAL = 4
    private const val DH_MAT_DIRT = 5
    private const val DH_MAT_LAVA = 6
    private const val DH_MAT_DEEPSLATE = 7
    private const val DH_MAT_SNOW = 8
    private const val DH_MAT_SAND = 9
    private const val DH_MAT_TERRACOTTA = 10
    private const val DH_MAT_NETHER_STONE = 11
    private const val DH_MAT_WATER = 12
    private const val DH_MAT_ILLUMINATED = 14

    data class RegisteredLodColumn(
        val keyId: Long,
        val detailLevel: Int,
        val originBlockX: Int,
        val originBlockY: Int,
        val originBlockZ: Int,
        val scaleBlocks: Float,
        val vkBuffer: Long,
        val vertexCount: Int,
    )

    fun interface LodColumnProvider {
        fun syncLodColumns(cameraBlockX: Int, cameraBlockY: Int, cameraBlockZ: Int)
    }

    /** Identity-based key for tracking live `BlazeVertexBufferWrapper` instances safely. */
    private class WrapperIdentityKey(val wrapper: Any) {
        private val hash: Int = System.identityHashCode(wrapper)
        override fun hashCode(): Int = hash
        override fun equals(other: Any?): Boolean = other is WrapperIdentityKey && other.wrapper === wrapper
    }

    private class DhWrapperState(val lodKeyId: Long) {
        @Volatile var hasMetadata: Boolean = false
        @Volatile var originBlockX: Int = 0
        @Volatile var originBlockY: Int = 0
        @Volatile var originBlockZ: Int = 0
        @Volatile var detailLevel: Int = 0
        @Volatile var materialFlags: Int = 0
        @Volatile var uploaded: Boolean = false

        // Held only while awaiting container metadata from BlazeDhTerrainRenderer.render(...)
        @Volatile var pendingRawBytes: ByteArray? = null
        @Volatile var pendingVertexCount: Int = 0
        @Volatile var pendingStrideBytes: Int = DH_VERTEX_STRIDE_BYTES
    }

    private class ContainerReflectionCache(cls: Class<*>) {
        val minCornerGetter = method(cls, "getMinCornerBlockPos")
        val minCornerField = if (minCornerGetter == null) cls.getField("minCornerBlockPos") else null
        val detailLevelGetter = method(cls, "getDetailLevel")
        val sectionPos = if (detailLevelGetter == null) cls.getField("pos") else null
        val sectionDetailLevel = if (sectionPos != null) Class.forName(
            "com.seibel.distanthorizons.core.pos.DhSectionPos", false, cls.classLoader,
        ).getMethod("getDetailLevel", Long::class.javaPrimitiveType) else null
        val opaqueGetter = method(cls, "getVboOpaqueWrappers")
        val opaqueField = if (opaqueGetter == null) cls.getField("vboOpaqueWrappers") else null
        val noSsaoGetter = method(cls, "getVboWithoutSsaoWrappers")
        val transparentGetter = method(cls, "getVboTransparentWrappers")
        val transparentField = if (transparentGetter == null) cls.getField("vboTransparentWrappers") else null

        private fun method(cls: Class<*>, name: String) =
            runCatching { cls.getMethod(name) }.getOrNull()
    }

    private class ListReflectionCache(cls: Class<*>) {
        val sizeMethod: Method? = runCatching { cls.getMethod("size") }.getOrNull()
        val getMethod: Method? = runCatching { cls.getMethod("get", Int::class.javaPrimitiveType) }.getOrNull()
    }

    private class Vec3iReflectionCache(cls: Class<*>) {
        val xField: Field? = runCatching { cls.getField("x") }.getOrNull()
        val yField: Field? = runCatching { cls.getField("y") }.getOrNull()
        val zField: Field? = runCatching { cls.getField("z") }.getOrNull()
        val getX: Method? = if (xField == null) runCatching { cls.getMethod("getX") }.getOrNull() else null
        val getY: Method? = if (yField == null) runCatching { cls.getMethod("getY") }.getOrNull() else null
        val getZ: Method? = if (zField == null) runCatching { cls.getMethod("getZ") }.getOrNull() else null

        fun readX(vec: Any): Int = xField?.getInt(vec) ?: (getX?.invoke(vec) as? Number)?.toInt() ?: 0
        fun readY(vec: Any): Int = yField?.getInt(vec) ?: (getY?.invoke(vec) as? Number)?.toInt() ?: 0
        fun readZ(vec: Any): Int = zField?.getInt(vec) ?: (getZ?.invoke(vec) as? Number)?.toInt() ?: 0
    }

    @JvmField
    val distantHorizonsLoaded: Boolean = StandaloneRenderPearlBridge.isModLoaded("distanthorizons")

    private val activeLodColumns = ConcurrentHashMap<Long, RegisteredLodColumn>()
    private val wrapperStates = ConcurrentHashMap<WrapperIdentityKey, DhWrapperState>()
    private val lodProviders = CopyOnWriteArrayList<LodColumnProvider>()
    private val nextDhKeySeq = AtomicLong(1L)
    private val pendingUploadCount = AtomicInteger(0)
    private val unpackScratchBuffer = ThreadLocal<ByteBuffer?>()

    private val containerCache = ConcurrentHashMap<Class<*>, ContainerReflectionCache>()
    private val listCache = ConcurrentHashMap<Class<*>, ListReflectionCache>()
    private val vec3iCache = ConcurrentHashMap<Class<*>, Vec3iReflectionCache>()

    @Volatile private var dhApiProbed = false
    @Volatile private var lastApiProbeAttemptNs = 0L

    @Volatile private var dhRenderingEnabledGetter: Method? = null
    @Volatile private var dhLodChunkDistGetter: Method? = null
    @Volatile private var dhRenderingConfigObj: Any? = null
    @Volatile private var dhQualityConfigObj: Any? = null

    // Per-frame DH rendering state (populated by BlazeDhTerrainRenderer hook).
    @JvmField @Volatile var dhRenderedThisFrame: Boolean = false
    @Volatile private var dhGbufferClearedThisFrame: Boolean = false
    @Volatile private var currentRenderPassTransparent: Boolean = false

    @JvmField val dhInvViewProjMatrix: FloatArray = FloatArray(16)
    @JvmField @Volatile var dhNearClip: Float = 16.0f
    @JvmField @Volatile var dhFarClip: Float = 1024.0f
    @JvmField @Volatile var dhColorTexture: VulkanGpuTexture? = null
    @JvmField @Volatile var dhColorTextureView: VulkanGpuTextureView? = null
    @JvmField @Volatile var dhDepthTexture: VulkanGpuTexture? = null
    @JvmField @Volatile var dhDepthTextureView: VulkanGpuTextureView? = null

    // Cached reflection handles for DH runtime objects.
    @Volatile private var renderPassWrapperField: Field? = null
    @Volatile private var metaRendererInstance: Any? = null
    @Volatile private var metaColorWrapperField: Field? = null
    @Volatile private var metaDepthWrapperField: Field? = null
    @Volatile private var textureWrapperGetTexture: Method? = null
    @Volatile private var textureWrapperGetView: Method? = null
    @Volatile private var dhInvMvmProjField: Field? = null
    @Volatile private var dhMatCreateJomlMethod: Method? = null
    @Volatile private var dhMatGetValuesMethod: Method? = null
    @Volatile private var dhNearClipField: Field? = null
    @Volatile private var dhFarClipField: Field? = null
    @Volatile private var renderUtilNearClipMethod: Method? = null
    @Volatile private var renderUtilFarClipMethod: Method? = null
    @Volatile private var dhReflectionInitAttempted: Boolean = false

    @Volatile private var lastNativeLodEnabled: Boolean? = null
    private var lastSyncBucket: Long = Long.MIN_VALUE

    @JvmStatic
    fun registerLodProvider(provider: LodColumnProvider) {
        lodProviders.addIfAbsent(provider)
    }

    @JvmStatic
    fun unregisterLodProvider(provider: LodColumnProvider) {
        lodProviders.remove(provider)
    }

    /**
     * Probes `DhApi$Delayed.configs` via reflection with retry backoff (every 2 seconds) until
     * Distant Horizons finishes initializing its delayed API bindings. Never permanently latches
     * failure during early startup.
     */
    private fun ensureDhApiProbed() {
        if (dhApiProbed || !distantHorizonsLoaded) return
        val now = System.nanoTime()
        if (lastApiProbeAttemptNs != 0L && (now - lastApiProbeAttemptNs) < API_PROBE_BACKOFF_NS) {
            return
        }
        lastApiProbeAttemptNs = now

        runCatching {
            val delayedCls = Class.forName("com.seibel.distanthorizons.api.DhApi\$Delayed")
            val configsField = delayedCls.getDeclaredField("configs")
            configsField.isAccessible = true
            val configsObj = configsField.get(null) ?: return@runCatching
            val graphicsMethod = configsObj.javaClass.getMethod("graphics")
            val graphicsObj = graphicsMethod.invoke(configsObj) ?: return@runCatching

            var boundAny = false

            fun configEntry(target: Any, methodName: String): Any? =
                runCatching { target.javaClass.getMethod(methodName).invoke(target) }.getOrNull()

            fun configGetter(entry: Any): Method? =
                runCatching { entry.javaClass.getMethod("getValue") }.getOrNull()
                    ?: runCatching { entry.javaClass.getMethod("get") }.getOrNull()

            configEntry(graphicsObj, "renderingEnabled")?.let { entry ->
                configGetter(entry)?.let { getter ->
                    dhRenderingConfigObj = entry
                    dhRenderingEnabledGetter = getter
                    boundAny = true
                }
            }

            val distanceEntry = configEntry(graphicsObj, "chunkRenderDistance")
                ?: configEntry(graphicsObj, "quality")?.let { configEntry(it, "lodChunkRenderDistanceRadius") }
            if (distanceEntry != null) {
                configGetter(distanceEntry)?.let { getter ->
                    dhQualityConfigObj = distanceEntry
                    dhLodChunkDistGetter = getter
                    boundAny = true
                }
            }

            if (boundAny) {
                dhApiProbed = true
                VulkanShaders.log.info("DistantHorizonsBridge bound reflection hooks to Distant Horizons DhApi")
            }
        }.onFailure { t ->
            VulkanShaders.log.debug("Distant Horizons DhApi not ready yet (will retry): {}", t.message)
        }
    }

    private fun ensureDhRuntimeReflection(sampleParam: Any?) {
        if (dhReflectionInitAttempted) return
        dhReflectionInitAttempted = true
        if (sampleParam != null) {
            dhNearClipField = runCatching { sampleParam.javaClass.getField("nearClipPlane") }.getOrNull()
            dhFarClipField = runCatching { sampleParam.javaClass.getField("farClipPlane") }.getOrNull()
        }
        val cl = sampleParam?.javaClass?.classLoader ?: DistantHorizonsBridge::class.java.classLoader
        runCatching {
            val metaCls = Class.forName("com.seibel.distanthorizons.common.render.blaze.BlazeDhMetaRenderer", false, cl)
            metaRendererInstance = metaCls.getField("INSTANCE").get(null)
            metaColorWrapperField = metaCls.getField("dhColorTextureWrapper")
            metaDepthWrapperField = metaCls.getField("dhDepthTextureWrapper")
            val texWrapCls = Class.forName(
                "com.seibel.distanthorizons.common.render.blaze.wrappers.texture.BlazeTextureWrapper",
                false,
                cl,
            )
            textureWrapperGetTexture = texWrapCls.getMethod("getTexture")
            textureWrapperGetView = texWrapCls.getMethod("getTextureView")
        }
        runCatching {
            val rpCls = Class.forName(
                "com.seibel.distanthorizons.common.render.blaze.wrappers.RenderPassWrapper",
                false,
                cl,
            )
            val f = rpCls.getDeclaredField("renderPass")
            f.isAccessible = true
            renderPassWrapperField = f
        }
        runCatching {
            val renderUtilCls = Class.forName("com.seibel.distanthorizons.core.util.RenderUtil", false, cl)
            renderUtilNearClipMethod = runCatching { renderUtilCls.getMethod("getNearClipPlaneInBlocks") }.getOrNull()
            renderUtilFarClipMethod = runCatching { renderUtilCls.getMethod("getFarClipPlaneDistanceInBlocks") }.getOrNull()
        }
        runCatching {
            val dhMatCls = Class.forName("com.seibel.distanthorizons.core.util.math.DhMat4f", false, cl)
            val apiMatCls = Class.forName("com.seibel.distanthorizons.api.objects.math.DhApiMat4f", false, cl)
            dhMatCreateJomlMethod = runCatching { dhMatCls.getMethod("createJomlMatrix", apiMatCls) }.getOrNull()
            dhMatGetValuesMethod = runCatching { apiMatCls.getMethod("getValuesAsArray") }.getOrNull()
        }
    }

    @JvmStatic
    fun isDistantHorizonsRenderingEnabled(): Boolean {
        if (!distantHorizonsLoaded) return false
        ensureDhApiProbed()
        val getter = dhRenderingEnabledGetter
        val target = dhRenderingConfigObj
        if (getter != null && target != null) {
            return runCatching { (getter.invoke(target) as? Boolean) ?: true }.getOrDefault(true)
        }
        return activeLodColumns.isNotEmpty()
    }

    /**
     * Returns true when DH's single-attachment raster composite (`copyToMcTexture`) and vanilla
     * fade post-process (`BlazeVanillaFadeRenderer.render`) must be suppressed.
     */
    @JvmStatic
    fun shouldSuppressDhRasterComposite(): Boolean = shouldCancelDhTerrainRaster() || dhRenderedThisFrame

    /**
     * Returns true when DH's raster terrain draw pass (`BlazeDhTerrainRenderer.render`) should be
     * cancelled because Tier 3 Path Tracing ray-traces DH LOD geometry directly from the TLAS.
     */
    @JvmStatic
    fun shouldCancelDhTerrainRaster(): Boolean =
        VulkanShaders.shaderPackActive && VulkanShaders.config.renderTier == RenderTier.TIER_3_PATH_TRACING

    @JvmStatic
    fun effectiveFarPlaneBlocks(vanillaDepthFar: Float): Float {
        var effective = vanillaDepthFar
        if (distantHorizonsLoaded) {
            ensureDhApiProbed()
            val getter = dhLodChunkDistGetter
            val target = dhQualityConfigObj
            if (getter != null && target != null && isDistantHorizonsRenderingEnabled()) {
                val lodChunks = runCatching { (getter.invoke(target) as? Number)?.toInt() ?: 0 }.getOrDefault(0)
                if (lodChunks > 0) {
                    effective = maxOf(effective, (lodChunks * 16).toFloat().coerceAtMost(4096.0f))
                }
            } else if (activeLodColumns.isNotEmpty()) {
                effective = maxOf(effective, 1024.0f)
            }
        }
        return effective
    }

    /**
     * Synchronizes Distant Horizons active state with native `vs_core_set_lod_enabled`.
     * When Distant Horizons is absent or toggled off mid-session, disables native LOD rendering
     * and retires any cached LOD BLASes/slices cleanly to prevent stale geometry or crashes.
     */
    @JvmStatic
    fun syncNativeLodEnabled(enabled: Boolean) {
        if (!NativeCoreBridge.initialized) {
            lastNativeLodEnabled = null
            return
        }
        if (lastNativeLodEnabled == enabled) return
        lastNativeLodEnabled = enabled
        NativeCoreBridge.setLodEnabled(enabled)
        if (!enabled) {
            val hadColumns = activeLodColumns.isNotEmpty()
            activeLodColumns.clear()
            wrapperStates.clear()
            pendingUploadCount.set(0)
            if (hadColumns) {
                NativeCoreBridge.clearLodSlices()
            }
            VulkanShaders.log.debug("Disabled native Distant Horizons LOD integration and retired stale LOD BLASes")
        }
    }

    @JvmStatic
    fun onBeginFrame(gameTimeTicks: Long, cameraBlockX: Int, cameraBlockY: Int, cameraBlockZ: Int) {
        dhRenderedThisFrame = false
        dhGbufferClearedThisFrame = false
        currentRenderPassTransparent = false
        val dhActive = isDistantHorizonsRenderingEnabled() || lodProviders.isNotEmpty()
        syncNativeLodEnabled(dhActive)
        if (dhActive) {
            ensureDhApiProbed()
            val bucket = gameTimeTicks / 10L
            if (bucket != lastSyncBucket) {
                lastSyncBucket = bucket
                for (provider in lodProviders) {
                    runCatching { provider.syncLodColumns(cameraBlockX, cameraBlockY, cameraBlockZ) }
                }
            }
        }
    }

    /**
     * Called from `DhTerrainRendererMixin` at `@At("HEAD")` of `BlazeDhTerrainRenderer.render(...)`.
     *
     * Syncs container metadata for LOD BLAS ingestion, captures DH's per-frame inverse matrix,
     * clip planes, and color/depth Vulkan textures, and prepares/clears DH's auxiliary G-Buffer
     * targets before the opaque pass opens.
     */
    @JvmStatic
    fun onBeforeDhTerrainRender(apiRenderEventParam: Any?, renderPass: Any?, bufferContainers: Any?) {
        ensureDhRuntimeReflection(apiRenderEventParam)
        val dhEnabled = isDistantHorizonsRenderingEnabled()
        syncNativeLodEnabled(dhEnabled)
        if (!dhEnabled) return
        syncContainerMetadata(bufferContainers)

        val passName = renderPass?.toString() ?: ""
        currentRenderPassTransparent = renderPass == true || passName.equals("TRANSPARENT", ignoreCase = true)

        captureDhFrameParams(apiRenderEventParam)
        refreshDhTargetHandles()

        if (VulkanShaders.shaderPackActive && !shouldCancelDhTerrainRaster()) {
            val colorTex = dhColorTexture
            if (colorTex != null && !colorTex.isClosed) {
                GbufferTargetManager.ensureDhTargetsSized(colorTex.getWidth(0), colorTex.getHeight(0))
                if (!currentRenderPassTransparent && !dhGbufferClearedThisFrame) {
                    GbufferTargetManager.clearDhGbufferTargets()
                    dhGbufferClearedThisFrame = true
                }
            }
        }
    }

    /**
     * Called from `DhTerrainRendererMixin`'s `@WrapOperation` on `RenderPassWrapper.setPipeline(...)`
     * in Tiers 0-2. Attaches Hyaloclast's 3-MRT DH G-Buffer color views
     * (`[dhColorView, dhNormalRoughnessView, dhMaterialPbrView]`) and returns the matching
     * Hyaloclast DH G-Buffer `RenderPipeline`.
     */
    @JvmStatic
    fun prepareDhGbufferPass(renderPassWrapper: Any?, originalPipeline: RenderPipeline): RenderPipeline {
        if (!VulkanShaders.shaderPackActive || shouldCancelDhTerrainRaster() || renderPassWrapper == null) {
            return originalPipeline
        }
        val colorTex = dhColorTexture ?: return originalPipeline
        val colorView = dhColorTextureView ?: return originalPipeline
        if (colorTex.isClosed) return originalPipeline

        GbufferTargetManager.ensureDhTargetsSized(colorTex.getWidth(0), colorTex.getHeight(0))
        val normalView = GbufferTargetManager.dhNormalRoughnessView as? VulkanGpuTextureView ?: return originalPipeline
        val matView = GbufferTargetManager.dhMaterialPbrView as? VulkanGpuTextureView ?: return originalPipeline

        val isTransparent = currentRenderPassTransparent ||
            originalPipeline.location.path.contains("transparent", ignoreCase = true)
        val stride = originalPipeline.vertexFormatBindings.firstOrNull()?.vertexSize ?: return originalPipeline
        val dhPipeline = GbufferPipelineRegistry.dhTerrainPipeline(stride, isTransparent) ?: return originalPipeline

        val rpField = renderPassWrapperField ?: runCatching {
            renderPassWrapper.javaClass.getDeclaredField("renderPass").apply { isAccessible = true }
        }.getOrNull()?.also { renderPassWrapperField = it } ?: return originalPipeline

        val renderPass = runCatching { rpField.get(renderPassWrapper) as? RenderPass }.getOrNull()
            ?: return originalPipeline

        if (StandaloneRenderPearlBridge.vulkanTurboLoaded) {
            val applied = runCatching {
                dev.vulkanturbo.api.shader.TurboShadersBridge.overridePendingPassColorViews(
                    renderPass,
                    arrayOf(colorView, normalView, matView),
                )
            }.getOrDefault(false)
            if (applied) {
                dhRenderedThisFrame = true
                return dhPipeline
            }
        }
        return originalPipeline
    }

    private fun captureDhFrameParams(apiRenderEventParam: Any?) {
        if (apiRenderEventParam != null) {
            runCatching {
                val field = dhInvMvmProjField ?: findFieldInHierarchy(
                    apiRenderEventParam.javaClass,
                    "dhInverseMvmProjectionMatrix",
                )?.also {
                    it.isAccessible = true
                    dhInvMvmProjField = it
                }
                val matObj = field?.get(apiRenderEventParam)
                if (matObj != null) {
                    val joml = dhMatCreateJomlMethod?.invoke(null, matObj) as? Matrix4f
                    if (joml != null) {
                        joml.get(dhInvViewProjMatrix)
                    } else {
                        val arr = dhMatGetValuesMethod?.invoke(matObj) as? FloatArray
                        if (arr != null && arr.size >= 16) {
                            System.arraycopy(arr, 0, dhInvViewProjMatrix, 0, 16)
                        }
                    }
                }
            }
        }
        runCatching {
            val nearVal = apiRenderEventParam?.let { dhNearClipField?.getFloat(it) }
                ?: (renderUtilNearClipMethod?.invoke(null) as? Number)?.toFloat()
            if (nearVal != null && nearVal.isFinite() && nearVal > 0f) {
                dhNearClip = nearVal
            }
        }
        runCatching {
            val farVal = apiRenderEventParam?.let { dhFarClipField?.getFloat(it) }
                ?: (renderUtilFarClipMethod?.invoke(null) as? Number)?.toFloat()
            if (farVal != null && farVal.isFinite() && farVal > 0f) {
                dhFarClip = farVal
            }
        }
    }

    private fun refreshDhTargetHandles() {
        val meta = metaRendererInstance ?: return
        runCatching {
            val colorWrapper = metaColorWrapperField?.get(meta)
            val depthWrapper = metaDepthWrapperField?.get(meta)
            val getTex = textureWrapperGetTexture
            val getView = textureWrapperGetView
            if (colorWrapper != null && getTex != null && getView != null) {
                dhColorTexture = (getTex.invoke(colorWrapper) as? GpuTexture) as? VulkanGpuTexture
                dhColorTextureView = (getView.invoke(colorWrapper) as? GpuTextureView) as? VulkanGpuTextureView
            }
            if (depthWrapper != null && getTex != null && getView != null) {
                dhDepthTexture = (getTex.invoke(depthWrapper) as? GpuTexture) as? VulkanGpuTexture
                dhDepthTextureView = (getView.invoke(depthWrapper) as? GpuTextureView) as? VulkanGpuTextureView
            }
        }
    }

    private fun findFieldInHierarchy(cls: Class<*>, name: String): Field? {
        var cur: Class<*>? = cls
        while (cur != null) {
            runCatching { return cur.getDeclaredField(name) }
            cur = cur.superclass
        }
        return null
    }

    /**
     * Called from `DhVertexBufferWrapperMixin` at `@At("HEAD")` of
     * `BlazeVertexBufferWrapper.uploadVertexBuffer(ByteBuffer, int)`.
     *
     * Supports DH 3.x's 24-byte `POS_META_COLOR_MATERIAL_NORMAL_TEXTURE_OFFSET` layout with a clean
     * 16-byte fallback if `remaining == vertexCount * 16L`.
     */
    @JvmStatic
    fun onDhVertexBufferUpload(wrapper: Any?, vertexBuffer: ByteBuffer?, vertexCount: Int) {
        if (wrapper == null || !VulkanShaders.vulkanReady) return
        if (!isDistantHorizonsRenderingEnabled()) {
            syncNativeLodEnabled(false)
            return
        }
        val key = WrapperIdentityKey(wrapper)

        if (vertexBuffer == null || vertexCount < 4) {
            val existing = wrapperStates[key]
            if (existing != null) {
                synchronized(existing) {
                    if (existing.pendingRawBytes != null) {
                        existing.pendingRawBytes = null
                        existing.pendingVertexCount = 0
                        pendingUploadCount.decrementAndGet()
                    }
                    if (existing.uploaded) {
                        existing.uploaded = false
                        unregisterLodMesh(existing.lodKeyId)
                    }
                }
            }
            return
        }

        val view = vertexBuffer.duplicate().order(ByteOrder.LITTLE_ENDIAN)
        val remaining = view.remaining().toLong()
        val strideBytes = when (remaining) {
            vertexCount.toLong() * DH_VERTEX_STRIDE_BYTES.toLong() -> DH_VERTEX_STRIDE_BYTES
            vertexCount.toLong() * DH_LEGACY_VERTEX_STRIDE_BYTES.toLong() -> DH_LEGACY_VERTEX_STRIDE_BYTES
            else -> return
        }

        val state = wrapperStates.computeIfAbsent(key) {
            DhWrapperState(DH_KEY_NAMESPACE or (nextDhKeySeq.getAndIncrement() and 0xFFFFFFFFL))
        }

        synchronized(state) {
            if (state.hasMetadata) {
                if (state.pendingRawBytes != null) {
                    state.pendingRawBytes = null
                    state.pendingVertexCount = 0
                    pendingUploadCount.decrementAndGet()
                }
                unpackAndUploadDirect(
                    state = state,
                    src = view,
                    srcBaseOffset = view.position(),
                    vertexCount = vertexCount,
                    srcStrideBytes = strideBytes,
                )
            } else {
                val wasPending = state.pendingRawBytes != null
                val byteLen = remaining.toInt()
                val copy = ByteArray(byteLen)
                view.get(copy, 0, byteLen)
                state.pendingRawBytes = copy
                state.pendingVertexCount = vertexCount
                state.pendingStrideBytes = strideBytes
                if (!wasPending) {
                    pendingUploadCount.incrementAndGet()
                }
            }
        }
    }

    /**
     * Called from `DhVertexBufferWrapperMixin` at `@At("HEAD")` of `BlazeVertexBufferWrapper.close()`.
     */
    @JvmStatic
    fun onDhVertexBufferClose(wrapper: Any?) {
        if (wrapper == null) return
        val removed = wrapperStates.remove(WrapperIdentityKey(wrapper)) ?: return
        synchronized(removed) {
            if (removed.pendingRawBytes != null) {
                removed.pendingRawBytes = null
                removed.pendingVertexCount = 0
                pendingUploadCount.decrementAndGet()
            }
            if (removed.uploaded) {
                removed.uploaded = false
                unregisterLodMesh(removed.lodKeyId)
            }
        }
    }

    /**
     * Reflectively iterates DH 3.x `IDhApiTerrainBufferContainer` instances to bind each opaque,
     * no-SSAO, or transparent `BlazeVertexBufferWrapper` to its world-space `minCornerBlockPos`
     * and `detailLevel`, flushing any pending CPU vertex uploads to
     * [NativeCoreBridge.uploadLodMeshCpu].
     */
    @JvmStatic
    fun syncContainerMetadata(bufferContainers: Any?) {
        if (bufferContainers == null || !VulkanShaders.vulkanReady || pendingUploadCount.get() <= 0) return
        runCatching {
            val listReflect = listCache.computeIfAbsent(bufferContainers.javaClass) { ListReflectionCache(it) }
            val sizeFn = listReflect.sizeMethod ?: return@runCatching
            val getFn = listReflect.getMethod ?: return@runCatching
            val count = (sizeFn.invoke(bufferContainers) as? Number)?.toInt() ?: return@runCatching
            if (count <= 0) return@runCatching

            for (i in 0 until count) {
                val container = getFn.invoke(bufferContainers, i) ?: continue
                val cReflect = containerCache.computeIfAbsent(container.javaClass) { ContainerReflectionCache(it) }
                val minPos = cReflect.minCornerGetter?.invoke(container)
                    ?: cReflect.minCornerField?.get(container) ?: continue
                val vReflect = vec3iCache.computeIfAbsent(minPos.javaClass) { Vec3iReflectionCache(it) }
                val originX = vReflect.readX(minPos)
                val originY = vReflect.readY(minPos)
                val originZ = vReflect.readZ(minPos)
                val detailLevel = (
                    cReflect.detailLevelGetter?.invoke(container)
                        ?: cReflect.sectionPos?.let { cReflect.sectionDetailLevel?.invoke(null, it.getLong(container)) }
                    ) as? Number ?: continue

                syncWrapperCollection(
                    collection = cReflect.opaqueGetter?.invoke(container)
                        ?: cReflect.opaqueField?.get(container),
                    originX = originX,
                    originY = originY,
                    originZ = originZ,
                    detailLevel = detailLevel.toInt(),
                    materialFlags = 0,
                )
                cReflect.noSsaoGetter?.let { noSsaoMethod ->
                    syncWrapperCollection(
                        collection = noSsaoMethod.invoke(container),
                        originX = originX,
                        originY = originY,
                        originZ = originZ,
                        detailLevel = detailLevel.toInt(),
                        materialFlags = 0,
                    )
                }
                syncWrapperCollection(
                    collection = cReflect.transparentGetter?.invoke(container)
                        ?: cReflect.transparentField?.get(container),
                    originX = originX,
                    originY = originY,
                    originZ = originZ,
                    detailLevel = detailLevel.toInt(),
                    materialFlags = WATER_MATERIAL_FLAGS,
                )
            }
        }.onFailure { t ->
            VulkanShaders.log.debug("DH container metadata sync skipped: {}", t.message)
        }
    }

    private fun syncWrapperCollection(
        collection: Any?,
        originX: Int,
        originY: Int,
        originZ: Int,
        detailLevel: Int,
        materialFlags: Int,
    ) {
        if (collection == null) return
        when (collection) {
            is Array<*> -> {
                for (item in collection) {
                    if (item != null) {
                        bindWrapperMetadata(item, originX, originY, originZ, detailLevel, materialFlags)
                    }
                }
            }
            is Iterable<*> -> {
                for (item in collection) {
                    if (item != null) {
                        bindWrapperMetadata(item, originX, originY, originZ, detailLevel, materialFlags)
                    }
                }
            }
        }
    }

    private fun bindWrapperMetadata(
        wrapper: Any,
        originX: Int,
        originY: Int,
        originZ: Int,
        detailLevel: Int,
        materialFlags: Int,
    ) {
        val key = WrapperIdentityKey(wrapper)
        val state = wrapperStates.computeIfAbsent(key) {
            DhWrapperState(DH_KEY_NAMESPACE or (nextDhKeySeq.getAndIncrement() and 0xFFFFFFFFL))
        }
        synchronized(state) {
            state.originBlockX = originX
            state.originBlockY = originY
            state.originBlockZ = originZ
            state.detailLevel = detailLevel
            state.materialFlags = materialFlags
            state.hasMetadata = true

            val raw = state.pendingRawBytes
            val vCount = state.pendingVertexCount
            val stride = state.pendingStrideBytes
            if (raw != null) {
                state.pendingRawBytes = null
                state.pendingVertexCount = 0
                pendingUploadCount.decrementAndGet()
                if (vCount >= 4) {
                    val srcBuf = ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN)
                    unpackAndUploadDirect(
                        state = state,
                        src = srcBuf,
                        srcBaseOffset = 0,
                        vertexCount = vCount,
                        srcStrideBytes = stride,
                    )
                }
            }
        }
    }

    private fun mapDhMaterialToHyaloclastFlags(irisMaterialId: Int, defaultFlags: Int): Int = when (irisMaterialId) {
        DH_MAT_WATER -> WATER_MATERIAL_FLAGS
        DH_MAT_LAVA -> LAVA_MATERIAL_FLAGS
        DH_MAT_LEAVES -> FOLIAGE_MATERIAL_FLAGS
        DH_MAT_ILLUMINATED -> EMISSIVE_MATERIAL_FLAGS
        DH_MAT_STONE, DH_MAT_DEEPSLATE, DH_MAT_NETHER_STONE -> SectionMaterialPacking.pack(MID_STONE, 0)
        DH_MAT_WOOD -> SectionMaterialPacking.pack(MID_WOOD, 0)
        DH_MAT_METAL -> SectionMaterialPacking.pack(MID_METAL, 0)
        DH_MAT_DIRT -> SectionMaterialPacking.pack(MID_DIRT, 0)
        DH_MAT_SNOW -> SectionMaterialPacking.pack(SectionMaterialPacking.ID_SNOW_ORGANIC, 0)
        DH_MAT_SAND -> SectionMaterialPacking.pack(MID_SAND, 0)
        DH_MAT_TERRACOTTA -> SectionMaterialPacking.pack(SectionMaterialPacking.ID_POLISHED_CERAMIC, 0)
        else -> defaultFlags
    }

    /**
     * Unpacks DH's 24-byte `POS_META_COLOR_MATERIAL_NORMAL_TEXTURE_OFFSET` vertex format (or legacy
     * 16-byte format) into 28-byte `LodVertexGpu` structs in an off-heap `MemoryUtil` direct buffer
     * and uploads to [NativeCoreBridge.uploadLodMeshCpu].
     *
     * DH 24-byte layout (little-endian):
     * - `0..7`: `vPositionAndMeta` (`uint16 x, y, z, meta`; byte 7 encodes `(skyLight << 4) | blockLight`)
     * - `8..11`: `vColor` (RGBA8_UNORM / ABGR int)
     * - `12..15`: `materialNormalTile` (byte 12 = DH material ID, byte 13 = `dhNormalIdx`, bytes 14..15 = tileId)
     * - `16..19`: `offsetData` (`ubyte offX, offY, offZ, flags`, in 1/256ths of a block)
     * - `20..23`: `uvNormal`
     *
     * 28-byte `LodVertexGpu` output:
     * - `0..11`: `float3(x, y, z) + offsetData.xyz / 256.0f`
     * - `12..15`: `color_abgr`
     * - `16..23`: `uv (0, 0)`
     * - `24..27`: `blockLightByte | (skyLightByte << 8) | (hyaloclastMaterialFlags << 16) | (dhNormalIdx << 24)`
     */
    private fun acquireUnpackScratchBuffer(minBytes: Int): ByteBuffer {
        val existing = unpackScratchBuffer.get()
        if (existing != null && existing.capacity() >= minBytes) {
            existing.clear()
            return existing
        }
        val cap = maxOf(minBytes, (existing?.capacity() ?: 0) * 2, 65536)
        val allocated = ByteBuffer.allocateDirect(cap).order(ByteOrder.LITTLE_ENDIAN)
        if (cap <= 16 * 1024 * 1024) {
            unpackScratchBuffer.set(allocated)
        }
        return allocated
    }

    private fun unpackAndUploadDirect(
        state: DhWrapperState,
        src: ByteBuffer,
        srcBaseOffset: Int,
        vertexCount: Int,
        srcStrideBytes: Int,
    ) {
        if (!NativeCoreBridge.initialized) return
        val dstByteSize = vertexCount * GPU_LOD_VERTEX_STRIDE_BYTES
        val dst = acquireUnpackScratchBuffer(dstByteSize)
        val is24Byte = srcStrideBytes == DH_VERTEX_STRIDE_BYTES
        for (v in 0 until vertexCount) {
            val sOff = srcBaseOffset + v * srcStrideBytes
            val dOff = v * GPU_LOD_VERTEX_STRIDE_BYTES

            var x = (src.getShort(sOff).toInt() and 0xFFFF).toFloat()
            var y = (src.getShort(sOff + 2).toInt() and 0xFFFF).toFloat()
            var z = (src.getShort(sOff + 4).toInt() and 0xFFFF).toFloat()
            val lightOffset = if (is24Byte) 7 else 6
            val lightByte = src.get(sOff + lightOffset).toInt() and 0xFF
            val blockLightByte = (lightByte and 0x0F) * 17
            val skyLightByte = ((lightByte ushr 4) and 0x0F) * 17
            val colorAbgr = src.getInt(sOff + 8)

            val dhMaterialId = src.get(sOff + 12).toInt() and 0xFF
            val dhNormalIdx = src.get(sOff + 13).toInt() and 0xFF
            if (is24Byte) {
                x += (src.get(sOff + 16).toInt() and 0xFF) / 256.0f
                y += (src.get(sOff + 17).toInt() and 0xFF) / 256.0f
                z += (src.get(sOff + 18).toInt() and 0xFF) / 256.0f
            }
            val materialFlags = mapDhMaterialToHyaloclastFlags(dhMaterialId, state.materialFlags)
            val packedMeta = blockLightByte or (skyLightByte shl 8) or
                ((materialFlags and 0xFF) shl 16) or (dhNormalIdx shl 24)

            dst.putFloat(dOff, x)
            dst.putFloat(dOff + 4, y)
            dst.putFloat(dOff + 8, z)
            dst.putInt(dOff + 12, colorAbgr)
            dst.putFloat(dOff + 16, 0.0f)
            dst.putFloat(dOff + 20, 0.0f)
            dst.putInt(dOff + 24, packedMeta)
        }
        dst.position(0)
        dst.limit(dstByteSize)

        val ok = NativeCoreBridge.uploadLodMeshCpu(
            lodKeyId = state.lodKeyId,
            level = state.detailLevel,
            originBlockX = state.originBlockX,
            originBlockY = state.originBlockY,
            originBlockZ = state.originBlockZ,
            scaleBlocks = 1.0f,
            vertices = dst,
            vertexCount = vertexCount,
            indices = null,
            indexCount = 0,
            materialFlags = state.materialFlags,
        )
        if (ok) {
            state.uploaded = true
            activeLodColumns[state.lodKeyId] = RegisteredLodColumn(
                keyId = state.lodKeyId,
                detailLevel = state.detailLevel,
                originBlockX = state.originBlockX,
                originBlockY = state.originBlockY,
                originBlockZ = state.originBlockZ,
                scaleBlocks = 1.0f,
                vkBuffer = 0L,
                vertexCount = vertexCount,
            )
        }
    }

    @JvmStatic
    fun registerLodMesh(
        lodKeyId: Long,
        detailLevel: Int,
        originBlockX: Int,
        originBlockY: Int,
        originBlockZ: Int,
        scaleBlocks: Float,
        vkVertexBuffer: Long,
        byteOffset: Long,
        byteSize: Long,
        vertexStride: Int,
        vertexCount: Int,
        vkIndexBuffer: Long = 0L,
        indexCount: Int = 0,
        albedoColorAbgr: Int = 0xFFFFFFFF.toInt(),
    ) {
        if (vkVertexBuffer == 0L || vertexCount < 3 || vertexStride < 12) return
        activeLodColumns[lodKeyId] = RegisteredLodColumn(
            keyId = lodKeyId,
            detailLevel = detailLevel,
            originBlockX = originBlockX,
            originBlockY = originBlockY,
            originBlockZ = originBlockZ,
            scaleBlocks = scaleBlocks,
            vkBuffer = vkVertexBuffer,
            vertexCount = vertexCount,
        )
        NativeCoreBridge.registerLodSlice(
            lodKeyId = lodKeyId,
            level = detailLevel,
            originBlockX = originBlockX,
            originBlockY = originBlockY,
            originBlockZ = originBlockZ,
            scaleBlocks = scaleBlocks,
            vkBuffer = vkVertexBuffer,
            byteOffset = byteOffset,
            byteSize = byteSize,
            vertexStride = vertexStride,
            vertexCount = vertexCount,
            indexBuffer = vkIndexBuffer,
            indexCount = indexCount,
            albedoColorAbgr = albedoColorAbgr,
        )
    }

    @JvmStatic
    fun unregisterLodMesh(lodKeyId: Long) {
        if (activeLodColumns.remove(lodKeyId) != null) {
            NativeCoreBridge.unregisterLodSlice(lodKeyId)
        }
    }

    @JvmStatic
    fun clearAll() {
        lastSyncBucket = Long.MIN_VALUE
        lastNativeLodEnabled = null
        dhRenderedThisFrame = false
        dhGbufferClearedThisFrame = false
        currentRenderPassTransparent = false
        dhColorTexture = null
        dhColorTextureView = null
        dhDepthTexture = null
        dhDepthTextureView = null
        wrapperStates.clear()
        pendingUploadCount.set(0)
        val hadColumns = activeLodColumns.isNotEmpty()
        activeLodColumns.clear()
        if (NativeCoreBridge.initialized) {
            NativeCoreBridge.setLodEnabled(false)
        }
        if (hadColumns) {
            NativeCoreBridge.clearLodSlices()
            VulkanShaders.log.debug("Cleared Distant Horizons LOD acceleration structures")
        }
    }

    @JvmStatic
    fun activeColumnCount(): Int = activeLodColumns.size
}
