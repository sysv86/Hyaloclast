package dev.vulkanshaders.pipeline

import com.mojang.blaze3d.systems.RenderSystem
import com.mojang.renderpearl.api.GpuFormat
import com.mojang.renderpearl.api.buffers.GpuBuffer
import com.mojang.renderpearl.api.commands.RenderPass
import com.mojang.renderpearl.api.textures.GpuTexture
import com.mojang.renderpearl.api.textures.GpuTextureView
import com.mojang.renderpearl.backend.vulkan.VulkanGpuTexture
import com.mojang.renderpearl.backend.vulkan.VulkanGpuTextureView
import dev.vulkanshaders.VulkanShaders
import dev.vulkanshaders.backend.NativeCoreBridge
import dev.vulkanshaders.bridge.StandaloneRenderPearlBridge
import dev.vulkanshaders.bridge.VulkanShadersTurboExtension
import dev.vulkanshaders.pack.ShaderPackManager
import dev.vulkanshaders.state.WorldStateCollector
import dev.vulkanshaders.texture.TextureRegistry
import net.minecraft.client.Minecraft
import net.minecraft.client.renderer.chunk.ChunkSectionLayer
import org.lwjgl.system.MemoryUtil
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Manages auxiliary 4-MRT G-Buffer, Distant Horizons auxiliary G-Buffer, HDR radiance, temporal
 * history, and FrameGen textures (`gNormalRoughness` `RGBA16_FLOAT`, `gMaterialPbr` `RGBA8_UNORM`,
 * `gVelocity` `RG16_FLOAT`, `dhNormalRoughness` `RGBA16_FLOAT`, `dhMaterialPbr` `RGBA8_UNORM`,
 * `prevHudlessColor` `RGBA8_UNORM`), attaches MRT `GpuTextureView`s to `RenderPearl` passes via
 * `TurboShadersBridge.overridePendingPassColorViews`, and binds their `VkImage` + `VkImageView`
 * handles into the native Rust `RenderGraph`.
 */
object GbufferTargetManager {
    const val VS_MOTION_UBO_BYTES: Int = 352

    private var width: Int = 0
    private var height: Int = 0

    private var dhWidth: Int = 0
    private var dhHeight: Int = 0

    var normalRoughnessTexture: GpuTexture? = null
        private set
    var normalRoughnessView: GpuTextureView? = null
        private set

    var materialPbrTexture: GpuTexture? = null
        private set
    var materialPbrView: GpuTextureView? = null
        private set

    var velocityTexture: GpuTexture? = null
        private set
    var velocityView: GpuTextureView? = null
        private set

    // Auxiliary Distant Horizons G-Buffer targets (sized to match DH's color texture).
    var dhNormalRoughnessTexture: GpuTexture? = null
        private set
    var dhNormalRoughnessView: GpuTextureView? = null
        private set

    var dhMaterialPbrTexture: GpuTexture? = null
        private set
    var dhMaterialPbrView: GpuTextureView? = null
        private set

    var hdrRadianceTexture: GpuTexture? = null
        private set
    var hdrRadianceView: GpuTextureView? = null
        private set

    var prevHudlessColorTexture: GpuTexture? = null
        private set
    var prevHudlessColorView: GpuTextureView? = null
        private set

    var interpolatedFrameTexture: GpuTexture? = null
        private set
    var interpolatedFrameView: GpuTextureView? = null
        private set

    // Typed VulkanGpuTexture accessors for explicit layout transitions before compute passes.
    val normalRoughnessVulkanTexture: VulkanGpuTexture?
        get() = normalRoughnessTexture?.takeIf { !it.isClosed } as? VulkanGpuTexture

    val materialPbrVulkanTexture: VulkanGpuTexture?
        get() = materialPbrTexture?.takeIf { !it.isClosed } as? VulkanGpuTexture

    val velocityVulkanTexture: VulkanGpuTexture?
        get() = velocityTexture?.takeIf { !it.isClosed } as? VulkanGpuTexture

    val dhNormalRoughnessVulkanTexture: VulkanGpuTexture?
        get() = dhNormalRoughnessTexture?.takeIf { !it.isClosed } as? VulkanGpuTexture

    val dhMaterialPbrVulkanTexture: VulkanGpuTexture?
        get() = dhMaterialPbrTexture?.takeIf { !it.isClosed } as? VulkanGpuTexture

    /**
     * 352-byte std140 UBO: prevViewProj (64B), cameraDeltaTime (16B), pbrParams (16B),
     * sunDirRain (16B), heldLightData (16B), invView (64B), packParams[10] (160B).
     */
    private var motionUbo: GpuBuffer? = null
    /** Companion 352-byte UBO identical to [motionUbo] with `pbrParams.x = 0.0` for draws without PBR companions. */
    private var motionUboNoPbr: GpuBuffer? = null
    @Volatile
    private var lastSyncedGraphBindingsSig: Long = 0L
    private val motionScratch: ByteBuffer = ByteBuffer.allocateDirect(VS_MOTION_UBO_BYTES).order(ByteOrder.nativeOrder())

    @JvmStatic
    fun ensureTargetsSized(targetWidth: Int, targetHeight: Int) {
        val w = targetWidth.coerceAtLeast(1)
        val h = targetHeight.coerceAtLeast(1)
        if (w == width && h == height &&
            normalRoughnessTexture != null && !normalRoughnessTexture!!.isClosed &&
            prevHudlessColorTexture != null && !prevHudlessColorTexture!!.isClosed
        ) {
            return
        }
        closeMainTargets()
        width = w
        height = h

        val device = try {
            RenderSystem.getDevice()
        } catch (_: Throwable) {
            return
        }

        val usage = GpuTexture.USAGE_RENDER_ATTACHMENT or
            GpuTexture.USAGE_TEXTURE_BINDING or
            GpuTexture.USAGE_COPY_SRC or
            GpuTexture.USAGE_COPY_DST

        normalRoughnessTexture = runCatching {
            device.createTexture({ "Hyaloclast GBuffer NormalRoughness" }, usage, GpuFormat.RGBA16_FLOAT, w, h, 1, 1)
        }.getOrNull()
        normalRoughnessView = normalRoughnessTexture?.let { runCatching { device.createTextureView(it) }.getOrNull() }

        materialPbrTexture = runCatching {
            device.createTexture({ "Hyaloclast GBuffer MaterialPBR" }, usage, GpuFormat.RGBA8_UNORM, w, h, 1, 1)
        }.getOrNull()
        materialPbrView = materialPbrTexture?.let { runCatching { device.createTextureView(it) }.getOrNull() }

        velocityTexture = runCatching {
            device.createTexture({ "Hyaloclast GBuffer Velocity" }, usage, GpuFormat.RG16_FLOAT, w, h, 1, 1)
        }.getOrNull()
        velocityView = velocityTexture?.let { runCatching { device.createTextureView(it) }.getOrNull() }

        prevHudlessColorTexture = runCatching {
            device.createTexture({ "Hyaloclast PrevHudlessColor" }, usage, GpuFormat.RGBA8_UNORM, w, h, 1, 1)
        }.getOrNull()
        prevHudlessColorView = prevHudlessColorTexture?.let { runCatching { device.createTextureView(it) }.getOrNull() }

        // hdrRadianceTexture & interpolatedFrameTexture are owned by Rust RenderGraph; skip duplicate VRAM allocations.
        hdrRadianceTexture = null
        hdrRadianceView = null
        interpolatedFrameTexture = null
        interpolatedFrameView = null
    }

    @JvmStatic
    fun ensureDhTargetsSized(targetWidth: Int, targetHeight: Int) {
        val w = targetWidth.coerceAtLeast(1)
        val h = targetHeight.coerceAtLeast(1)
        if (w == dhWidth && h == dhHeight &&
            dhNormalRoughnessTexture != null && !dhNormalRoughnessTexture!!.isClosed &&
            dhMaterialPbrTexture != null && !dhMaterialPbrTexture!!.isClosed
        ) {
            return
        }
        closeDhTargets()
        dhWidth = w
        dhHeight = h

        val device = try {
            RenderSystem.getDevice()
        } catch (_: Throwable) {
            return
        }

        val usage = GpuTexture.USAGE_RENDER_ATTACHMENT or
            GpuTexture.USAGE_TEXTURE_BINDING or
            GpuTexture.USAGE_COPY_SRC or
            GpuTexture.USAGE_COPY_DST

        dhNormalRoughnessTexture = runCatching {
            device.createTexture({ "Hyaloclast DH GBuffer NormalRoughness" }, usage, GpuFormat.RGBA16_FLOAT, w, h, 1, 1)
        }.getOrNull()
        dhNormalRoughnessView = dhNormalRoughnessTexture?.let { runCatching { device.createTextureView(it) }.getOrNull() }

        dhMaterialPbrTexture = runCatching {
            device.createTexture({ "Hyaloclast DH GBuffer MaterialPBR" }, usage, GpuFormat.RGBA8_UNORM, w, h, 1, 1)
        }.getOrNull()
        dhMaterialPbrView = dhMaterialPbrTexture?.let { runCatching { device.createTextureView(it) }.getOrNull() }
    }

    @JvmStatic
    fun clearDhGbufferTargets() {
        runCatching {
            val encoder = RenderSystem.getDevice().createCommandEncoder()
            dhNormalRoughnessTexture?.takeIf { !it.isClosed }?.let {
                encoder.clearColorTexture(it, zeroClear)
            }
            dhMaterialPbrTexture?.takeIf { !it.isClosed }?.let {
                encoder.clearColorTexture(it, zeroClear)
            }
        }
    }

    private val zeroClear = org.joml.Vector4f(0f, 0f, 0f, 0f)

    @Volatile
    private var lastBoundMrtPass: RenderPass? = null
    @Volatile
    private var lastBoundMrtFrame: Int = -1

    /**
     * Guards the shared 4-MRT view override so a pass that was already set up for this frame is not
     * rebound (re-clearing pending state).
     */
    private fun shouldBindMrt(renderPass: RenderPass): Boolean {
        val frame = WorldStateCollector.frameIndex
        if (lastBoundMrtPass === renderPass && lastBoundMrtFrame == frame) return false
        lastBoundMrtPass = renderPass
        lastBoundMrtFrame = frame
        return true
    }

    @JvmStatic
    fun hasWorldGbuffer(renderPass: RenderPass): Boolean =
        lastBoundMrtPass === renderPass && lastBoundMrtFrame == WorldStateCollector.frameIndex

    /**
     * Packs the `std140` `VsMotion` buffer from the 528-byte `WorldUBO`:
     * - `0..63`:    `prevViewProj` (`WorldUBO` `64..127`)
     * - `64..79`:   `cameraDeltaTime` = `(cameraDelta.xyz, gameTimeSec)` (`WorldUBO` `272..287`)
     * - `80..95`:   `pbrParams` = `(hasPbr, windHook, wavesHook, 0.0)`
     * - `96..111`:  `sunDirRain` = `(sunDir.xyz, rain)` (`WorldUBO` `288..303`)
     * - `112..127`: `heldLightData` = `(heldLight, thunder, eyeInFluid, depthFar)` (`WorldUBO` `304..319`)
     * - `128..191`: `invView` (`WorldUBO` `192..255`, when `dst.capacity() >= 192`)
     * - `192..351`: `packParams[10]` (160B from `WorldUBO` `320..479`, when `dst.capacity() >= 352`)
     */
    internal fun packVsMotionBuffer(
        ubo: ByteBuffer,
        dst: ByteBuffer,
        hasPbr: Boolean,
        hasWindHook: Boolean,
        hasWavesHook: Boolean,
    ) {
        dst.clear()
        val hasInvView = dst.capacity() >= 192
        val hasPackParams = hasInvView && dst.capacity() >= VS_MOTION_UBO_BYTES && ubo.capacity() >= 480
        if (ubo.isDirect && dst.isDirect) {
            val srcAddr = MemoryUtil.memAddress0(ubo)
            val dstAddr = MemoryUtil.memAddress0(dst)
            // 0..63: prevViewProj (from WorldUBO 64..127)
            MemoryUtil.memCopy(srcAddr + 64L, dstAddr, 64L)
            // 64..79: cameraDeltaTime (cameraDelta.xyz + gameTimeSec from WorldUBO 272..287)
            MemoryUtil.memCopy(srcAddr + 272L, dstAddr + 64L, 16L)
            // 96..111: sunDirRain (sunDir.xyz + rain from WorldUBO 288..303)
            MemoryUtil.memCopy(srcAddr + 288L, dstAddr + 96L, 16L)
            if (hasInvView) {
                // 128..191: invView (from WorldUBO 192..255)
                MemoryUtil.memCopy(srcAddr + 192L, dstAddr + 128L, 64L)
                if (hasPackParams) {
                    // 192..351: packParams[10] (160 bytes from WorldUBO 320..479)
                    MemoryUtil.memCopy(srcAddr + 320L, dstAddr + 192L, 160L)
                }
            }
        } else {
            dst.put(0, ubo, 64, 64)
            dst.put(64, ubo, 272, 16)
            dst.put(96, ubo, 288, 16)
            if (hasInvView) {
                dst.put(128, ubo, 192, 64)
                if (hasPackParams) {
                    dst.put(192, ubo, 320, 160)
                }
            }
        }
        // 80..95: pbrParams
        dst.putFloat(80, if (hasPbr) 1f else 0f)
        dst.putFloat(84, if (hasWindHook) 1f else 0f)
        dst.putFloat(88, if (hasWavesHook) 1f else 0f)
        dst.putFloat(92, 0f)
        // 112..127: heldLightData = (heldLight, thunder, eyeInFluid, depthFar)
        // WorldUBO 304 = thunder, 308 = eyeInFluid (int), 312 = depthFar, 316 = heldLight
        val thunder = ubo.getFloat(304)
        val eyeInFluid = ubo.getInt(308).toFloat()
        val depthFar = ubo.getFloat(312)
        val heldLight = ubo.getFloat(316)
        dst.putFloat(112, heldLight)
        dst.putFloat(116, thunder)
        dst.putFloat(120, eyeInFluid)
        dst.putFloat(124, depthFar)
        dst.position(0)
        dst.limit(if (hasPackParams) VS_MOTION_UBO_BYTES else if (hasInvView) 192 else 128)
    }

    /**
     * Uploads the current motion-vector and pack-parameter data into the 352-byte `VsMotion` UBOs.
     * Must run OUTSIDE any open render pass (`writeToBuffer` is an encoder command). Called from
     * [prepareTerrainMrtPass] before the main world pass opens.
     */
    private fun refreshMotionUniformData() {
        val device = runCatching { RenderSystem.getDevice() }.getOrNull() ?: return
        if (motionUbo == null || motionUbo!!.isClosed) {
            motionUbo = runCatching {
                device.createBuffer(
                    { "Hyaloclast GBuffer VsMotion" },
                    GpuBuffer.USAGE_UNIFORM or GpuBuffer.USAGE_COPY_DST,
                    VS_MOTION_UBO_BYTES.toLong(),
                )
            }.getOrNull() ?: run {
                VulkanShaders.log.warn(
                    "Could not create the {}-byte VsMotion UBO; G-Buffer motion vectors are disabled",
                    VS_MOTION_UBO_BYTES,
                )
                return
            }
        }
        if (motionUboNoPbr == null || motionUboNoPbr!!.isClosed) {
            motionUboNoPbr = runCatching {
                device.createBuffer(
                    { "Hyaloclast GBuffer VsMotion (NoPBR)" },
                    GpuBuffer.USAGE_UNIFORM or GpuBuffer.USAGE_COPY_DST,
                    VS_MOTION_UBO_BYTES.toLong(),
                )
            }.getOrNull()
        }
        val ubo = WorldStateCollector.uniformBufferReadOnly()
        val pack = ShaderPackManager.activePack
        val hasPbr = TextureRegistry.hasLabPbrCompanions()
        val hasWindHook = pack?.resolvedHooks?.containsKey("vertex_displace") == true
        val hasWavesHook = pack?.resolvedHooks?.containsKey("water_waves") == true

        packVsMotionBuffer(
            ubo = ubo,
            dst = motionScratch,
            hasPbr = hasPbr,
            hasWindHook = hasWindHook,
            hasWavesHook = hasWavesHook,
        )
        // ModelViewMat is the raster view; Minecraft applies bobbing in ProjMat.
        WorldStateCollector.writeInverseRasterView(motionScratch, 128)

        val encoder = device.createCommandEncoder()
        runCatching {
            encoder.writeToBuffer(motionUbo!!.slice(), motionScratch)
        }.onFailure { t ->
            VulkanShaders.log.warn("Could not upload VsMotion UBO contents: {}", t.message)
        }

        motionUboNoPbr?.takeIf { !it.isClosed }?.let { noPbrBuf ->
            motionScratch.putFloat(80, 0f)
            motionScratch.position(0)
            motionScratch.limit(VS_MOTION_UBO_BYTES)
            runCatching {
                encoder.writeToBuffer(noPbrBuf.slice(), motionScratch)
            }.onFailure { t ->
                VulkanShaders.log.warn("Could not upload VsMotion (NoPBR) UBO contents: {}", t.message)
            }
        }
    }

    /**
     * Sets the `VsMotion` UBO and LabPBR atlas uniforms on [renderPass]. Must be called AFTER the
     * G-Buffer pipeline is bound (`setPipeline`).
     *
     * When [sampler0View] is supplied (entity / block-entity / item draws), looks up that albedo
     * texture's dedicated companion PBR pair in [TextureRegistry]; if none exists, binds the
     * `pbrParams.x = 0` UBO so the shader skips PBR sampling on that texture.
     */
    private fun bindGbufferUniforms(renderPass: RenderPass, sampler0View: GpuTextureView? = null) {
        val albedoVk = (sampler0View as? VulkanGpuTextureView)?.texture()?.vkImage() ?: 0L
        val companion = if (albedoVk != 0L) TextureRegistry.getCompanionForAlbedo(albedoVk) else null
        val usePbr = if (sampler0View != null) {
            companion != null && !companion.isFlatFallback
        } else {
            TextureRegistry.hasLabPbrCompanions()
        }

        val live = if (usePbr) motionUbo else (motionUboNoPbr ?: motionUbo)
        if (live == null || live.isClosed) return
        runCatching {
            renderPass.setUniform("VsMotion", live)
        }.onFailure { t ->
            VulkanShaders.log.warn("Could not bind VsMotion uniform: {}", t.message)
        }
        val normalView = companion?.normalView ?: TextureRegistry.normalAtlasView
        val normalSampler = companion?.normalSampler ?: TextureRegistry.normalAtlasSampler
        val specView = companion?.specularView ?: TextureRegistry.specularAtlasView
        val specSampler = companion?.specularSampler ?: TextureRegistry.specularAtlasSampler
        runCatching {
            if (normalView != null && normalSampler != null) {
                renderPass.setUniform("VsNormalAtlas", normalView, normalSampler)
            }
            if (specView != null && specSampler != null) {
                renderPass.setUniform("VsSpecularAtlas", specView, specSampler)
            }
        }.onFailure { t ->
            VulkanShaders.log.warn("Could not bind LabPBR atlas uniforms: {}", t.message)
        }
    }

    /**
     * Stages and clears the 4 MRT color attachment `GpuTextureView`s (`[mainColor, gNormalRoughness, gMaterialPbr, gVelocity]`)
     * before `RenderSystem.getDevice().createCommandEncoder().createRenderPass(...)` opens the terrain render pass.
     */
    @JvmStatic
    fun prepareTerrainMrtPass() {
        if (!VulkanShaders.shaderPackActive) return
        val mc = Minecraft.getInstance()
        val mainTarget = mc.gameRenderer.mainRenderTarget()
        val colorTex = mainTarget.colorTexture ?: return
        ensureTargetsSized(colorTex.getWidth(0), colorTex.getHeight(0))
        runCatching {
            val encoder = RenderSystem.getDevice().createCommandEncoder()
            normalRoughnessTexture?.takeIf { !it.isClosed }?.let {
                encoder.clearColorTexture(it, zeroClear)
            }
            materialPbrTexture?.takeIf { !it.isClosed }?.let {
                encoder.clearColorTexture(it, zeroClear)
            }
            velocityTexture?.takeIf { !it.isClosed }?.let {
                encoder.clearColorTexture(it, zeroClear)
            }
        }
        if (TextureRegistry.normalAtlasView == null || TextureRegistry.specularAtlasView == null) {
            TextureRegistry.ensureCompanionAtlasesBuilt()
        }
        refreshMotionUniformData()
    }

    private val legacyRasterFlag: Boolean = System.getenv("VS_LEGACY_RASTER") == "1"

    private fun legacyRasterRequested(): Boolean = legacyRasterFlag

    @JvmStatic
    fun onBeginTerrainPass(renderPass: RenderPass, layer: ChunkSectionLayer) {
        if (VulkanShadersTurboExtension.shouldSkipWorldRasterization()) return
        val mc = Minecraft.getInstance()
        val mainTarget = mc.gameRenderer.mainRenderTarget()
        val colorTex = mainTarget.colorTexture ?: return
        val depthTex = mainTarget.depthTexture ?: return

        ensureTargetsSized(colorTex.getWidth(0), colorTex.getHeight(0))

        val colorView = mainTarget.colorTextureView as? VulkanGpuTextureView
        val nView = normalRoughnessView as? VulkanGpuTextureView
        val mView = materialPbrView as? VulkanGpuTextureView
        val vView = velocityView as? VulkanGpuTextureView
        if (!legacyRasterRequested() && shouldBindMrt(renderPass)) {
            if (StandaloneRenderPearlBridge.vulkanTurboLoaded) {
                runCatching {
                    dev.vulkanturbo.api.shader.TurboShadersBridge.invalidateRenderPassStateCache(renderPass)
                }
            }
            overrideMrtViews(renderPass, colorView, nView, mView, vView)
        }

        if (layer == ChunkSectionLayer.SOLID && NativeCoreBridge.initialized) {
            syncNativeGraphBindings(
                colorTex = colorTex,
                colorView = mainTarget.colorTextureView,
                depthTex = depthTex,
                depthView = mainTarget.depthTextureView,
            )
        }
    }

    /**
     * Called by the region renderer AFTER the G-Buffer pipeline is bound (`setPipeline`).
     */
    @JvmStatic
    fun onTerrainPipelineBound(renderPass: RenderPass) {
        if (!VulkanShaders.shaderPackActive || legacyRasterRequested() || VulkanShadersTurboExtension.shouldSkipWorldRasterization()) return
        bindGbufferUniforms(renderPass, sampler0View = null)
    }

    /**
     * Called by `PreparedRenderTypeMixin` after binding a G-Buffer entity/block-entity/particle
     * pipeline so the draw receives its matching LabPBR companion pair (or `pbrParams.x = 0`).
     */
    @JvmStatic
    fun onWorldFeaturePipelineBound(renderPass: RenderPass, sampler0View: GpuTextureView?) {
        if (!VulkanShaders.shaderPackActive || legacyRasterRequested() || VulkanShadersTurboExtension.shouldSkipWorldRasterization()) return
        bindGbufferUniforms(renderPass, sampler0View = sampler0View)
    }

    /**
     * Binds the shared 4 MRT color views and the `VsMotion` UBO for the main world pass.
     * Skipped in Tier 3 (`shouldSkipWorldRasterization() == true`) so overlay/gizmo passes that run
     * with 1 color attachment after world rasterization is skipped do not receive 4 MRT color views.
     */
    @JvmStatic
    fun onBeginWorldPass(renderPass: RenderPass) {
        if (!VulkanShaders.shaderPackActive || legacyRasterRequested() || VulkanShadersTurboExtension.shouldSkipWorldRasterization()) return
        val colorTex = Minecraft.getInstance().gameRenderer.mainRenderTarget().colorTexture ?: return
        ensureTargetsSized(colorTex.getWidth(0), colorTex.getHeight(0))
        val colorView = Minecraft.getInstance().gameRenderer.mainRenderTarget().colorTextureView as? VulkanGpuTextureView
        if (shouldBindMrt(renderPass)) {
            overrideMrtViews(
                renderPass,
                colorView,
                normalRoughnessView as? VulkanGpuTextureView,
                materialPbrView as? VulkanGpuTextureView,
                velocityView as? VulkanGpuTextureView,
            )
        }
        // Uniforms are bound after setPipeline() to match RenderPearl's pipeline-reset contract.
    }

    private fun overrideMrtViews(
        renderPass: RenderPass,
        colorView: VulkanGpuTextureView?,
        nView: VulkanGpuTextureView?,
        mView: VulkanGpuTextureView?,
        vView: VulkanGpuTextureView?,
    ) {
        if (StandaloneRenderPearlBridge.vulkanTurboLoaded &&
            colorView != null && nView != null && mView != null && vView != null
        ) {
            runCatching {
                dev.vulkanturbo.api.shader.TurboShadersBridge.overridePendingPassColorViews(
                    renderPass,
                    arrayOf(colorView, nView, mView, vView),
                )
            }
        }
    }

    @JvmStatic
    fun syncNativeGraphBindings(
        colorTex: GpuTexture?,
        colorView: GpuTextureView?,
        depthTex: GpuTexture?,
        depthView: GpuTextureView?,
    ) {
        if (!NativeCoreBridge.initialized || colorTex == null || depthTex == null) return
        ensureTargetsSized(colorTex.getWidth(0), colorTex.getHeight(0))

        val albedoVk = (colorTex as? VulkanGpuTexture)?.vkImage() ?: 0L
        val albedoViewVk = (colorView as? VulkanGpuTextureView)?.vkImageView() ?: 0L
        val normalVk = (normalRoughnessTexture as? VulkanGpuTexture)?.vkImage() ?: 0L
        val normalViewVk = (normalRoughnessView as? VulkanGpuTextureView)?.vkImageView() ?: 0L
        val matVk = (materialPbrTexture as? VulkanGpuTexture)?.vkImage() ?: 0L
        val matViewVk = (materialPbrView as? VulkanGpuTextureView)?.vkImageView() ?: 0L
        val velVk = (velocityTexture as? VulkanGpuTexture)?.vkImage() ?: 0L
        val velViewVk = (velocityView as? VulkanGpuTextureView)?.vkImageView() ?: 0L
        val depthVk = (depthTex as? VulkanGpuTexture)?.vkImage() ?: 0L
        val depthViewVk = (depthView as? VulkanGpuTextureView)?.vkImageView() ?: 0L

        val prevHudVk = (prevHudlessColorTexture as? VulkanGpuTexture)?.vkImage() ?: 0L
        val prevHudViewVk = (prevHudlessColorView as? VulkanGpuTextureView)?.vkImageView() ?: 0L

        var sig = albedoVk
        sig = sig * 31L + albedoViewVk
        sig = sig * 31L + normalVk
        sig = sig * 31L + normalViewVk
        sig = sig * 31L + matVk
        sig = sig * 31L + matViewVk
        sig = sig * 31L + velVk
        sig = sig * 31L + velViewVk
        sig = sig * 31L + depthVk
        sig = sig * 31L + depthViewVk
        sig = sig * 31L + prevHudVk
        sig = sig * 31L + prevHudViewVk
        if (sig == lastSyncedGraphBindingsSig && albedoVk != 0L) {
            return
        }
        lastSyncedGraphBindingsSig = sig

        NativeCoreBridge.bindGbufferViews(
            albedoImg = albedoVk,
            albedoView = albedoViewVk,
            normalImg = normalVk,
            normalView = normalViewVk,
            materialImg = matVk,
            materialView = matViewVk,
            velocityImg = velVk,
            velocityView = velViewVk,
            depthImg = depthVk,
            depthView = depthViewVk,
            radianceImg = 0L,
            radianceView = 0L,
        )

        NativeCoreBridge.bindPostTargets(
            hudlessImg = albedoVk,
            hudlessView = albedoViewVk,
            prevHudlessImg = prevHudVk,
            prevHudlessView = prevHudViewVk,
            uiImg = albedoVk,
            uiView = albedoViewVk,
            interpolatedImg = 0L,
            interpolatedView = 0L,
        )
    }

    private fun closeMainTargets() {
        runCatching { normalRoughnessView?.close() }
        runCatching { normalRoughnessTexture?.close() }
        runCatching { materialPbrView?.close() }
        runCatching { materialPbrTexture?.close() }
        runCatching { velocityView?.close() }
        runCatching { velocityTexture?.close() }
        runCatching { hdrRadianceView?.close() }
        runCatching { hdrRadianceTexture?.close() }
        runCatching { prevHudlessColorView?.close() }
        runCatching { prevHudlessColorTexture?.close() }
        runCatching { interpolatedFrameView?.close() }
        runCatching { interpolatedFrameTexture?.close() }
        runCatching { motionUbo?.close() }
        runCatching { motionUboNoPbr?.close() }
        motionUbo = null
        motionUboNoPbr = null
        lastBoundMrtPass = null
        lastBoundMrtFrame = -1
        lastSyncedGraphBindingsSig = 0L

        normalRoughnessView = null
        normalRoughnessTexture = null
        materialPbrView = null
        materialPbrTexture = null
        velocityView = null
        velocityTexture = null
        hdrRadianceView = null
        hdrRadianceTexture = null
        prevHudlessColorView = null
        prevHudlessColorTexture = null
        interpolatedFrameView = null
        interpolatedFrameTexture = null
        width = 0
        height = 0
    }

    @JvmStatic
    fun closeDhTargets() {
        runCatching { dhNormalRoughnessView?.close() }
        runCatching { dhNormalRoughnessTexture?.close() }
        runCatching { dhMaterialPbrView?.close() }
        runCatching { dhMaterialPbrTexture?.close() }
        dhNormalRoughnessView = null
        dhNormalRoughnessTexture = null
        dhMaterialPbrView = null
        dhMaterialPbrTexture = null
        dhWidth = 0
        dhHeight = 0
    }

    /**
     * Releases cached G-buffer, Distant Horizons, and post-processing targets and clears cached
     * dimensions, MRT pass state, and native binding signatures on display resize or runtime
     * Render Tier switches without leaking or leaving stale handles bound in native core.
     */
    @JvmStatic
    fun invalidateForResize() {
        if (NativeCoreBridge.initialized) {
            NativeCoreBridge.bindGbufferViews(
                albedoImg = 0L,
                albedoView = 0L,
                normalImg = 0L,
                normalView = 0L,
                materialImg = 0L,
                materialView = 0L,
                velocityImg = 0L,
                velocityView = 0L,
                depthImg = 0L,
                depthView = 0L,
                radianceImg = 0L,
                radianceView = 0L,
            )
            NativeCoreBridge.bindPostTargets(
                hudlessImg = 0L,
                hudlessView = 0L,
                prevHudlessImg = 0L,
                prevHudlessView = 0L,
                uiImg = 0L,
                uiView = 0L,
                interpolatedImg = 0L,
                interpolatedView = 0L,
            )
        }
        closeAll()
    }

    @JvmStatic
    fun closeAll() {
        closeMainTargets()
        closeDhTargets()
    }
}
