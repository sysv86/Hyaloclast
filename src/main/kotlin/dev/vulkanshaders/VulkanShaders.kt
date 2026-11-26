package dev.vulkanshaders

import com.mojang.renderpearl.api.device.GpuDevice
import dev.vulkanshaders.backend.NativeCoreBridge
import dev.vulkanshaders.backend.VulkanDeviceContext
import dev.vulkanshaders.bridge.DistantHorizonsBridge
import dev.vulkanshaders.bridge.StandaloneRenderPearlBridge
import dev.vulkanshaders.bridge.VulkanShadersTurboExtension
import dev.vulkanshaders.config.ConfigStore
import dev.vulkanshaders.config.FrameGenMode
import dev.vulkanshaders.config.LightingMode
import dev.vulkanshaders.config.RuntimePerformanceGuard
import dev.vulkanshaders.config.VulkanShadersConfig
import dev.vulkanshaders.diagnostics.HyaloclastDebugHud
import dev.vulkanshaders.diagnostics.HyaloclastErrorLog
import dev.vulkanshaders.gui.ShaderNotices
import dev.vulkanshaders.gui.frameGenLabel
import dev.vulkanshaders.input.HyaloclastKeybinds
import dev.vulkanshaders.pack.ShaderPackManager
import dev.vulkanshaders.pipeline.GbufferPipelineRegistry
import dev.vulkanshaders.pipeline.GbufferTargetManager
import dev.vulkanshaders.pipeline.SectionSliceRegistry
import dev.vulkanshaders.sdk.ProprietaryLibraryManager
import dev.vulkanshaders.state.WorldStateCollector
import dev.vulkanshaders.texture.TextureRegistry
import net.fabricmc.api.ClientModInitializer
import net.fabricmc.loader.api.FabricLoader
import net.minecraft.client.Minecraft
import net.minecraft.client.multiplayer.ClientLevel
import net.minecraft.network.chat.Component
import net.minecraft.network.chat.MutableComponent
import net.minecraft.network.chat.Style
import net.minecraft.network.chat.TextColor
import org.slf4j.Logger
import org.slf4j.LoggerFactory

/**
 * Client entrypoint (`VulkanShaders` internal engine, distributed publicly as **Hyaloclast** under
 * distribution codename **`IGNIS`**).
 */
object VulkanShaders : ClientModInitializer {
    const val MOD_ID: String = "vulkanshaders"
    const val INTERNAL_NAME: String = "VulkanShaders"
    const val PUBLIC_NAME: String = "Hyaloclast"
    const val CODENAME: String = "IGNIS"

    private const val BRAND_NAME: String = "Hyaloclast"

    // Sampled from docs/icons/hyaloclast.svg face-grad — the banner at the top of README.md:
    // #FFF2A6 -> #FFCE47 -> #FF9624 -> #EA5E12 -> #BD3808.
    private val BRAND_GRADIENT = intArrayOf(0xFFF2A6, 0xFFCE47, 0xFF9624, 0xEA5E12, 0xBD3808)
    private const val BRAND_BODY_COLOR: Int = 0xFF9624

    private fun gradientRgb(t: Float): Int {
        val last = BRAND_GRADIENT.size - 1
        val pos = t.coerceIn(0f, 1f) * last
        val idx = pos.toInt().coerceAtMost(last - 1)
        val f = pos - idx
        val a = BRAND_GRADIENT[idx]
        val b = BRAND_GRADIENT[idx + 1]
        val ar = (a shr 16) and 0xFF
        val ag = (a shr 8) and 0xFF
        val ab = a and 0xFF
        val br = (b shr 16) and 0xFF
        val bg = (b shr 8) and 0xFF
        val bb = b and 0xFF
        val r = (ar + (br - ar) * f).toInt()
        val g = (ag + (bg - ag) * f).toInt()
        val bl = (ab + (bb - ab) * f).toInt()
        return (r shl 16) or (g shl 8) or bl
    }
    private const val TELEMETRY_POLL_INTERVAL_TICKS: Int = 10

    val log: Logger = LoggerFactory.getLogger(MOD_ID)

    @Volatile
    var config: VulkanShadersConfig = VulkanShadersConfig()
        private set

    @JvmField @Volatile var shaderPackActive: Boolean = false
    @JvmField @Volatile var vulkanReady: Boolean = false
    // Session-only kill switch: the render thread can be blocked inside VulkanCommandEncoder.submit
    // before the tick-driven performance guard ever gets a chance to run. A separate watchdog
    // thread flips this when no world-graph frame completes in time; config on disk is untouched.
    @JvmField @Volatile var emergencyBypass: Boolean = false
    @Volatile private var lastWorldGraphFrameNanos: Long = 0L
    @Volatile private var pendingChatAnnouncement: Boolean = false
    @Volatile private var lastSyncedGpuProfilerEnabled: Boolean? = null
    @Volatile private var lastDimensionKey: Any? = null
    @Volatile private var pendingDeferredTier2Config: VulkanShadersConfig? = null
    private var telemetryTickCounter: Int = 0

    // TEMPORARY headless switch hook: VS_TEST_TIER_SWITCH=T1_TO_T2 (or T3_TO_T2).
    private val testTierSwitchMode: String? = System.getenv("VS_TEST_TIER_SWITCH")
    private val testTierSwitchSource: LightingMode? = when (testTierSwitchMode) {
        "T1_TO_T2" -> LightingMode.COMPUTE
        "T3_TO_T2" -> LightingMode.PATH
        else -> null
    }
    private var testTierSwitchCountdown: Int = if (testTierSwitchSource == null) -1 else 120

    override fun onInitializeClient() {
        HyaloclastErrorLog.installCrashHooks()
        HyaloclastKeybinds.register()
        if (StandaloneRenderPearlBridge.isStandaloneMode()) {
            StandaloneRenderPearlBridge.ensureStandaloneInitialized()
        }

        config = ConfigStore.load()
        startRenderWatchdog()
        if (!StandaloneRenderPearlBridge.vulkanTurboLoaded && config.enabled) {
            config = config.copy(enabled = false, selectedPackId = "off")
            ConfigStore.save(config)
            log.warn("Pyroclast (VulkanTurbo) is missing. Hyaloclast shaders were disabled safely.")
        }

        val version = FabricLoader.getInstance()
            .getModContainer(MOD_ID)
            .map { it.metadata.version.friendlyString }
            .orElse("development")

        log.info("{} ({} / codename {}) {} initializing", INTERNAL_NAME, PUBLIC_NAME, CODENAME, version)
        if (testTierSwitchSource != null) {
            log.info("VS_TEST_TIER_SWITCH armed: {} -> TIER_2 after 120 in-world ticks", testTierSwitchSource)
        }

        runCatching {
            val gameDir = FabricLoader.getInstance().gameDir.toAbsolutePath()
            NativeCoreBridge.setGameDir(gameDir.toString())
            ProprietaryLibraryManager.initialize(gameDir)
        }
        ShaderPackManager.initialize(config)
        refreshRuntimeState()
    }

    internal fun initializeConfig(loadedConfig: VulkanShadersConfig) {
        config = loadedConfig
    }

    @JvmStatic
    fun updateConfig(newConfig: VulkanShadersConfig): Boolean {
        // Any later config change (rapid switch, watchdog rollback) supersedes a queued activation.
        pendingDeferredTier2Config = null
        clearEmergencyBypass()
        if (!StandaloneRenderPearlBridge.vulkanTurboLoaded && newConfig.enabled &&
            !newConfig.selectedPackId.equals("off", ignoreCase = true)
        ) {
            ShaderNotices.frameGeneration(
                Component.translatable("vulkanshaders.screen.title"),
                Component.translatable("vulkanshaders.screen.requires_pyroclast"),
            )
            return false
        }
        val prev = config
        // Entering TIER_2 rebuilds RT pipelines and recompiles the pack. Queue that work behind a
        // few successful world-graph frames so no in-flight frame waits on a stale semaphore.
        val deferTier2 = newConfig.renderTier == LightingMode.HYBRID &&
            prev.renderTier != LightingMode.HYBRID
        if (deferTier2) {
            pendingDeferredTier2Config = newConfig
            log.info("TIER_2 activation deferred: config persisted, rebuild queued for frame boundary")
        }
        val packId = newConfig.selectedPackId
        val frameGenChanged = prev.frameGenMode != newConfig.frameGenMode
        // Only a tier or pack change requires unzipping and recompiling shader modules.
        val shaderPathChanged = prev.selectedPackId != packId || prev.renderTier != newConfig.renderTier
        val pipelineChanged = shaderPathChanged || frameGenChanged ||
            prev.enableSer != newConfig.enableSer ||
            prev.denoiserMode != newConfig.denoiserMode || prev.upscalerMode != newConfig.upscalerMode ||
            prev.hdrEnabled != newConfig.hdrEnabled || prev.enableOverlayPass != newConfig.enableOverlayPass
        val oldOverrides = prev.activePackOverrides(packId)
        val newOverrides = newConfig.activePackOverrides(packId)
        val specChanged = ShaderPackManager.activePack?.manifest?.settings?.any { setting ->
            !setting.runtime && setting.appliesToTier(newConfig.renderTier.tierIndex) &&
                (oldOverrides[setting.key] ?: setting.defaultValue) !=
                (newOverrides[setting.key] ?: setting.defaultValue)
        } == true
        if (shaderPathChanged || frameGenChanged) NativeCoreBridge.waitPresentIdle()
        // Deferred TIER_2 skips the pipeline rebuild; applyDeferredTier2() replays it later.
        if (pipelineChanged && !deferTier2) {
            val pipelineStatus = NativeCoreBridge.configurePipeline(newConfig)
            if (frameGenChanged && newConfig.frameGenMode != FrameGenMode.NONE && pipelineStatus < 0) {
                return false
            }
        } else if (!deferTier2 && (prev.shadowResolution != newConfig.shadowResolution ||
            prev.shadowDistanceChunks != newConfig.shadowDistanceChunks)
        ) {
            NativeCoreBridge.setShadowConfig(newConfig.shadowResolution, newConfig.shadowDistanceChunks)
        }
        // Keep the LIVE config on the previous tier until applyDeferredTier2() has rebuilt the
        // native graph. Committing TIER_2 here would make the deferred frames run TIER_2 Kotlin
        // logic (world rasterization, DH targets, blob shadows) against a TIER_3 native graph,
        // which stalls the render queue until RenderPearl's 5 s semaphore timeout fires.
        config = if (deferTier2) newConfig.copy(renderTier = prev.renderTier) else newConfig
        ConfigStore.save(newConfig)
        NativeCoreBridge.setRenderScale(newConfig.effectiveUpscalerQuality.renderScale)
        NativeCoreBridge.setHdrParams(newConfig.hdrPeakBrightnessNits, newConfig.hdrPaperWhiteNits)
        if (prev.enableGpuProfiler != newConfig.enableGpuProfiler) {
            syncGpuProfilerState(newConfig.enableGpuProfiler, force = true)
        }
        if (shaderPathChanged && !deferTier2) {
            GbufferTargetManager.invalidateForResize()
            GbufferPipelineRegistry.invalidateShaderSourceCache()
            ShaderPackManager.selectPack(newConfig.selectedPackId)
        } else if (!deferTier2 && specChanged) {
            ShaderPackManager.syncActivePackSpecializationConstants()
            GbufferPipelineRegistry.invalidateShaderSourceCache()
        }
        if (shaderPathChanged || oldOverrides != newOverrides || frameGenChanged ||
            prev.denoiserMode != newConfig.denoiserMode || prev.upscalerMode != newConfig.upscalerMode ||
            prev.upscalerQuality != newConfig.upscalerQuality || prev.shadowResolution != newConfig.shadowResolution ||
            prev.shadowDistanceChunks != newConfig.shadowDistanceChunks
        ) {
            WorldStateCollector.resetHistory()
        }
        refreshRuntimeState()
        if (prev.renderTier != config.renderTier) {
            RuntimePerformanceGuard.resetForStateChange()
        }
        if (shaderPackActive) {
            TextureRegistry.ensureCompanionAtlasesBuilt()
        }
        return true
    }

    /** Called by the render graph once a world frame has been fully recorded. */
    @JvmStatic
    fun markWorldGraphFrame() {
        lastWorldGraphFrameNanos = System.nanoTime()
    }

    /** Any explicit config change from the GUI is an intentional request to return to shaders. */
    @JvmStatic
    fun clearEmergencyBypass() {
        emergencyBypass = false
    }

    /**
     * Session-only emergency shutdown. Keeps the user's saved config and selected pack intact so
     * the shader can be restored from the GUI or by restarting the game.
     */
    @JvmStatic
    fun emergencyDisableForSession(reason: String) {
        if (emergencyBypass) return
        emergencyBypass = true
        shaderPackActive = false
        val msg = "$reason — Hyaloclast shaders disabled for this session."
        log.warn(msg)
        HyaloclastErrorLog.recordError("RenderWatchdog", msg)
        runCatching { Minecraft.getInstance() }.getOrNull()?.execute {
            runCatching {
                ShaderNotices.showStatusNotice(
                    Component.literal("Hyaloclast Watchdog"),
                    Component.literal(msg),
                )
                Minecraft.getInstance().gui.hud.getChat()
                    .addClientSystemMessage(buildGradientMessage(": $msg"))
            }
        }
    }

    private fun startRenderWatchdog() {
        Thread({
            while (!Thread.currentThread().isInterrupted) {
                Thread.sleep(250L)
                val mc = runCatching { Minecraft.getInstance() }.getOrNull() ?: continue
                if (mc.level == null || mc.isPaused) {
                    lastWorldGraphFrameNanos = System.nanoTime()
                    continue
                }
                if (!shaderPackActive || emergencyBypass) continue
                val last = lastWorldGraphFrameNanos
                if (last > 0L && System.nanoTime() - last > 2_000_000_000L) {
                    emergencyDisableForSession("No render-graph frame completed for 2 s")
                }
            }
        }, "hyaloclast-render-watchdog").apply {
            isDaemon = true
            start()
        }
    }

    @JvmStatic
    fun hasPendingDeferredTier2(): Boolean = pendingDeferredTier2Config != null

    /**
     * Second phase of a TIER_2 switch: replays the rebuild deferred by [updateConfig] once the
     * previous tier has rendered clean frames. Called from the client's frame-boundary task queue.
     */
    @JvmStatic
    fun applyDeferredTier2(framesWaited: Int = 0) {
        val pending = pendingDeferredTier2Config ?: return
        if (pending.renderTier != LightingMode.HYBRID) {
            pendingDeferredTier2Config = null
            log.warn("Discarded stale deferred TIER_2 activation; current tier is {}", config.renderTier)
            return
        }
        pendingDeferredTier2Config = null
        // The normal switch path waits for in-flight present work before rebuilding; do the same
        // here instead of rebuilding while a previous-tier frame is still being submitted.
        NativeCoreBridge.waitPresentIdle()
        val pipelineStatus = NativeCoreBridge.configurePipeline(pending)
        GbufferTargetManager.invalidateForResize()
        GbufferPipelineRegistry.invalidateShaderSourceCache()
        ShaderPackManager.selectPack(pending.selectedPackId)
        config = pending
        WorldStateCollector.resetHistory()
        NativeCoreBridge.resetTemporalHistory()
        NativeCoreBridge.markHistoryReset()
        refreshRuntimeState()
        RuntimePerformanceGuard.resetForStateChange()
        log.info(
            "deferred TIER_2 applied after {} frames (configurePipeline status={})",
            framesWaited,
            pipelineStatus,
        )
    }

    /**
     * Fast path for `runtime = true` shader pack settings: updates in-memory config and pushes
     * `WorldUBO.packParams` on the next frame without unzipping, recompiling SPIR-V, or reloading PNGs.
     */
    @JvmStatic
    fun updateRuntimePackOverrides(newConfig: VulkanShadersConfig, persistToDisk: Boolean = false) {
        config = newConfig
        if (persistToDisk) {
            ConfigStore.save(newConfig)
        }
    }

    @JvmStatic
    fun refreshRuntimeState() {
        val wasActive = shaderPackActive
        val backendOk = StandaloneRenderPearlBridge.vulkanTurboLoaded &&
            StandaloneRenderPearlBridge.isBackendAllowed() &&
            StandaloneRenderPearlBridge.isVulkanBackendActive()
        vulkanReady = backendOk
        shaderPackActive = !emergencyBypass && config.enabled && backendOk && ShaderPackManager.activePack != null
        StandaloneRenderPearlBridge.refreshBridgeState()
        if (wasActive != shaderPackActive) {
            val client: Minecraft? = Minecraft.getInstance()
            if (client?.level == null) return
            VulkanShadersTurboExtension.resetPerWorldState()
            SectionSliceRegistry.clear()
            // Replace the meshes while keeping the occlusion graph's records of loaded chunks.
            client.levelExtractor.allChanged()
        }
    }

    /**
     * Keeps the active pack and chunk pipeline in sync with Pyroclast's live enable switch.
     * Called through a guarded bridge so VulkanTurbo can remain independent of this mod.
     */
    @JvmStatic
    fun onPyroclastStateChanged(enabled: Boolean) {
        if (!enabled) {
            config = config.copy(enabled = false)
            ConfigStore.save(config)
            ShaderPackManager.selectPack("off")
        } else if (!config.enabled && !config.selectedPackId.equals("off", ignoreCase = true)) {
            config = config.copy(enabled = true)
            ConfigStore.save(config)
            ShaderPackManager.selectPack(config.selectedPackId)
        }
        refreshRuntimeState()
        if (shaderPackActive) {
            TextureRegistry.ensureCompanionAtlasesBuilt()
        }
    }

    @JvmStatic
    fun onRendererReady(device: GpuDevice?) {
        val backend = try {
            device?.getDeviceInfo()?.backendName()
        } catch (_: Throwable) {
            null
        }
        vulkanReady = backend.equals("Vulkan", ignoreCase = true)
        if (vulkanReady) {
            VulkanDeviceContext.initialize(device)
            if (NativeCoreBridge.initialized) {
                NativeCoreBridge.setRenderScale(config.effectiveUpscalerQuality.renderScale)
                NativeCoreBridge.setHdrParams(config.hdrPeakBrightnessNits, config.hdrPaperWhiteNits)
                syncGpuProfilerState(config.enableGpuProfiler, force = true)
                if (config.enabled && !config.selectedPackId.equals("off", ignoreCase = true)) {
                    val compiledOk = ShaderPackManager.selectPack(config.selectedPackId)
                    log.info(
                        "Initial shader pack '{}' compilation: ok={}, activePack='{}', error={}",
                        config.selectedPackId,
                        compiledOk,
                        ShaderPackManager.activePack?.manifest?.id,
                        ShaderPackManager.lastPackError,
                    )
                }
            }
        }
        refreshRuntimeState()
        if (shaderPackActive) {
            TextureRegistry.ensureCompanionAtlasesBuilt()
        }
        pendingChatAnnouncement = true
    }

    @JvmStatic
    fun onClientTick(mc: Minecraft) {
        HyaloclastKeybinds.tick(mc)
        ShaderPackManager.notifyRenderAlive()
        RuntimePerformanceGuard.onClientTick(mc)

        // TEMPORARY headless verification hook; remove after TIER_2 switch is validated in-game.
        val testSourceTier = testTierSwitchSource
        if (testSourceTier != null && testTierSwitchCountdown >= 0 && mc.level != null && shaderPackActive) {
            if (--testTierSwitchCountdown <= 0) {
                testTierSwitchCountdown = -1
                if (config.renderTier == testSourceTier) {
                    log.info("VS_TEST_TIER_SWITCH: requesting {} -> HYBRID", testSourceTier)
                    updateConfig(config.copy(renderTier = LightingMode.HYBRID))
                } else {
                    log.warn(
                        "VS_TEST_TIER_SWITCH: expected {}, found {}; switch not triggered",
                        testSourceTier,
                        config.renderTier,
                    )
                }
            }
        }

        if (!StandaloneRenderPearlBridge.vulkanTurboLoaded) {
            NativeCoreBridge.drainNativeErrorsToLog()
            ShaderNotices.pollErrorToastInteraction(mc)
        }

        if (NativeCoreBridge.initialized) {
            syncGpuProfilerState(config.enableGpuProfiler)
            telemetryTickCounter++
            if (telemetryTickCounter >= TELEMETRY_POLL_INTERVAL_TICKS) {
                telemetryTickCounter = 0
                NativeCoreBridge.pollTelemetryToDebugHud()
                NativeCoreBridge.checkDeviceLostRecovery()
            }
        }

        val level = mc.level
        if (level != null) {
            val dimKey = level.dimension()
            if (dimKey != lastDimensionKey) {
                val wasInWorld = lastDimensionKey != null
                lastDimensionKey = dimKey
                onDimensionChanged(resolveDimensionId(level), resetTemporal = wasInWorld)
            }
        }

        val mode = config.frameGenMode
        if (mode != FrameGenMode.NONE && NativeCoreBridge.takeFrameGenFailure() == mode.ordinal) {
            updateConfig(config.copy(frameGenMode = FrameGenMode.NONE))
            ShaderNotices.frameGeneration(
                Component.translatable("vulkanshaders.screen.framegen.disabled"),
                Component.translatable("vulkanshaders.screen.framegen.stopped", mode.frameGenLabel()),
            )
        }
        if (!pendingChatAnnouncement) return
        if (mc.player == null || mc.level == null) return
        pendingChatAnnouncement = false

        val packName = ShaderPackManager.activePack?.manifest?.name ?: "Off"
        val body = when {
            !StandaloneRenderPearlBridge.vulkanTurboLoaded ->
                ": Pyroclast (VulkanTurbo) is missing — shader engine disabled."
            !StandaloneRenderPearlBridge.isVulkanBackendActive() -> ": Vulkan backend required — shader engine disabled."
            !config.enabled -> ": Disabled in config."
            ShaderPackManager.activePack == null -> ": No shader pack active."
            else -> ": Pack '$packName' loaded (${config.renderTier.name})."
        }
        val msg = buildGradientMessage(body)
        mc.gui.hud.getChat().addClientSystemMessage(msg)
    }

    @JvmStatic
    fun onDisplayResized() {
        GbufferTargetManager.invalidateForResize()
        WorldStateCollector.resetHistory()
        NativeCoreBridge.onDisplayResized()
        NativeCoreBridge.markHistoryReset()
    }

    @JvmStatic
    fun onResourcePacksReloaded() {
        WorldStateCollector.resetHistory()
        NativeCoreBridge.resetTemporalHistory()
        NativeCoreBridge.markHistoryReset()
        TextureRegistry.closeCompanionAtlases()
        NativeCoreBridge.onResourcePacksReloaded()
        if (shaderPackActive) {
            TextureRegistry.ensureCompanionAtlasesBuilt()
        }
        if (NativeCoreBridge.initialized) {
            NativeCoreBridge.syncCustomDescriptorSetsToBridge()
        }
    }

    @JvmStatic
    fun onDimensionChanged(dimensionId: Int, resetTemporal: Boolean = true) {
        log.info("Dimension changed: id={} (0=Overworld, 1=Nether, 2=End)", dimensionId)
        NativeCoreBridge.setDimension(dimensionId)
        if (resetTemporal) {
            WorldStateCollector.resetHistory()
            NativeCoreBridge.resetTemporalHistory()
            NativeCoreBridge.markHistoryReset()
        }
    }

    @JvmStatic
    fun onWorldUnloading() {
        VulkanShadersTurboExtension.resetPerWorldState()
        WorldStateCollector.resetHistory()
        NativeCoreBridge.resetTemporalHistory()
        NativeCoreBridge.markHistoryReset()
    }

    @JvmStatic
    fun onWorldUnloaded() {
        lastDimensionKey = null
        lastSyncedGpuProfilerEnabled = null
        telemetryTickCounter = 0
        VulkanShadersTurboExtension.resetPerWorldState()
        if (StandaloneRenderPearlBridge.isStandaloneMode()) {
            StandaloneRenderPearlBridge.resetStandaloneWorldState()
        }
        WorldStateCollector.resetHistory()
        NativeCoreBridge.setDimension(0)
        NativeCoreBridge.resetTemporalHistory()
        NativeCoreBridge.markHistoryReset()
        NativeCoreBridge.onWorldUnloaded()
        HyaloclastDebugHud.clearNativeMetrics()
        GbufferTargetManager.closeAll()
        SectionSliceRegistry.clear()
        DistantHorizonsBridge.clearAll()
        pendingChatAnnouncement = true
    }

    private fun syncGpuProfilerState(enabled: Boolean, force: Boolean = false) {
        if (!NativeCoreBridge.initialized) {
            lastSyncedGpuProfilerEnabled = null
            return
        }
        if (force || lastSyncedGpuProfilerEnabled != enabled) {
            lastSyncedGpuProfilerEnabled = enabled
            NativeCoreBridge.setGpuProfilerEnabled(enabled)
        }
    }

    private fun resolveDimensionId(level: ClientLevel): Int {
        val dimType = level.dimensionType()
        // The End has skylight in Minecraft 26.3; identify it by dimension id first.
        return when (level.dimension().identifier().toString().substringAfter(':')) {
            "the_end" -> 2
            "the_nether" -> 1
            else -> when {
                dimType.hasCeiling() -> 1
                !dimType.hasSkyLight() -> 2
                else -> 0
            }
        }
    }

    internal fun buildGradientBrandComponent(): MutableComponent {
        val root = Component.empty()
        val last = (BRAND_NAME.length - 1).coerceAtLeast(1)
        for (i in BRAND_NAME.indices) {
            val t = i.toFloat() / last.toFloat()
            root.append(
                Component.literal(BRAND_NAME.substring(i, i + 1)).withStyle(
                    Style.EMPTY.withColor(TextColor.fromRgb(gradientRgb(t))).withBold(true),
                ),
            )
        }
        return root
    }

    /** Full chat line: banner-gradient brand plus body tinted with a color from the same banner. */
    internal fun buildGradientMessage(body: String): MutableComponent =
        buildGradientBrandComponent()
            .append(Component.literal(body).withStyle(Style.EMPTY.withColor(TextColor.fromRgb(BRAND_BODY_COLOR)).withBold(false)))

    internal fun buildGradientMessage(body: Component): MutableComponent =
        buildGradientMessage(": ${body.string}")
}
