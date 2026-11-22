package dev.vulkanshaders

import com.mojang.renderpearl.api.device.GpuDevice
import dev.vulkanshaders.backend.NativeCoreBridge
import dev.vulkanshaders.backend.VulkanDeviceContext
import dev.vulkanshaders.bridge.DistantHorizonsBridge
import dev.vulkanshaders.bridge.StandaloneRenderPearlBridge
import dev.vulkanshaders.bridge.VulkanShadersTurboExtension
import dev.vulkanshaders.config.ConfigStore
import dev.vulkanshaders.config.FrameGenMode
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
import net.minecraft.ChatFormatting
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
    private const val GRADIENT_START_R = 0x00
    private const val GRADIENT_START_G = 0xE5
    private const val GRADIENT_START_B = 0xFF
    private const val GRADIENT_END_R = 0x7B
    private const val GRADIENT_END_G = 0x61
    private const val GRADIENT_END_B = 0xFF
    private const val TELEMETRY_POLL_INTERVAL_TICKS: Int = 10

    val log: Logger = LoggerFactory.getLogger(MOD_ID)

    @Volatile
    var config: VulkanShadersConfig = VulkanShadersConfig()
        private set

    @JvmField @Volatile var shaderPackActive: Boolean = false
    @JvmField @Volatile var vulkanReady: Boolean = false
    @Volatile private var pendingChatAnnouncement: Boolean = false
    @Volatile private var lastSyncedGpuProfilerEnabled: Boolean? = null
    @Volatile private var lastDimensionKey: Any? = null
    private var telemetryTickCounter: Int = 0

    override fun onInitializeClient() {
        HyaloclastErrorLog.installCrashHooks()
        HyaloclastKeybinds.register()
        if (StandaloneRenderPearlBridge.isStandaloneMode()) {
            StandaloneRenderPearlBridge.ensureStandaloneInitialized()
        }

        config = ConfigStore.load()
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
        val packId = newConfig.selectedPackId
        val frameGenChanged = prev.frameGenMode != newConfig.frameGenMode
        val packOrTierChanged = prev.selectedPackId != packId ||
            prev.renderTier != newConfig.renderTier || prev.enabled != newConfig.enabled
        val oldOverrides = prev.activePackOverrides(packId)
        val newOverrides = newConfig.activePackOverrides(packId)
        val specChanged = ShaderPackManager.activePack?.manifest?.settings?.any { setting ->
            !setting.runtime && setting.appliesToTier(newConfig.renderTier.tierIndex) &&
                (oldOverrides[setting.key] ?: setting.defaultValue) !=
                (newOverrides[setting.key] ?: setting.defaultValue)
        } == true
        if (frameGenChanged || packOrTierChanged) NativeCoreBridge.waitPresentIdle()
        val pipelineStatus = NativeCoreBridge.configurePipeline(newConfig)
        if (frameGenChanged && newConfig.frameGenMode != FrameGenMode.NONE && pipelineStatus < 0) {
            return false
        }
        config = newConfig
        ConfigStore.save(newConfig)
        NativeCoreBridge.setRenderScale(newConfig.effectiveUpscalerQuality.renderScale)
        NativeCoreBridge.setHdrParams(newConfig.hdrPeakBrightnessNits, newConfig.hdrPaperWhiteNits)
        if (prev.enableGpuProfiler != newConfig.enableGpuProfiler) {
            syncGpuProfilerState(newConfig.enableGpuProfiler, force = true)
        }
        if (packOrTierChanged) {
            GbufferTargetManager.invalidateForResize()
            GbufferPipelineRegistry.invalidateShaderSourceCache()
            ShaderPackManager.selectPack(newConfig.selectedPackId)
        } else if (specChanged) {
            ShaderPackManager.syncActivePackSpecializationConstants()
            GbufferPipelineRegistry.invalidateShaderSourceCache()
        }
        if (packOrTierChanged || oldOverrides != newOverrides || frameGenChanged ||
            prev.denoiserMode != newConfig.denoiserMode || prev.upscalerMode != newConfig.upscalerMode ||
            prev.upscalerQuality != newConfig.upscalerQuality || prev.shadowResolution != newConfig.shadowResolution ||
            prev.shadowDistanceChunks != newConfig.shadowDistanceChunks
        ) {
            WorldStateCollector.resetHistory()
        }
        refreshRuntimeState()
        if (shaderPackActive) {
            TextureRegistry.ensureCompanionAtlasesBuilt()
        }
        return true
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
        shaderPackActive = config.enabled && backendOk && ShaderPackManager.activePack != null
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
        val msg = buildGradientBrandComponent()
            .append(Component.literal(body).withStyle(Style.EMPTY.withColor(ChatFormatting.WHITE).withBold(false)))
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
        return when {
            dimType.hasCeiling() -> 1
            !dimType.hasSkyLight() -> 2
            else -> 0
        }
    }

    internal fun buildGradientBrandComponent(): MutableComponent {
        val root = Component.empty()
        val last = (BRAND_NAME.length - 1).coerceAtLeast(1)
        for (i in BRAND_NAME.indices) {
            val t = i.toFloat() / last.toFloat()
            val r = (GRADIENT_START_R + (GRADIENT_END_R - GRADIENT_START_R) * t).toInt()
            val g = (GRADIENT_START_G + (GRADIENT_END_G - GRADIENT_START_G) * t).toInt()
            val b = (GRADIENT_START_B + (GRADIENT_END_B - GRADIENT_START_B) * t).toInt()
            val rgb = (r shl 16) or (g shl 8) or b
            root.append(
                Component.literal(BRAND_NAME.substring(i, i + 1)).withStyle(
                    Style.EMPTY.withColor(TextColor.fromRgb(rgb)).withBold(true),
                ),
            )
        }
        return root
    }
}
