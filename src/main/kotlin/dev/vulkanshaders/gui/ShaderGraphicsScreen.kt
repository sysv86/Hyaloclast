package dev.vulkanshaders.gui

import dev.vulkanshaders.VulkanShaders
import dev.vulkanshaders.backend.NativeCoreBridge
import dev.vulkanshaders.config.DenoiserMode
import dev.vulkanshaders.config.FrameGenMode
import dev.vulkanshaders.config.HardwareAutoCalibrator
import dev.vulkanshaders.config.LightingMode
import dev.vulkanshaders.config.UpscalerMode
import dev.vulkanshaders.config.UpscalerQuality
import dev.vulkanshaders.pack.ShaderPackManager
import dev.vulkanshaders.sdk.NativePlatform
import dev.vulkanshaders.sdk.OsFamily
import dev.vulkanshaders.sdk.ProprietaryLibraryManager
import dev.vulkanshaders.state.WorldStateCollector
import net.minecraft.ChatFormatting
import net.minecraft.client.Options
import net.minecraft.client.gui.components.Button
import net.minecraft.client.gui.components.Tooltip
import net.minecraft.client.gui.layouts.LinearLayout
import net.minecraft.client.gui.screens.Screen
import net.minecraft.client.gui.screens.options.OptionsSubScreen
import net.minecraft.network.chat.CommonComponents
import net.minecraft.network.chat.Component

class ShaderGraphicsScreen(
    lastScreen: Screen,
    options: Options,
) : OptionsSubScreen(lastScreen, options, TITLE) {

    private var frameGenButton: Button? = null
    private var displayedFrameGenMode = VulkanShaders.config.frameGenMode
    private var lastLibraryStateVersion: Long = ProprietaryLibraryManager.stateVersion
    private var calibrationFeedback: Component? = null

    companion object {
        @JvmField
        val TITLE: Component = Component.translatable("vulkanshaders.screen.graphics.title")

        private val SHADOW_RESOLUTIONS = listOf(1024, 2048, 4096)
        private val SHADOW_RESOLUTIONS_HIGH_END = listOf(1024, 2048, 4096, 8192)
        private val SHADOW_DISTANCE_CHUNKS = listOf(8, 12, 16, 24, 32)
        private val SHADOW_DISTANCE_CHUNKS_HIGH_END = listOf(8, 12, 16, 24, 32, 48)

        internal fun isMacOsOrMoltenVk(): Boolean {
            if (NativePlatform.detect().os == OsFamily.MACOS) return true
            val driver = runCatching { NativeCoreBridge.getHardwareProfile().driver }.getOrDefault("")
            return driver.contains("MoltenVK", ignoreCase = true)
        }

        internal fun tierTooltipKey(tier: LightingMode): String = when (tier) {
            LightingMode.CLASSIC -> "vulkanshaders.screen.mode.tooltip.classic"
            LightingMode.COMPUTE -> "vulkanshaders.screen.mode.tooltip.compute"
            LightingMode.HYBRID -> "vulkanshaders.screen.mode.tooltip.hybrid"
            LightingMode.PATH -> "vulkanshaders.screen.mode.tooltip.path"
        }

        internal fun upscalerTooltipKey(mode: UpscalerMode): String = when (mode) {
            UpscalerMode.NONE -> "vulkanshaders.screen.upscaler.tooltip.none"
            UpscalerMode.TSR -> "vulkanshaders.screen.upscaler.tooltip.tsr"
            UpscalerMode.FSR3 -> "vulkanshaders.screen.upscaler.tooltip.fsr3"
            UpscalerMode.DLSS -> "vulkanshaders.screen.upscaler.tooltip.dlss"
            UpscalerMode.XESS -> "vulkanshaders.screen.upscaler.tooltip.xess"
        }

        internal fun denoiserTooltipKey(mode: DenoiserMode): String = when (mode) {
            DenoiserMode.NONE -> "vulkanshaders.screen.denoiser.tooltip.none"
            DenoiserMode.SVGF -> "vulkanshaders.screen.denoiser.tooltip.svgf"
            DenoiserMode.NRD -> "vulkanshaders.screen.denoiser.tooltip.nrd"
            DenoiserMode.DLSS_RR -> "vulkanshaders.screen.denoiser.tooltip.dlss_rr"
        }

        internal fun frameGenTooltipKey(mode: FrameGenMode): String = when (mode) {
            FrameGenMode.NONE -> "vulkanshaders.screen.framegen.tooltip.none"
            FrameGenMode.OPTICAL_FLOW -> "vulkanshaders.screen.framegen.tooltip.optical_flow"
            FrameGenMode.FSR3_FG -> "vulkanshaders.screen.framegen.tooltip.fsr3_fg"
            FrameGenMode.DLSS_FG -> "vulkanshaders.screen.framegen.tooltip.dlss_fg"
        }

        internal fun upscalerLabelKey(mode: UpscalerMode): String = when (mode) {
            UpscalerMode.NONE -> "vulkanshaders.screen.upscaler.mode.none"
            UpscalerMode.TSR -> "vulkanshaders.screen.upscaler.mode.tsr"
            UpscalerMode.FSR3 -> "vulkanshaders.screen.upscaler.mode.fsr3"
            UpscalerMode.DLSS -> "vulkanshaders.screen.upscaler.mode.dlss"
            UpscalerMode.XESS -> "vulkanshaders.screen.upscaler.mode.xess"
        }

        internal fun denoiserLabelKey(mode: DenoiserMode): String = when (mode) {
            DenoiserMode.NONE -> "vulkanshaders.screen.denoiser.mode.none"
            DenoiserMode.SVGF -> "vulkanshaders.screen.denoiser.mode.svgf"
            DenoiserMode.NRD -> "vulkanshaders.screen.denoiser.mode.nrd"
            DenoiserMode.DLSS_RR -> "vulkanshaders.screen.denoiser.mode.dlss_rr"
        }

        internal fun tierAvailable(
            tier: LightingMode,
            caps: Int,
            macOsMoltenVk: Boolean = isMacOsOrMoltenVk(),
        ): Boolean = when (tier) {
            LightingMode.CLASSIC, LightingMode.COMPUTE -> true
            LightingMode.HYBRID ->
                (caps and NativeCoreBridge.CAP_RAY_QUERY) != 0 &&
                    (caps and NativeCoreBridge.CAP_RAY_TRACING_AS) != 0
            LightingMode.PATH ->
                (caps and NativeCoreBridge.CAP_RAY_TRACING_PIPELINE) != 0 &&
                    (caps and NativeCoreBridge.CAP_RAY_TRACING_AS) != 0
        }

        internal fun upscalerAvailable(mode: UpscalerMode, caps: Int): Boolean = when (mode) {
            UpscalerMode.NONE, UpscalerMode.TSR -> true
            UpscalerMode.FSR3 -> (caps and NativeCoreBridge.CAP_FSR3) != 0
            UpscalerMode.XESS -> (caps and NativeCoreBridge.CAP_XESS) != 0
            UpscalerMode.DLSS -> (caps and NativeCoreBridge.CAP_DLSS_SR) != 0
        }

        internal fun denoiserAvailable(mode: DenoiserMode, caps: Int): Boolean = when (mode) {
            DenoiserMode.NONE, DenoiserMode.SVGF -> true
            DenoiserMode.NRD -> (caps and NativeCoreBridge.CAP_NRD) != 0
            DenoiserMode.DLSS_RR -> (caps and NativeCoreBridge.CAP_DLSS_RR) != 0
        }

        internal fun frameGenAvailable(mode: FrameGenMode, caps: Int): Boolean = when (mode) {
            FrameGenMode.NONE, FrameGenMode.OPTICAL_FLOW -> true
            FrameGenMode.FSR3_FG -> (caps and NativeCoreBridge.CAP_FSR3) != 0
            FrameGenMode.DLSS_FG -> (caps and NativeCoreBridge.CAP_DLSS_FG) != 0
        }

        private fun <T> nextAvailable(current: T, all: List<T>, available: (T) -> Boolean): T {
            val enabled = all.filter(available)
            if (enabled.isEmpty()) return current
            val idx = enabled.indexOf(current)
            return enabled[(idx + 1) % enabled.size]
        }

        private fun unavailableSuffix(available: Boolean): Component =
            if (available) Component.empty() else Component.literal("\n").append(
                Component.translatable("vulkanshaders.screen.graphics.unavailable"),
            )
    }

    override fun init() {
        layout.removeChildren()
        super.init()
    }

    override fun tick() {
        val libVersion = ProprietaryLibraryManager.stateVersion
        if (libVersion != lastLibraryStateVersion) {
            lastLibraryStateVersion = libVersion
            refreshWidgets()
            return
        }
        val mode = VulkanShaders.config.frameGenMode
        if (displayedFrameGenMode != mode) {
            displayedFrameGenMode = mode
            frameGenButton?.message = frameGenLabel(mode)
            frameGenButton?.setTooltip(Tooltip.create(Component.translatable(frameGenTooltipKey(mode))))
        }
    }

    private fun refreshWidgets() {
        val scroll = list?.scrollAmount() ?: 0.0
        rebuildWidgets()
        list?.setScrollAmount(scroll)
    }

    private fun frameGenLabel(mode: FrameGenMode): Component =
        Component.translatable("vulkanshaders.screen.framegen").append(": ").append(mode.frameGenLabel())

    private fun resetTemporalBuffers() {
        WorldStateCollector.resetHistory()
        NativeCoreBridge.resetTemporalHistory()
        NativeCoreBridge.markHistoryReset()
    }

    override fun addOptions() {
        val list = this.list ?: return
        val cfg = VulkanShaders.config
        val caps = NativeCoreBridge.queryCapabilities()
        val highEndGpu = NativeCoreBridge.isHighEndGpu()
        val macOsMoltenVk = isMacOsOrMoltenVk()
        val hardwareRtAvailable = tierAvailable(LightingMode.HYBRID, caps, macOsMoltenVk) ||
            tierAvailable(LightingMode.PATH, caps, macOsMoltenVk)

        val shadowResOptions = if (highEndGpu) SHADOW_RESOLUTIONS_HIGH_END else SHADOW_RESOLUTIONS
        val shadowDistOptions = if (highEndGpu) SHADOW_DISTANCE_CHUNKS_HIGH_END else SHADOW_DISTANCE_CHUNKS

        val supportedTiers = ShaderPackManager.activePack?.manifest?.supportedTiers
        val tierOk = { tier: LightingMode ->
            tierAvailable(tier, caps, macOsMoltenVk) &&
                (supportedTiers == null || supportedTiers.contains(tier.tierIndex))
        }
        val tierTooltip = Component.translatable(tierTooltipKey(cfg.renderTier))
            .append(unavailableSuffix(tierOk(cfg.renderTier)))
        if (macOsMoltenVk && !hardwareRtAvailable) {
            tierTooltip.append(Component.literal("\n")).append(
                Component.translatable("vulkanshaders.screen.graphics.macos_rt_unavailable")
                    .withStyle(ChatFormatting.GOLD),
            )
        }
        val tierButton = Button.builder(
            Component.translatable("vulkanshaders.screen.mode")
                .append(": ")
                .append(Component.translatable(cfg.renderTier.translationKey)),
        ) {
            val next = nextAvailable(cfg.renderTier, LightingMode.entries, tierOk)
            calibrationFeedback = null
            if (VulkanShaders.updateConfig(VulkanShaders.config.copy(renderTier = next))) {
                resetTemporalBuffers()
            }
            refreshWidgets()
        }.tooltip(Tooltip.create(tierTooltip)).build()

        val upscalerOk = { mode: UpscalerMode -> upscalerAvailable(mode, caps) }
        val upscalerButton = Button.builder(
            Component.translatable("vulkanshaders.screen.upscaler")
                .append(": ")
                .append(Component.translatable(upscalerLabelKey(cfg.upscalerMode))),
        ) {
            val next = nextAvailable(cfg.upscalerMode, UpscalerMode.entries, upscalerOk)
            calibrationFeedback = null
            NativeCoreBridge.waitPresentIdle()
            if (VulkanShaders.updateConfig(VulkanShaders.config.copy(upscalerMode = next))) {
                resetTemporalBuffers()
            }
            refreshWidgets()
        }.tooltip(
            Tooltip.create(
                Component.translatable(upscalerTooltipKey(cfg.upscalerMode))
                    .append(unavailableSuffix(upscalerOk(cfg.upscalerMode)))
                    .append(pendingRestartUpscalerSuffix()),
            ),
        ).build()

        val upscalerQualityActive = cfg.upscalerMode != UpscalerMode.NONE
        val upscalerQualityTooltip = Component.translatable("vulkanshaders.screen.upscaler_quality.tooltip")
        if (!upscalerQualityActive) {
            upscalerQualityTooltip.append(Component.literal("\n")).append(
                Component.translatable("vulkanshaders.screen.upscaler_quality.requires_upscaler")
                    .withStyle(ChatFormatting.GRAY),
            )
        }
        val upscalerQualityButton = Button.builder(
            Component.translatable("vulkanshaders.screen.upscaler_quality")
                .append(": ")
                .append(Component.translatable(cfg.effectiveUpscalerQuality.translationKey)),
        ) {
            val entries = UpscalerQuality.entries
            val idx = entries.indexOf(cfg.upscalerQuality).coerceAtLeast(0)
            val next = entries[(idx + 1) % entries.size]
            calibrationFeedback = null
            if (VulkanShaders.updateConfig(VulkanShaders.config.copy(upscalerQuality = next))) {
                resetTemporalBuffers()
            }
            refreshWidgets()
        }.tooltip(Tooltip.create(upscalerQualityTooltip)).build()
        upscalerQualityButton.active = upscalerQualityActive

        val denoiserOk = { mode: DenoiserMode -> denoiserAvailable(mode, caps) }
        val denoiserButton = Button.builder(
            Component.translatable("vulkanshaders.screen.denoiser")
                .append(": ")
                .append(Component.translatable(denoiserLabelKey(cfg.denoiserMode))),
        ) {
            val next = nextAvailable(cfg.denoiserMode, DenoiserMode.entries, denoiserOk)
            calibrationFeedback = null
            if (VulkanShaders.updateConfig(VulkanShaders.config.copy(denoiserMode = next))) {
                resetTemporalBuffers()
            }
            refreshWidgets()
        }.tooltip(
            Tooltip.create(
                Component.translatable(denoiserTooltipKey(cfg.denoiserMode))
                    .append(unavailableSuffix(denoiserOk(cfg.denoiserMode)))
                    .append(pendingRestartDenoiserSuffix()),
            ),
        ).build()

        displayedFrameGenMode = cfg.frameGenMode
        val frameGenButton = Button.builder(frameGenLabel(cfg.frameGenMode)) {
            minecraft.gui.setScreen(ShaderFrameGenScreen(this, options))
        }.tooltip(
            Tooltip.create(Component.translatable(frameGenTooltipKey(cfg.frameGenMode))),
        ).build()
        this.frameGenButton = frameGenButton

        val shadowsActive = cfg.renderTier.tierIndex <= 1
        val shadowResTooltip = Component.translatable("vulkanshaders.screen.shadow_res.tooltip")
        val shadowDistTooltip = Component.translatable("vulkanshaders.screen.shadow_dist.tooltip")
        if (!shadowsActive) {
            val rtNote = Component.translatable("vulkanshaders.screen.shadows.disabled_in_rt")
                .withStyle(ChatFormatting.GRAY)
            shadowResTooltip.append(Component.literal("\n")).append(rtNote)
            shadowDistTooltip.append(Component.literal("\n")).append(rtNote.copy())
        }

        val shadowResButton = Button.builder(
            Component.translatable("vulkanshaders.screen.shadow_res")
                .append(": ${cfg.shadowResolution}"),
        ) {
            val idx = shadowResOptions.indexOf(cfg.shadowResolution)
            val next = shadowResOptions[(idx + 1) % shadowResOptions.size]
            calibrationFeedback = null
            VulkanShaders.updateConfig(VulkanShaders.config.copy(shadowResolution = next))
            refreshWidgets()
        }.tooltip(Tooltip.create(shadowResTooltip)).build()
        shadowResButton.active = shadowsActive

        val shadowDistButton = Button.builder(
            Component.translatable("vulkanshaders.screen.shadow_dist")
                .append(": ")
                .append(Component.translatable("vulkanshaders.screen.shadow_dist.value", cfg.shadowDistanceChunks)),
        ) {
            val idx = shadowDistOptions.indexOf(cfg.shadowDistanceChunks)
            val next = shadowDistOptions[(idx + 1) % shadowDistOptions.size]
            calibrationFeedback = null
            VulkanShaders.updateConfig(VulkanShaders.config.copy(shadowDistanceChunks = next))
            refreshWidgets()
        }.tooltip(Tooltip.create(shadowDistTooltip)).build()
        shadowDistButton.active = shadowsActive

        val hdrButton = Button.builder(
            Component.translatable("vulkanshaders.screen.hdr")
                .append(": ")
                .append(if (cfg.hdrEnabled) CommonComponents.OPTION_ON else CommonComponents.OPTION_OFF),
        ) {
            calibrationFeedback = null
            VulkanShaders.updateConfig(VulkanShaders.config.copy(hdrEnabled = !cfg.hdrEnabled))
            refreshWidgets()
        }.tooltip(
            Tooltip.create(Component.translatable("vulkanshaders.screen.hdr.tooltip")),
        ).build()

        val overlayDepthActive = cfg.renderTier == LightingMode.PATH
        val overlayDepthTooltip = Component.translatable("vulkanshaders.screen.overlay_depth.tooltip")
        if (!overlayDepthActive) {
            overlayDepthTooltip.append(Component.literal("\n")).append(
                Component.translatable("vulkanshaders.screen.overlay_depth.requires_pt")
                    .withStyle(ChatFormatting.GRAY),
            )
        }
        val overlayDepthButton = Button.builder(
            Component.translatable("vulkanshaders.screen.overlay_depth")
                .append(": ")
                .append(if (cfg.enableOverlayPass) CommonComponents.OPTION_ON else CommonComponents.OPTION_OFF),
        ) {
            calibrationFeedback = null
            VulkanShaders.updateConfig(VulkanShaders.config.copy(enableOverlayPass = !cfg.enableOverlayPass))
            refreshWidgets()
        }.tooltip(Tooltip.create(overlayDepthTooltip)).build()
        overlayDepthButton.active = overlayDepthActive

        val f3DebugButton = Button.builder(
            Component.translatable("vulkanshaders.screen.f3_debug")
                .append(": ")
                .append(if (cfg.showF3DebugInfo) CommonComponents.OPTION_ON else CommonComponents.OPTION_OFF),
        ) {
            calibrationFeedback = null
            VulkanShaders.updateConfig(VulkanShaders.config.copy(showF3DebugInfo = !cfg.showF3DebugInfo))
            refreshWidgets()
        }.tooltip(
            Tooltip.create(Component.translatable("vulkanshaders.screen.f3_debug.tooltip")),
        ).build()

        val gpuProfilerButton = Button.builder(
            Component.translatable("vulkanshaders.screen.gpu_profiler")
                .append(": ")
                .append(if (cfg.enableGpuProfiler) CommonComponents.OPTION_ON else CommonComponents.OPTION_OFF),
        ) {
            calibrationFeedback = null
            VulkanShaders.updateConfig(VulkanShaders.config.copy(enableGpuProfiler = !cfg.enableGpuProfiler))
            refreshWidgets()
        }.tooltip(
            Tooltip.create(Component.translatable("vulkanshaders.screen.gpu_profiler.tooltip")),
        ).build()

        val librariesLabel = if (ProprietaryLibraryManager.restartRequired) {
            Component.translatable("vulkanshaders.libraries.button")
                .append(Component.literal(" — "))
                .append(
                    Component.translatable("vulkanshaders.libraries.status.restart")
                        .withStyle(ChatFormatting.GOLD),
                )
        } else {
            Component.translatable("vulkanshaders.libraries.button")
        }
        val librariesButton = Button.builder(librariesLabel) {
            minecraft.gui.setScreen(ShaderLibrariesScreen(this, options))
        }.tooltip(
            Tooltip.create(Component.translatable("vulkanshaders.libraries.button.tooltip")),
        ).build()

        calibrationFeedback?.let { list.addHeader(it) }
        if (ProprietaryLibraryManager.restartRequired) {
            list.addHeader(
                Component.translatable("vulkanshaders.libraries.restart_banner")
                    .withStyle(ChatFormatting.GOLD),
            )
        }

        // Section 1: Pipeline & Ray Tracing
        list.addHeader(
            Component.translatable("vulkanshaders.screen.graphics.section.pipeline")
                .withStyle(ChatFormatting.GOLD),
        )
        list.addBig(tierButton)
        list.addSmall(hdrButton, overlayDepthButton)

        // Section 2: Upscaling & Frame Generation
        list.addHeader(
            Component.translatable("vulkanshaders.screen.graphics.section.upscaling")
                .withStyle(ChatFormatting.GOLD),
        )
        list.addSmall(upscalerButton, denoiserButton)
        list.addBig(upscalerQualityButton)
        list.addBig(frameGenButton)

        // Section 3: Shadows (Tier 0-1)
        list.addHeader(
            Component.translatable("vulkanshaders.screen.graphics.section.shadows")
                .withStyle(ChatFormatting.GOLD),
        )
        list.addSmall(shadowResButton, shadowDistButton)

        // Section 4: Diagnostics & Libraries
        list.addHeader(
            Component.translatable("vulkanshaders.screen.graphics.section.diagnostics")
                .withStyle(ChatFormatting.GOLD),
        )
        list.addSmall(f3DebugButton, gpuProfilerButton)
        list.addBig(librariesButton)

        lastLibraryStateVersion = ProprietaryLibraryManager.stateVersion
    }

    override fun addFooter() {
        val macOsMoltenVk = isMacOsOrMoltenVk()
        val footer = LinearLayout.horizontal().spacing(10)
        footer.addChild(
            Button.builder(Component.translatable("vulkanshaders.screen.graphics.auto_calibrate")) {
                val result = HardwareAutoCalibrator.calibrateCurrent(
                    currentConfig = VulkanShaders.config,
                    manifest = ShaderPackManager.activePack?.manifest,
                )
                val calibratedConfig = if (macOsMoltenVk && result.config.renderTier.tierIndex >= 2) {
                    result.config.copy(renderTier = LightingMode.COMPUTE)
                } else {
                    result.config
                }
                if (VulkanShaders.updateConfig(calibratedConfig)) {
                    resetTemporalBuffers()
                    calibrationFeedback = Component.translatable(
                        "vulkanshaders.screen.graphics.auto_calibrate.done",
                        result.summary,
                    ).withStyle(ChatFormatting.GREEN)
                }
                refreshWidgets()
            }.width(150).tooltip(buildAutoCalibrateTooltip()).build(),
        )
        footer.addChild(
            Button.builder(CommonComponents.GUI_DONE) { onClose() }
                .width(150)
                .build(),
        )
        layout.addToFooter(footer)
    }

    private fun pendingRestartUpscalerSuffix(): Component {
        val pending = UpscalerMode.entries.any { ProprietaryLibraryManager.isPendingRestartForUpscaler(it) }
        return if (pending) {
            Component.literal("\n").append(
                Component.translatable("vulkanshaders.libraries.restart_required_short")
                    .withStyle(ChatFormatting.GOLD),
            )
        } else {
            Component.empty()
        }
    }

    private fun pendingRestartDenoiserSuffix(): Component {
        val pending = DenoiserMode.entries.any { ProprietaryLibraryManager.isPendingRestartForDenoiser(it) }
        return if (pending) {
            Component.literal("\n").append(
                Component.translatable("vulkanshaders.libraries.restart_required_short")
                    .withStyle(ChatFormatting.GOLD),
            )
        } else {
            Component.empty()
        }
    }

    private fun buildAutoCalibrateTooltip(): Tooltip {
        val base = Component.translatable("vulkanshaders.screen.graphics.auto_calibrate.tooltip")
        val feedback = calibrationFeedback
        return if (feedback != null) {
            Tooltip.create(Component.empty().append(feedback).append(Component.literal("\n")).append(base))
        } else {
            Tooltip.create(base)
        }
    }
}
