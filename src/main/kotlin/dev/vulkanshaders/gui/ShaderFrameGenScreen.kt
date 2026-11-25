package dev.vulkanshaders.gui

import dev.vulkanshaders.VulkanShaders
import dev.vulkanshaders.backend.NativeCoreBridge
import dev.vulkanshaders.config.FrameGenMode
import dev.vulkanshaders.gui.ShaderPacksScreen.Companion.text
import dev.vulkanshaders.sdk.ProprietaryLibraryManager
import dev.vulkanshaders.state.WorldStateCollector
import net.minecraft.ChatFormatting
import net.minecraft.client.Options
import net.minecraft.client.gui.components.Button
import net.minecraft.client.gui.components.Tooltip
import net.minecraft.client.gui.screens.Screen
import net.minecraft.client.gui.screens.options.OptionsSubScreen
import net.minecraft.network.chat.Component

internal class ShaderFrameGenScreen(
    parent: Screen,
    options: Options,
) : OptionsSubScreen(parent, options, text("framegen")) {
    private var displayedMode = VulkanShaders.config.frameGenMode

    override fun init() {
        layout.removeChildren()
        super.init()
    }

    override fun tick() {
        if (VulkanShaders.config.frameGenMode != displayedMode) {
            refreshWidgets()
        }
    }

    override fun addOptions() {
        val list = this.list ?: return
        displayedMode = VulkanShaders.config.frameGenMode
        for (mode in FrameGenMode.entries) {
            val name = mode.frameGenLabel()
            val reason = unavailableReason(mode)
            val label = if (mode == displayedMode) text("active", name).withStyle(ChatFormatting.GREEN) else name
            val tooltip = Component.translatable(ShaderGraphicsScreen.frameGenTooltipKey(mode))
            reason?.let { tooltip.append("\n").append(it.copy().withStyle(ChatFormatting.YELLOW)) }
            val button = Button.builder(label) { select(mode) }
                .tooltip(Tooltip.create(tooltip))
                .build()
            // Disable unavailable modes upfront instead of allowing clicks that fail with a toast.
            button.active = reason == null || mode == displayedMode
            list.addBig(button)
        }
    }

    private fun refreshWidgets() {
        val scroll = list?.scrollAmount() ?: 0.0
        rebuildWidgets()
        list?.setScrollAmount(scroll)
    }

    private fun select(mode: FrameGenMode) {
        if (mode == VulkanShaders.config.frameGenMode) return
        val reason = unavailableReason(mode)
        if (reason != null) {
            ShaderNotices.frameGeneration(text("framegen.unavailable"), reason)
            return
        }
        NativeCoreBridge.waitPresentIdle()
        if (!VulkanShaders.updateConfig(VulkanShaders.config.copy(frameGenMode = mode))) {
            ShaderNotices.frameGeneration(text("framegen.unavailable"), text("framegen.failed", mode.frameGenLabel()))
            return
        }
        WorldStateCollector.resetHistory()
        NativeCoreBridge.resetTemporalHistory()
        NativeCoreBridge.markHistoryReset()
        ShaderNotices.frameGeneration(
            text(if (mode == FrameGenMode.NONE) "framegen.disabled" else "framegen.enabled"),
            mode.frameGenLabel(),
        )
        refreshWidgets()
    }

    private fun unavailableReason(mode: FrameGenMode): Component? = when {
        mode == FrameGenMode.NONE -> null
        !VulkanShaders.shaderPackActive -> text("framegen.requires_pack")
        ShaderGraphicsScreen.isMacOsOrMoltenVk() &&
            (mode == FrameGenMode.DLSS_FG || mode == FrameGenMode.FSR3_FG) ->
            text("framegen.macos_unsupported", mode.frameGenLabel())
        !NativeCoreBridge.canEnableFrameGen(mode.ordinal) &&
            ProprietaryLibraryManager.isPendingRestartForFrameGen(mode) ->
            Component.translatable("vulkanshaders.libraries.restart_required_short")
        !NativeCoreBridge.canEnableFrameGen(mode.ordinal) -> text("framegen.unsupported", mode.frameGenLabel())
        else -> null
    }
}

internal fun FrameGenMode.frameGenLabel(): Component =
    text("framegen.mode.${name.lowercase()}")
