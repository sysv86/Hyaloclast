package dev.vulkanshaders.gui

import com.mojang.blaze3d.Blaze3D
import dev.vulkanshaders.bridge.StandaloneRenderPearlBridge
import dev.vulkanshaders.diagnostics.HyaloclastErrorLog
import net.minecraft.ChatFormatting
import net.minecraft.client.Minecraft
import net.minecraft.client.Options
import net.minecraft.client.gui.GuiGraphicsExtractor
import net.minecraft.client.gui.components.AbstractWidget
import net.minecraft.client.gui.components.Button
import net.minecraft.client.gui.layouts.LinearLayout
import net.minecraft.client.gui.narration.NarrationElementOutput
import net.minecraft.client.gui.screens.Screen
import net.minecraft.client.gui.screens.TitleScreen
import net.minecraft.client.gui.screens.options.OptionsSubScreen
import net.minecraft.network.chat.Component
import net.minecraft.util.FormattedCharSequence
import java.nio.file.Files

/**
 * Vanilla-styled `OptionsSubScreen` displaying the Hyaloclast session error log and structured
 * `.vsl` / GLSL shader compiler diagnostics (exact file and line number).
 *
 * When Pyroclast is loaded and no Hyaloclast-specific shader compile diagnostic is active,
 * delegates to Pyroclast's unified `VulkanErrorLogScreen` while preserving this class signature.
 */
class ShaderErrorLogScreen(
    private val returnScreen: Screen?,
    options: Options,
) : OptionsSubScreen(returnScreen ?: TitleScreen(), options, Component.translatable("vulkanshaders.screen.error_log.title")) {
    private var copyButton: Button? = null
    private var copiedResetTicks: Int = 0

    override fun init() {
        if (StandaloneRenderPearlBridge.vulkanTurboLoaded && HyaloclastErrorLog.latestShaderDiagnostic == null) {
            val delegated = dev.vulkanturbo.api.shader.TurboShadersBridge.createLogScreen(returnScreen, options)
            if (delegated != null && delegated !is ShaderErrorLogScreen) {
                minecraft.gui.setScreen(delegated)
                return
            }
        }
        HyaloclastErrorLog.markAllRead()
        layout.removeChildren()
        super.init()
    }

    override fun addOptions() {
        val list = this.list ?: return
        val maxTextWidth = LOG_WIDGET_WIDTH - 8

        HyaloclastErrorLog.latestShaderDiagnostic?.let { diag ->
            list.addHeader(
                Component.translatable("vulkanshaders.screen.error_log.shader_location", diag.formatLocation())
                    .withStyle(ChatFormatting.RED),
            )
            val msgLines = font.split(Component.literal(diag.message), maxTextWidth)
            msgLines.forEachIndexed { idx, segment ->
                list.addBig(LogLineWidget(segment, COLOR_DIAGNOSTIC, idx))
            }
        }

        val fullLog = HyaloclastErrorLog.formatFullLogText()
        var currentColor = COLOR_NORMAL
        var rowIndex = 0
        for (rawLine in fullLog.lineSequence()) {
            val lineColor = classifyLineColor(rawLine, currentColor)
            currentColor = lineColor

            val wrapped: List<FormattedCharSequence> = if (rawLine.isEmpty()) {
                listOf(FormattedCharSequence.EMPTY)
            } else {
                font.split(Component.literal(rawLine), maxTextWidth)
            }
            for (segment in wrapped) {
                list.addBig(LogLineWidget(segment, lineColor, rowIndex++))
            }
        }
    }

    override fun addFooter() {
        // Keep the footer strictly aligned to the 310px column (78 + 4 + 78 + 4 + 70 + 4 + 72).
        val footer = LinearLayout.horizontal().spacing(4)

        val copyBtn = Button.builder(Component.translatable("vulkanshaders.screen.error_log.copy")) {
            val text = HyaloclastErrorLog.formatFullLogText()
            runCatching {
                minecraft.keyboardHandler.clipboard = text
            }
            copyButton?.message = Component.translatable("vulkanshaders.screen.error_log.copied")
            copiedResetTicks = 40
        }.width(78).build()
        copyButton = copyBtn
        footer.addChild(copyBtn)

        footer.addChild(
            Button.builder(Component.translatable("vulkanshaders.screen.error_log.open_folder")) {
                runCatching {
                    val dir = HyaloclastErrorLog.logsDir()
                    Files.createDirectories(dir)
                    Blaze3D.openPath(dir)
                }
            }.width(78).build(),
        )

        footer.addChild(
            Button.builder(Component.translatable("vulkanshaders.screen.error_log.clear")) {
                HyaloclastErrorLog.clear()
                rebuildWidgets()
            }.width(70).build(),
        )

        footer.addChild(
            Button.builder(Component.translatable("vulkanshaders.screen.error_log.close")) {
                onClose()
            }.width(72).build(),
        )

        layout.addToFooter(footer)
    }

    override fun onClose() {
        minecraft.gui.setScreen(returnScreen)
    }

    override fun tick() {
        super.tick()
        if (copiedResetTicks > 0) {
            copiedResetTicks--
            if (copiedResetTicks == 0) {
                copyButton?.message = Component.translatable("vulkanshaders.screen.error_log.copy")
            }
        }
    }

    private fun classifyLineColor(rawLine: String, previousColor: Int): Int = when {
        rawLine.contains("[ShaderCompile]") || rawLine.contains("[DeviceLost]") -> COLOR_DIAGNOSTIC
        rawLine.contains("[ERROR]") -> COLOR_ERROR
        rawLine.contains("[WARN]") -> COLOR_WARN
        rawLine.contains("[INFO]") -> COLOR_NORMAL
        rawLine.startsWith("    ") -> if (previousColor == COLOR_ERROR || previousColor == COLOR_DIAGNOSTIC) COLOR_ERROR_DETAIL else COLOR_DETAIL
        rawLine.startsWith("===") || rawLine.startsWith("--") -> COLOR_HEADER
        else -> COLOR_NORMAL
    }

    private class LogLineWidget(
        private val line: FormattedCharSequence,
        private val textColor: Int,
        private val rowIndex: Int,
    ) : AbstractWidget(0, 0, LOG_WIDGET_WIDTH, LOG_LINE_HEIGHT, Component.empty()) {
        init {
            this.active = false
        }

        override fun extractWidgetRenderState(
            guiGraphics: GuiGraphicsExtractor,
            mouseX: Int,
            mouseY: Int,
            partialTick: Float,
        ) {
            val bg = if ((rowIndex and 1) == 0) COLOR_ROW_EVEN else COLOR_ROW_ODD
            guiGraphics.fill(this.x, this.y, this.x + this.width, this.y + this.height, bg)
            val font = Minecraft.getInstance().font
            val textY = this.y + ((this.height - font.lineHeight) / 2).coerceAtLeast(1)
            guiGraphics.text(font, line, this.x + 4, textY, textColor)
        }

        override fun updateWidgetNarration(output: NarrationElementOutput) {
            defaultButtonNarrationText(output)
        }
    }

    companion object {
        private const val LOG_WIDGET_WIDTH: Int = 310
        private const val LOG_LINE_HEIGHT: Int = 13
        private const val COLOR_DIAGNOSTIC: Int = 0xFFFF8855.toInt()
        private const val COLOR_ERROR: Int = 0xFFFF5555.toInt()
        private const val COLOR_ERROR_DETAIL: Int = 0xFFE8A0A0.toInt()
        private const val COLOR_WARN: Int = 0xFFFFFF55.toInt()
        private const val COLOR_HEADER: Int = 0xFFCCCCCC.toInt()
        private const val COLOR_NORMAL: Int = 0xFFE6E6E6.toInt()
        private const val COLOR_DETAIL: Int = 0xFFB8B8B8.toInt()
        private const val COLOR_ROW_EVEN: Int = 0x55101014
        private const val COLOR_ROW_ODD: Int = 0x3318181E
    }
}
