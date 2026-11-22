package dev.vulkanshaders.gui

import com.mojang.blaze3d.Blaze3D
import dev.vulkanshaders.VulkanShaders
import dev.vulkanshaders.backend.NativeCoreBridge
import dev.vulkanshaders.bridge.StandaloneRenderPearlBridge
import dev.vulkanshaders.diagnostics.HyaloclastErrorLog
import dev.vulkanshaders.pack.LoadedShaderPack
import dev.vulkanshaders.pack.ShaderPackManager
import net.minecraft.ChatFormatting
import net.minecraft.client.Options
import net.minecraft.client.gui.components.Button
import net.minecraft.client.gui.components.Tooltip
import net.minecraft.client.gui.layouts.LinearLayout
import net.minecraft.client.gui.screens.AlertScreen
import net.minecraft.client.gui.screens.Screen
import net.minecraft.client.gui.screens.TitleScreen
import net.minecraft.client.gui.screens.options.OptionsSubScreen
import net.minecraft.network.chat.CommonComponents
import net.minecraft.network.chat.Component
import java.nio.file.Path

class ShaderPacksScreen(
    private val returnScreen: Screen?,
    options: Options,
) : OptionsSubScreen(returnScreen ?: TitleScreen(), options, text("title")) {
    override fun onClose() {
        minecraft.gui.setScreen(returnScreen)
    }

    private var importStatus: Component? = null

    private fun createLogScreen(): Screen =
        if (StandaloneRenderPearlBridge.vulkanTurboLoaded) {
            dev.vulkanturbo.api.shader.TurboShadersBridge.createLogScreen(this, options)
                ?: ShaderErrorLogScreen(this, options)
        } else {
            ShaderErrorLogScreen(this, options)
        }

    private fun canEnableShaders(): Boolean =
        StandaloneRenderPearlBridge.vulkanTurboLoaded &&
            StandaloneRenderPearlBridge.isBackendAllowed() &&
            StandaloneRenderPearlBridge.isVulkanBackendActive() &&
            NativeCoreBridge.initialized

    override fun init() {
        layout.removeChildren()
        layout.setFooterHeight(56)
        super.init()
    }

    override fun addOptions() {
        val list = this.list ?: return
        val cfg = VulkanShaders.config
        val canEnable = canEnableShaders()
        val active = ShaderPackManager.activePack.takeIf { VulkanShaders.shaderPackActive }

        if (!canEnable) {
            list.addHeader(
                text(
                    when {
                        !StandaloneRenderPearlBridge.vulkanTurboLoaded -> "requires_pyroclast"
                        StandaloneRenderPearlBridge.isVulkanBackendActive() -> "runtime_unavailable"
                        else -> "requires_vulkan"
                    },
                ).withStyle(ChatFormatting.YELLOW),
            )
        }
        importStatus?.let(list::addHeader)
        ShaderPackManager.lastPackError?.takeIf { it.isNotBlank() }?.let {
            list.addBig(Button.builder(text("load_error").withStyle(ChatFormatting.RED)) {
                minecraft.gui.setScreen(createLogScreen())
            }.build())
        }
        if (HyaloclastErrorLog.errorCount() > 0) {
            list.addBig(
                Button.builder(
                    Component.translatable("vulkanshaders.screen.packs.view_log").withStyle(ChatFormatting.RED),
                ) {
                    minecraft.gui.setScreen(createLogScreen())
                }.build(),
            )
        }

        val off = !cfg.enabled || cfg.selectedPackId.equals("off", ignoreCase = true) || cfg.selectedPackId.isBlank()
        list.addBig(Button.builder(packLabel(text("pack.none"), off)) {
            if (!off) {
                VulkanShaders.updateConfig(VulkanShaders.config.copy(selectedPackId = "off"))
                refreshWidgets()
            }
        }.build())

        for (pack in ShaderPackManager.discoveredPacks) {
            val selected = active?.manifest?.id == pack.manifest.id
            list.addBig(
                Button.builder(packLabel(Component.literal(pack.manifest.name), selected)) {
                    if (!selected) {
                        NativeCoreBridge.startPackPrewarm(pack.rootPath)
                        val currentConfig = VulkanShaders.config
                        if (currentConfig.selectedPackId == pack.manifest.id && currentConfig.enabled) {
                            ShaderPackManager.selectPack(pack.manifest.id)
                            VulkanShaders.refreshRuntimeState()
                        } else {
                            VulkanShaders.updateConfig(currentConfig.copy(enabled = true, selectedPackId = pack.manifest.id))
                        }
                        refreshWidgets()
                    }
                }.tooltip(Tooltip.create(packDetails(pack))).build().apply {
                    this.active = canEnable
                },
            )
        }
    }

    override fun addFooter() {
        val footer = LinearLayout.vertical().spacing(4)
        val files = footer.addChild(LinearLayout.horizontal().spacing(5))
        val settings = footer.addChild(LinearLayout.horizontal().spacing(5))

        files.addChild(
            Button.builder(text("import")) { minecraft.gui.setScreen(ShaderPackImportScreen(this, options)) }
                .width(100).tooltip(Tooltip.create(text("import.tooltip"))).build(),
        )
        files.addChild(Button.builder(text("open_folder")) {
            runCatching { Blaze3D.openPath(ShaderPackManager.shaderpacksDir()) }
                .onFailure { showDetails(text("folder_error"), Component.literal(it.message ?: it.javaClass.simpleName)) }
        }.width(100).build())
        files.addChild(Button.builder(text("reload")) { reloadPacks() }.width(100).build())

        val pack = ShaderPackManager.activePack.takeIf { VulkanShaders.shaderPackActive }
        settings.addChild(Button.builder(text("pack_settings")) {
            if (pack != null) minecraft.gui.setScreen(PackSettingsScreen(this, options, pack))
        }.width(110).build().apply { active = pack != null && pack.manifest.settings.isNotEmpty() })
        settings.addChild(Button.builder(text("graphics")) {
            minecraft.gui.setScreen(ShaderGraphicsScreen(this, options))
        }.width(90).build().apply { active = canEnableShaders() })
        settings.addChild(Button.builder(CommonComponents.GUI_DONE) { onClose() }.width(100).build())

        layout.addToFooter(footer)
    }

    override fun onFilesDrop(files: List<Path>) {
        importArchives(files)
    }

    internal fun importArchives(files: List<Path>) {
        if (files.isEmpty()) return
        val sourceScreen = minecraft.gui.screen() ?: this
        val imported = mutableListOf<LoadedShaderPack>()
        val errors = mutableListOf<Component>()
        runCatching { ShaderPackManager.reloadAll() }.onFailure {
            showDetails(text("import.error"), Component.literal(it.message ?: it.javaClass.simpleName), sourceScreen)
            return
        }
        for (file in files) {
            runCatching { ShaderPackManager.importArchive(file) }
                .onSuccess { imported.add(it) }
                .onFailure { error ->
                    val reason = ShaderPackImportScreen.describeImportError(error)
                    errors.add(text("import.file_error", file.fileName ?: file, reason))
                }
        }
        if (imported.isNotEmpty()) {
            importStatus = if (imported.size == 1) {
                text("import.success", imported.single().manifest.name.take(45))
            } else {
                text("import.success_many", imported.size)
            }
            runCatching { ShaderPackManager.reloadAll() }.onFailure {
                errors.add(Component.literal(it.message ?: it.javaClass.simpleName))
            }
            if (sourceScreen !== this) minecraft.gui.setScreen(this)
            refreshWidgets()
        }
        if (errors.isNotEmpty()) {
            val details = Component.empty()
            errors.take(5).forEachIndexed { index, error ->
                if (index > 0) details.append("\n")
                details.append(error)
            }
            if (errors.size > 5) details.append("\n").append(text("import.more_errors", errors.size - 5))
            showDetails(text("import.error"), details, if (imported.isEmpty()) sourceScreen else this)
        }
    }

    private fun reloadPacks() {
        importStatus = null
        val result = ShaderPackManager.reloadActivePackInWorld()
        if (!result.success) {
            val details = result.diagnostic?.formatShort()
                ?: result.errorSummary
                ?: "Failed to reload shader pack."
            showDetails(text("reload_error"), Component.literal(details))
        }
        refreshWidgets()
    }

    private fun refreshWidgets() {
        val scroll = list?.scrollAmount() ?: 0.0
        rebuildWidgets()
        list?.setScrollAmount(scroll)
    }

    private fun showDetails(title: Component, message: Component, returnTo: Screen = this) {
        minecraft.gui.setScreen(object : AlertScreen(
            { minecraft.gui.setScreen(returnTo) }, title, message,
        ) {
            override fun onClose() {
                minecraft.gui.setScreen(returnTo)
            }
        })
    }

    private fun packLabel(name: Component, active: Boolean): Component =
        if (active) text("active", name).withStyle(ChatFormatting.GREEN) else name

    private fun packDetails(pack: LoadedShaderPack): Component {
        val manifest = pack.manifest
        val details = text("pack.info", manifest.version, manifest.author)
        if (pack.isBuiltin) details.append("\n").append(text("pack.builtin"))
        if (manifest.description.isNotBlank()) details.append("\n\n").append(manifest.description)
        return details
    }

    companion object {
        internal fun text(key: String, vararg args: Any) =
            Component.translatable("vulkanshaders.screen.$key", *args)
    }
}
