package dev.vulkanshaders.gui

import com.mojang.blaze3d.Blaze3D
import dev.vulkanshaders.VulkanShaders
import dev.vulkanshaders.backend.NativeCoreBridge
import dev.vulkanshaders.bridge.StandaloneRenderPearlBridge
import dev.vulkanshaders.diagnostics.HyaloclastErrorLog
import dev.vulkanshaders.pack.LoadedShaderPack
import dev.vulkanshaders.pack.PackLocalization
import dev.vulkanshaders.pack.ShaderPackManager
import net.minecraft.ChatFormatting
import net.minecraft.client.Options
import net.minecraft.client.gui.components.Button
import net.minecraft.client.gui.components.EditBox
import net.minecraft.client.gui.components.OptionsList
import net.minecraft.client.gui.components.StringWidget
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
    private var searchQuery: String = ""
    private var searchBox: EditBox? = null

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

    private fun unavailableReason(): Component = text(
        when {
            !StandaloneRenderPearlBridge.vulkanTurboLoaded -> "requires_pyroclast"
            StandaloneRenderPearlBridge.isVulkanBackendActive() -> "runtime_unavailable"
            else -> "requires_vulkan"
        },
    )

    private fun shouldShowSearch(): Boolean =
        ShaderPackManager.discoveredPacks.size > SEARCH_THRESHOLD || searchQuery.isNotEmpty()

    override fun init() {
        layout.removeChildren()
        layout.setHeaderHeight(if (shouldShowSearch()) HEADER_WITH_SEARCH_HEIGHT else HEADER_DEFAULT_HEIGHT)
        layout.setFooterHeight(FOOTER_HEIGHT)
        super.init()
    }

    override fun addTitle() {
        searchBox = null
        if (!shouldShowSearch()) {
            super.addTitle()
            return
        }

        val header = layout.addToHeader(LinearLayout.vertical().spacing(4))
        header.defaultCellSetting().alignHorizontallyCenter()
        header.addChild(StringWidget(title, font))

        val field = header.addChild(EditBox(font, COLUMN_WIDTH, 20, text("packs.search")))
        field.setMaxLength(128)
        field.setHint(text("packs.search"))
        field.value = searchQuery
        field.setResponder { newQuery ->
            if (newQuery != searchQuery) {
                searchQuery = newQuery
                refreshOptionsList(resetScroll = true)
            }
        }
        searchBox = field
    }

    override fun addOptions() {
        val list = this.list ?: return
        val cfg = VulkanShaders.config
        val canEnable = canEnableShaders()
        val active = ShaderPackManager.activePack.takeIf { VulkanShaders.shaderPackActive }

        if (!canEnable) {
            list.addHeader(unavailableReason().copy().withStyle(ChatFormatting.YELLOW))
        }
        importStatus?.let(list::addHeader)

        // Show a single deduplicated error/log entry rather than two identical red buttons.
        val hasPackError = !ShaderPackManager.lastPackError.isNullOrBlank()
        val errorCount = HyaloclastErrorLog.errorCount()
        if (hasPackError || errorCount > 0) {
            val errorLabel = when {
                hasPackError -> text("load_error")
                else -> Component.translatable("vulkanshaders.screen.packs.view_log_count", errorCount)
            }.withStyle(ChatFormatting.RED)
            list.addBig(
                Button.builder(errorLabel) {
                    minecraft.gui.setScreen(createLogScreen())
                }.build(),
            )
        }

        val query = searchQuery.trim()
        val off = !cfg.enabled || cfg.selectedPackId.equals("off", ignoreCase = true) || cfg.selectedPackId.isBlank()
        if (query.isEmpty()) {
            list.addBig(
                Button.builder(packLabel(text("pack.none"), off)) {
                    if (!off) {
                        VulkanShaders.updateConfig(VulkanShaders.config.copy(selectedPackId = "off"))
                        refreshWidgets()
                    }
                }.build(),
            )
        }

        var matchedAny = false
        for (pack in ShaderPackManager.discoveredPacks) {
            val manifest = localizedManifest(pack)
            if (query.isNotEmpty() &&
                !manifest.name.contains(query, ignoreCase = true) &&
                !manifest.id.contains(query, ignoreCase = true) &&
                !manifest.author.contains(query, ignoreCase = true)
            ) {
                continue
            }
            matchedAny = true
            val selected = active?.manifest?.id == manifest.id
            val tooltip = packDetails(pack, canEnable)
            list.addBig(
                Button.builder(packLabel(Component.literal(manifest.name), selected)) {
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
                }.tooltip(Tooltip.create(tooltip)).build().apply {
                    this.active = canEnable
                },
            )
        }

        if (query.isNotEmpty() && !matchedAny) {
            list.addBig(StringWidget(text("packs.no_matches").withStyle(ChatFormatting.GRAY), font))
        }
    }

    override fun addFooter() {
        val footer = LinearLayout.vertical().spacing(4)
        val files = footer.addChild(LinearLayout.horizontal().spacing(FOOTER_GAP))
        val actions = footer.addChild(LinearLayout.horizontal().spacing(FOOTER_GAP))

        // Align both footer rows to the exact same 3 x 100px grid (310px total).
        files.addChild(
            Button.builder(text("import")) { minecraft.gui.setScreen(ShaderPackImportScreen(this, options)) }
                .width(FOOTER_BUTTON_WIDTH)
                .tooltip(Tooltip.create(text("import.tooltip")))
                .build(),
        )
        files.addChild(
            Button.builder(text("open_folder")) {
                runCatching { Blaze3D.openPath(ShaderPackManager.shaderpacksDir()) }
                    .onFailure { showDetails(text("folder_error"), Component.literal(it.message ?: it.javaClass.simpleName)) }
            }.width(FOOTER_BUTTON_WIDTH)
                .tooltip(Tooltip.create(text("open_folder.tooltip")))
                .build(),
        )
        files.addChild(
            Button.builder(text("reload")) { reloadPacks() }
                .width(FOOTER_BUTTON_WIDTH)
                .tooltip(Tooltip.create(text("reload.tooltip")))
                .build(),
        )

        val pack = ShaderPackManager.activePack.takeIf { VulkanShaders.shaderPackActive }
        val hasPackSettings = pack != null && pack.manifest.settings.isNotEmpty()
        val packSettingsButton = Button.builder(text("pack_settings")) {
            if (pack != null) minecraft.gui.setScreen(PackSettingsScreen(this, options, pack))
        }.width(FOOTER_BUTTON_WIDTH).apply {
            if (!hasPackSettings) {
                tooltip(Tooltip.create(text("pack_settings.disabled_tooltip")))
            }
        }.build().apply { active = hasPackSettings }
        actions.addChild(packSettingsButton)

        val canEnable = canEnableShaders()
        val graphicsButton = Button.builder(text("graphics")) {
            minecraft.gui.setScreen(ShaderGraphicsScreen(this, options))
        }.width(FOOTER_BUTTON_WIDTH).apply {
            if (!canEnable) {
                tooltip(Tooltip.create(unavailableReason()))
            }
        }.build().apply { active = canEnable }
        actions.addChild(graphicsButton)

        actions.addChild(
            Button.builder(CommonComponents.GUI_DONE) { onClose() }
                .width(FOOTER_BUTTON_WIDTH)
                .build(),
        )

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

    private fun refreshOptionsList(resetScroll: Boolean = false) {
        val listWidget = this.list ?: return
        val scroll = if (resetScroll) 0.0 else listWidget.scrollAmount()
        clearOptionsListEntries(listWidget)
        addOptions()
        listWidget.setScrollAmount(scroll)
    }

    private fun clearOptionsListEntries(listWidget: OptionsList) {
        val cleared = runCatching {
            var cls: Class<*>? = listWidget.javaClass
            var method: java.lang.reflect.Method? = null
            while (cls != null && method == null) {
                method = cls.declaredMethods.firstOrNull { it.name == "clearEntries" && it.parameterCount == 0 }
                cls = cls.superclass
            }
            if (method != null) {
                method.isAccessible = true
                method.invoke(listWidget)
                true
            } else {
                false
            }
        }.getOrDefault(false)
        if (!cleared) {
            runCatching { listWidget.children().clear() }
        }
    }

    private fun refreshWidgets() {
        val scroll = list?.scrollAmount() ?: 0.0
        val wasSearchFocused = searchBox?.isFocused == true
        rebuildWidgets()
        list?.setScrollAmount(scroll)
        if (wasSearchFocused) {
            searchBox?.let { box ->
                this.focused = box
                box.isFocused = true
            }
        }
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

    private fun localizedManifest(pack: LoadedShaderPack) = PackLocalization.localizePack(
        pack,
        minecraft.languageManager.selected,
    )

    private fun packDetails(pack: LoadedShaderPack, canEnable: Boolean): Component {
        val manifest = localizedManifest(pack)
        val details = text("pack.info", manifest.version, manifest.author)
        if (pack.isBuiltin) details.append("\n").append(text("pack.builtin"))
        if (manifest.description.isNotBlank()) details.append("\n\n").append(manifest.description)
        if (!canEnable) {
            details.append("\n\n").append(unavailableReason().copy().withStyle(ChatFormatting.YELLOW))
        }
        return details
    }

    companion object {
        private const val COLUMN_WIDTH: Int = 310
        private const val HEADER_DEFAULT_HEIGHT: Int = 33
        private const val HEADER_WITH_SEARCH_HEIGHT: Int = 56
        private const val FOOTER_HEIGHT: Int = 56
        private const val FOOTER_BUTTON_WIDTH: Int = 100
        private const val FOOTER_GAP: Int = 5
        private const val SEARCH_THRESHOLD: Int = 4

        internal fun text(key: String, vararg args: Any) =
            Component.translatable("vulkanshaders.screen.$key", *args)
    }
}
