package dev.vulkanshaders.gui

import dev.vulkanshaders.gui.ShaderPacksScreen.Companion.text
import dev.vulkanshaders.pack.ShaderPackManager
import net.minecraft.ChatFormatting
import net.minecraft.client.Options
import net.minecraft.client.gui.components.Button
import net.minecraft.client.gui.components.EditBox
import net.minecraft.client.gui.components.StringWidget
import net.minecraft.client.gui.components.Tooltip
import net.minecraft.client.gui.layouts.LinearLayout
import net.minecraft.client.gui.screens.options.OptionsSubScreen
import net.minecraft.network.chat.CommonComponents
import net.minecraft.network.chat.Component
import java.nio.file.FileSystems
import java.nio.file.Files
import java.nio.file.InvalidPathException
import java.nio.file.Path

internal class ShaderPackImportScreen(
    private val parent: ShaderPacksScreen,
    options: Options,
) : OptionsSubScreen(parent, options, text("import.title")) {
    private val home = Path.of(System.getProperty("user.home"))
    private var directory: Path? = home.resolve("Downloads").takeIf(Files::isDirectory) ?: home
    private var invalidPath: String? = null
    private var importErrorBanner: Component? = null

    override fun init() {
        layout.removeChildren()
        layout.setHeaderHeight(60)
        super.init()
    }

    override fun addTitle() {
        val header = layout.addToHeader(LinearLayout.vertical().spacing(5))
        header.defaultCellSetting().alignHorizontallyCenter()
        header.addChild(StringWidget(title, font))
        val location = header.addChild(LinearLayout.horizontal().spacing(5))
        val field = location.addChild(EditBox(font, 230, 20, text("import.path")))
        field.setMaxLength(4096)
        field.setHint(text("import.path"))
        field.value = invalidPath ?: directory?.toString().orEmpty()
        location.addChild(Button.builder(text("import.open")) {
            val value = field.value.trim().removeSurrounding("\"")
            val file = try {
                (directory ?: home).resolve(value).normalize()
            } catch (_: InvalidPathException) {
                null
            }
            when {
                value.isBlank() || file == null -> {
                    invalidPath = field.value
                    importErrorBanner = null
                    rebuildWidgets()
                }
                Files.isDirectory(file) -> browse(file)
                else -> validateAndForwardImport(listOf(file))
            }
        }.width(75).build())
    }

    override fun addOptions() {
        val list = this.list ?: return
        list.addHeader(text("import.security_hint").withStyle(ChatFormatting.GRAY))
        if (invalidPath != null) list.addHeader(text("import.invalid_path").withStyle(ChatFormatting.RED))
        importErrorBanner?.let { list.addHeader(it) }

        val folder = directory
        val entries = runCatching {
            if (folder == null) {
                FileSystems.getDefault().rootDirectories.toList()
            } else {
                Files.list(folder).use { stream ->
                    stream.filter { path ->
                        !path.fileName.toString().startsWith(".") &&
                            (Files.isDirectory(path) || ShaderPackManager.isImportableArchive(path))
                    }.sorted(compareBy<Path>({ !Files.isDirectory(it) }, { it.fileName.toString().lowercase() }))
                        .toList()
                }
            }
        }.getOrElse {
            list.addHeader(text("import.cannot_read"))
            return
        }
        if (entries.isEmpty()) list.addHeader(text("import.empty_folder"))

        for (path in entries) {
            val isDirectory = Files.isDirectory(path)
            val name = path.fileName?.toString() ?: path.toString()
            val label = if (isDirectory && path.fileName != null) {
                text("import.folder_entry", name)
            } else {
                Component.literal(name)
            }
            list.addBig(Button.builder(label) {
                if (isDirectory) browse(path) else validateAndForwardImport(listOf(path))
            }.tooltip(Tooltip.create(Component.literal(path.toString()))).build())
        }
    }

    override fun addFooter() {
        val footer = layout.addToFooter(LinearLayout.horizontal().spacing(5))
        footer.addChild(Button.builder(text("import.up")) { browse(directory?.parent) }
            .width(100).build().apply { active = directory != null })
        footer.addChild(Button.builder(text("import.home")) { browse(home) }.width(100).build())
        footer.addChild(Button.builder(CommonComponents.GUI_BACK) { onClose() }.width(100).build())
    }

    override fun onFilesDrop(files: List<Path>) {
        validateAndForwardImport(files)
    }

    private fun validateAndForwardImport(files: List<Path>) {
        importErrorBanner = null
        val invalid = files.firstOrNull { !ShaderPackManager.isImportableArchive(it) }
        if (invalid != null) {
            importErrorBanner = text(
                "import.file_error",
                invalid.fileName?.toString() ?: invalid.toString(),
                text("import.format"),
            ).withStyle(ChatFormatting.RED)
            rebuildWidgets()
            return
        }
        parent.importArchives(files)
    }

    private fun browse(path: Path?) {
        directory = path
        invalidPath = null
        importErrorBanner = null
        rebuildWidgets()
    }

    companion object {
        /**
         * Formats a user-facing localized error component for any exception thrown by
         * [ShaderPackManager.importArchive].
         */
        internal fun describeImportError(error: Throwable): Component = when (error) {
            is ShaderPackManager.DuplicatePackException -> text("import.duplicate")
            is ShaderPackManager.UnsafeArchiveException -> text("import.unsafe_archive")
            is ShaderPackManager.InvalidManifestException ->
                text("import.invalid_manifest", error.message ?: "pack.toml")
            is ShaderPackManager.ArchiveFormatException -> text("import.format")
            else -> Component.literal(error.message ?: error.javaClass.simpleName)
        }
    }
}
