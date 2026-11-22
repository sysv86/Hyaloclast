package dev.vulkanshaders.gui

import com.mojang.blaze3d.Blaze3D
import dev.vulkanshaders.sdk.AcquisitionPolicy
import dev.vulkanshaders.sdk.LibraryInstallState
import dev.vulkanshaders.sdk.LibraryStatusEntry
import dev.vulkanshaders.sdk.ProprietaryLibraryManager
import net.minecraft.ChatFormatting
import net.minecraft.client.Minecraft
import net.minecraft.client.Options
import net.minecraft.client.gui.components.Button
import net.minecraft.client.gui.components.Tooltip
import net.minecraft.client.gui.layouts.LinearLayout
import net.minecraft.client.gui.screens.Screen
import net.minecraft.client.gui.screens.options.OptionsSubScreen
import net.minecraft.network.chat.CommonComponents
import net.minecraft.network.chat.Component
import net.minecraft.network.chat.MutableComponent

internal class ShaderLibrariesScreen(
    lastScreen: Screen,
    options: Options,
) : OptionsSubScreen(lastScreen, options, TITLE) {

    private var lastRenderedVersion: Long = -1L

    companion object {
        @JvmField
        val TITLE: Component = Component.translatable("vulkanshaders.libraries.title")
    }

    override fun init() {
        layout.removeChildren()
        layout.setFooterHeight(33)
        ProprietaryLibraryManager.refresh()
        super.init()
    }

    override fun tick() {
        val currentVersion = ProprietaryLibraryManager.stateVersion
        if (currentVersion != lastRenderedVersion) {
            val scroll = list?.scrollAmount() ?: 0.0
            rebuildWidgets()
            list?.setScrollAmount(scroll)
        }
    }

    override fun addOptions() {
        val list = this.list ?: return
        lastRenderedVersion = ProprietaryLibraryManager.stateVersion

        list.addHeader(
            Component.translatable(
                "vulkanshaders.libraries.platform",
                ProprietaryLibraryManager.platform.displayLabel,
            ).withStyle(ChatFormatting.GRAY),
        )

        list.addHeader(
            Component.translatable("vulkanshaders.libraries.manual_instructions")
                .withStyle(ChatFormatting.GRAY),
        )

        if (ProprietaryLibraryManager.restartRequired) {
            list.addHeader(
                Component.translatable("vulkanshaders.libraries.restart_banner")
                    .withStyle(ChatFormatting.GOLD),
            )
        }

        for (entry in ProprietaryLibraryManager.entries()) {
            val button = Button.builder(buildRowLabel(entry)) { onEntryClicked(entry) }
                .tooltip(Tooltip.create(buildRowTooltip(entry)))
                .build()
            button.active = entry.state != LibraryInstallState.UNSUPPORTED &&
                entry.state != LibraryInstallState.DOWNLOADING
            list.addBig(button)
        }
    }

    override fun addFooter() {
        val row = LinearLayout.horizontal().spacing(5)
        row.addChild(
            Button.builder(Component.translatable("vulkanshaders.libraries.open_folder")) {
                val dir = ProprietaryLibraryManager.ensureNativesDir()
                runCatching { Blaze3D.openPath(dir) }.onFailure {
                    ShaderNotices.libraryNotice(
                        TITLE,
                        Component.literal(dir.toString()),
                    )
                }
            }.width(100).tooltip(
                Tooltip.create(
                    Component.translatable(
                        "vulkanshaders.libraries.open_folder.tooltip",
                        ProprietaryLibraryManager.nativesDir().toString(),
                    ),
                ),
            ).build(),
        )
        row.addChild(
            Button.builder(Component.translatable("vulkanshaders.screen.reload")) {
                ProprietaryLibraryManager.refresh()
                val scroll = list?.scrollAmount() ?: 0.0
                rebuildWidgets()
                list?.setScrollAmount(scroll)
            }.width(100).build(),
        )
        row.addChild(
            Button.builder(CommonComponents.GUI_DONE) { onClose() }
                .width(100)
                .build(),
        )
        layout.addToFooter(row)
    }

    private fun onEntryClicked(entry: LibraryStatusEntry) {
        val libName = Component.translatable(entry.spec.id.translationKey)
        when (entry.state) {
            LibraryInstallState.AVAILABLE_TO_DOWNLOAD, LibraryInstallState.ERROR -> {
                if (entry.spec.policy == AcquisitionPolicy.DOWNLOADABLE) {
                    val started = ProprietaryLibraryManager.startDownload(entry.spec.id) { ok, err ->
                        Minecraft.getInstance().execute {
                            if (ok) {
                                ShaderNotices.libraryNotice(
                                    Component.translatable("vulkanshaders.libraries.toast.downloaded_title"),
                                    Component.translatable("vulkanshaders.libraries.toast.downloaded_body", libName),
                                )
                            } else {
                                ShaderNotices.libraryNotice(
                                    Component.translatable("vulkanshaders.libraries.toast.failed_title"),
                                    Component.literal(err ?: "Download failed"),
                                )
                            }
                        }
                    }
                    if (started) {
                        val scroll = list?.scrollAmount() ?: 0.0
                        rebuildWidgets()
                        list?.setScrollAmount(scroll)
                    }
                } else {
                    openManualFolder(entry, libName)
                }
            }
            LibraryInstallState.MANUAL_REQUIRED -> openManualFolder(entry, libName)
            LibraryInstallState.RESTART_REQUIRED -> {
                ShaderNotices.libraryNotice(
                    Component.translatable("vulkanshaders.libraries.toast.restart_title"),
                    Component.translatable("vulkanshaders.libraries.toast.restart_body", libName),
                )
            }
            LibraryInstallState.SYSTEM_MISSING -> {
                ShaderNotices.libraryNotice(
                    libName,
                    Component.translatable("vulkanshaders.libraries.toast.system_driver"),
                )
            }
            LibraryInstallState.ACTIVE, LibraryInstallState.INSTALLED -> {
                val location = entry.detectedPath?.fileName?.toString()
                    ?: entry.spec.expectedFileName
                    ?: ""
                ShaderNotices.libraryNotice(
                    libName,
                    Component.translatable(
                        if (entry.state == LibraryInstallState.ACTIVE) "vulkanshaders.libraries.toast.active_body"
                        else "vulkanshaders.libraries.toast.installed_body",
                        location,
                    ),
                )
            }
            LibraryInstallState.DOWNLOADING, LibraryInstallState.UNSUPPORTED -> Unit
        }
    }

    private fun openManualFolder(entry: LibraryStatusEntry, libName: Component) {
        val dir = ProprietaryLibraryManager.ensureNativesDir()
        runCatching { Blaze3D.openPath(dir) }
        val expected = entry.spec.expectedFileName ?: ""
        ShaderNotices.libraryNotice(
            libName,
            Component.translatable("vulkanshaders.libraries.toast.manual_body", expected),
        )
    }

    private fun buildStatusBadge(entry: LibraryStatusEntry): MutableComponent = when (entry.state) {
        LibraryInstallState.ACTIVE ->
            Component.translatable("vulkanshaders.libraries.badge.found").withStyle(ChatFormatting.GREEN)
        LibraryInstallState.INSTALLED ->
            Component.translatable("vulkanshaders.libraries.badge.version_mismatch").withStyle(ChatFormatting.YELLOW)
        LibraryInstallState.RESTART_REQUIRED ->
            Component.translatable("vulkanshaders.libraries.status.restart").withStyle(ChatFormatting.GOLD)
        LibraryInstallState.DOWNLOADING ->
            Component.translatable("vulkanshaders.libraries.status.downloading", entry.progressPercent)
                .withStyle(ChatFormatting.AQUA)
        LibraryInstallState.AVAILABLE_TO_DOWNLOAD ->
            Component.translatable("vulkanshaders.libraries.badge.missing")
                .append(Component.literal(" "))
                .append(Component.translatable("vulkanshaders.libraries.status.download"))
                .withStyle(ChatFormatting.AQUA)
        LibraryInstallState.SYSTEM_MISSING ->
            Component.translatable("vulkanshaders.libraries.badge.missing")
                .append(Component.literal(" "))
                .append(Component.translatable("vulkanshaders.libraries.status.system_missing"))
                .withStyle(ChatFormatting.GRAY)
        LibraryInstallState.MANUAL_REQUIRED ->
            Component.translatable("vulkanshaders.libraries.badge.missing")
                .append(Component.literal(" "))
                .append(Component.translatable("vulkanshaders.libraries.status.manual"))
                .withStyle(ChatFormatting.YELLOW)
        LibraryInstallState.UNSUPPORTED ->
            Component.translatable("vulkanshaders.libraries.badge.unsupported_os").withStyle(ChatFormatting.DARK_GRAY)
        LibraryInstallState.ERROR ->
            Component.translatable("vulkanshaders.libraries.status.error").withStyle(ChatFormatting.RED)
    }

    private fun buildRowLabel(entry: LibraryStatusEntry): Component {
        val name = Component.translatable(entry.spec.id.translationKey)
        val badge = buildStatusBadge(entry)
        return Component.empty().append(name).append(Component.literal(" — ")).append(badge)
    }

    private fun buildRowTooltip(entry: LibraryStatusEntry): Component {
        val tip = Component.translatable(entry.spec.notesKey)
        if (entry.state == LibraryInstallState.INSTALLED) {
            tip.append("\n").append(
                Component.translatable("vulkanshaders.libraries.tooltip.version_mismatch")
                    .withStyle(ChatFormatting.YELLOW),
            )
        }
        if (entry.state == LibraryInstallState.MANUAL_REQUIRED || entry.spec.policy == AcquisitionPolicy.MANUAL_INSTALL) {
            val expected = entry.spec.expectedFileName ?: ProprietaryLibraryManager.platform.sharedLibraryExtension
            tip.append("\n").append(
                Component.translatable(
                    "vulkanshaders.libraries.tooltip.manual_steps",
                    expected,
                    ProprietaryLibraryManager.nativesDir().toString(),
                ).withStyle(ChatFormatting.AQUA),
            )
        }
        tip.append("\n").append(
            Component.translatable("vulkanshaders.libraries.tooltip.license", entry.spec.id.licenseName)
                .withStyle(ChatFormatting.GRAY),
        )
        tip.append("\n").append(
            Component.translatable("vulkanshaders.libraries.tooltip.source", entry.spec.id.officialPageUrl)
                .withStyle(ChatFormatting.GRAY),
        )
        entry.detectedPath?.let { path ->
            tip.append("\n").append(
                Component.translatable("vulkanshaders.libraries.tooltip.path", path.toString())
                    .withStyle(ChatFormatting.DARK_GRAY),
            )
        } ?: entry.spec.expectedFileName?.let { fileName ->
            tip.append("\n").append(
                Component.translatable("vulkanshaders.libraries.tooltip.expected", fileName)
                    .withStyle(ChatFormatting.DARK_GRAY),
            )
        }
        entry.errorMessage?.let { err ->
            tip.append("\n").append(
                Component.translatable("vulkanshaders.libraries.tooltip.error", err)
                    .withStyle(ChatFormatting.RED),
            )
        }
        if (entry.state == LibraryInstallState.RESTART_REQUIRED) {
            tip.append("\n").append(
                Component.translatable("vulkanshaders.libraries.restart_banner")
                    .withStyle(ChatFormatting.GOLD),
            )
        }
        return tip
    }
}
