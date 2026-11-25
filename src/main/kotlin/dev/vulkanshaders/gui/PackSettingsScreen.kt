package dev.vulkanshaders.gui

import com.google.gson.GsonBuilder
import com.google.gson.JsonObject
import com.google.gson.JsonParser
import dev.vulkanshaders.VulkanShaders
import dev.vulkanshaders.config.RenderTier
import dev.vulkanshaders.pack.LoadedShaderPack
import dev.vulkanshaders.pack.PackPresetDefinition
import dev.vulkanshaders.pack.PackSettingDefinition
import dev.vulkanshaders.pack.PackLocalization
import dev.vulkanshaders.pack.SettingType
import dev.vulkanshaders.pack.ShaderPackManager
import dev.vulkanshaders.pipeline.GbufferPipelineRegistry
import dev.vulkanshaders.state.WorldStateCollector
import net.minecraft.ChatFormatting
import net.minecraft.client.Minecraft
import net.minecraft.client.Options
import net.minecraft.client.gui.GuiGraphicsExtractor
import net.minecraft.client.gui.components.AbstractSliderButton
import net.minecraft.client.gui.components.AbstractWidget
import net.minecraft.client.gui.components.Button
import net.minecraft.client.gui.components.EditBox
import net.minecraft.client.gui.components.OptionsList
import net.minecraft.client.gui.components.StringWidget
import net.minecraft.client.gui.components.Tooltip
import net.minecraft.client.gui.layouts.LinearLayout
import net.minecraft.client.gui.narration.NarrationElementOutput
import net.minecraft.client.gui.screens.Screen
import net.minecraft.client.gui.screens.TitleScreen
import net.minecraft.client.gui.screens.options.OptionsSubScreen
import net.minecraft.network.chat.CommonComponents
import net.minecraft.network.chat.Component
import net.minecraft.network.chat.Style
import net.minecraft.network.chat.TextColor
import java.util.Locale
import kotlin.math.abs
import kotlin.math.round
import kotlin.math.roundToInt

private fun localizedManifest(pack: LoadedShaderPack) =
    PackLocalization.localize(
        pack.manifest,
        PackLocalization.load(pack.rootPath, Minecraft.getInstance().languageManager.selected),
    )

/**
 * Dynamic vanilla-styled settings screen generated from a `.vsp` shader pack's `[settings.*]` definitions.
 * Supports Low/Medium/High/Ultra quality presets, category filter, live search, modified-only filtering,
 * per-setting tooltips and reset buttons, JSON profile copy/paste via clipboard, live `WorldUBO.packParams`
 * updates (`runtime = true`), and native `VkSpecializationInfo` pipeline specialization (`runtime = false`).
 */
class PackSettingsScreen(
    private val returnScreen: Screen?,
    options: Options,
    private val pack: LoadedShaderPack,
) : OptionsSubScreen(
    returnScreen ?: TitleScreen(),
    options,
    Component.translatable("vulkanshaders.screen.pack_settings.title", localizedManifest(pack).name),
) {
    private val manifest = localizedManifest(pack)
    private var selectedCategory: String? = null
    private var searchQuery: String = ""
    private var modifiedOnly: Boolean = false
    private var searchBox: EditBox? = null
    private var presetButton: Button? = null
    private var statusBanner: Component? = null
    private var dirtyRuntimeOverrides: Boolean = false

    private fun activeTier(): RenderTier = VulkanShaders.config.renderTier

    private fun visibleSettings(tierIndex: Int = activeTier().tierIndex): List<PackSettingDefinition> =
        manifest.settings.filter { it.appliesToTier(tierIndex) }

    private fun availablePresets(tierIndex: Int = activeTier().tierIndex): List<PackPresetDefinition> {
        val declared = manifest.presets.filter { it.appliesToTier(tierIndex) }
        if (declared.isNotEmpty()) return declared
        val visible = visibleSettings(tierIndex)
        if (visible.isEmpty()) return emptyList()
        return buildDefaultQualityPresets(visible, tierIndex)
    }

    private fun visibleCategories(tierIndex: Int = activeTier().tierIndex): List<String> {
        val seen = LinkedHashSet<String>()
        for (setting in visibleSettings(tierIndex)) {
            seen.add(resolveCategory(setting))
        }
        return seen.toList()
    }

    override fun init() {
        layout.removeChildren()
        val tierIndex = activeTier().tierIndex
        if (selectedCategory != null && selectedCategory !in visibleCategories(tierIndex)) {
            selectedCategory = null
        }
        // Compact 2-row header (56px) keeps the list usable at small window sizes and high GUI scales.
        layout.setHeaderHeight(56)
        super.init()
    }

    override fun addTitle() {
        val tierIndex = activeTier().tierIndex
        val categories = visibleCategories(tierIndex)

        val header = layout.addToHeader(LinearLayout.vertical().spacing(4))
        header.defaultCellSetting().alignHorizontallyCenter()
        header.addChild(StringWidget(title, font))

        // Single 310px toolbar: Search (130) + gap (4) + Category cycle (100) + gap (4) + Modified toggle (72).
        val toolbar = header.addChild(LinearLayout.horizontal().spacing(4))
        val field = toolbar.addChild(
            EditBox(font, 130, 20, Component.translatable("vulkanshaders.screen.pack_settings.search")),
        )
        field.setMaxLength(128)
        field.setHint(Component.translatable("vulkanshaders.screen.pack_settings.search"))
        field.value = searchQuery
        field.setResponder { newQuery ->
            if (newQuery != searchQuery) {
                searchQuery = newQuery
                statusBanner = null
                refreshOptionsList(resetScroll = true)
            }
        }
        searchBox = field

        val catName = selectedCategory?.let { Component.literal(it) }
            ?: Component.translatable("vulkanshaders.screen.pack_settings.tab.all")
        val catLabel = Component.translatable("vulkanshaders.screen.pack_settings.category_cycle", catName)
            .let { if (selectedCategory != null) it.withStyle(ChatFormatting.YELLOW) else it }
        val categoryButton = Button.builder(catLabel) {
            val allOptions: List<String?> = listOf(null) + categories
            val idx = allOptions.indexOf(selectedCategory).coerceAtLeast(0)
            selectedCategory = allOptions[(idx + 1) % allOptions.size]
            statusBanner = null
            refreshWidgets(resetScroll = true)
        }.width(100)
            .tooltip(Tooltip.create(Component.translatable("vulkanshaders.screen.pack_settings.category_cycle.tooltip")))
            .build()
            .apply { active = categories.isNotEmpty() }
        toolbar.addChild(categoryButton)

        val modState = if (modifiedOnly) CommonComponents.OPTION_ON else CommonComponents.OPTION_OFF
        val modLabel = Component.translatable("vulkanshaders.screen.pack_settings.modified_short", modState)
            .let { if (modifiedOnly) it.withStyle(ChatFormatting.YELLOW) else it }
        toolbar.addChild(
            Button.builder(modLabel) {
                modifiedOnly = !modifiedOnly
                statusBanner = null
                refreshWidgets(resetScroll = true)
            }.width(72)
                .tooltip(Tooltip.create(Component.translatable("vulkanshaders.screen.pack_settings.modified_only.tooltip")))
                .build(),
        )
    }

    override fun addOptions() {
        val list = this.list ?: return
        val packId = pack.manifest.id
        presetButton = null

        statusBanner?.let { list.addHeader(it) }

        val tierIndex = activeTier().tierIndex
        val presets = availablePresets(tierIndex)
        if (presets.isNotEmpty()) {
            val activePreset = detectActivePreset(tierIndex)
            val btn = Button.builder(buildPresetButtonLabel(activePreset)) {
                val currentPresets = availablePresets()
                if (currentPresets.isNotEmpty()) {
                    val current = detectActivePreset()
                    val nextIndex = if (current == null) {
                        0
                    } else {
                        val idx = currentPresets.indexOfFirst { it.id == current.id }
                        if (idx < 0) 0 else (idx + 1) % currentPresets.size
                    }
                    applyPreset(currentPresets[nextIndex])
                }
            }.tooltip(buildPresetTooltip(activePreset)).build()
            presetButton = btn
            list.addBig(btn)
        }

        val query = searchQuery.trim()
        val filtered = visibleSettings().filter { setting ->
            val cat = resolveCategory(setting)
            val matchesTab = selectedCategory == null || cat == selectedCategory
            val matchesMod = !modifiedOnly || isModified(packId, setting)
            val matchesQuery = query.isEmpty() ||
                setting.label.contains(query, ignoreCase = true) ||
                setting.key.contains(query, ignoreCase = true) ||
                (setting.description?.contains(query, ignoreCase = true) == true) ||
                cat.contains(query, ignoreCase = true)
            matchesTab && matchesMod && matchesQuery
        }

        if (filtered.isEmpty()) {
            list.addBig(
                StringWidget(
                    Component.translatable("vulkanshaders.screen.pack_settings.no_matches")
                        .withStyle(ChatFormatting.GRAY),
                    font,
                ),
            )
            return
        }

        if (selectedCategory == null) {
            val grouped = LinkedHashMap<String, MutableList<PackSettingDefinition>>()
            for (setting in filtered) {
                grouped.getOrPut(resolveCategory(setting)) { mutableListOf() }.add(setting)
            }
            for ((category, groupSettings) in grouped) {
                list.addHeader(Component.literal(category).withStyle(ChatFormatting.GOLD))
                for (setting in groupSettings) {
                    addSettingWidget(list, packId, setting)
                }
            }
        } else {
            for (setting in filtered) {
                addSettingWidget(list, packId, setting)
            }
        }
    }

    private fun addSettingWidget(
        list: OptionsList,
        packId: String,
        setting: PackSettingDefinition,
    ) {
        val currentVal = currentValue(packId, setting)
        val modified = isModified(packId, setting)
        val tooltip = buildSettingTooltip(setting)

        when (setting.type) {
            SettingType.BOOL -> {
                val enabled = currentVal.equals("true", ignoreCase = true) || currentVal == "1"
                val control = Button.builder(
                    settingLabel(setting.label, modified)
                        .append(": ")
                        .append(if (enabled) CommonComponents.OPTION_ON else CommonComponents.OPTION_OFF),
                ) {
                    statusBanner = null
                    setOverride(packId, setting, (!enabled).toString())
                    refreshWidgets()
                }.width(if (modified) 150 else 310).tooltip(tooltip).build()
                if (modified) {
                    list.addSmall(control, createResetButton(packId, setting, modified = true))
                } else {
                    list.addBig(control)
                }
            }
            SettingType.ENUM -> {
                val control = Button.builder(
                    settingLabel(setting.label, modified).append(": ${displayOptionLabel(setting, currentVal)}"),
                ) {
                    val opts = setting.options
                    if (opts.isNotEmpty()) {
                        val idx = opts.indexOfFirst { it.equals(currentVal, ignoreCase = true) }.coerceAtLeast(0)
                        val next = opts[(idx + 1) % opts.size]
                        statusBanner = null
                        setOverride(packId, setting, next)
                        refreshWidgets()
                    }
                }.width(if (modified) 150 else 310).tooltip(tooltip).build()
                if (modified) {
                    list.addSmall(control, createResetButton(packId, setting, modified = true))
                } else {
                    list.addBig(control)
                }
            }
            SettingType.INT, SettingType.FLOAT -> {
                val min = setting.min ?: 0.0
                val max = (setting.max ?: 10.0).coerceAtLeast(min + 1e-4)
                val cur = (currentVal.toDoubleOrNull() ?: setting.defaultValue.toDoubleOrNull() ?: min)
                    .coerceIn(min, max)
                val step = setting.step ?: if (setting.type == SettingType.INT) 1.0 else 0.05
                val resetBtn = createResetButton(packId, setting, modified)
                val slider = NumericSettingSlider(
                    label = setting.label,
                    isInt = setting.type == SettingType.INT,
                    minVal = min,
                    maxVal = max,
                    step = step,
                    initialVal = cur,
                    initialModified = modified,
                    width = 150,
                    onValueChanged = { formatted ->
                        statusBanner = null
                        setOverride(packId, setting, formatted)
                        val nowModified = isModified(packId, setting)
                        resetBtn.active = nowModified
                        nowModified
                    },
                )
                slider.setTooltip(tooltip)
                list.addSmall(slider, resetBtn)
            }
            SettingType.COLOR_RGB -> addColorRgbEditor(list, packId, setting, currentVal, modified, tooltip)
        }
    }

    private fun createResetButton(
        packId: String,
        setting: PackSettingDefinition,
        modified: Boolean,
    ): Button = Button.builder(
        Component.literal("↺ ").append(Component.translatable("vulkanshaders.screen.reset")),
    ) {
        statusBanner = null
        resetSingleSetting(packId, setting)
        refreshWidgets()
    }.width(150)
        .tooltip(
            Tooltip.create(
                Component.translatable("vulkanshaders.screen.pack_settings.default_hint", setting.defaultValue),
            ),
        )
        .build()
        .apply { active = modified }

    private fun addColorRgbEditor(
        list: OptionsList,
        packId: String,
        setting: PackSettingDefinition,
        currentVal: String,
        modified: Boolean,
        tooltip: Tooltip,
    ) {
        val defaultRgb = WorldStateCollector.parseColorRgb(setting.defaultValue) ?: 0xFFFFFF
        val initialRgb = WorldStateCollector.parseColorRgb(currentVal) ?: defaultRgb
        val (r0, g0, b0) = WorldStateCollector.unpackColorRgb(initialRgb)
        val rgbChannels = intArrayOf(r0, g0, b0)

        val previewBadge = ColorPreviewBadgeWidget(setting.label, rgbChannels, modified)
        previewBadge.setTooltip(tooltip)

        lateinit var resetColorButton: Button
        val onColorChanged = {
            val packed = (rgbChannels[0] shl 16) or (rgbChannels[1] shl 8) or rgbChannels[2]
            val hex = WorldStateCollector.formatColorRgb(packed)
            statusBanner = null
            setOverride(packId, setting, hex)
            val nowModified = isModified(packId, setting)
            previewBadge.updateModified(nowModified)
            previewBadge.syncMessage()
            resetColorButton.active = nowModified
        }

        val redSlider = RgbChannelSlider(
            translationKey = "vulkanshaders.settings.color.red",
            channelIndex = 0,
            rgbChannels = rgbChannels,
            onChannelChanged = onColorChanged,
        ).apply { setTooltip(tooltip) }
        val greenSlider = RgbChannelSlider(
            translationKey = "vulkanshaders.settings.color.green",
            channelIndex = 1,
            rgbChannels = rgbChannels,
            onChannelChanged = onColorChanged,
        ).apply { setTooltip(tooltip) }
        val blueSlider = RgbChannelSlider(
            translationKey = "vulkanshaders.settings.color.blue",
            channelIndex = 2,
            rgbChannels = rgbChannels,
            onChannelChanged = onColorChanged,
        ).apply { setTooltip(tooltip) }
        resetColorButton = Button.builder(
            Component.literal("↺ ").append(Component.translatable("vulkanshaders.settings.color.reset")),
        ) {
            statusBanner = null
            resetSingleSetting(packId, setting)
            refreshWidgets()
        }.width(150)
            .tooltip(
                Tooltip.create(
                    Component.translatable("vulkanshaders.screen.pack_settings.default_hint", setting.defaultValue),
                ),
            )
            .build()
            .apply { active = modified }

        list.addBig(previewBadge)
        list.addSmall(redSlider, greenSlider)
        list.addSmall(blueSlider, resetColorButton)
    }

    override fun addFooter() {
        // Keep the footer strictly aligned to the 310px column (74 + 4 + 74 + 4 + 74 + 4 + 76).
        val footer = layout.addToFooter(LinearLayout.horizontal().spacing(4))

        footer.addChild(
            Button.builder(Component.translatable("vulkanshaders.screen.pack_settings.copy_profile")) {
                copyProfileToClipboard()
            }.width(74).build(),
        )
        footer.addChild(
            Button.builder(Component.translatable("vulkanshaders.screen.pack_settings.paste_profile")) {
                pasteProfileFromClipboard()
            }.width(74).build(),
        )

        val resetKey = if (selectedCategory == null) {
            "vulkanshaders.screen.pack_settings.reset_all"
        } else {
            "vulkanshaders.screen.pack_settings.reset_tab"
        }
        footer.addChild(
            Button.builder(Component.translatable(resetKey)) {
                resetCurrentScope()
            }.width(74).build(),
        )

        footer.addChild(
            Button.builder(CommonComponents.GUI_DONE) {
                this.onClose()
            }.width(76).build(),
        )
    }

    private fun detectActivePreset(tierIndex: Int = activeTier().tierIndex): PackPresetDefinition? {
        val presets = availablePresets(tierIndex)
        if (presets.isEmpty()) return null
        val visible = visibleSettings(tierIndex)
        val packId = pack.manifest.id
        return presets.firstOrNull { preset ->
            visible.all { setting ->
                val expected = preset.values[setting.key] ?: setting.defaultValue
                val actual = currentValue(packId, setting)
                !valuesDiffer(setting, actual, expected)
            }
        }
    }

    private fun localizedPresetLabel(preset: PackPresetDefinition): Component = when (preset.id.lowercase(Locale.ROOT)) {
        "low" -> Component.translatable("vulkanshaders.screen.pack_settings.preset.low")
        "medium" -> Component.translatable("vulkanshaders.screen.pack_settings.preset.medium")
        "high" -> Component.translatable("vulkanshaders.screen.pack_settings.preset.high")
        "ultra" -> Component.translatable("vulkanshaders.screen.pack_settings.preset.ultra")
        else -> Component.literal(preset.label)
    }

    private fun buildPresetButtonLabel(activePreset: PackPresetDefinition?): Component {
        val valueComponent = if (activePreset != null) {
            localizedPresetLabel(activePreset)
        } else {
            Component.translatable("vulkanshaders.screen.pack_settings.preset.custom")
                .withStyle(ChatFormatting.YELLOW)
        }
        return Component.translatable("vulkanshaders.screen.pack_settings.preset", valueComponent)
    }

    private fun buildPresetTooltip(activePreset: PackPresetDefinition?): Tooltip {
        val text = Component.empty()
        val desc = activePreset?.description?.trim().orEmpty()
        if (desc.isNotEmpty()) {
            text.append(Component.literal(desc))
            text.append(Component.literal("\n"))
        }
        text.append(
            Component.translatable("vulkanshaders.screen.pack_settings.preset.tooltip")
                .withStyle(ChatFormatting.GRAY),
        )
        return Tooltip.create(text)
    }

    private fun syncPresetButton() {
        val button = presetButton ?: return
        val activePreset = detectActivePreset()
        button.message = buildPresetButtonLabel(activePreset)
        button.setTooltip(buildPresetTooltip(activePreset))
    }

    private fun applyPreset(preset: PackPresetDefinition) {
        val packId = pack.manifest.id
        val tier = activeTier()
        val visible = visibleSettings(tier.tierIndex)
        val newOverrides = LinkedHashMap<String, String>()

        for (setting in visible) {
            val rawPresetVal = preset.values[setting.key]
            if (rawPresetVal != null) {
                val normalized = validateAndNormalizeSettingValue(setting, rawPresetVal) ?: rawPresetVal
                if (valuesDiffer(setting, normalized, setting.defaultValue)) {
                    newOverrides[setting.key] = normalized
                }
            }
        }

        val updated = VulkanShaders.config.withPackOverrides(packId, tier, newOverrides)
        if (!VulkanShaders.updateConfig(updated)) return
        dirtyRuntimeOverrides = false

        statusBanner = Component.translatable(
            "vulkanshaders.screen.pack_settings.preset_applied",
            localizedPresetLabel(preset),
        ).withStyle(ChatFormatting.GREEN)
        refreshWidgets()
    }

    private fun copyProfileToClipboard() {
        val packId = pack.manifest.id
        val tier = activeTier()
        val visibleKeys = visibleSettings(tier.tierIndex).mapTo(HashSet()) { it.key }
        val overrides = VulkanShaders.config.activePackOverrides(packId, tier)
            .filterKeys { it in visibleKeys }
        val root = JsonObject()
        root.addProperty("packId", packId)
        root.addProperty("tier", tier.tierIndex)
        val overridesObj = JsonObject()
        for ((k, v) in overrides.toSortedMap()) {
            overridesObj.addProperty(k, v)
        }
        root.add("overrides", overridesObj)
        val json = PROFILE_GSON.toJson(root)
        val mc = Minecraft.getInstance()
        runCatching { mc.keyboardHandler.clipboard = json }
        statusBanner = Component.translatable("vulkanshaders.screen.pack_settings.copied_profile")
            .withStyle(ChatFormatting.GREEN)
        refreshWidgets()
    }

    private fun pasteProfileFromClipboard() {
        val mc = Minecraft.getInstance()
        val raw = runCatching { mc.keyboardHandler.clipboard }.getOrNull()?.trim().orEmpty()
        if (raw.isEmpty()) return

        val parsedResult = runCatching {
            val elem = JsonParser.parseString(raw)
            if (!elem.isJsonObject) return@runCatching null
            val obj = elem.asJsonObject
            if (obj.has("overrides") && obj.get("overrides").isJsonObject) {
                obj.getAsJsonObject("overrides") to true
            } else {
                obj to false
            }
        }.getOrNull() ?: return

        val (parsedObj, hasOverridesObject) = parsedResult

        val tier = activeTier()
        val visible = visibleSettings(tier.tierIndex)
        val settingsByKey = visible.associateBy { it.key }
        val packId = pack.manifest.id
        val oldPerPack = VulkanShaders.config.activePackOverrides(packId, tier)
        val perPack = if (hasOverridesObject) {
            LinkedHashMap()
        } else {
            LinkedHashMap(oldPerPack)
        }

        var appliedCount = 0

        for ((key, jsonVal) in parsedObj.entrySet()) {
            val setting = settingsByKey[key] ?: continue
            if (!jsonVal.isJsonPrimitive) continue
            val validated = validateAndNormalizeSettingValue(setting, jsonVal.asString) ?: continue
            if (valuesDiffer(setting, validated, setting.defaultValue)) {
                perPack[key] = validated
            } else {
                perPack.remove(key)
            }
            appliedCount++
        }

        if (appliedCount == 0 && !hasOverridesObject) return

        val requiresRecompile = visible.any { setting ->
            if (setting.runtime) return@any false
            val oldVal = oldPerPack[setting.key] ?: setting.defaultValue
            val newVal = perPack[setting.key] ?: setting.defaultValue
            valuesDiffer(setting, oldVal, newVal)
        }

        val updated = VulkanShaders.config.withPackOverrides(packId, tier, perPack)
        if (requiresRecompile) {
            dirtyRuntimeOverrides = false
            VulkanShaders.updateConfig(updated)
            ShaderPackManager.syncActivePackSpecializationConstants()
            GbufferPipelineRegistry.invalidateShaderSourceCache()
        } else {
            dirtyRuntimeOverrides = false
            VulkanShaders.updateRuntimePackOverrides(updated, persistToDisk = true)
        }

        statusBanner = Component.translatable("vulkanshaders.screen.pack_settings.pasted_profile", appliedCount)
            .withStyle(ChatFormatting.GREEN)
        refreshWidgets()
    }

    private fun validateAndNormalizeSettingValue(setting: PackSettingDefinition, raw: String): String? {
        val trimmed = raw.trim()
        return when (setting.type) {
            SettingType.BOOL -> when {
                trimmed.equals("true", ignoreCase = true) || trimmed == "1" -> "true"
                trimmed.equals("false", ignoreCase = true) || trimmed == "0" -> "false"
                else -> null
            }
            SettingType.ENUM -> setting.options.firstOrNull { it.equals(trimmed, ignoreCase = true) }
            SettingType.INT -> {
                val parsed = trimmed.toIntOrNull() ?: trimmed.toDoubleOrNull()?.roundToInt() ?: return null
                val min = setting.min?.roundToInt() ?: Int.MIN_VALUE
                val max = setting.max?.roundToInt() ?: Int.MAX_VALUE
                val clamped = parsed.coerceIn(min, max)
                clamped.toString()
            }
            SettingType.FLOAT -> {
                val parsed = trimmed.toDoubleOrNull() ?: return null
                if (!parsed.isFinite()) return null
                val min = setting.min ?: -Double.MAX_VALUE
                val max = setting.max ?: Double.MAX_VALUE
                val clamped = parsed.coerceIn(min, max)
                formatFloat(clamped)
            }
            SettingType.COLOR_RGB -> {
                val rgb = WorldStateCollector.parseColorRgb(trimmed) ?: return null
                WorldStateCollector.formatColorRgb(rgb)
            }
        }
    }

    private fun resetCurrentScope() {
        val packId = pack.manifest.id
        val tier = activeTier()
        val visible = visibleSettings(tier.tierIndex)
        val perPack = LinkedHashMap(VulkanShaders.config.activePackOverrides(packId, tier))
        val scope = selectedCategory

        if (scope == null) {
            val requiresRecompile = visible.any { setting ->
                !setting.runtime && perPack.containsKey(setting.key)
            }
            val updated = VulkanShaders.config.withPackOverrides(packId, tier, emptyMap())
            if (requiresRecompile) {
                dirtyRuntimeOverrides = false
                VulkanShaders.updateConfig(updated)
                ShaderPackManager.syncActivePackSpecializationConstants()
                GbufferPipelineRegistry.invalidateShaderSourceCache()
            } else {
                dirtyRuntimeOverrides = false
                VulkanShaders.updateRuntimePackOverrides(updated, persistToDisk = true)
            }
            statusBanner = Component.translatable("vulkanshaders.screen.pack_settings.reset_defaults_done")
                .withStyle(ChatFormatting.GREEN)
            refreshWidgets()
            return
        }

        var requiresRecompile = false
        for (setting in visible) {
            if (resolveCategory(setting) == scope) {
                if (perPack.remove(setting.key) != null && !setting.runtime) {
                    requiresRecompile = true
                }
            }
        }
        val updated = VulkanShaders.config.withPackOverrides(packId, tier, perPack)
        if (requiresRecompile) {
            dirtyRuntimeOverrides = false
            VulkanShaders.updateConfig(updated)
            ShaderPackManager.syncActivePackSpecializationConstants()
            GbufferPipelineRegistry.invalidateShaderSourceCache()
        } else {
            dirtyRuntimeOverrides = false
            VulkanShaders.updateRuntimePackOverrides(updated, persistToDisk = true)
        }
        statusBanner = Component.translatable("vulkanshaders.screen.pack_settings.reset_defaults_done")
            .withStyle(ChatFormatting.GREEN)
        refreshWidgets()
    }

    private fun resetSingleSetting(packId: String, setting: PackSettingDefinition) {
        val tier = activeTier()
        val perPack = LinkedHashMap(VulkanShaders.config.activePackOverrides(packId, tier))
        perPack.remove(setting.key)
        val updated = VulkanShaders.config.withPackOverrides(packId, tier, perPack)
        if (setting.runtime) {
            dirtyRuntimeOverrides = false
            VulkanShaders.updateRuntimePackOverrides(updated, persistToDisk = true)
        } else {
            dirtyRuntimeOverrides = false
            VulkanShaders.updateConfig(updated)
            ShaderPackManager.syncActivePackSpecializationConstants()
            GbufferPipelineRegistry.invalidateShaderSourceCache()
        }
        syncPresetButton()
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

    private fun refreshWidgets(keepSearchFocus: Boolean = false, resetScroll: Boolean = false) {
        val scroll = if (resetScroll) 0.0 else (list?.scrollAmount() ?: 0.0)
        val wasFocused = keepSearchFocus || (searchBox?.isFocused == true)
        rebuildWidgets()
        list?.setScrollAmount(scroll)
        if (wasFocused) {
            searchBox?.let { box ->
                this.focused = box
                box.isFocused = true
            }
        }
    }

    private fun buildSettingTooltip(setting: PackSettingDefinition): Tooltip {
        val text = Component.empty()
        val desc = setting.description?.trim().orEmpty()
        if (desc.isNotEmpty()) {
            text.append(Component.literal(desc))
            text.append(Component.literal("\n"))
        }
        text.append(
            Component.translatable("vulkanshaders.screen.pack_settings.default_hint", setting.defaultValue)
                .withStyle(ChatFormatting.GRAY),
        )
        text.append(Component.literal("\n"))
        if (setting.runtime) {
            text.append(
                Component.translatable("vulkanshaders.screen.pack_settings.runtime_hint")
                    .withStyle(ChatFormatting.GREEN),
            )
        } else {
            text.append(
                Component.translatable("vulkanshaders.screen.pack_settings.recompile_hint")
                    .withStyle(ChatFormatting.GOLD),
            )
        }
        return Tooltip.create(text)
    }

    private fun settingLabel(label: String, modified: Boolean) =
        if (modified) {
            Component.literal("* $label").withStyle(ChatFormatting.YELLOW)
        } else {
            Component.literal(label)
        }

    private fun currentValue(packId: String, setting: PackSettingDefinition): String =
        VulkanShaders.config.activePackOverrides(packId, activeTier())[setting.key] ?: setting.defaultValue

    private fun displayOptionLabel(setting: PackSettingDefinition, rawValue: String): String {
        val optionIndex = setting.options.indexOfFirst { it.equals(rawValue, ignoreCase = true) }
        return setting.optionLabels.getOrNull(optionIndex)?.takeIf(String::isNotBlank) ?: rawValue
    }

    private fun isModified(packId: String, setting: PackSettingDefinition): Boolean {
        val overrideVal = VulkanShaders.config.activePackOverrides(packId, activeTier())[setting.key] ?: return false
        return valuesDiffer(setting, overrideVal, setting.defaultValue)
    }

    private fun valuesDiffer(setting: PackSettingDefinition, current: String, defaultVal: String): Boolean =
        when (setting.type) {
            SettingType.BOOL -> {
                val c = current.equals("true", ignoreCase = true) || current == "1"
                val d = defaultVal.equals("true", ignoreCase = true) || defaultVal == "1"
                c != d
            }
            SettingType.INT -> {
                val c = current.toIntOrNull()
                val d = defaultVal.toIntOrNull()
                if (c != null && d != null) c != d else current != defaultVal
            }
            SettingType.FLOAT -> {
                val c = current.toDoubleOrNull()
                val d = defaultVal.toDoubleOrNull()
                if (c != null && d != null) abs(c - d) > 1e-5 else current != defaultVal
            }
            SettingType.ENUM -> !current.equals(defaultVal, ignoreCase = true)
            SettingType.COLOR_RGB -> {
                val c = WorldStateCollector.parseColorRgb(current)
                val d = WorldStateCollector.parseColorRgb(defaultVal)
                if (c != null && d != null) c != d else !current.equals(defaultVal, ignoreCase = true)
            }
        }

    private fun setOverride(packId: String, setting: PackSettingDefinition, value: String) {
        val tier = activeTier()
        val perPack = LinkedHashMap(VulkanShaders.config.activePackOverrides(packId, tier))
        if (valuesDiffer(setting, value, setting.defaultValue)) {
            perPack[setting.key] = value
        } else {
            perPack.remove(setting.key)
        }
        val updated = VulkanShaders.config.withPackOverrides(packId, tier, perPack)
        if (setting.runtime) {
            // Fast path: live WorldUBO update without recompiling shaders or blocking on disk JSON writes
            dirtyRuntimeOverrides = true
            VulkanShaders.updateRuntimePackOverrides(updated, persistToDisk = false)
        } else {
            dirtyRuntimeOverrides = false
            VulkanShaders.updateConfig(updated)
            ShaderPackManager.syncActivePackSpecializationConstants()
            GbufferPipelineRegistry.invalidateShaderSourceCache()
        }
        syncPresetButton()
    }

    private fun flushPendingOverrides() {
        if (dirtyRuntimeOverrides) {
            dirtyRuntimeOverrides = false
            VulkanShaders.updateRuntimePackOverrides(VulkanShaders.config, persistToDisk = true)
        }
    }

    override fun onClose() {
        flushPendingOverrides()
        minecraft.gui.setScreen(returnScreen)
    }

    override fun removed() {
        flushPendingOverrides()
        super.removed()
    }

    private class RgbChannelSlider(
        private val translationKey: String,
        private val channelIndex: Int,
        private val rgbChannels: IntArray,
        private val onChannelChanged: () -> Unit,
    ) : AbstractSliderButton(
        0,
        0,
        150,
        DEFAULT_HEIGHT,
        Component.empty(),
        rgbChannels[channelIndex].coerceIn(0, 255) / 255.0,
    ) {
        init {
            updateMessage()
        }

        override fun updateMessage() {
            val channelValue = (this.value * 255.0).roundToInt().coerceIn(0, 255)
            this.message = Component.translatable(translationKey, channelValue)
        }

        override fun applyValue() {
            val channelValue = (this.value * 255.0).roundToInt().coerceIn(0, 255)
            if (rgbChannels[channelIndex] != channelValue) {
                rgbChannels[channelIndex] = channelValue
                onChannelChanged()
            }
        }
    }

    private class ColorPreviewBadgeWidget(
        private val settingLabel: String,
        private val rgbChannels: IntArray,
        private var modified: Boolean,
    ) : AbstractWidget(0, 0, 310, 20, Component.empty()) {
        init {
            this.active = false
            syncMessage()
        }

        fun updateModified(isModified: Boolean) {
            modified = isModified
        }

        fun syncMessage() {
            val packed = currentRgb24()
            val hex = WorldStateCollector.formatColorRgb(packed)
            val prefix = if (modified) "* $settingLabel: " else "$settingLabel: "
            this.message = Component.literal(prefix)
                .append(Component.literal(hex).withStyle(Style.EMPTY.withColor(TextColor.fromRgb(packed))))
        }

        override fun extractWidgetRenderState(
            guiGraphics: GuiGraphicsExtractor,
            mouseX: Int,
            mouseY: Int,
            partialTick: Float,
        ) {
            val packed = currentRgb24()
            val argb = 0xFF000000.toInt() or packed
            val x = this.x
            val y = this.y
            val w = this.width
            val h = this.height

            guiGraphics.fill(x, y, x + w, y + h, 0xCC181820.toInt())
            guiGraphics.outline(x, y, w, h, 0xFF555566.toInt())

            val swatchSize = 14
            val swatchX = x + 4
            val swatchY = y + (h - swatchSize) / 2
            guiGraphics.fill(swatchX, swatchY, swatchX + swatchSize, swatchY + swatchSize, argb)
            guiGraphics.outline(swatchX, swatchY, swatchSize, swatchSize, 0xFFFFFFFF.toInt())

            val font = Minecraft.getInstance().font
            val textY = y + (h - font.lineHeight) / 2
            val hex = WorldStateCollector.formatColorRgb(packed)
            val labelText = if (modified) {
                Component.literal("* $settingLabel: ").withStyle(ChatFormatting.YELLOW)
            } else {
                Component.literal("$settingLabel: ")
            }
            guiGraphics.text(font, labelText, swatchX + swatchSize + 6, textY, 0xFFEAEAEA.toInt())

            val labelWidth = font.width(labelText)
            val badgeText = Component.literal("$hex (${rgbChannels[0]}, ${rgbChannels[1]}, ${rgbChannels[2]})")
            guiGraphics.text(font, badgeText, swatchX + swatchSize + 6 + labelWidth, textY, 0xFFFFFFFF.toInt())
        }

        override fun updateWidgetNarration(output: NarrationElementOutput) {
            defaultButtonNarrationText(output)
        }

        private fun currentRgb24(): Int =
            ((rgbChannels[0] and 0xFF) shl 16) or
                ((rgbChannels[1] and 0xFF) shl 8) or
                (rgbChannels[2] and 0xFF)
    }

    private class NumericSettingSlider(
        private val label: String,
        private val isInt: Boolean,
        private val minVal: Double,
        private val maxVal: Double,
        private val step: Double,
        initialVal: Double,
        initialModified: Boolean,
        width: Int = 150,
        private val onValueChanged: (String) -> Boolean,
    ) : AbstractSliderButton(
        0,
        0,
        width,
        20,
        Component.empty(),
        ((initialVal - minVal) / (maxVal - minVal)).coerceIn(0.0, 1.0),
    ) {
        private var modified: Boolean = initialModified

        init {
            updateMessage()
        }

        private fun snappedValue(): Double {
            val raw = minVal + this.value * (maxVal - minVal)
            val s = if (step > 1e-6) round((raw - minVal) / step) * step + minVal else raw
            return s.coerceIn(minVal, maxVal)
        }

        private fun formatValue(v: Double): String = if (isInt) {
            v.roundToInt().toString()
        } else {
            formatFloat(v)
        }

        override fun updateMessage() {
            val formatted = formatValue(snappedValue())
            this.message = if (modified) {
                Component.literal("* $label: $formatted").withStyle(ChatFormatting.YELLOW)
            } else {
                Component.literal("$label: $formatted")
            }
        }

        override fun applyValue() {
            modified = onValueChanged(formatValue(snappedValue()))
            updateMessage()
        }
    }

    companion object {
        private val PROFILE_GSON = GsonBuilder().setPrettyPrinting().create()

        /**
         * Builds fallback Low / Medium / High / Ultra presets for shader packs that declare
         * `[settings.*]` without explicit `[presets.*]` entries.
         */
        internal fun buildDefaultQualityPresets(
            settings: List<PackSettingDefinition>,
            tierIndex: Int,
        ): List<PackPresetDefinition> {
            data class ProfileSpec(val id: String, val label: String, val order: Int, val factor: Double)
            val profiles = listOf(
                ProfileSpec("low", "Low", 0, 0.25),
                ProfileSpec("medium", "Medium", 1, 0.50),
                ProfileSpec("high", "High", 2, 0.75),
                ProfileSpec("ultra", "Ultra", 3, 1.00),
            )
            return profiles.map { spec ->
                val values = LinkedHashMap<String, String>()
                for (setting in settings) {
                    val v = when (setting.type) {
                        SettingType.BOOL -> {
                            if (spec.id == "low") "false" else setting.defaultValue
                        }
                        SettingType.ENUM -> {
                            if (setting.options.isEmpty()) {
                                setting.defaultValue
                            } else {
                                val idx = ((setting.options.size - 1) * spec.factor).roundToInt()
                                    .coerceIn(0, setting.options.lastIndex)
                                setting.options[idx]
                            }
                        }
                        SettingType.INT -> {
                            val min = setting.min
                            val max = setting.max
                            if (min != null && max != null && max > min) {
                                val raw = min + (max - min) * spec.factor
                                val step = setting.step ?: 1.0
                                val snapped = if (step > 1e-6) round((raw - min) / step) * step + min else raw
                                snapped.coerceIn(min, max).roundToInt().toString()
                            } else {
                                setting.defaultValue
                            }
                        }
                        SettingType.FLOAT -> {
                            val min = setting.min
                            val max = setting.max
                            if (min != null && max != null && max > min) {
                                val raw = min + (max - min) * spec.factor
                                val step = setting.step ?: 0.05
                                val snapped = if (step > 1e-6) round((raw - min) / step) * step + min else raw
                                formatFloat(snapped.coerceIn(min, max))
                            } else {
                                setting.defaultValue
                            }
                        }
                        SettingType.COLOR_RGB -> setting.defaultValue
                    }
                    values[setting.key] = v
                }
                PackPresetDefinition(
                    id = spec.id,
                    label = spec.label,
                    order = spec.order,
                    minTier = tierIndex,
                    maxTier = tierIndex,
                    values = values,
                )
            }
        }

        private fun formatFloat(v: Double): String =
            String.format(Locale.ROOT, "%.4f", v).trimEnd('0').trimEnd('.').ifEmpty { "0" }

        private fun resolveCategory(setting: PackSettingDefinition): String {
            val explicit = setting.category?.trim()
            if (!explicit.isNullOrEmpty()) return explicit
            val probe = "${setting.key} ${setting.label}".lowercase(Locale.ROOT)
            return when {
                probe.containsAny("water", "wave", "wet", "puddle", "caustic", "underwater") -> "Water"
                probe.containsAny("fog", "sky", "cloud", "godray", "wind", "atmo", "weather", "volumetric") -> "Atmosphere"
                probe.containsAny("sun", "torch", "light", "gi_", "bounce", "shadow", "emissive", "ambient", "moon") -> "Lighting"
                probe.containsAny("tone", "bloom", "exposure", "grade", "color", "saturation", "contrast", "vignette", "post", "gamma") -> "Post"
                else -> "General"
            }
        }

        private fun String.containsAny(vararg needles: String): Boolean =
            needles.any { this.contains(it) }
    }
}
