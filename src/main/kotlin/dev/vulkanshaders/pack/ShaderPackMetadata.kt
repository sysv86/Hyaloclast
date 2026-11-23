package dev.vulkanshaders.pack

import dev.vulkanshaders.state.WorldStateCollector
import java.lang.Float.floatToRawIntBits
import java.nio.file.Path

enum class SettingType {
    BOOL,
    INT,
    FLOAT,
    ENUM,
    COLOR_RGB,
}

enum class PipelineGraphMode {
    EXTEND,
    CUSTOM,
}

data class PackPipelineConfig(
    val mode: PipelineGraphMode = PipelineGraphMode.EXTEND,
    val gbufferMrtCount: Int = 4,
    val modules: List<String> = emptyList(),
)

data class PackTargetDefinition(
    val name: String,
    val format: String = "rgba16f",
    val scale: Float = 1.0f,
    val mipLevels: Int = 1,
    val arrayLayers: Int = 1,
    val clearColor: FloatArray? = null,
    val temporal: Boolean = false,
    val gbufferAttachment: Int? = null,
)

data class PackBufferDefinition(
    val name: String,
    val sizeBytes: Long = 65536L,
    val clearEachFrame: Boolean = false,
    val persistent: Boolean = true,
)

data class PackPassDefinition(
    val id: String,
    val kind: String = "compute",
    val shader: String,
    val minTier: Int = 0,
    val maxTier: Int = 3,
    val reads: List<String> = emptyList(),
    val writes: List<String> = emptyList(),
    val before: String? = null,
    val after: String? = null,
    val replaces: String? = null,
    val bindings: Map<String, String> = emptyMap(),
    val scale: Float = 1.0f,
    val enabled: Boolean = true,
)

data class PackSettingDefinition(
    val key: String,
    val label: String,
    val type: SettingType,
    val defaultValue: String,
    val min: Double? = null,
    val max: Double? = null,
    val step: Double? = null,
    val options: List<String> = emptyList(),
    val optionLabels: List<String> = emptyList(),
    val runtime: Boolean = true,
    val category: String? = null,
    val description: String? = null,
    val minTier: Int = 0,
    val maxTier: Int = 3,
) {
    fun appliesToTier(tierIndex: Int): Boolean = tierIndex in minTier..maxTier

    fun resolveRawU32Bits(overrideRaw: String?): Int {
        val raw = overrideRaw ?: defaultValue
        return when (type) {
            SettingType.BOOL -> if (raw.equals("true", ignoreCase = true) || raw == "1") 1 else 0
            SettingType.INT -> raw.toIntOrNull() ?: defaultValue.toIntOrNull() ?: 0
            SettingType.ENUM -> options.indexOfFirst { it.equals(raw, ignoreCase = true) }.coerceAtLeast(0)
            SettingType.FLOAT -> {
                val f = raw.toFloatOrNull() ?: defaultValue.toFloatOrNull() ?: 0.0f
                floatToRawIntBits(f)
            }
            SettingType.COLOR_RGB -> {
                val rgb24 = WorldStateCollector.parseColorRgb(raw)
                    ?: WorldStateCollector.parseColorRgb(defaultValue)
                val f = if (rgb24 != null) {
                    WorldStateCollector.packRgbAsFloat(rgb24)
                } else {
                    raw.toFloatOrNull() ?: defaultValue.toFloatOrNull() ?: 0.0f
                }
                floatToRawIntBits(f)
            }
        }
    }
}

data class PackPresetDefinition(
    val id: String,
    val label: String,
    val description: String? = null,
    val order: Int = 0,
    val minTier: Int = 0,
    val maxTier: Int = 3,
    val values: Map<String, String> = emptyMap(),
) {
    fun appliesToTier(tierIndex: Int): Boolean = tierIndex in minTier..maxTier
}

data class PackResourceDefinition(
    val name: String,
    val path: String,
    val format: String,
)

data class ShaderPackManifest(
    val id: String,
    val name: String,
    val version: String,
    val author: String,
    val description: String,
    val extendsPackId: String? = null,
    val supportedTiers: List<Int>,
    val settings: List<PackSettingDefinition>,
    val presets: List<PackPresetDefinition> = emptyList(),
    val resources: List<PackResourceDefinition>,
    val hooks: Map<String, String>,
    val pipeline: PackPipelineConfig = PackPipelineConfig(),
    val targets: List<PackTargetDefinition> = emptyList(),
    val buffers: List<PackBufferDefinition> = emptyList(),
    val passes: List<PackPassDefinition> = emptyList(),
) {
    /**
     * Validates manifest invariants (`id`, `tiers`, `hooks`, `settings` bounds/options,
     * `VSP Graph v2` `[pipeline]`/`[targets]`/`[buffers]`/`[[passes]]`, and
     * `MAX_RUNTIME_PACK_PARAMS = 48`). Throws [IllegalArgumentException] on invalid declarations.
     */
    fun validateOrThrow(allowInheritedPresetKeys: Boolean = false) {
        require(id.isNotBlank()) { "pack.id must not be blank" }
        require(id.all { it.isLetterOrDigit() || it == '-' || it == '_' }) {
            "pack.id '$id' may only contain ASCII alphanumeric, '-', or '_' characters"
        }
        require(name.isNotBlank()) { "pack.name must not be blank for pack '$id'" }
        require(supportedTiers.isNotEmpty()) { "capabilities.tiers must list at least one tier (0..3)" }
        for (tier in supportedTiers) {
            require(tier in 0..3) { "Unsupported tier index $tier in pack '$id' (expected 0..3)" }
        }
        for ((stage, hookPath) in hooks) {
            require(stage in VALID_HOOK_STAGES) { "Unknown hook stage '$stage' in pack '$id'" }
            require(hookPath.isNotBlank() && !hookPath.contains("..")) {
                "Invalid hook path '$hookPath' for stage '$stage' in pack '$id'"
            }
        }
        require(pipeline.gbufferMrtCount in 4..8) {
            "pipeline.gbuffer_MRT_count in pack '$id' must be in 4..8 (got ${pipeline.gbufferMrtCount})"
        }
        for (modPath in pipeline.modules) {
            require(modPath.isNotBlank() && !modPath.contains("..")) {
                "Invalid pipeline.modules path '$modPath' in pack '$id'"
            }
        }
        for (tgt in targets) {
            require(tgt.name.isNotBlank()) { "Target name must not be blank in pack '$id'" }
            require(tgt.scale in 0.125f..4.0f) {
                "Target '${tgt.name}' scale (${tgt.scale}) must be in 0.125..4.0 in pack '$id'"
            }
            if (tgt.gbufferAttachment != null) {
                require(tgt.gbufferAttachment in 4..7) {
                    "Target '${tgt.name}' gbuffer_attachment (${tgt.gbufferAttachment}) must be in 4..7"
                }
            }
        }
        for (buf in buffers) {
            require(buf.name.isNotBlank()) { "Buffer name must not be blank in pack '$id'" }
            require(buf.sizeBytes in 64L..(256L * 1024L * 1024L)) {
                "Buffer '${buf.name}' size_bytes (${buf.sizeBytes}) out of bounds in pack '$id'"
            }
        }
        val seenPassIds = HashSet<String>()
        for (pass in passes) {
            require(pass.id.isNotBlank()) { "Pass id must not be blank in pack '$id'" }
            require(seenPassIds.add(pass.id)) { "Duplicate pass id '${pass.id}' in pack '$id'" }
            require(pass.shader.isNotBlank() && !pass.shader.contains("..")) {
                "Invalid shader path '${pass.shader}' in pass '${pass.id}' of pack '$id'"
            }
            require(pass.minTier in 0..3 && pass.maxTier in 0..3 && pass.minTier <= pass.maxTier) {
                "Invalid tier range ${pass.minTier}..${pass.maxTier} in pass '${pass.id}' of pack '$id'"
            }
        }
        var runtimeCount = 0
        for (s in settings) {
            if (s.type == SettingType.ENUM) {
                require(s.options.isNotEmpty()) {
                    "Enum setting '${s.key}' in pack '$id' must define a non-empty options list"
                }
            }
            if (s.min != null && s.max != null) {
                require(s.min <= s.max) {
                    "Setting '${s.key}' in pack '$id' has min (${s.min}) > max (${s.max})"
                }
            }
            require(s.minTier in 0..3 && s.maxTier in 0..3 && s.minTier <= s.maxTier) {
                "Setting '${s.key}' in pack '$id' has invalid tier range ${s.minTier}..${s.maxTier}"
            }
            if (s.runtime) {
                runtimeCount++
            }
        }
        require(runtimeCount <= 40) {
            "Pack '$id' declares $runtimeCount runtime settings, exceeding WorldUBO capacity (40; slots 40..47 reserved for biome telemetry)"
        }

        val settingsByKey = settings.associateBy { it.key }
        val seenPresetIds = HashSet<String>()
        for (preset in presets) {
            val pid = preset.id.trim()
            require(pid.isNotEmpty() && pid.all { isAsciiIdChar(it) }) {
                "Invalid preset identifier '${preset.id}' in pack '$id'"
            }
            require(!pid.equals("custom", ignoreCase = true)) {
                "Preset id 'custom' is reserved in pack '$id'"
            }
            require(seenPresetIds.add(pid)) {
                "Duplicate preset id '${preset.id}' in pack '$id'"
            }
            require(preset.label.isNotBlank()) {
                "Preset '${preset.id}' label must not be blank in pack '$id'"
            }
            require(preset.minTier in 0..3 && preset.maxTier in 0..3 && preset.minTier <= preset.maxTier) {
                "Preset '${preset.id}' in pack '$id' has invalid tier range ${preset.minTier}..${preset.maxTier}"
            }
            for ((settingKey, rawVal) in preset.values) {
                val setting = settingsByKey[settingKey]
                if (setting == null) {
                    if (allowInheritedPresetKeys && !extendsPackId.isNullOrBlank()) {
                        continue
                    }
                    throw IllegalArgumentException(
                        "Preset '${preset.id}' in pack '$id' references unknown setting '$settingKey'",
                    )
                }
                validatePresetValue(id, preset.id, setting, rawVal)
            }
        }
    }

    // PackManifest::build_parameter_layout in vsp-compiler assigns runtime slots by key.
    internal val runtimeSettings: List<PackSettingDefinition> = settings
        .filter { it.runtime }
        .sortedBy { it.key }

    /**
     * Computes `(constant_id, raw_u32_bits)` specialization constant overrides for all
     * compile-time settings (`runtime == false`), matching `PackManifest::FIRST_SPEC_CONSTANT_ID = 100`
     * sorted by setting key in `vsp-compiler`.
     */
    fun buildSpecializationOverrides(
        userOverrides: Map<String, String>,
        activeTierIndex: Int? = null,
    ): List<Pair<Int, Int>> {
        val sortedCompileTime = settings
            .filter { !it.runtime }
            .sortedBy { it.key }
        val out = ArrayList<Pair<Int, Int>>(sortedCompileTime.size)
        var constId = FIRST_SPEC_CONSTANT_ID
        for (setting in sortedCompileTime) {
            val effectiveOverride = if (activeTierIndex == null || setting.appliesToTier(activeTierIndex)) {
                userOverrides[setting.key]
            } else {
                null
            }
            val bits = setting.resolveRawU32Bits(effectiveOverride)
            out.add(constId to bits)
            constId++
        }
        return out
    }

    /**
     * Returns a new [ShaderPackManifest] that inherits any unoverridden hooks, settings,
     * resources, targets, buffers, and passes from [parent] (`extends = "<parent_id>"`).
     */
    fun inheritFrom(parent: ShaderPackManifest): ShaderPackManifest {
        val mergedHooks = LinkedHashMap<String, String>(parent.hooks)
        mergedHooks.putAll(this.hooks)

        val parentSettingsByKey = parent.settings.associateBy { it.key }
        val ownSettingKeys = this.settings.mapTo(HashSet()) { it.key }
        val mergedSettings = buildList {
            for (s in parent.settings) {
                if (s.key !in ownSettingKeys) add(s)
            }
            for (s in this@ShaderPackManifest.settings) {
                val inherited = parentSettingsByKey[s.key]
                add(
                    if (inherited != null && (s.category == null || s.description == null)) {
                        s.copy(
                            category = s.category ?: inherited.category,
                            description = s.description ?: inherited.description,
                        )
                    } else {
                        s
                    },
                )
            }
        }

        val ownResNames = this.resources.mapTo(HashSet()) { it.name }
        val mergedResources = buildList {
            for (r in parent.resources) {
                if (r.name !in ownResNames) add(r)
            }
            addAll(this@ShaderPackManifest.resources)
        }

        val ownTargetNames = this.targets.mapTo(HashSet()) { it.name }
        val mergedTargets = buildList {
            for (t in parent.targets) {
                if (t.name !in ownTargetNames) add(t)
            }
            addAll(this@ShaderPackManifest.targets)
        }

        val ownBufferNames = this.buffers.mapTo(HashSet()) { it.name }
        val mergedBuffers = buildList {
            for (b in parent.buffers) {
                if (b.name !in ownBufferNames) add(b)
            }
            addAll(this@ShaderPackManifest.buffers)
        }

        val ownPassIds = this.passes.mapTo(HashSet()) { it.id }
        val mergedPasses = buildList {
            for (p in parent.passes) {
                if (p.id !in ownPassIds) add(p)
            }
            addAll(this@ShaderPackManifest.passes)
        }

        val ownPresetIds = this.presets.mapTo(HashSet()) { it.id }
        val mergedPresets = buildList {
            for (p in parent.presets) {
                if (p.id !in ownPresetIds) add(p)
            }
            addAll(this@ShaderPackManifest.presets)
        }.withIndex()
            .sortedWith(compareBy<IndexedValue<PackPresetDefinition>>({ it.value.order }, { it.index }))
            .map { it.value }

        val merged = copy(
            settings = mergedSettings,
            presets = mergedPresets,
            resources = mergedResources,
            hooks = mergedHooks,
            targets = mergedTargets,
            buffers = mergedBuffers,
            passes = mergedPasses,
        )
        merged.validateOrThrow(allowInheritedPresetKeys = false)
        return merged
    }

    companion object {
        const val FIRST_SPEC_CONSTANT_ID: Int = 100

        val VALID_HOOK_STAGES: Set<String> = setOf(
            "vertex_displace",
            "gbuffer_pack",
            "surface",
            "emissive_sample",
            "restir_reservoir",
            "direct_light",
            "block_light",
            "indirect_gi",
            "water_waves",
            "water_surface",
            "sky",
            "clouds",
            "volumetric_media",
            "fog",
            "ao_filter",
            "post_grade",
        )

        fun fromToml(tomlText: String, fallbackId: String): ShaderPackManifest {
            val sections = VspTomlParser.parse(tomlText)
            val pack = sections["pack"].orEmpty()
            val caps = sections["capabilities"].orEmpty()
            val hooksSec = sections["hooks"].orEmpty()
            val pipeSec = sections["pipeline"].orEmpty()

            val id = str(pack["id"]) ?: fallbackId
            val name = str(pack["name"]) ?: id
            val version = str(pack["version"]) ?: "0.1.0"
            val author = str(pack["author"]) ?: "Unknown"
            val description = str(pack["description"]) ?: ""
            val extendsPackId = str(pack["extends"])?.takeIf { it.isNotBlank() }

            val tiers = (caps["tiers"] as? VspTomlParser.Value.Arr)?.items
                ?.mapNotNull { (it as? VspTomlParser.Value.Num)?.value?.toInt() }
                ?.takeIf { it.isNotEmpty() }
                ?: listOf(0)

            val pipeMode = when (str(pipeSec["mode"])?.lowercase()) {
                "custom" -> PipelineGraphMode.CUSTOM
                else -> PipelineGraphMode.EXTEND
            }
            val gbufferMrtCount = ((pipeSec["gbuffer_MRT_count"] ?: pipeSec["gbuffer_mrt_count"]) as? VspTomlParser.Value.Num)
                ?.value?.toInt() ?: 4
            val pipeModules = (pipeSec["modules"] as? VspTomlParser.Value.Arr)?.items
                ?.mapNotNull { str(it) }
                ?: emptyList()
            val pipelineConfig = PackPipelineConfig(
                mode = pipeMode,
                gbufferMrtCount = gbufferMrtCount,
                modules = pipeModules,
            )

            val settings = mutableListOf<PackSettingDefinition>()
            val presetSections = linkedMapOf<String, Map<String, VspTomlParser.Value>>()
            val presetValueSections = linkedMapOf<String, MutableMap<String, String>>()
            val resources = mutableListOf<PackResourceDefinition>()
            val targets = mutableListOf<PackTargetDefinition>()
            val buffers = mutableListOf<PackBufferDefinition>()
            val passSections = linkedMapOf<Int, Map<String, VspTomlParser.Value>>()
            val passBindingsSections = linkedMapOf<Int, Map<String, String>>()

            for ((sectionName, fields) in sections) {
                when {
                    sectionName.startsWith("settings.") -> {
                        val key = sectionName.removePrefix("settings.")
                        val typeStr = (str(fields["type"]) ?: "float").lowercase()
                        val type = when (typeStr) {
                            "bool" -> SettingType.BOOL
                            "int" -> SettingType.INT
                            "enum" -> SettingType.ENUM
                            "color_rgb" -> SettingType.COLOR_RGB
                            else -> SettingType.FLOAT
                        }
                        val defaultVal = when (val d = fields["default"]) {
                            is VspTomlParser.Value.Str -> d.value
                            is VspTomlParser.Value.Bool -> d.value.toString()
                            is VspTomlParser.Value.Num -> if (d.isFloat) d.value.toString() else d.value.toInt().toString()
                            is VspTomlParser.Value.Arr -> d.items.mapNotNull { item ->
                                when (item) {
                                    is VspTomlParser.Value.Num -> if (item.isFloat) item.value.toString() else item.value.toInt().toString()
                                    is VspTomlParser.Value.Str -> item.value
                                    else -> null
                                }
                            }.joinToString(", ")
                            else -> "0"
                        }
                        val opts = (fields["options"] as? VspTomlParser.Value.Arr)?.items
                            ?.mapNotNull { str(it) }
                            ?: emptyList()
                        val category = (str(fields["category"]) ?: str(fields["group"]))
                            ?.trim()
                            ?.takeIf { it.isNotEmpty() }
                        val settingDescription = (str(fields["description"]) ?: str(fields["tooltip"]))
                            ?.trim()
                            ?.takeIf { it.isNotEmpty() }
                        settings.add(
                            PackSettingDefinition(
                                key = key,
                                label = str(fields["label"]) ?: key,
                                type = type,
                                defaultValue = defaultVal,
                                min = (fields["min"] as? VspTomlParser.Value.Num)?.value,
                                max = (fields["max"] as? VspTomlParser.Value.Num)?.value,
                                step = (fields["step"] as? VspTomlParser.Value.Num)?.value,
                                options = opts,
                                runtime = (fields["runtime"] as? VspTomlParser.Value.Bool)?.value ?: true,
                                category = category,
                                description = settingDescription,
                                minTier = (fields["min_tier"] as? VspTomlParser.Value.Num)?.value?.toInt() ?: 0,
                                maxTier = (fields["max_tier"] as? VspTomlParser.Value.Num)?.value?.toInt() ?: 3,
                            ),
                        )
                    }
                    sectionName.startsWith("presets.") -> {
                        val rest = sectionName.removePrefix("presets.")
                        if (rest.endsWith(".values")) {
                            val presetId = rest.removeSuffix(".values")
                            if (presetId.isNotEmpty() && !presetId.contains('.')) {
                                val valueMap = presetValueSections.getOrPut(presetId) { linkedMapOf() }
                                for ((k, v) in fields) {
                                    val serialized = settingValueToString(v)
                                    if (serialized != null) {
                                        valueMap[k] = serialized
                                    }
                                }
                            }
                        } else if (!rest.contains('.')) {
                            presetSections[rest] = fields
                        }
                    }
                    sectionName.startsWith("resources.") -> {
                        val resName = sectionName.removePrefix("resources.")
                        resources.add(
                            PackResourceDefinition(
                                name = resName,
                                path = str(fields["path"]) ?: "",
                                format = str(fields["format"]) ?: "RGBA8_UNORM",
                            ),
                        )
                    }
                    sectionName.startsWith("targets.") -> {
                        val tgtName = sectionName.removePrefix("targets.")
                        targets.add(
                            PackTargetDefinition(
                                name = tgtName,
                                format = str(fields["format"]) ?: "rgba16f",
                                scale = (fields["scale"] as? VspTomlParser.Value.Num)?.value?.toFloat() ?: 1.0f,
                                mipLevels = (fields["mip_levels"] as? VspTomlParser.Value.Num)?.value?.toInt() ?: 1,
                                arrayLayers = (fields["array_layers"] as? VspTomlParser.Value.Num)?.value?.toInt() ?: 1,
                                clearColor = (fields["clear_color"] as? VspTomlParser.Value.Arr)?.items
                                    ?.mapNotNull { (it as? VspTomlParser.Value.Num)?.value?.toFloat() }
                                    ?.takeIf { it.size == 4 }
                                    ?.toFloatArray(),
                                temporal = (fields["temporal"] as? VspTomlParser.Value.Bool)?.value ?: false,
                                gbufferAttachment = (fields["gbuffer_attachment"] as? VspTomlParser.Value.Num)?.value?.toInt(),
                            ),
                        )
                    }
                    sectionName.startsWith("buffers.") -> {
                        val bufName = sectionName.removePrefix("buffers.")
                        buffers.add(
                            PackBufferDefinition(
                                name = bufName,
                                sizeBytes = (fields["size_bytes"] as? VspTomlParser.Value.Num)?.value?.toLong() ?: 65536L,
                                clearEachFrame = (fields["clear_each_frame"] as? VspTomlParser.Value.Bool)?.value ?: false,
                                persistent = (fields["persistent"] as? VspTomlParser.Value.Bool)?.value ?: true,
                            ),
                        )
                    }
                    sectionName.startsWith("passes.") -> {
                        val rest = sectionName.removePrefix("passes.")
                        if (rest.endsWith(".bindings")) {
                            val idx = rest.removeSuffix(".bindings").toIntOrNull()
                            if (idx != null) {
                                passBindingsSections[idx] = fields.mapValues { (_, v) -> str(v) ?: "" }
                                    .filterValues { it.isNotEmpty() }
                            }
                        } else {
                            val idx = rest.toIntOrNull()
                            if (idx != null) {
                                passSections[idx] = fields
                            }
                        }
                    }
                }
            }

            val passes = passSections.entries.sortedBy { it.key }.mapNotNull { (idx, fields) ->
                val passId = str(fields["id"]) ?: return@mapNotNull null
                val shader = str(fields["shader"]) ?: return@mapNotNull null
                val inlineBindings = (fields["bindings"] as? VspTomlParser.Value.Tbl)?.entries
                    ?.mapValues { (_, v) -> str(v) ?: "" }
                    ?.filterValues { it.isNotEmpty() }
                    .orEmpty()
                val mergedBindings = LinkedHashMap<String, String>()
                mergedBindings.putAll(inlineBindings)
                mergedBindings.putAll(passBindingsSections[idx].orEmpty())

                PackPassDefinition(
                    id = passId,
                    kind = str(fields["kind"]) ?: "compute",
                    shader = shader,
                    minTier = (fields["min_tier"] as? VspTomlParser.Value.Num)?.value?.toInt() ?: 0,
                    maxTier = (fields["max_tier"] as? VspTomlParser.Value.Num)?.value?.toInt() ?: 3,
                    reads = (fields["reads"] as? VspTomlParser.Value.Arr)?.items?.mapNotNull { str(it) }.orEmpty(),
                    writes = (fields["writes"] as? VspTomlParser.Value.Arr)?.items?.mapNotNull { str(it) }.orEmpty(),
                    before = str(fields["before"]),
                    after = str(fields["after"]),
                    replaces = str(fields["replaces"]),
                    bindings = mergedBindings,
                    scale = (fields["scale"] as? VspTomlParser.Value.Num)?.value?.toFloat() ?: 1.0f,
                    enabled = (fields["enabled"] as? VspTomlParser.Value.Bool)?.value ?: true,
                )
            }

            val hooks = linkedMapOf<String, String>()
            for ((k, v) in hooksSec) {
                val path = str(v) ?: continue
                hooks[k] = path
            }

            val presets = presetSections.entries.mapIndexed { declIdx, (presetId, fields) ->
                val mergedValues = LinkedHashMap<String, String>()
                val inlineValues = (fields["values"] as? VspTomlParser.Value.Tbl)?.entries.orEmpty()
                for ((k, v) in inlineValues) {
                    val serialized = settingValueToString(v)
                    if (serialized != null) {
                        mergedValues[k] = serialized
                    }
                }
                mergedValues.putAll(presetValueSections[presetId].orEmpty())
                val presetDesc = str(fields["description"])?.trim()?.takeIf { it.isNotEmpty() }
                declIdx to PackPresetDefinition(
                    id = presetId,
                    label = str(fields["label"]) ?: "",
                    description = presetDesc,
                    order = (fields["order"] as? VspTomlParser.Value.Num)?.value?.toInt() ?: 0,
                    minTier = (fields["min_tier"] as? VspTomlParser.Value.Num)?.value?.toInt() ?: 0,
                    maxTier = (fields["max_tier"] as? VspTomlParser.Value.Num)?.value?.toInt() ?: 3,
                    values = mergedValues,
                )
            }.sortedWith(compareBy<Pair<Int, PackPresetDefinition>>({ it.second.order }, { it.first }))
                .map { it.second }

            val manifest = ShaderPackManifest(
                id = id,
                name = name,
                version = version,
                author = author,
                description = description,
                extendsPackId = extendsPackId,
                supportedTiers = tiers,
                settings = settings,
                presets = presets,
                resources = resources,
                hooks = hooks,
                pipeline = pipelineConfig,
                targets = targets,
                buffers = buffers,
                passes = passes,
            )
            manifest.validateOrThrow(allowInheritedPresetKeys = !extendsPackId.isNullOrBlank())
            return manifest
        }

        private fun isAsciiIdChar(c: Char): Boolean =
            c in 'a'..'z' || c in 'A'..'Z' || c in '0'..'9' || c == '-' || c == '_'

        private fun settingValueToString(v: VspTomlParser.Value?): String? = when (v) {
            is VspTomlParser.Value.Str -> v.value
            is VspTomlParser.Value.Bool -> v.value.toString()
            is VspTomlParser.Value.Num -> if (v.isFloat) v.value.toString() else v.value.toLong().toString()
            is VspTomlParser.Value.Arr -> v.items.mapNotNull { item ->
                when (item) {
                    is VspTomlParser.Value.Num -> if (item.isFloat) item.value.toString() else item.value.toLong().toString()
                    is VspTomlParser.Value.Str -> item.value
                    else -> null
                }
            }.joinToString(", ")
            else -> null
        }

        private fun validatePresetValue(
            packId: String,
            presetId: String,
            setting: PackSettingDefinition,
            rawVal: String,
        ) {
            val trimmed = rawVal.trim()
            when (setting.type) {
                SettingType.BOOL -> {
                    require(trimmed.equals("true", ignoreCase = true) || trimmed.equals("false", ignoreCase = true)) {
                        "Preset '$presetId' in pack '$packId' has invalid bool value '$rawVal' for '${setting.key}'"
                    }
                }
                SettingType.INT -> {
                    val parsed = trimmed.toIntOrNull()
                        ?: throw IllegalArgumentException(
                            "Preset '$presetId' in pack '$packId' has invalid int value '$rawVal' for '${setting.key}'",
                        )
                    if (setting.min != null) {
                        require(parsed.toDouble() >= setting.min - 1e-6) {
                            "Preset '$presetId' in pack '$packId' value $parsed for '${setting.key}' is below min (${setting.min})"
                        }
                    }
                    if (setting.max != null) {
                        require(parsed.toDouble() <= setting.max + 1e-6) {
                            "Preset '$presetId' in pack '$packId' value $parsed for '${setting.key}' exceeds max (${setting.max})"
                        }
                    }
                }
                SettingType.FLOAT -> {
                    val parsed = trimmed.toDoubleOrNull()
                    require(parsed != null && parsed.isFinite()) {
                        "Preset '$presetId' in pack '$packId' has invalid float value '$rawVal' for '${setting.key}'"
                    }
                    if (setting.min != null) {
                        require(parsed >= setting.min - 1e-6) {
                            "Preset '$presetId' in pack '$packId' value $parsed for '${setting.key}' is below min (${setting.min})"
                        }
                    }
                    if (setting.max != null) {
                        require(parsed <= setting.max + 1e-6) {
                            "Preset '$presetId' in pack '$packId' value $parsed for '${setting.key}' exceeds max (${setting.max})"
                        }
                    }
                }
                SettingType.ENUM -> {
                    require(setting.options.any { it.equals(trimmed, ignoreCase = true) }) {
                        "Preset '$presetId' in pack '$packId' value '$rawVal' for enum '${setting.key}' is not in options ${setting.options}"
                    }
                }
                SettingType.COLOR_RGB -> {
                    require(WorldStateCollector.parseColorRgb(trimmed) != null) {
                        "Preset '$presetId' in pack '$packId' has invalid color_rgb value '$rawVal' for '${setting.key}'"
                    }
                }
            }
        }

        private fun str(v: VspTomlParser.Value?): String? = (v as? VspTomlParser.Value.Str)?.value
    }
}

data class LoadedShaderPack(
    val manifest: ShaderPackManifest,
    val rootPath: Path,
    val isBuiltin: Boolean,
    val resolvedHooks: Map<String, String>,
)
