package dev.vulkanshaders.pipeline

import com.mojang.renderpearl.api.pipeline.BlendFunction
import com.mojang.blaze3d.systems.RenderSystem
import com.mojang.blaze3d.vertex.DefaultVertexFormat
import com.mojang.renderpearl.api.GpuFormat
import com.mojang.renderpearl.api.pipeline.BindGroupLayout
import com.mojang.renderpearl.api.vertex.VertexFormat
import com.mojang.renderpearl.api.pipeline.ColorTargetState
import com.mojang.renderpearl.api.pipeline.CompareOp
import com.mojang.renderpearl.api.pipeline.DepthStencilState
import com.mojang.renderpearl.api.pipeline.PolygonMode
import com.mojang.renderpearl.api.pipeline.PrimitiveTopology
import com.mojang.renderpearl.api.pipeline.RenderPipeline
import com.mojang.renderpearl.api.pipeline.ShaderType
import com.mojang.renderpearl.api.pipeline.UniformType
import dev.vulkanshaders.VulkanShaders
import dev.vulkanshaders.backend.NativeCoreBridge
import dev.vulkanshaders.bridge.VulkanShadersTurboExtension
import dev.vulkanshaders.pack.LoadedShaderPack
import dev.vulkanshaders.pack.SettingType
import dev.vulkanshaders.pack.ShaderPackManager
import dev.vulkanshaders.state.WorldStateCollector
import net.minecraft.client.Minecraft
import net.minecraft.client.renderer.BindGroupLayouts
import net.minecraft.client.renderer.RenderPipelines
import net.minecraft.client.renderer.chunk.ChunkSectionLayer
import net.minecraft.resources.Identifier
import java.nio.charset.StandardCharsets
import java.nio.file.Files
import java.util.Locale
import java.util.Optional
import java.util.concurrent.ConcurrentHashMap
import java.util.function.UnaryOperator

/**
 * Registers 40-byte (`IGNIS_TERRAIN_40B`) 4-target MRT G-Buffer `RenderPipeline` instances for
 * **VulkanShaders (Hyaloclast / IGNIS)** and provides world pipeline replacement across terrain
 * (`SOLID`, `CUTOUT`, `TRANSLUCENT`, `TRIPWIRE`), entities, block entities, particles, water,
 * sky, and clouds.
 *
 * Uses a comment/string-aware GLSL lexer (`GlslHookLexer`) for `.vsl` hook extraction and
 * automatic helper namespace isolation (`vs_ns_<stage>_<ident>`) when multiple hooks in the same
 * shader stage define colliding helper identifiers.
 */
object GbufferPipelineRegistry {
    private val SHADER_ID = Identifier.fromNamespaceAndPath("vulkanshaders", "core/gbuffer_terrain_region")
    private val ENTITY_SHADER_ID = Identifier.fromNamespaceAndPath("vulkanshaders", "core/gbuffer_entity")
    private val PARTICLE_SHADER_ID = Identifier.fromNamespaceAndPath("vulkanshaders", "core/gbuffer_particle")
    private val DH_TERRAIN_SHADER_ID = Identifier.fromNamespaceAndPath("vulkanshaders", "core/gbuffer_dh_terrain")

    private fun dhVertexFormat(extended: Boolean): VertexFormat {
        val builder = VertexFormat.builder(0)
            .addAttribute("vPositionAndMeta", GpuFormat.RGBA16_UINT)
            .addAttribute("vColor", GpuFormat.RGBA8_UNORM)
            .addAttribute("materialNormalTile", GpuFormat.RGBA8_UINT)
        if (extended) {
            builder.addAttribute("offsetData", GpuFormat.RGBA8_UINT)
                .addAttribute("uvNormal", GpuFormat.RGBA8_UINT)
        }
        return builder.build()
    }

    /** Distant Horizons 3.x terrain sampler and uniform layout matching RenderPipelineBuilderWrapper. */
    @JvmField
    val DH_TERRAIN_LAYOUT: BindGroupLayout = BindGroupLayout.builder()
        .withUniform("uLightMap", UniformType.COMBINED_IMAGE_SAMPLER)
        .withUniform("uBlockAtlas", UniformType.COMBINED_IMAGE_SAMPLER)
        .withUniform("vertUniqueUniformBlock", UniformType.UNIFORM_BUFFER)
        .withUniform("vertSharedUniformBlock", UniformType.UNIFORM_BUFFER)
        .withUniform("fragUniformBlock", UniformType.UNIFORM_BUFFER)
        .build()

    /**
     * Extra uniform block consumed by the bundled G-Buffer shaders for true motion vectors.
     * `prevViewProj` (64B) + `cameraDeltaTime` (16B) + `pbrParams` (16B) + `sunDirRain` (16B) +
     * `heldLightData` (16B) = 128 bytes std140.
     * Populated by `GbufferTargetManager` from `WorldStateCollector` every world pass.
     */
    @JvmField
    val VS_MOTION_LAYOUT: BindGroupLayout = BindGroupLayout.builder()
        .withUniform("VsMotion", UniformType.UNIFORM_BUFFER)
        .build()

    /** LabPBR 1.3 companion atlases built from `<sprite>_n.png` / `<sprite>_s.png`. */
    @JvmField
    val VS_PBR_LAYOUT: BindGroupLayout = BindGroupLayout.builder()
        .withUniform("VsNormalAtlas", UniformType.COMBINED_IMAGE_SAMPLER)
        .withUniform("VsSpecularAtlas", UniformType.COMBINED_IMAGE_SAMPLER)
        .build()

    private val MRT_TARGET_0_ALBEDO = ColorTargetState(Optional.empty(), GpuFormat.RGBA8_UNORM, ColorTargetState.WRITE_ALL)
    private val MRT_TARGET_1_NORMAL = ColorTargetState(Optional.empty(), GpuFormat.RGBA16_FLOAT, ColorTargetState.WRITE_ALL)
    private val MRT_TARGET_2_MATERIAL = ColorTargetState(Optional.empty(), GpuFormat.RGBA8_UNORM, ColorTargetState.WRITE_ALL)
    private val MRT_TARGET_3_VELOCITY = ColorTargetState(Optional.empty(), GpuFormat.RG16_FLOAT, ColorTargetState.WRITE_ALL)

    private val MRT_TARGET_0_TRANSLUCENT = ColorTargetState(
        Optional.of(BlendFunction.TRANSLUCENT),
        GpuFormat.RGBA8_UNORM,
        ColorTargetState.WRITE_ALL,
    )

    @Volatile
    private var registered = false

    @JvmField var solidGbufferPipeline: RenderPipeline? = null
    @JvmField var cutoutGbufferPipeline: RenderPipeline? = null
    @JvmField var translucentGbufferPipeline: RenderPipeline? = null
    @JvmField var wireframeGbufferPipeline: RenderPipeline? = null
    @JvmField var vanillaWireframeMultidraw: RenderPipeline? = null

    @JvmField var entitySolidPipeline: RenderPipeline? = null
    @JvmField var entityCutoutPipeline: RenderPipeline? = null
    @JvmField var entityTranslucentPipeline: RenderPipeline? = null
    @JvmField var entityTranslucentEmissivePipeline: RenderPipeline? = null
    @JvmField var blockSolidPipeline: RenderPipeline? = null
    @JvmField var blockCutoutPipeline: RenderPipeline? = null
    @JvmField var blockTranslucentPipeline: RenderPipeline? = null
    @JvmField var blockEntityPipeline: RenderPipeline? = null
    @JvmField var particlePipeline: RenderPipeline? = null

    private val dhOpaquePipelines = mutableMapOf<Int, RenderPipeline>()
    private val dhTransparentPipelines = mutableMapOf<Int, RenderPipeline>()

    fun dhTerrainPipeline(vertexStride: Int, transparent: Boolean): RenderPipeline? =
        (if (transparent) dhTransparentPipelines else dhOpaquePipelines)[vertexStride]

    private val worldPipelineOverrides = ConcurrentHashMap<Identifier, RenderPipeline>()

    private val cachedShaderSources = ConcurrentHashMap<String, String>()

    @JvmStatic
    fun registerAll(
        litBlocksSnippet: RenderPipeline.Snippet,
        entitySnippet: RenderPipeline.Snippet,
        particleSnippet: RenderPipeline.Snippet,
        vanillaWireframe: RenderPipeline?,
        registerOptional: UnaryOperator<RenderPipeline>,
    ) {
        if (registered) return
        vanillaWireframeMultidraw = vanillaWireframe

        val ignis40BSnippet = RenderPipeline.builder(litBlocksSnippet)
            .withBindGroupLayout(BindGroupLayouts.PROJECTION)
            .withBindGroupLayout(BindGroupLayouts.TERRAIN_INFO)
            .withBindGroupLayout(VS_MOTION_LAYOUT)
            .withBindGroupLayout(VS_PBR_LAYOUT)
            .withVertexShader(SHADER_ID)
            .withFragmentShader(SHADER_ID)
            .withVertexBinding(0, VulkanShadersTurboExtension.IGNIS_TERRAIN_40B)
            .withShaderDefine("MULTIDRAW_TERRAIN")
            .withShaderDefine("EXTENDED_VERTEX_40B")
            .withPushConstantSize(16)
            .buildSnippet()

        solidGbufferPipeline = registerOptional.apply(
            RenderPipeline.builder(ignis40BSnippet)
                .withLocation(Identifier.fromNamespaceAndPath("vulkanshaders", "pipeline/solid_gbuffer_region"))
                .withColorTargetState(0, MRT_TARGET_0_ALBEDO)
                .withColorTargetState(1, MRT_TARGET_1_NORMAL)
                .withColorTargetState(2, MRT_TARGET_2_MATERIAL)
                .withColorTargetState(3, MRT_TARGET_3_VELOCITY)
                .build(),
        )
        cutoutGbufferPipeline = registerOptional.apply(
            RenderPipeline.builder(ignis40BSnippet)
                .withLocation(Identifier.fromNamespaceAndPath("vulkanshaders", "pipeline/cutout_gbuffer_region"))
                .withShaderDefine("ALPHA_CUTOUT", 0.5f)
                .withColorTargetState(0, MRT_TARGET_0_ALBEDO)
                .withColorTargetState(1, MRT_TARGET_1_NORMAL)
                .withColorTargetState(2, MRT_TARGET_2_MATERIAL)
                .withColorTargetState(3, MRT_TARGET_3_VELOCITY)
                .build(),
        )
        translucentGbufferPipeline = registerOptional.apply(
            RenderPipeline.builder(ignis40BSnippet)
                .withLocation(Identifier.fromNamespaceAndPath("vulkanshaders", "pipeline/translucent_gbuffer_region"))
                .withShaderDefine("ALPHA_CUTOUT", 0.1f)
                .withColorTargetState(0, MRT_TARGET_0_ALBEDO)
                .withColorTargetState(1, MRT_TARGET_1_NORMAL)
                .withColorTargetState(2, MRT_TARGET_2_MATERIAL)
                .withColorTargetState(3, MRT_TARGET_3_VELOCITY)
                .build(),
        )
        wireframeGbufferPipeline = registerOptional.apply(
            RenderPipeline.builder(ignis40BSnippet)
                .withLocation(Identifier.fromNamespaceAndPath("vulkanshaders", "pipeline/wireframe_gbuffer_region"))
                .withPolygonMode(PolygonMode.WIREFRAME)
                .withColorTargetState(0, MRT_TARGET_0_ALBEDO)
                .withColorTargetState(1, MRT_TARGET_1_NORMAL)
                .withColorTargetState(2, MRT_TARGET_2_MATERIAL)
                .withColorTargetState(3, MRT_TARGET_3_VELOCITY)
                .build(),
        )

        registerEntityAndParticlePipelines(entitySnippet, particleSnippet, registerOptional)
        registerDhPipelines(registerOptional)

        registered = true
    }

    /** Builds 4-target MRT variants of the vanilla entity/block-entity/particle snippets. */
    private fun registerEntityAndParticlePipelines(
        entitySnippet: RenderPipeline.Snippet,
        particleSnippet: RenderPipeline.Snippet,
        registerOptional: UnaryOperator<RenderPipeline>,
    ) {
        fun build(
            base: RenderPipeline.Snippet,
            name: String,
            shader: Identifier,
            translucent: Boolean = false,
            cutout: Float? = null,
            vertexFormat: VertexFormat? = null,
            extraDefines: List<String> = emptyList(),
        ): RenderPipeline {
            val builder = RenderPipeline.builder(base)
                .withBindGroupLayout(VS_MOTION_LAYOUT)
                .withBindGroupLayout(VS_PBR_LAYOUT)
                .withLocation(Identifier.fromNamespaceAndPath("vulkanshaders", "pipeline/$name"))
                .withVertexShader(shader)
                .withFragmentShader(shader)
                .withColorTargetState(0, if (translucent) MRT_TARGET_0_TRANSLUCENT else MRT_TARGET_0_ALBEDO)
                .withColorTargetState(1, MRT_TARGET_1_NORMAL)
                .withColorTargetState(2, MRT_TARGET_2_MATERIAL)
                .withColorTargetState(3, MRT_TARGET_3_VELOCITY)
            if (vertexFormat != null) builder.withVertexBinding(0, vertexFormat)
            if (cutout != null) builder.withShaderDefine("ALPHA_CUTOUT", cutout)
            for (define in extraDefines) {
                builder.withShaderDefine(define)
            }
            return builder.build()
        }

        entitySolidPipeline = registerOptional.apply(
            build(entitySnippet, "entity_solid_gbuffer", ENTITY_SHADER_ID),
        )
        entityCutoutPipeline = registerOptional.apply(
            build(entitySnippet, "entity_cutout_gbuffer", ENTITY_SHADER_ID, cutout = 0.5f),
        )
        entityTranslucentPipeline = registerOptional.apply(
            build(entitySnippet, "entity_translucent_gbuffer", ENTITY_SHADER_ID, translucent = true),
        )
        entityTranslucentEmissivePipeline = registerOptional.apply(
            build(
                entitySnippet,
                "entity_translucent_emissive_gbuffer",
                ENTITY_SHADER_ID,
                translucent = true,
                extraDefines = listOf("EMISSIVE"),
            ),
        )
        blockSolidPipeline = registerOptional.apply(
            build(
                entitySnippet,
                "block_solid_gbuffer",
                ENTITY_SHADER_ID,
                vertexFormat = DefaultVertexFormat.BLOCK,
                extraDefines = listOf("NO_OVERLAY_UV1", "NO_NORMAL"),
            ),
        )
        blockCutoutPipeline = registerOptional.apply(
            build(
                entitySnippet,
                "block_cutout_gbuffer",
                ENTITY_SHADER_ID,
                cutout = 0.5f,
                vertexFormat = DefaultVertexFormat.BLOCK,
                extraDefines = listOf("NO_OVERLAY_UV1", "NO_NORMAL"),
            ),
        )
        blockTranslucentPipeline = registerOptional.apply(
            build(
                entitySnippet,
                "block_translucent_gbuffer",
                ENTITY_SHADER_ID,
                translucent = true,
                vertexFormat = DefaultVertexFormat.BLOCK,
                extraDefines = listOf("NO_OVERLAY_UV1", "NO_NORMAL"),
            ),
        )
        blockEntityPipeline = blockSolidPipeline
        particlePipeline = registerOptional.apply(
            build(particleSnippet, "particle_gbuffer", PARTICLE_SHADER_ID, translucent = true),
        )
    }

    /** Registers 3-MRT G-Buffer pipelines for Distant Horizons 3.x opaque and transparent terrain. */
    private fun registerDhPipelines(registerOptional: UnaryOperator<RenderPipeline>) {
        val dhSnippet = RenderPipeline.builder()
            .withCull(true)
            .withPolygonMode(PolygonMode.FILL)
            .withPrimitiveTopology(PrimitiveTopology.TRIANGLES)
            .withDepthStencilState(DepthStencilState(CompareOp.GREATER_THAN, true))
            .withBindGroupLayout(DH_TERRAIN_LAYOUT)
            .withVertexShader(DH_TERRAIN_SHADER_ID)
            .withFragmentShader(DH_TERRAIN_SHADER_ID)
            .buildSnippet()

        for (extended in listOf(false, true)) {
            val format = dhVertexFormat(extended)
            val stride = format.vertexSize
            for (transparent in listOf(false, true)) {
                val pass = if (transparent) "transparent" else "opaque"
                val target = if (transparent) MRT_TARGET_0_TRANSLUCENT else MRT_TARGET_0_ALBEDO
                val builder = RenderPipeline.builder(dhSnippet)
                    .withLocation(Identifier.fromNamespaceAndPath("vulkanshaders", "pipeline/dh_${pass}_${stride}b_gbuffer"))
                    .withVertexBinding(0, format)
                    .withColorTargetState(0, target)
                    .withColorTargetState(1, MRT_TARGET_1_NORMAL)
                    .withColorTargetState(2, MRT_TARGET_2_MATERIAL)
                if (!extended) builder.withShaderDefine("VS_DH_PACKED_16")
                val pipelines = if (transparent) dhTransparentPipelines else dhOpaquePipelines
                pipelines[stride] = registerOptional.apply(builder.build())
            }
        }
    }

    @JvmStatic
    fun isReady(): Boolean = registered && solidGbufferPipeline != null

    @JvmStatic
    fun invalidateShaderSourceCache() {
        cachedShaderSources.clear()
        runCatching {
            val mc = Minecraft.getInstance()
            if (mc != null) {
                mc.execute {
                    runCatching {
                        val device = RenderSystem.getDevice()
                        device.javaClass.getMethod("clearPipelineCache").invoke(device)
                    }
                }
            }
        }
    }

    /**
     * Resolves vertex and fragment sources for the bundled G-Buffer shaders. Prefers a same-named
     * override inside the active `.vsp` pack, then falls back to
     * `/assets/vulkanshaders/shaders/core/`. The active pack's `.vsl` hook *bodies* and any helper
     * functions are extracted via a comment/string-aware GLSL lexer and injected at stable
     * `// @hook <stage>` markers.
     */
    @JvmStatic
    fun resolveShader(id: Identifier, type: ShaderType): String? {
        val stem = when (id) {
            SHADER_ID -> "gbuffer_terrain_region"
            ENTITY_SHADER_ID -> "gbuffer_entity"
            PARTICLE_SHADER_ID -> "gbuffer_particle"
            DH_TERRAIN_SHADER_ID -> "gbuffer_dh_terrain"
            else -> return null
        }
        val tierIndex = VulkanShaders.config.renderTier.tierIndex
        val cacheKey = "$stem:$type:$tierIndex"
        cachedShaderSources[cacheKey]?.let { return it }

        val activeRoot = ShaderPackManager.activePack?.rootPath
        var source: String? = null
        if (activeRoot != null && Files.isDirectory(activeRoot)) {
            val packOverride = activeRoot.resolve("shaders/core/$stem.${if (type == ShaderType.VERTEX) "vsh" else "fsh"}")
            if (Files.isRegularFile(packOverride)) {
                source = runCatching { Files.readString(packOverride) }.getOrNull()
            }
        }
        if (source == null) {
            val path = "/assets/vulkanshaders/shaders/core/$stem.${if (type == ShaderType.VERTEX) "vsh" else "fsh"}"
            source = GbufferPipelineRegistry::class.java.getResourceAsStream(path)?.use { stream ->
                String(stream.readAllBytes(), StandardCharsets.UTF_8)
            } ?: return null
        }
        val pack = ShaderPackManager.activePack
        val stageIsFragment = type == ShaderType.FRAGMENT
        var result: String? = null
        if (pack != null && NativeCoreBridge.nativeAvailable) {
            val nativePreprocessed = NativeCoreBridge.preprocessRasterShader(
                source = source,
                packDir = pack.rootPath,
                stageIsFragment = stageIsFragment,
            )
            if (!nativePreprocessed.isNullOrBlank()) {
                val wanted = wantedStagesFor(stem, type)
                val hasUnresolvedHook = wanted.any { stage ->
                    pack.resolvedHooks.containsKey(stage) && nativePreprocessed.contains("// @hook $stage")
                }
                if (!hasUnresolvedHook) {
                    result = nativePreprocessed
                }
            }
        }
        if (result == null) {
            val params = if (pack != null) packFloatParams(pack) else emptyMap()
            val specs = if (pack != null) packSpecDefines(pack) else emptyMap()
            val runtimeSlots = if (pack != null) packRuntimeParamSlots(pack) else emptyMap()
            result = applyPackHooks(source, stem, type, pack?.resolvedHooks.orEmpty(), params, specs, runtimeSlots)
        }
        cachedShaderSources[cacheKey] = result
        return result
    }

    private fun wantedStagesFor(stem: String, type: ShaderType): List<String> = when {
        type == ShaderType.VERTEX && stem == "gbuffer_terrain_region" ->
            listOf("vertex_displace", "water_waves")
        type == ShaderType.FRAGMENT && stem in setOf("gbuffer_terrain_region", "gbuffer_entity") ->
            listOf("surface", "gbuffer_pack")
        else -> emptyList()
    }

    private fun packFloatParams(pack: LoadedShaderPack): Map<String, Float> {
        val cfg = VulkanShaders.config
        val tierIndex = cfg.renderTier.tierIndex
        val overrides = cfg.activePackOverrides(pack.manifest.id)
        return pack.manifest.settings.associate { setting ->
            val overrideRaw = if (setting.appliesToTier(tierIndex)) overrides[setting.key] else null
            setting.key to WorldStateCollector.settingAsFloat(setting, overrideRaw)
        }
    }

    private fun packRuntimeParamSlots(pack: LoadedShaderPack): Map<String, Int> {
        val out = LinkedHashMap<String, Int>()
        val runtimeSettings = pack.manifest.runtimeSettings
        val limit = runtimeSettings.size.coerceAtMost(40)
        for (i in 0 until limit) {
            out[runtimeSettings[i].key] = i
        }
        return out
    }

    private fun packSpecDefines(pack: LoadedShaderPack): Map<String, String> {
        val cfg = VulkanShaders.config
        val tierIndex = cfg.renderTier.tierIndex
        val overrides = cfg.activePackOverrides(pack.manifest.id)
        val out = LinkedHashMap<String, String>()
        for (setting in pack.manifest.settings) {
            if (setting.runtime) continue
            val overrideRaw = if (setting.appliesToTier(tierIndex)) overrides[setting.key] else null
            val raw = overrideRaw ?: setting.defaultValue
            val expr = when (setting.type) {
                SettingType.BOOL -> if (raw.equals("true", ignoreCase = true) || raw == "1") "true" else "false"
                SettingType.INT -> (raw.toIntOrNull() ?: setting.defaultValue.toIntOrNull() ?: 0).toString()
                SettingType.ENUM -> setting.options.indexOfFirst { it.equals(raw, ignoreCase = true) }.coerceAtLeast(0).toString()
                SettingType.FLOAT, SettingType.COLOR_RGB -> "${WorldStateCollector.settingAsFloat(setting, overrideRaw)}f"
            }
            out[setting.key] = expr
        }
        return out
    }

    internal data class AppliedHook(
        val stage: String,
        val helperCode: String,
        val body: String,
    )

    /**
     * Extracts the *actual body* (plus any file-local helper functions) of the hook function for
     * [stage] from a resolved `.vsl` source using a comment/string-aware GLSL lexer and injects it
     * into the G-Buffer GLSL at its `// @hook <stage>` marker.
     *
     * If two hooks in the same shader stage define helper functions/globals with colliding names,
     * those colliding identifiers are automatically isolated under `vs_ns_<stage>_<ident>`.
     */
    @JvmOverloads
    internal fun applyPackHooks(
        source: String,
        stem: String,
        type: ShaderType,
        hooks: Map<String, String>,
        params: Map<String, Float> = emptyMap(),
        specDefines: Map<String, String> = emptyMap(),
        runtimeParamSlots: Map<String, Int> = emptyMap(),
    ): String {
        val wanted = wantedStagesFor(stem, type)
        val rawApplied = wanted.mapNotNull { stage ->
            hooks[stage]?.let { text -> extractHookHelperAndBody(text, stage) }
        }
        if (rawApplied.isEmpty() || !wanted.any { source.contains("// @hook $it") }) return source

        val applied = isolateCollidingHelpers(rawApplied)
        val withDefines = injectAfterVersion(source, buildHookDefines(applied, params, specDefines, runtimeParamSlots))
        val withPrelude = injectAfterVsMotionOrBeforeMain(withDefines, buildHookDeclarations(applied))
        return applied.fold(withPrelude) { acc, hook ->
            acc.replace("// @hook ${hook.stage}", hookCall(hook.stage))
        }
    }

    /** Strips `#include` directives and `// begin include: ... // end include:` blocks. */
    private fun stripIncludeBlocks(source: String): String {
        val sb = StringBuilder()
        var includeDepth = 0
        for (line in source.lineSequence()) {
            val trimmed = line.trim()
            if (trimmed.startsWith("// begin include:")) {
                includeDepth++
                continue
            }
            if (trimmed.startsWith("// end include:")) {
                includeDepth = (includeDepth - 1).coerceAtLeast(0)
                continue
            }
            if (includeDepth > 0 || trimmed.startsWith("#include")) {
                continue
            }
            sb.appendLine(line)
        }
        return sb.toString()
    }

    /**
     * Masks out `//` single-line comments, `/* ... */` multi-line block comments, and `"..."`
     * string literals with spaces (preserving newlines and exact character indices) so `{`/`}`
     * brace matching and identifier scanning never trip over braces or tokens inside comments.
     */
    internal fun maskCommentsAndStrings(src: String): String {
        val out = CharArray(src.length) { i -> src[i] }
        var i = 0
        val n = src.length
        while (i < n) {
            val c = src[i]
            val next = if (i + 1 < n) src[i + 1] else '\u0000'
            if (c == '/' && next == '/') {
                out[i] = ' '
                out[i + 1] = ' '
                i += 2
                while (i < n && src[i] != '\n') {
                    if (src[i] != '\r') out[i] = ' '
                    i++
                }
            } else if (c == '/' && next == '*') {
                out[i] = ' '
                out[i + 1] = ' '
                i += 2
                while (i < n) {
                    if (src[i] == '*' && i + 1 < n && src[i + 1] == '/') {
                        out[i] = ' '
                        out[i + 1] = ' '
                        i += 2
                        break
                    }
                    if (src[i] != '\n' && src[i] != '\r') {
                        out[i] = ' '
                    }
                    i++
                }
            } else if (c == '"') {
                out[i] = ' '
                i++
                while (i < n) {
                    val sc = src[i]
                    if (sc == '\\' && i + 1 < n) {
                        out[i] = ' '
                        out[i + 1] = ' '
                        i += 2
                        continue
                    }
                    if (sc != '\n' && sc != '\r') out[i] = ' '
                    i++
                    if (sc == '"') break
                }
            } else {
                i++
            }
        }
        return String(out)
    }

    /** Comment- and string-aware extraction of the annotated hook function body and helper functions. */
    internal fun extractHookHelperAndBody(hookSource: String, stage: String): AppliedHook? {
        val cleaned = stripIncludeBlocks(hookSource)
        val masked = maskCommentsAndStrings(cleaned)
        val annotated = Regex("""@hook\s*\(\s*stage\s*=\s*\"${Regex.escape(stage)}\"[^)]*\)""").find(cleaned)
        val helperEnd = if (annotated != null) {
            val lineStart = cleaned.lastIndexOf('\n', annotated.range.first).let { if (it < 0) 0 else it + 1 }
            val prefixOnLine = cleaned.substring(lineStart, annotated.range.first).trim()
            if (prefixOnLine == "//" || prefixOnLine.startsWith("/*")) lineStart else annotated.range.first
        } else {
            Regex("""\b\w*${Regex.escape(stage)}\w*\s*\(""").find(masked)?.range?.first ?: return null
        }
        val searchFrom = annotated?.range?.last ?: helperEnd
        val open = masked.indexOf('{', searchFrom).takeIf { it >= 0 } ?: return null
        var depth = 0
        var close = -1
        for (i in open until masked.length) {
            when (masked[i]) {
                '{' -> depth++
                '}' -> {
                    depth--
                    if (depth == 0) {
                        close = i
                        break
                    }
                }
            }
        }
        if (close < 0) return null
        val body = cleaned.substring(open + 1, close).trim('\n', '\r')
        val rawBefore = cleaned.substring(0, helperEnd)
        val maskedBefore = masked.substring(0, helperEnd)
        val hasRealHelperTokens = maskedBefore.any { !it.isWhitespace() }
        val helperCode = if (!hasRealHelperTokens) "" else rawBefore.trim()
        return AppliedHook(stage = stage, helperCode = helperCode, body = body)
    }

    /** Collects top-level helper function and struct names declared in `helperCode`. */
    private fun collectTopLevelHelperNames(helperCode: String): Set<String> {
        val extra = LinkedHashSet<String>()
        Regex("""^\s*struct\s+([A-Za-z_][A-Za-z0-9_]*)\b""", RegexOption.MULTILINE).findAll(helperCode).forEach { m ->
            val n = m.groupValues[1]
            if (!n.startsWith("Vs") && !n.startsWith("vs")) extra.add(n)
        }
        Regex("""^\s*const\s+[A-Za-z_][A-Za-z0-9_]*\s+([A-Za-z_][A-Za-z0-9_]*)\s*(?:\[[^\]]*\])?\s*=""", RegexOption.MULTILINE).findAll(helperCode).forEach { m ->
            val n = m.groupValues[1]
            if (!n.startsWith("VS_") && !n.startsWith("vs")) extra.add(n)
        }
        if (helperCode.isBlank()) return emptySet()
        val masked = maskCommentsAndStrings(helperCode)
        val names = LinkedHashSet<String>()
        var braceDepth = 0
        var parenDepth = 0
        val tokens = mutableListOf<String>()
        var i = 0
        val n = masked.length
        while (i < n) {
            val c = masked[i]
            when {
                c == '{' -> {
                    if (braceDepth == 0 && parenDepth == 0 && tokens.isNotEmpty()) {
                        val candidate = tokens.last()
                        if (candidate !in GLSL_KEYWORDS) {
                            names.add(candidate)
                        }
                        tokens.clear()
                    }
                    braceDepth++
                    i++
                }
                c == '}' -> {
                    braceDepth = (braceDepth - 1).coerceAtLeast(0)
                    if (braceDepth == 0) tokens.clear()
                    i++
                }
                c == '(' -> {
                    if (braceDepth == 0 && parenDepth == 0 && tokens.isNotEmpty()) {
                        val candidate = tokens.last()
                        if (candidate !in GLSL_KEYWORDS) {
                            names.add(candidate)
                        }
                    }
                    parenDepth++
                    i++
                }
                c == ')' -> {
                    parenDepth = (parenDepth - 1).coerceAtLeast(0)
                    i++
                }
                c == ';' && braceDepth == 0 && parenDepth == 0 -> {
                    tokens.clear()
                    i++
                }
                c.isLetter() || c == '_' -> {
                    val start = i
                    i++
                    while (i < n && (masked[i].isLetterOrDigit() || masked[i] == '_')) i++
                    if (braceDepth == 0 && parenDepth == 0) {
                        tokens.add(masked.substring(start, i))
                    }
                }
                else -> i++
            }
        }
        names.addAll(extra)
        return names
    }

    /**
     * Rewrites identifiers in `src` using `renameMap`, skipping comments, string literals,
     * preprocessor directives, and struct member access (`.ident`).
     */
    private fun rewriteIdentifiersLexed(src: String, renameMap: Map<String, String>): String {
        if (renameMap.isEmpty()) return src
        val masked = maskCommentsAndStrings(src)
        val sb = StringBuilder(src.length + 64)
        var i = 0
        val n = src.length
        while (i < n) {
            val mc = masked[i]
            if (mc.isLetter() || mc == '_') {
                val start = i
                i++
                while (i < n && (masked[i].isLetterOrDigit() || masked[i] == '_')) i++
                val ident = src.substring(start, i)
                var prevNonWs = start
                while (prevNonWs > 0 && masked[prevNonWs - 1] in listOf(' ', '\t')) {
                    prevNonWs--
                }
                val isMemberAccess = prevNonWs > 0 && src[prevNonWs - 1] == '.'
                val replacement = if (!isMemberAccess) renameMap[ident] else null
                sb.append(replacement ?: ident)
            } else {
                sb.append(src[i])
                i++
            }
        }
        return sb.toString()
    }

    /**
     * Isolates any helper identifiers that collide across multiple hooks injected into the same
     * shader stage (`vs_ns_<stage>_<ident>`), while keeping unique helper names untouched.
     */
    private fun isolateCollidingHelpers(hooks: List<AppliedHook>): List<AppliedHook> {
        if (hooks.size <= 1) return hooks
        val counts = HashMap<String, Int>()
        val perHookNames = hooks.map { h ->
            val names = collectTopLevelHelperNames(h.helperCode)
            for (name in names) {
                counts[name] = (counts[name] ?: 0) + 1
            }
            names
        }
        val colliding = counts.filterValues { it > 1 }.keys
        if (colliding.isEmpty()) return hooks
        return hooks.mapIndexed { idx, hook ->
            val localColliding = perHookNames[idx].intersect(colliding)
            if (localColliding.isEmpty()) {
                hook
            } else {
                val renameMap = localColliding.associateWith { "vs_ns_${hook.stage}_$it" }
                hook.copy(
                    helperCode = rewriteIdentifiersLexed(hook.helperCode, renameMap),
                    body = rewriteIdentifiersLexed(hook.body, renameMap),
                )
            }
        }
    }

    private val GLSL_KEYWORDS = setOf(
        "void", "bool", "int", "uint", "float", "double",
        "vec2", "vec3", "vec4", "ivec2", "ivec3", "ivec4", "uvec2", "uvec3", "uvec4",
        "mat2", "mat3", "mat4", "in", "out", "inout", "const", "struct",
        "if", "else", "for", "while", "do", "return", "break", "continue", "discard",
        "VsSurface", "VsVertexInput", "VsSkyState", "VsPostContext",
    )

    private fun paramMacroExpr(key: String, fallback: Float, runtimeParamSlots: Map<String, Int>): String {
        val slot = runtimeParamSlots[key]
        if (slot != null && slot in 0 until 40) {
            val comp = "xyzw"[slot and 3]
            return "(vsMotion.packParams[${slot ushr 2}].$comp)"
        }
        return "${fallback}f"
    }

    /** Preprocessor `#define`s for pack parameters and spec constants (safe right after `#version`/`#extension`). */
    private fun buildHookDefines(
        applied: List<AppliedHook>,
        params: Map<String, Float>,
        specDefines: Map<String, String>,
        runtimeParamSlots: Map<String, Int> = emptyMap(),
    ): String = buildString {
        appendLine("// --- Hyaloclast .vsl pack defines ---")
        appendLine("#define VS_SPEC_ENABLE_WIND ${specDefines["enable_wind"] ?: "true"}")
        for ((k, v) in specDefines) {
            if (k != "enable_wind") {
                appendLine("#define VS_SPEC_${k.uppercase(Locale.ROOT)} $v")
            }
        }
        appendLine("#define VS_PARAM_WATER_WAVE_STRENGTH ${paramMacroExpr("water_wave_strength", params["water_wave_strength"] ?: 0.35f, runtimeParamSlots)}")
        appendLine("#define VS_PARAM_FOG_DENSITY ${paramMacroExpr("fog_density", params["fog_density"] ?: 0.002f, runtimeParamSlots)}")
        for ((k, v) in params) {
            if (k != "water_wave_strength" && k != "fog_density") {
                appendLine("#define VS_PARAM_${k.uppercase(Locale.ROOT)} ${paramMacroExpr(k, v, runtimeParamSlots)}")
            }
        }
        if (applied.any { it.stage == "water_waves" }) {
            appendLine("#define VS_HOOK_WATER_WAVES_BOUND 1")
        }
        appendLine("// --- end .vsl pack defines ---")
    }

    /** VSP structs, helper code, and hook functions (injected after `} vsMotion;` so `vsMotion` is in scope). */
    private fun buildHookDeclarations(applied: List<AppliedHook>): String = buildString {
        appendLine("// --- Hyaloclast .vsl hook bodies (generated from the active shader pack) ---")
        appendLine("struct VsHookWorld { float gameTimeSec; float rain; float heldLight; vec3 sunDir; };")
        appendLine("VsHookWorld vsHookWorld;")
        appendLine("#define vsWorld vsHookWorld")
        appendLine("struct VsVertexInput { uint materialFlags; };")
        appendLine("struct VsSurface { vec3 albedo; vec3 normal; float roughness; float metallic; float emission; float ao; float subsurface; float f0; };")
        for (hook in applied) {
            if (hook.helperCode.isNotBlank()) {
                appendLine(hook.helperCode)
            }
            when (hook.stage) {
                "vertex_displace" ->
                    appendLine("void vsHookVertexDisplace(inout vec3 localPos, VsVertexInput v) {\n${hook.body}\n}")
                "water_waves" ->
                    appendLine("void vsHookWaterWaves(inout float strength, inout float speed, inout float scale) {\n${hook.body}\n}")
                "surface" ->
                    appendLine("void vsHookSurface(inout VsSurface s, uint flags, uint bounce) {\n${hook.body}\n}")
                "gbuffer_pack" ->
                    appendLine("void vsHookGbufferPack(in VsSurface s, inout vec4 rt0, inout vec4 rt1, inout vec4 rt2, inout vec2 rt3) {\n${hook.body}\n}")
            }
        }
        appendLine("// --- end .vsl hook bodies ---")
    }

    private fun hookCall(stage: String): String = when (stage) {
        "vertex_displace" -> """
            vsHookWorld = VsHookWorld(vsMotion.cameraDeltaTime.w, vsMotion.sunDirRain.w,
                                      vsMotion.heldLightData.x, vsMotion.sunDirRain.xyz);
            vsHookVertexDisplace(localPos, VsVertexInput(vertexHookFlags));
        """.trimIndent()
        "water_waves" -> """
            vsHookWorld = VsHookWorld(vsMotion.cameraDeltaTime.w, vsMotion.sunDirRain.w,
                                      vsMotion.heldLightData.x, vsMotion.sunDirRain.xyz);
            float vsWaveStrength = 0.35;
            float vsWaveSpeed = 1.8;
            float vsWaveScale = 2.2;
            vsHookWaterWaves(vsWaveStrength, vsWaveSpeed, vsWaveScale);
        """.trimIndent()
        "surface" -> """
            vsHookWorld = VsHookWorld(vsMotion.cameraDeltaTime.w, vsMotion.sunDirRain.w,
                                      vsMotion.heldLightData.x, vsMotion.sunDirRain.xyz);
            VsSurface vsHookSurf = VsSurface(pow(max(color.rgb, vec3(0.0)), vec3(2.2)), N, roughness, metallic, emission, 1.0, subsurface, f0);
            vsHookSurface(vsHookSurf, materialFlags, 0u);
            N = vsHookSurf.normal;
            color.rgb = pow(max(vsHookSurf.albedo, vec3(0.0)), vec3(1.0 / 2.2));
            fragColor.rgb = color.rgb;
            metallic = vsHookSurf.metallic;
            roughness = vsHookSurf.roughness;
            f0 = vsHookSurf.f0;
            emission = vsHookSurf.emission;
            subsurface = vsHookSurf.subsurface;
        """.trimIndent()
        "gbuffer_pack" -> """
            vsHookWorld = VsHookWorld(vsMotion.cameraDeltaTime.w, vsMotion.sunDirRain.w,
                                      vsMotion.heldLightData.x, vsMotion.sunDirRain.xyz);
            VsSurface vsPackSurf = VsSurface(pow(max(color.rgb, vec3(0.0)), vec3(2.2)), N, roughness, metallic, emission, 1.0, subsurface, f0);
            vsHookGbufferPack(vsPackSurf, fragColor, outNormalRoughness, outMaterialPbr, outVelocity);
        """.trimIndent()
        else -> ""
    }

    /** Inserts generated declarations after `#version`/`#extension`, never before them. */
    private fun injectAfterVersion(source: String, prelude: String): String {
        val lines = source.lineSequence().toList()
        var insertAt = 0
        for ((i, line) in lines.withIndex()) {
            val t = line.trimStart()
            if (t.startsWith("#version") || t.startsWith("#extension")) insertAt = i + 1
        }
        return buildString {
            if (insertAt == 0) appendLine(prelude)
            for (i in lines.indices) {
                appendLine(lines[i])
                if (i == insertAt - 1) appendLine(prelude)
            }
        }
    }

    /**
     * Inserts hook struct/function declarations after `} vsMotion;` (or immediately before
     * `void main()` if `vsMotion` is absent) so `VS_PARAM_*` macros referencing `vsMotion`
     * resolve without `'vsMotion' : undeclared identifier`.
     */
    private fun injectAfterVsMotionOrBeforeMain(source: String, declarations: String): String {
        val lines = source.lineSequence().toList()
        val vsMotionEndRegex = Regex("""\}\s*vsMotion\s*;""")
        val vsMotionLine = lines.indexOfFirst { vsMotionEndRegex.containsMatchIn(it) }
        if (vsMotionLine >= 0) {
            return buildString {
                for (i in lines.indices) {
                    appendLine(lines[i])
                    if (i == vsMotionLine) appendLine(declarations)
                }
            }
        }
        val mainLineRegex = Regex("""\bvoid\s+main\s*\(""")
        val mainLine = lines.indexOfFirst { mainLineRegex.containsMatchIn(it) }
        if (mainLine >= 0) {
            return buildString {
                for (i in lines.indices) {
                    if (i == mainLine) appendLine(declarations)
                    appendLine(lines[i])
                }
            }
        }
        return injectAfterVersion(source, declarations)
    }

    private val legacyRasterFlag: Boolean = System.getenv("VS_LEGACY_RASTER") == "1"

    @JvmStatic
    private fun legacyRasterRequested(): Boolean = legacyRasterFlag

    @JvmStatic
    fun isLegacyRasterRequested(): Boolean = legacyRasterRequested()

    fun resolveTerrainPipeline(
        layer: ChunkSectionLayer,
        multidrawOverride: RenderPipeline?,
        defaultPipeline: RenderPipeline,
    ): RenderPipeline {
        if (legacyRasterRequested()) return defaultPipeline
        if (!isReady() || !VulkanShaders.shaderPackActive) return defaultPipeline
        if (multidrawOverride != null && multidrawOverride === vanillaWireframeMultidraw) {
            return wireframeGbufferPipeline ?: defaultPipeline
        }
        if (multidrawOverride != null) {
            return defaultPipeline
        }
        return when (layer) {
            ChunkSectionLayer.SOLID -> solidGbufferPipeline ?: defaultPipeline
            ChunkSectionLayer.CUTOUT -> cutoutGbufferPipeline ?: defaultPipeline
            ChunkSectionLayer.TRANSLUCENT -> translucentGbufferPipeline ?: defaultPipeline
        }
    }

    @JvmStatic
    fun resolveWorldPipeline(vanillaPipeline: RenderPipeline): RenderPipeline {
        if (legacyRasterRequested()) return vanillaPipeline
        if (!isReady() || !VulkanShaders.shaderPackActive) return vanillaPipeline
        if (worldPipelineOverrides.isNotEmpty()) {
            worldPipelineOverrides[vanillaPipeline.location]?.let { return it }
        }

        return when (vanillaPipeline) {
            RenderPipelines.SOLID_TERRAIN_MULTIDRAW -> solidGbufferPipeline ?: vanillaPipeline
            RenderPipelines.CUTOUT_TERRAIN_MULTIDRAW -> cutoutGbufferPipeline ?: vanillaPipeline
            RenderPipelines.TRANSLUCENT_TERRAIN_MULTIDRAW -> translucentGbufferPipeline ?: vanillaPipeline
            RenderPipelines.WIREFRAME_MULTIDRAW -> wireframeGbufferPipeline ?: vanillaPipeline
            RenderPipelines.ENTITY_SOLID, RenderPipelines.ENTITY_SOLID_Z_OFFSET_FORWARD,
            RenderPipelines.ENTITY_SOLID_GLINT -> entitySolidPipeline ?: vanillaPipeline
            RenderPipelines.ENTITY_CUTOUT, RenderPipelines.ENTITY_CUTOUT_CULL,
            RenderPipelines.ENTITY_CUTOUT_Z_OFFSET, RenderPipelines.ENTITY_CUTOUT_DISSOLVE,
            RenderPipelines.ARMOR_CUTOUT_NO_CULL, RenderPipelines.ARMOR_DECAL_CUTOUT_NO_CULL,
            RenderPipelines.WOLF_ARMOR_CRACKS,
            RenderPipelines.ITEM_CUTOUT -> entityCutoutPipeline ?: vanillaPipeline
            RenderPipelines.ENTITY_TRANSLUCENT_EMISSIVE -> entityTranslucentEmissivePipeline ?: vanillaPipeline
            RenderPipelines.ENTITY_TRANSLUCENT,
            RenderPipelines.ENTITY_TRANSLUCENT_CULL, RenderPipelines.ITEM_TRANSLUCENT,
            RenderPipelines.BEACON_BEAM_TRANSLUCENT -> entityTranslucentPipeline ?: vanillaPipeline
            RenderPipelines.SOLID_BLOCK -> blockSolidPipeline ?: vanillaPipeline
            RenderPipelines.CUTOUT_BLOCK -> blockCutoutPipeline ?: vanillaPipeline
            RenderPipelines.TRANSLUCENT_BLOCK -> blockTranslucentPipeline ?: vanillaPipeline
            RenderPipelines.OPAQUE_PARTICLE, RenderPipelines.TRANSLUCENT_PARTICLE,
            RenderPipelines.WEATHER -> particlePipeline ?: vanillaPipeline
            else -> vanillaPipeline
        }
    }
}
