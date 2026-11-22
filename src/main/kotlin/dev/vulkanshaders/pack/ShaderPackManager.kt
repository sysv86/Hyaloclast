package dev.vulkanshaders.pack

import dev.vulkanshaders.VulkanShaders
import dev.vulkanshaders.backend.NativeCoreBridge
import dev.vulkanshaders.config.VulkanShadersConfig
import dev.vulkanshaders.diagnostics.HyaloclastErrorLog
import dev.vulkanshaders.gui.ShaderNotices
import dev.vulkanshaders.pipeline.GbufferPipelineRegistry
import dev.vulkanshaders.state.WorldStateCollector
import dev.vulkanshaders.texture.TextureRegistry
import net.fabricmc.loader.api.FabricLoader
import net.minecraft.client.Minecraft
import java.io.ByteArrayInputStream
import java.io.IOException
import java.io.InputStream
import java.nio.charset.StandardCharsets
import java.nio.file.Files
import java.nio.file.LinkOption
import java.nio.file.Path
import java.util.Comparator
import java.util.zip.ZipException
import java.util.zip.ZipFile
import javax.imageio.ImageIO

/**
 * Discovers, validates, imports, and loads `.vsp` shader packages from `<gameDir>/shaderpacks`
 * and built-in `.vsp` resources, including strict zip path-traversal / zip-bomb protection,
 * required `pack.toml` field validation, `extends` inheritance, `VSP Graph v2` programmable DAG
 * passes/targets/buffers, native SPIR-V compilation via `NativeCoreBridge.compileActivePack`,
 * specialization constant synchronization, live in-world hot-reload, and `.trying` crash-guard
 * fallback.
 */
object ShaderPackManager {
    private data class BuiltinPackSpec(
        val id: String,
        val files: List<String>,
        val legacyFilesToPrune: List<String> = emptyList(),
    )

    data class ReloadResult(
        val success: Boolean,
        val packName: String? = null,
        val errorSummary: String? = null,
        val diagnostic: HyaloclastErrorLog.ShaderCompileDiagnostic? = null,
    )

    internal const val BUILTIN_PACK_ID = "hyaloclast-sunlit-vanilla"
    internal const val SHOWCASE_PACK_ID = "hyaloclast-obsidian-showcase"

    internal const val MAX_ARCHIVE_ENTRIES: Int = 4096
    internal const val MAX_ARCHIVE_UNCOMPRESSED_BYTES: Long = 256L * 1024L * 1024L // 256 MB
    internal const val MAX_SINGLE_ENTRY_BYTES: Long = 64L * 1024L * 1024L // 64 MB
    private val DRIVE_LETTER_PATH_REGEX = Regex("^[A-Za-z]:.*")

    internal val LEGACY_BUILTIN_PACK_IDS: Set<String> = setOf(
        "hyaloclast-default",
        "hyaloclast-vanilla-reverie",
        "hyaloclast-golden-solstice",
        "hyaloclast-nordic-borealis",
        "hyaloclast-cinder-mist",
    )

    private val BUILTIN_PACKS = listOf(
        BuiltinPackSpec(
            id = BUILTIN_PACK_ID,
            files = listOf(
                "pack.toml",
                "CREDITS.md",
                "THIRD_PARTY_NOTICES.txt",
                "include/vulkanshaders/v1/common.vsl",
                "include/vulkanshaders/v1/lighting.vsl",
                "include/vulkanshaders/v1/restir.vsl",
                "shaders/core/csm_shadow.vsh",
                "shaders/core/csm_shadow_28b.vsh",
                "shaders/core/csm_shadow.fsh",
                "shaders/core/depth_blit.vsh",
                "shaders/core/depth_blit.fsh",
                "shaders/compute/wind_displace.comp",
                "shaders/compute/sharc_resolve.comp",
                "shaders/compute/volumetric_fog.comp",
                "shaders/compute/compute_lighting.comp",
                "shaders/compute/rayquery_lighting.comp",
                "shaders/compute/ssgi.comp",
                "shaders/compute/denoiser_prepare.comp",
                "shaders/compute/denoiser_recompose.comp",
                "shaders/compute/svgf_temporal.comp",
                "shaders/compute/svgf_atrous.comp",
                "shaders/compute/tsr_upscale.comp",
                "shaders/compute/post_tonemap.comp",
                "shaders/compute/hdr_compose.comp",
                "shaders/compute/optical_flow_fg.comp",
                "shaders/rt/pathtrace.rgen",
                "shaders/rt/pathtrace.rchit",
                "shaders/rt/pathtrace.rahit",
                "shaders/rt/pathtrace.rmiss",
                "resources/blue_noise_64x64.png",
                "hooks/vertex_displace.vsl",
                "hooks/surface.vsl",
                "hooks/direct_light.vsl",
                "hooks/block_light.vsl",
                "hooks/water_waves.vsl",
                "hooks/water_surface.vsl",
                "hooks/sky.vsl",
                "hooks/clouds.vsl",
                "hooks/fog.vsl",
                "hooks/post_grade.vsl",
            ),
            legacyFilesToPrune = listOf(
                "shaders/compute/mc_depth.comp",
                "shaders/core/gbuffer_terrain.vsh",
                "shaders/core/gbuffer_terrain.fsh",
            ),
        ),
        BuiltinPackSpec(
            id = SHOWCASE_PACK_ID,
            files = listOf(
                "pack.toml",
                "hooks/vertex_displace.vsl",
                "hooks/surface.vsl",
                "hooks/direct_light.vsl",
                "hooks/block_light.vsl",
                "hooks/water_waves.vsl",
                "hooks/water_surface.vsl",
                "hooks/sky.vsl",
                "hooks/clouds.vsl",
                "hooks/fog.vsl",
                "hooks/post_grade.vsl",
            ),
        ),
    )

    internal val BUILTIN_PACK_IDS: List<String> = BUILTIN_PACKS.map { it.id }
    private val BUILTIN_PACK_ORDER: Map<String, Int> =
        BUILTIN_PACKS.mapIndexed { index, spec -> spec.id to index }.toMap()

    @Volatile
    var discoveredPacks: List<LoadedShaderPack> = emptyList()
        private set

    @Volatile
    var activePack: LoadedShaderPack? = null
        private set

    @Volatile
    var lastPackError: String? = null
        private set

    private fun resolveGameDir(): Path =
        runCatching { FabricLoader.getInstance().gameDir }
            .getOrElse { Path.of(System.getProperty("user.dir") ?: ".") }

    private fun resolveConfigDir(): Path =
        runCatching { FabricLoader.getInstance().configDir }
            .getOrElse { resolveGameDir().resolve("config") }

    fun shaderpacksDir(): Path =
        resolveGameDir().resolve("shaderpacks")

    private fun extractedCacheDir(): Path =
        resolveGameDir().resolve("shaderpacks/.extracted")

    private fun tryingMarkerPath(): Path =
        resolveConfigDir().resolve("${VulkanShaders.MOD_ID}.trying")

    fun initialize(config: VulkanShadersConfig) {
        ensureBuiltinPackExtracted()
        reloadAll()

        val marker = tryingMarkerPath()
        if (Files.exists(marker)) {
            val crashedId = runCatching { Files.readString(marker).trim() }.getOrNull()
            val crashGuardMsg = "Crash-guard activated for pack '${crashedId ?: "unknown"}'; reverted to safe mode."
            lastPackError = crashGuardMsg
            HyaloclastErrorLog.recordError("ShaderPack", crashGuardMsg)
            runCatching { Files.deleteIfExists(marker) }
            activePack = null
            VulkanShaders.updateConfig(VulkanShaders.config.copy(enabled = false, selectedPackId = "off"))
            return
        }

        if (config.enabled && !config.selectedPackId.equals("off", ignoreCase = true) && NativeCoreBridge.initialized) {
            selectPack(config.selectedPackId)
        }
    }

    fun reloadAll() {
        val dir = shaderpacksDir()
        Files.createDirectories(dir)
        val rawPacks = mutableListOf<LoadedShaderPack>()
        Files.list(dir).use { stream ->
            stream.forEach { entry ->
                if (entry.fileName.toString().startsWith(".")) return@forEach
                val loaded = tryLoadPack(entry)
                if (loaded != null) {
                    rawPacks.add(loaded)
                }
            }
        }

        val byId = rawPacks.associateBy { it.manifest.id }
        val resolvedPacks = rawPacks.map { pack ->
            val parentId = pack.manifest.extendsPackId
            val parent = if (parentId != null) byId[parentId] else null
            if (parent != null) {
                val mergedHooks = LinkedHashMap<String, String>(parent.resolvedHooks)
                mergedHooks.putAll(pack.resolvedHooks)
                pack.copy(
                    manifest = pack.manifest.inheritFrom(parent.manifest),
                    resolvedHooks = mergedHooks,
                )
            } else {
                pack
            }
        }.filterNot { pack ->
            // Keep retired packs available as inheritance parents and preserve local edits.
            pack.manifest.id in LEGACY_BUILTIN_PACK_IDS &&
                Files.isDirectory(pack.rootPath, LinkOption.NOFOLLOW_LINKS) &&
                Files.isRegularFile(pack.rootPath.resolve(".builtin-stamp"), LinkOption.NOFOLLOW_LINKS)
        }.sortedWith(
            compareBy<LoadedShaderPack>(
                { BUILTIN_PACK_ORDER[it.manifest.id] ?: Int.MAX_VALUE },
                { it.manifest.name.lowercase() },
            ),
        )

        discoveredPacks = resolvedPacks
        VulkanShaders.log.info("Discovered {} shader pack(s) in {}", resolvedPacks.size, dir)
    }

    /**
     * Live in-world hot-reload of the currently active shader pack without requiring a world rejoin.
     * Preserves all per-pack settings/overrides, re-reads `pack.toml` and `.vsl`/GLSL files from disk,
     * recompiles native SPIR-V pipelines, refreshes specialization constants and companion atlases,
     * and surfaces any compilation diagnostic via [HyaloclastErrorLog] and [ShaderNotices].
     */
    @JvmStatic
    fun reloadActivePackInWorld(): ReloadResult {
        val cfg = VulkanShaders.config
        return try {
            reloadAll()
            if (!cfg.enabled || cfg.selectedPackId.isBlank() || cfg.selectedPackId.equals("off", ignoreCase = true)) {
                activePack = null
                lastPackError = null
                VulkanShaders.refreshRuntimeState()
                return ReloadResult(success = true, packName = null)
            }

            NativeCoreBridge.waitPresentIdle()
            val ok = selectPack(cfg.selectedPackId)
            if (!ok) {
                val err = lastPackError ?: "Failed to reload shader pack '${cfg.selectedPackId}'"
                val diag = HyaloclastErrorLog.latestShaderDiagnostic
                VulkanShaders.refreshRuntimeState()
                return ReloadResult(
                    success = false,
                    packName = cfg.selectedPackId,
                    errorSummary = err,
                    diagnostic = diag,
                )
            }

            syncActivePackSpecializationConstants()
            GbufferPipelineRegistry.invalidateShaderSourceCache()
            WorldStateCollector.resetHistory()
            NativeCoreBridge.resetTemporalHistory()
            NativeCoreBridge.markHistoryReset()
            VulkanShaders.refreshRuntimeState()
            if (VulkanShaders.shaderPackActive) {
                TextureRegistry.ensureCompanionAtlasesBuilt()
            }

            runCatching {
                val client = Minecraft.getInstance()
                if (client?.level != null) {
                    client.levelExtractor.allChanged()
                }
            }

            val packName = activePack?.manifest?.name ?: cfg.selectedPackId
            runCatching { ShaderNotices.showPackReloadedToast(packName) }
            ReloadResult(success = true, packName = packName)
        } catch (t: Throwable) {
            val msg = t.message ?: t.javaClass.simpleName
            lastPackError = msg
            HyaloclastErrorLog.recordError("ShaderPack", "Hot-reload failed: $msg", throwable = t)
            ReloadResult(success = false, packName = cfg.selectedPackId, errorSummary = msg)
        }
    }

    internal class ArchiveFormatException(message: String? = null) : IOException(message)
    internal class DuplicatePackException(message: String? = null) : IOException(message)
    internal class UnsafeArchiveException(message: String) : IOException(message)
    internal class InvalidManifestException(message: String) : IOException(message)

    internal fun isImportableArchive(path: Path): Boolean {
        val extension = path.fileName?.toString().orEmpty().substringAfterLast('.', "").lowercase()
        return (extension == "vsp" || extension == "zip") && Files.isRegularFile(path)
    }

    internal fun importArchive(source: Path): LoadedShaderPack {
        if (!isImportableArchive(source)) {
            throw ArchiveFormatException("Expected a .vsp or .zip archive file.")
        }

        val directory = shaderpacksDir()
        Files.createDirectories(directory)
        val staging = Files.createTempDirectory(directory, ".import-")
        try {
            try {
                unpackArchive(source, staging)
            } catch (e: UnsafeArchiveException) {
                throw e
            } catch (e: InvalidManifestException) {
                throw e
            } catch (e: ZipException) {
                throw ArchiveFormatException("Malformed zip archive: ${e.message ?: "invalid zip structure"}")
            }

            val manifestPath = staging.resolve("pack.toml")
            if (!Files.isRegularFile(manifestPath, LinkOption.NOFOLLOW_LINKS)) {
                throw InvalidManifestException("Archive is missing a root pack.toml file.")
            }
            val manifestText = Files.readString(manifestPath, StandardCharsets.UTF_8)
            validateRequiredManifestFields(manifestText)

            val pack = try {
                loadDirectoryPack(staging, fallbackId = "")
            } catch (e: IllegalArgumentException) {
                throw InvalidManifestException(e.message ?: "Invalid pack.toml manifest.")
            } ?: throw InvalidManifestException("Archive is missing a valid pack.toml manifest.")

            val id = pack.manifest.id
            if (id.equals("off", ignoreCase = true)) {
                throw InvalidManifestException("The shader pack ID 'off' is reserved.")
            }
            if (discoveredPacks.any { it.manifest.id == id } ||
                Files.exists(directory.resolve("$id.zip")) ||
                Files.exists(directory.resolve("$id.vsp"))
            ) {
                throw DuplicatePackException("A shader pack with ID '$id' is already installed.")
            }
            val destination = directory.resolve("$id.vsp")
            Files.move(staging, destination)
            return pack.copy(rootPath = destination)
        } finally {
            runCatching { deleteTree(staging) }
                .onFailure { VulkanShaders.log.warn("Could not remove temporary shader pack {}", staging, it) }
        }
    }

    /**
     * Validates that a `pack.toml` file explicitly declares the required `[pack]` metadata fields
     * (`id`, `name`, `version`) before parsing/importing.
     */
    internal fun validateRequiredManifestFields(tomlText: String) {
        val sections = try {
            VspTomlParser.parse(tomlText)
        } catch (t: Throwable) {
            throw InvalidManifestException("Malformed pack.toml syntax: ${t.message ?: t.javaClass.simpleName}")
        }
        val pack = sections["pack"]
            ?: throw InvalidManifestException("pack.toml is missing required [pack] section.")

        val missing = mutableListOf<String>()
        val id = (pack["id"] as? VspTomlParser.Value.Str)?.value?.trim().orEmpty()
        val name = (pack["name"] as? VspTomlParser.Value.Str)?.value?.trim().orEmpty()
        val version = (pack["version"] as? VspTomlParser.Value.Str)?.value?.trim().orEmpty()

        if (id.isEmpty()) missing.add("pack.id")
        if (name.isEmpty()) missing.add("pack.name")
        if (version.isEmpty()) missing.add("pack.version")

        if (missing.isNotEmpty()) {
            throw InvalidManifestException("pack.toml is missing required field(s): ${missing.joinToString(", ")}.")
        }
        if (!id.all { it.isLetterOrDigit() || it == '-' || it == '_' }) {
            throw InvalidManifestException("pack.id '$id' may only contain ASCII letters, digits, '-', or '_'.")
        }
    }

    /**
     * Validates a zip entry name against path-traversal attacks: rejects empty names, NUL bytes,
     * absolute POSIX/UNC paths, Windows drive-letter prefixes, and `..` segments.
     */
    internal fun validateArchiveEntryPath(rawEntryName: String): String {
        if (rawEntryName.isBlank()) {
            throw UnsafeArchiveException("Archive contains an empty entry path.")
        }
        if (rawEntryName.indexOf('\u0000') >= 0) {
            throw UnsafeArchiveException("Archive entry contains a NUL byte: '$rawEntryName'.")
        }
        val normalized = rawEntryName.replace('\\', '/')
        if (normalized.startsWith("/") || normalized.startsWith("//")) {
            throw UnsafeArchiveException("Rejected absolute archive path: '$rawEntryName'.")
        }
        if (DRIVE_LETTER_PATH_REGEX.matches(normalized)) {
            throw UnsafeArchiveException("Rejected drive-letter archive path: '$rawEntryName'.")
        }
        val segments = normalized.split('/')
        for (segment in segments) {
            if (segment == "..") {
                throw UnsafeArchiveException("Rejected path-traversal archive entry: '$rawEntryName'.")
            }
        }
        return normalized
    }

    fun selectPack(packId: String): Boolean {
        GbufferPipelineRegistry.invalidateShaderSourceCache()
        WorldStateCollector.resetHistory()

        if (packId.isBlank() || packId.equals("off", ignoreCase = true)) {
            activePack = null
            lastPackError = null
            clearTryingMarker()
            return true
        }

        val target = discoveredPacks.firstOrNull { it.manifest.id == packId }
        if (target == null) {
            val notFoundMsg = "Shader pack '$packId' not found."
            lastPackError = notFoundMsg
            HyaloclastErrorLog.recordError("ShaderPack", notFoundMsg)
            activePack = null
            clearTryingMarker()
            return false
        }

        writeTryingMarker(target.manifest.id)
        val compileRoot = if (Files.isDirectory(target.rootPath)) {
            target.rootPath
        } else {
            extractZipPackForCompilation(target)
        }

        if (compileRoot == null) {
            activePack = null
            clearTryingMarker()
            return false
        }
        val compiledCount = NativeCoreBridge.compileActivePack(compileRoot)
        VulkanShaders.log.info("Compiled {} native shader module(s) for pack '{}'", compiledCount, target.manifest.id)
        if (NativeCoreBridge.lastCompilationError != null) {
            VulkanShaders.log.warn("Native shader compilation error/diagnostic: {}", NativeCoreBridge.lastCompilationError)
        }
        if (compiledCount < 0) {
            val rawErr = NativeCoreBridge.lastCompilationError
                ?: "Failed to compile shader pack '${target.manifest.name}' (status $compiledCount)"
            val diagnostic = HyaloclastErrorLog.recordShaderCompileError(
                packName = target.manifest.name,
                rawError = rawErr,
                statusCode = compiledCount,
                showToast = true,
            )
            lastPackError = diagnostic.formatShort()
            activePack = null
            clearTryingMarker()
            return false
        }

        uploadPackResources(target, compileRoot)
        NativeCoreBridge.syncCustomDescriptorSetsToBridge()

        activePack = target
        lastPackError = null
        syncActivePackSpecializationConstants()
        GbufferPipelineRegistry.invalidateShaderSourceCache()
        clearTryingMarker()
        VulkanShaders.log.info(
            "Activated shader pack '{}' v{} (tiers: {}, hooks: {}, customPasses: {})",
            target.manifest.name,
            target.manifest.version,
            target.manifest.supportedTiers,
            target.resolvedHooks.keys,
            target.manifest.passes.size,
        )
        return true
    }

    /**
     * Pushes any non-runtime (`runtime == false`) specialization constant values for the active
     * shader pack to the native Rust pipeline manager so `VkPipeline`s are specialized immediately.
     */
    @JvmStatic
    fun syncActivePackSpecializationConstants() {
        val pack = activePack ?: return
        val cfg = VulkanShaders.config
        val overrides = cfg.activePackOverrides(pack.manifest.id)
        val specPairs = pack.manifest.buildSpecializationOverrides(overrides, cfg.renderTier.tierIndex)
        NativeCoreBridge.updateSpecConstants(specPairs)
    }

    private data class DecodedRgba(val width: Int, val height: Int, val rgba: ByteArray)

    private fun resolveInheritanceChain(pack: LoadedShaderPack): List<LoadedShaderPack> {
        val chain = mutableListOf<LoadedShaderPack>()
        val visited = HashSet<String>()
        var current: LoadedShaderPack? = pack
        while (current != null && visited.add(current.manifest.id)) {
            chain.add(current)
            val parentId = current.manifest.extendsPackId
            current = if (parentId != null) {
                discoveredPacks.firstOrNull { it.manifest.id == parentId }
            } else {
                null
            }
        }
        return chain
    }

    private fun uploadPackResources(pack: LoadedShaderPack, compileRoot: Path?) {
        if (pack.manifest.resources.isEmpty()) return
        val chain = resolveInheritanceChain(pack)
        val mergedTomlFields = linkedMapOf<String, Map<String, VspTomlParser.Value>>()
        for (ancestor in chain.asReversed()) {
            val ancestorRoot = if (ancestor.manifest.id == pack.manifest.id) compileRoot else ancestor.rootPath
            val parsed = runCatching {
                val text = readPackFileBytes(ancestor, ancestorRoot, "pack.toml")
                    ?.toString(Charsets.UTF_8).orEmpty()
                VspTomlParser.parse(text)
            }.getOrDefault(emptyMap())
            mergedTomlFields.putAll(parsed)
        }

        var uploaded = 0
        pack.manifest.resources.forEachIndexed { bindingIndex, res ->
            val bytes = chain.firstNotNullOfOrNull { candidate ->
                val candidateRoot = if (candidate.manifest.id == pack.manifest.id) compileRoot else candidate.rootPath
                readPackFileBytes(candidate, candidateRoot, res.path)
            }
            if (bytes == null) {
                VulkanShaders.log.warn("Pack resource '{}' declared in '{}' was not found", res.path, pack.manifest.id)
                return@forEachIndexed
            }
            val decoded = decodeRgba8(bytes)
            if (decoded == null) {
                VulkanShaders.log.warn("Pack resource '{}' is not a decodable image", res.path)
                return@forEachIndexed
            }
            val fields = mergedTomlFields["resources.${res.name}"].orEmpty()
            val filterLinear = (fields["filter"] as? VspTomlParser.Value.Str)?.value
                ?.equals("linear", ignoreCase = true) ?: false
            val wrapValue = (fields["wrap"] as? VspTomlParser.Value.Str)?.value
            val wrapRepeat = wrapValue?.equals("clamp", ignoreCase = true)?.not() ?: true
            val status = NativeCoreBridge.uploadPackTexture(
                slot = bindingIndex,
                width = decoded.width,
                height = decoded.height,
                rgba = decoded.rgba,
                filterLinear = filterLinear,
                wrapRepeat = wrapRepeat,
            )
            if (status == 0 || status == 2) {
                uploaded++
            } else {
                VulkanShaders.log.warn(
                    "Native upload of pack resource '{}' (binding {}) returned {}",
                    res.path, bindingIndex, status,
                )
            }
        }
        VulkanShaders.log.info(
            "Uploaded {}/{} pack resource texture(s) for '{}'",
            uploaded, pack.manifest.resources.size, pack.manifest.id,
        )
    }

    private fun readPackFileBytes(pack: LoadedShaderPack, compileRoot: Path?, relPath: String): ByteArray? {
        if (compileRoot != null && Files.isDirectory(compileRoot)) {
            val file = compileRoot.resolve(relPath).normalize()
            if (!file.startsWith(compileRoot) || !Files.isRegularFile(file)) return null
            return runCatching { Files.readAllBytes(file) }.getOrNull()
        }
        val zipPath = if (Files.isDirectory(pack.rootPath)) null else pack.rootPath
        if (zipPath == null) return null
        return runCatching {
            ZipFile(zipPath.toFile()).use { zip ->
                val entry = zip.getEntry(relPath) ?: return null
                zip.getInputStream(entry).use { it.readBytes() }
            }
        }.getOrNull()
    }

    private fun decodeRgba8(png: ByteArray): DecodedRgba? = runCatching {
        val image = ImageIO.read(ByteArrayInputStream(png)) ?: return null
        val w = image.width
        val h = image.height
        if (w <= 0 || h <= 0) return null
        val argb = image.getRGB(0, 0, w, h, null, 0, w)
        val rgba = ByteArray(w * h * 4)
        var i = 0
        for (pixel in argb) {
            rgba[i++] = (pixel ushr 16 and 0xFF).toByte()
            rgba[i++] = (pixel ushr 8 and 0xFF).toByte()
            rgba[i++] = (pixel and 0xFF).toByte()
            rgba[i++] = (pixel ushr 24 and 0xFF).toByte()
        }
        DecodedRgba(w, h, rgba)
    }.getOrNull()

    private fun extractZipPackForCompilation(pack: LoadedShaderPack): Path? {
        return try {
            val outRoot = extractedCacheDir().resolve(pack.manifest.id).normalize()
            deleteTree(outRoot)
            Files.createDirectories(outRoot)
            unpackArchive(pack.rootPath, outRoot)
            outRoot
        } catch (t: Throwable) {
            val unpackMsg = "Failed to unpack '${pack.manifest.name}': ${t.message}"
            lastPackError = unpackMsg
            HyaloclastErrorLog.recordError("ShaderPack", unpackMsg, throwable = t)
            null
        }
    }

    internal fun unpackArchive(archive: Path, directory: Path) {
        val normalizedRoot = directory.toAbsolutePath().normalize()
        ZipFile(archive.toFile()).use { zip ->
            val manifestEntry = zip.getEntry("pack.toml")
            if (manifestEntry == null || manifestEntry.isDirectory) {
                throw ArchiveFormatException("Archive must contain a root pack.toml file.")
            }

            var entryCount = 0
            var totalUncompressedBytes = 0L
            val entries = zip.entries()
            while (entries.hasMoreElements()) {
                val entry = entries.nextElement()
                entryCount++
                if (entryCount > MAX_ARCHIVE_ENTRIES) {
                    throw UnsafeArchiveException(
                        "Archive exceeds maximum allowed entry count ($MAX_ARCHIVE_ENTRIES).",
                    )
                }

                val safeName = validateArchiveEntryPath(entry.name)
                val declaredSize = entry.size
                if (declaredSize > MAX_SINGLE_ENTRY_BYTES) {
                    throw UnsafeArchiveException(
                        "Archive entry '$safeName' exceeds maximum allowed file size (${MAX_SINGLE_ENTRY_BYTES / (1024 * 1024)} MB).",
                    )
                }

                val dest = normalizedRoot.resolve(safeName).normalize()
                if (!dest.startsWith(normalizedRoot) || dest == normalizedRoot) {
                    throw UnsafeArchiveException("Archive entry escapes target directory: '${entry.name}'.")
                }

                if (entry.isDirectory) {
                    Files.createDirectories(dest)
                    continue
                }

                Files.createDirectories(dest.parent)
                if (Files.isSymbolicLink(dest) || Files.isSymbolicLink(dest.parent)) {
                    throw UnsafeArchiveException("Rejected symbolic link target inside archive: '${entry.name}'.")
                }

                val remainingBudget = MAX_ARCHIVE_UNCOMPRESSED_BYTES - totalUncompressedBytes
                val entryLimit = minOf(MAX_SINGLE_ENTRY_BYTES, remainingBudget)
                zip.getInputStream(entry).use { input ->
                    val written = copyBounded(input, dest, entryLimit, safeName)
                    totalUncompressedBytes += written
                    if (totalUncompressedBytes > MAX_ARCHIVE_UNCOMPRESSED_BYTES) {
                        throw UnsafeArchiveException(
                            "Archive exceeds maximum allowed uncompressed size (${MAX_ARCHIVE_UNCOMPRESSED_BYTES / (1024 * 1024)} MB).",
                        )
                    }
                }
            }
        }
    }

    private fun copyBounded(input: InputStream, dest: Path, maxBytes: Long, entryName: String): Long {
        val buf = ByteArray(32 * 1024)
        var total = 0L
        Files.newOutputStream(dest).use { out ->
            while (true) {
                val n = input.read(buf)
                if (n < 0) break
                total += n
                if (total > maxBytes) {
                    throw UnsafeArchiveException(
                        "Archive entry '$entryName' exceeded maximum allowed uncompressed size.",
                    )
                }
                out.write(buf, 0, n)
            }
        }
        return total
    }

    private fun deleteTree(path: Path) {
        if (!Files.exists(path, LinkOption.NOFOLLOW_LINKS)) return
        Files.walk(path).use { walk ->
            walk.sorted(Comparator.reverseOrder()).forEach(Files::deleteIfExists)
        }
    }

    private fun ensureBuiltinPackExtracted() {
        val packsDir = shaderpacksDir()
        val cl = ShaderPackManager::class.java.classLoader
        for (spec in BUILTIN_PACKS) {
            extractBuiltinPack(packsDir, cl, spec)
        }
    }

    private fun extractBuiltinPack(packsDir: Path, cl: ClassLoader, spec: BuiltinPackSpec) {
        val targetRoot = packsDir.resolve("${spec.id}.vsp")
        for (legacyRel in spec.legacyFilesToPrune) {
            runCatching { Files.deleteIfExists(targetRoot.resolve(legacyRel)) }
        }

        val digest = java.security.MessageDigest.getInstance("SHA-256")
        val payload = LinkedHashMap<String, ByteArray>(spec.files.size)
        for (rel in spec.files) {
            val resourcePath = "builtin-packs/${spec.id}.vsp/$rel"
            val bytes = cl.getResourceAsStream(resourcePath)?.use { it.readAllBytes() } ?: continue
            payload[rel] = bytes
            digest.update(rel.toByteArray(Charsets.UTF_8))
            digest.update(0)
            digest.update(bytes)
            digest.update(0)
        }
        if (payload.isEmpty()) {
            return
        }

        val expectedStamp = digest.digest().joinToString("") { "%02x".format(it) }
        val stampFile = targetRoot.resolve(".builtin-stamp")
        val existingStamp = runCatching {
            if (Files.isRegularFile(stampFile)) Files.readString(stampFile).trim() else null
        }.getOrNull()
        val allPresent = existingStamp == expectedStamp &&
            payload.keys.all { rel -> Files.isRegularFile(targetRoot.resolve(rel)) }
        if (allPresent) {
            return
        }

        Files.createDirectories(targetRoot)
        for ((rel, bytes) in payload) {
            val outFile = targetRoot.resolve(rel)
            Files.createDirectories(outFile.parent)
            Files.write(outFile, bytes)
        }
        runCatching { Files.writeString(stampFile, expectedStamp) }
    }

    private fun tryLoadPack(entry: Path): LoadedShaderPack? {
        return try {
            if (Files.isDirectory(entry)) {
                loadDirectoryPack(entry)
            } else if (Files.isRegularFile(entry) &&
                (entry.fileName.toString().endsWith(".vsp") || entry.fileName.toString().endsWith(".zip"))
            ) {
                loadZipPack(entry)
            } else {
                null
            }
        } catch (t: Throwable) {
            val invalidMsg = "Invalid pack '${entry.fileName}': ${t.message}"
            lastPackError = invalidMsg
            HyaloclastErrorLog.recordError("ShaderPack", invalidMsg, throwable = t)
            null
        }
    }

    private fun loadDirectoryPack(
        dir: Path,
        fallbackId: String = dir.fileName.toString().removeSuffix(".vsp"),
    ): LoadedShaderPack? {
        val manifestPath = dir.resolve("pack.toml")
        if (!Files.isRegularFile(manifestPath)) return null
        val manifest = ShaderPackManifest.fromToml(
            Files.readString(manifestPath, StandardCharsets.UTF_8),
            fallbackId,
        )
        val normalizedDir = dir.toAbsolutePath().normalize()
        val hooks = linkedMapOf<String, String>()
        for ((stage, relPath) in manifest.hooks) {
            val hookFile = normalizedDir.resolve(relPath).normalize()
            if (hookFile.startsWith(normalizedDir) && Files.isRegularFile(hookFile)) {
                hooks[stage] = resolveIncludes(Files.readString(hookFile, StandardCharsets.UTF_8)) { inc ->
                    val localInc = normalizedDir.resolve("include/$inc").normalize()
                    if (localInc.startsWith(normalizedDir) && Files.isRegularFile(localInc)) {
                        Files.readString(localInc, StandardCharsets.UTF_8)
                    } else {
                        null
                    }
                }
            }
        }
        return LoadedShaderPack(
            manifest = manifest,
            rootPath = dir,
            isBuiltin = manifest.id in BUILTIN_PACK_ORDER,
            resolvedHooks = hooks,
        )
    }

    private fun loadZipPack(zipPath: Path): LoadedShaderPack? {
        ZipFile(zipPath.toFile()).use { zip ->
            var count = 0
            val entries = zip.entries()
            while (entries.hasMoreElements()) {
                val e = entries.nextElement()
                count++
                if (count > MAX_ARCHIVE_ENTRIES) {
                    throw UnsafeArchiveException("Archive '${zipPath.fileName}' exceeds $MAX_ARCHIVE_ENTRIES entries.")
                }
                validateArchiveEntryPath(e.name)
            }
            val manifestEntry = zip.getEntry("pack.toml") ?: return null
            val tomlText = zip.getInputStream(manifestEntry).bufferedReader(StandardCharsets.UTF_8).readText()
            val fallbackId = zipPath.fileName.toString().removeSuffix(".vsp").removeSuffix(".zip")
            val manifest = ShaderPackManifest.fromToml(tomlText, fallbackId)
            val hooks = linkedMapOf<String, String>()
            for ((stage, relPath) in manifest.hooks) {
                val entry = zip.getEntry(relPath) ?: continue
                val raw = zip.getInputStream(entry).bufferedReader(StandardCharsets.UTF_8).readText()
                hooks[stage] = resolveIncludes(raw) { inc ->
                    zip.getEntry("include/$inc")?.let {
                        zip.getInputStream(it).bufferedReader(StandardCharsets.UTF_8).readText()
                    }
                }
            }
            return LoadedShaderPack(
                manifest = manifest,
                rootPath = zipPath,
                isBuiltin = false,
                resolvedHooks = hooks,
            )
        }
    }

    internal fun resolveIncludes(
        source: String,
        depth: Int = 0,
        alreadyIncluded: MutableSet<String> = HashSet(),
        loader: (String) -> String?,
    ): String {
        if (depth > 8) throw IOException("Exceeded maximum #include nesting depth")
        val sb = StringBuilder(source.length)
        for (line in source.lineSequence()) {
            val trimmed = line.trim()
            if (trimmed.startsWith("#include <") && trimmed.endsWith(">")) {
                val inc = trimmed.removePrefix("#include <").removeSuffix(">").trim()
                if (!alreadyIncluded.add(inc)) {
                    continue
                }
                val included = loader(inc)
                if (included != null) {
                    sb.append("// begin include: ").append(inc).append('\n')
                    sb.append(resolveIncludes(included, depth + 1, alreadyIncluded, loader)).append('\n')
                    sb.append("// end include: ").append(inc).append('\n')
                    continue
                }
            }
            sb.append(line).append('\n')
        }
        return sb.toString()
    }

    private fun writeTryingMarker(packId: String) {
        val marker = tryingMarkerPath()
        runCatching {
            Files.createDirectories(marker.parent)
            Files.writeString(marker, packId)
        }
    }

    private fun clearTryingMarker() {
        runCatching { Files.deleteIfExists(tryingMarkerPath()) }
    }
}
