package dev.vulkanshaders.config

import com.google.gson.Gson
import com.google.gson.GsonBuilder
import com.google.gson.JsonDeserializer
import dev.vulkanshaders.VulkanShaders
import net.fabricmc.loader.api.FabricLoader
import java.nio.charset.StandardCharsets
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.StandardCopyOption

object ConfigStore {
    private val LEGACY_BUILTIN_PACK_IDS: Set<String> = setOf(
        "hyaloclast-default",
        "hyaloclast-vanilla-reverie",
        "hyaloclast-golden-solstice",
        "hyaloclast-nordic-borealis",
        "hyaloclast-cinder-mist",
    )

    private val gson: Gson = GsonBuilder()
        .setPrettyPrinting()
        .registerTypeAdapter(
            LightingMode::class.java,
            JsonDeserializer { json, _, _ ->
                LightingMode.fromToken(runCatching { json.asString }.getOrNull()) ?: LightingMode.COMPUTE
            },
        )
        .create()

    private fun resolveConfigDir(): Path =
        runCatching { FabricLoader.getInstance().configDir }
            .getOrElse { Path.of(System.getProperty("user.dir") ?: ".").resolve("config") }

    fun configPath(): Path =
        resolveConfigDir().resolve("${VulkanShaders.MOD_ID}.json")

    fun load(): VulkanShadersConfig {
        val path = configPath()
        if (!Files.exists(path)) {
            val defaults = VulkanShadersConfig()
            save(defaults)
            return defaults
        }
        return try {
            Files.newBufferedReader(path, StandardCharsets.UTF_8).use { reader ->
                val raw = gson.fromJson(reader, VulkanShadersConfig::class.java) ?: VulkanShadersConfig()
                sanitize(raw)
            }
        } catch (t: Throwable) {
            VulkanShaders.log.warn("Failed to read {}; using defaults", path, t)
            VulkanShadersConfig()
        }
    }

    @Suppress("USELESS_ELVIS", "SENSELESS_COMPARISON")
    internal fun sanitize(raw: VulkanShadersConfig): VulkanShadersConfig {
        val defaults = VulkanShadersConfig()
        val rawPackId = (raw.selectedPackId ?: defaults.selectedPackId).trim().ifEmpty { defaults.selectedPackId }
        val safePackId = if (rawPackId in LEGACY_BUILTIN_PACK_IDS) defaults.selectedPackId else rawPackId
        val safeTier = raw.renderTier ?: defaults.renderTier
        val safeUpscaler = raw.upscalerMode ?: defaults.upscalerMode
        val safeQuality = raw.upscalerQuality ?: defaults.upscalerQuality
        val safeDenoiser = raw.denoiserMode ?: defaults.denoiserMode
        val safeFrameGen = raw.frameGenMode ?: defaults.frameGenMode

        val rawLegacy = raw.packOverrides ?: emptyMap()
        val rawByTier = raw.packOverridesByTier ?: emptyMap()
        val sanitizedByTier = LinkedHashMap<String, Map<Int, Map<String, String>>>()

        if (rawByTier.isEmpty() && rawLegacy.isNotEmpty()) {
            for ((packId, overrides) in rawLegacy) {
                val cleanPackId = packId.trim()
                if (cleanPackId.isEmpty() || overrides.isNullOrEmpty()) continue
                val cleanOverrides = LinkedHashMap<String, String>()
                for ((k, v) in overrides) {
                    if (k.isNotBlank() && v != null) {
                        cleanOverrides[k] = v
                    }
                }
                if (cleanOverrides.isNotEmpty()) {
                    sanitizedByTier[cleanPackId] = mapOf(safeTier.tierIndex to cleanOverrides)
                }
            }
        } else {
            for ((packId, tierMap) in rawByTier) {
                val cleanPackId = packId.trim()
                if (cleanPackId.isEmpty() || tierMap.isNullOrEmpty()) continue
                val cleanTierMap = LinkedHashMap<Int, Map<String, String>>()
                for ((tierIdx, overrides) in tierMap) {
                    if (tierIdx !in 0..3 || overrides.isNullOrEmpty()) continue
                    val cleanOverrides = LinkedHashMap<String, String>()
                    for ((k, v) in overrides) {
                        if (k.isNotBlank() && v != null) {
                            cleanOverrides[k] = v
                        }
                    }
                    if (cleanOverrides.isNotEmpty()) {
                        cleanTierMap[tierIdx] = cleanOverrides
                    }
                }
                if (cleanTierMap.isNotEmpty()) {
                    sanitizedByTier[cleanPackId] = cleanTierMap
                }
            }
        }

        return raw.copy(
            selectedPackId = safePackId,
            renderTier = safeTier,
            upscalerMode = safeUpscaler,
            upscalerQuality = safeQuality,
            denoiserMode = safeDenoiser,
            frameGenMode = safeFrameGen,
            shadowResolution = raw.shadowResolution.coerceIn(512, 8192),
            shadowDistanceChunks = raw.shadowDistanceChunks.coerceIn(4, 64),
            hdrPeakBrightnessNits = if (raw.hdrPeakBrightnessNits.isFinite()) {
                raw.hdrPeakBrightnessNits.coerceIn(200f, 4000f)
            } else {
                defaults.hdrPeakBrightnessNits
            },
            hdrPaperWhiteNits = if (raw.hdrPaperWhiteNits.isFinite()) {
                raw.hdrPaperWhiteNits.coerceIn(80f, 1000f)
            } else {
                defaults.hdrPaperWhiteNits
            },
            showF3DebugInfo = raw.showF3DebugInfo,
            enableGpuProfiler = raw.enableGpuProfiler,
            packOverrides = emptyMap(),
            packOverridesByTier = sanitizedByTier,
        )
    }

    fun save(config: VulkanShadersConfig) {
        val path = configPath()
        val sanitized = sanitize(config)
        val tmpPath = path.resolveSibling("${path.fileName}.tmp")
        try {
            Files.createDirectories(path.parent)
            Files.newBufferedWriter(tmpPath, StandardCharsets.UTF_8).use { writer ->
                gson.toJson(sanitized, writer)
            }
            try {
                Files.move(tmpPath, path, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE)
            } catch (_: Throwable) {
                Files.move(tmpPath, path, StandardCopyOption.REPLACE_EXISTING)
            }
        } catch (t: Throwable) {
            runCatching { Files.deleteIfExists(tmpPath) }
            VulkanShaders.log.error("Failed to write {}", path, t)
        }
    }
}
