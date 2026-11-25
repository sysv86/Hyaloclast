package dev.vulkanshaders.pack

import com.google.gson.JsonElement
import com.google.gson.JsonParser
import java.nio.charset.StandardCharsets
import java.nio.file.Files
import java.nio.file.Path
import java.util.Locale
import java.util.concurrent.ConcurrentHashMap
import java.util.zip.ZipFile

internal object PackLocalization {
    private const val MAX_TRANSLATION_BYTES = 1024 * 1024

    private data class CacheKey(
        val rootPath: Path,
        val locale: String,
        val lastModifiedMillis: Long,
    )

    private val translationCache = ConcurrentHashMap<CacheKey, Map<String, String>>()

    fun clearCache() {
        translationCache.clear()
    }

    /**
     * Loads merged translation keys for a single pack root using the fallback chain:
     * `en_us.json` -> base language `<lang>.json` (when `<lang>_<region>` is requested) ->
     * exact `<lang>_<region>.json`.
     */
    fun load(rootPath: Path, locale: String): Map<String, String> {
        val normalizedLocale = normalizeLocale(locale) ?: return emptyMap()
        val mtime = runCatching { Files.getLastModifiedTime(rootPath).toMillis() }.getOrDefault(0L)
        val cacheKey = CacheKey(rootPath.toAbsolutePath().normalize(), normalizedLocale, mtime)
        translationCache[cacheKey]?.let { return it }

        val translations = linkedMapOf<String, String>()
        readLocale(rootPath, "en_us")?.let(translations::putAll)

        val baseLanguage = normalizedLocale.substringBefore('_')
        if (baseLanguage.isNotEmpty() && baseLanguage != normalizedLocale && baseLanguage != "en_us") {
            readLocale(rootPath, baseLanguage)?.let(translations::putAll)
        }

        if (normalizedLocale != "en_us") {
            readLocale(rootPath, normalizedLocale)?.let(translations::putAll)
        }

        val frozen = translations.toMap()
        translationCache[cacheKey] = frozen
        return frozen
    }

    /**
     * Loads merged translations across an `extends` inheritance chain (base parent first,
     * child pack last so child entries override parent entries).
     */
    fun loadWithInheritance(
        pack: LoadedShaderPack,
        locale: String,
        allPacks: Collection<LoadedShaderPack> = ShaderPackManager.discoveredPacks,
    ): Map<String, String> {
        val chain = resolveRootChain(pack, allPacks)
        if (chain.size == 1) {
            return load(chain[0], locale)
        }
        val merged = linkedMapOf<String, String>()
        for (root in chain) {
            merged.putAll(load(root, locale))
        }
        return merged
    }

    fun localizePack(
        pack: LoadedShaderPack,
        locale: String,
        allPacks: Collection<LoadedShaderPack> = ShaderPackManager.discoveredPacks,
    ): ShaderPackManifest {
        val translations = loadWithInheritance(pack, locale, allPacks)
        return localize(pack.manifest, translations)
    }

    fun localize(manifest: ShaderPackManifest, translations: Map<String, String>): ShaderPackManifest {
        if (translations.isEmpty()) {
            return manifest
        }

        fun translated(key: String, fallback: String?): String? =
            translations[key]?.trim()?.takeIf(String::isNotEmpty) ?: fallback

        fun translatedCategory(settingKey: String, rawCategory: String?): String? {
            val perSetting = translated("settings.$settingKey.category", null)
            if (perSetting != null) return perSetting
            val rawTrimmed = rawCategory?.trim()?.takeIf(String::isNotEmpty) ?: return null
            val catKey = rawTrimmed.lowercase(Locale.ROOT).replace(Regex("[^a-z0-9]+"), "_").trim('_')
            return translated("categories.$catKey", rawTrimmed)
        }

        return manifest.copy(
            name = translated("pack.name", manifest.name) ?: manifest.name,
            description = translated("pack.description", manifest.description) ?: manifest.description,
            settings = manifest.settings.map { setting ->
                val prefix = "settings.${setting.key}"
                setting.copy(
                    label = translated("$prefix.label", setting.label) ?: setting.label,
                    category = translatedCategory(setting.key, setting.category),
                    description = translated("$prefix.description", setting.description),
                    optionLabels = setting.options.map { option ->
                        translated("$prefix.options.$option", option) ?: option
                    },
                )
            },
            presets = manifest.presets.map { preset ->
                val prefix = "presets.${preset.id}"
                preset.copy(
                    label = translated("$prefix.label", preset.label) ?: preset.label,
                    description = translated("$prefix.description", preset.description),
                )
            },
        )
    }

    internal fun normalizeLocale(locale: String): String? {
        val normalized = locale.trim().lowercase(Locale.ROOT).replace('-', '_')
        if (normalized.isBlank() || normalized.any { !it.isLetterOrDigit() && it != '_' }) {
            return null
        }
        return normalized
    }

    private fun resolveRootChain(
        pack: LoadedShaderPack,
        allPacks: Collection<LoadedShaderPack>,
    ): List<Path> {
        val byId = allPacks.associateBy { it.manifest.id }
        val visited = linkedSetOf<String>()
        val chain = ArrayList<Path>(4)
        var current: LoadedShaderPack? = pack
        while (current != null && visited.add(current.manifest.id)) {
            chain.add(current.rootPath)
            val parentId = current.manifest.extendsPackId?.trim()?.takeIf(String::isNotEmpty)
            current = parentId?.let(byId::get)
        }
        chain.reverse()
        return chain
    }

    private fun readLocale(rootPath: Path, locale: String): Map<String, String>? {
        val bytes = runCatching {
            if (Files.isDirectory(rootPath)) {
                val root = rootPath.toAbsolutePath().normalize()
                val file = root.resolve("lang/$locale.json").normalize()
                if (!file.startsWith(root) || !Files.isRegularFile(file) || Files.size(file) > MAX_TRANSLATION_BYTES) {
                    return null
                }
                Files.readAllBytes(file)
            } else {
                val targetName = "lang/$locale.json"
                ZipFile(rootPath.toFile()).use { zip ->
                    val entry = zip.getEntry(targetName) ?: run {
                        val entries = zip.entries()
                        var matched: java.util.zip.ZipEntry? = null
                        while (entries.hasMoreElements()) {
                            val candidate = entries.nextElement()
                            if (candidate.isDirectory) continue
                            val normalizedName = candidate.name
                                .replace('\\', '/')
                                .removePrefix("./")
                                .lowercase(Locale.ROOT)
                            if (normalizedName == targetName) {
                                matched = candidate
                                break
                            }
                        }
                        matched
                    } ?: return null
                    if (entry.isDirectory || entry.size > MAX_TRANSLATION_BYTES) return null
                    zip.getInputStream(entry).use { it.readNBytes(MAX_TRANSLATION_BYTES + 1) }
                }.takeIf { it.size <= MAX_TRANSLATION_BYTES } ?: return null
            }
        }.getOrNull() ?: return null

        return runCatching {
            val root = JsonParser.parseString(String(bytes, StandardCharsets.UTF_8))
            val flattened = linkedMapOf<String, String>()
            flatten(root, "", flattened)
            flattened
        }.getOrNull()
    }

    private fun flatten(element: JsonElement, prefix: String, output: MutableMap<String, String>) {
        when {
            element.isJsonObject -> element.asJsonObject.entrySet().forEach { (key, value) ->
                val path = if (prefix.isEmpty()) key else "$prefix.$key"
                flatten(value, path, output)
            }
            element.isJsonPrimitive && element.asJsonPrimitive.isString -> {
                if (prefix.isNotEmpty()) output[prefix] = element.asString
            }
        }
    }
}
