package dev.vulkanshaders.pack

import com.google.gson.JsonElement
import com.google.gson.JsonParser
import java.nio.charset.StandardCharsets
import java.nio.file.Files
import java.nio.file.Path
import java.util.Locale
import java.util.zip.ZipFile

internal object PackLocalization {
    private const val MAX_TRANSLATION_BYTES = 1024 * 1024

    fun load(rootPath: Path, locale: String): Map<String, String> {
        val normalizedLocale = locale.lowercase(Locale.ROOT).replace('-', '_')
        if (normalizedLocale.isBlank() || normalizedLocale.any { !it.isLetterOrDigit() && it != '_' }) {
            return emptyMap()
        }

        val translations = linkedMapOf<String, String>()
        readLocale(rootPath, "en_us")?.let(translations::putAll)
        if (normalizedLocale != "en_us") {
            readLocale(rootPath, normalizedLocale)?.let(translations::putAll)
        }
        return translations
    }

    fun localize(manifest: ShaderPackManifest, translations: Map<String, String>): ShaderPackManifest {
        fun translated(key: String, fallback: String?): String? =
            translations[key]?.trim()?.takeIf(String::isNotEmpty) ?: fallback

        return manifest.copy(
            name = translated("pack.name", manifest.name) ?: manifest.name,
            description = translated("pack.description", manifest.description) ?: manifest.description,
            settings = manifest.settings.map { setting ->
                val prefix = "settings.${setting.key}"
                setting.copy(
                    label = translated("$prefix.label", setting.label) ?: setting.label,
                    category = translated("$prefix.category", setting.category),
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
                ZipFile(rootPath.toFile()).use { zip ->
                    val entry = zip.getEntry("lang/$locale.json") ?: return null
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
