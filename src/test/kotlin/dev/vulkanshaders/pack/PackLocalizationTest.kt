package dev.vulkanshaders.pack

import java.nio.file.Files
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream
import kotlin.test.Test
import kotlin.test.assertEquals

class PackLocalizationTest {
    @Test
    fun requestedLocaleOverridesBaseLanguageAndEnglishAndFallsBackToManifestText() {
        val root = Files.createTempDirectory("vsp-locale-test")
        try {
            PackLocalization.clearCache()
            val lang = Files.createDirectories(root.resolve("lang"))
            Files.writeString(
                lang.resolve("en_us.json"),
                """
                {
                  "pack": { "name": "English Pack" },
                  "categories": { "lighting": "Lighting EN" },
                  "settings": {
                    "sun": { "label": "Sun", "options": { "agx": "AgX EN" } },
                    "fog": { "label": "Fog" }
                  },
                  "presets": {
                    "high": { "label": "High EN", "description": "High EN desc" }
                  }
                }
                """.trimIndent(),
            )
            Files.writeString(
                lang.resolve("fr.json"),
                """
                {
                  "categories": { "lighting": "Eclairage" },
                  "settings": {
                    "sun": { "label": "Soleil", "options": { "agx": "AgX FR" } }
                  },
                  "presets": {
                    "high": { "label": "Eleve" }
                  }
                }
                """.trimIndent(),
            )
            Files.writeString(
                lang.resolve("fr_ca.json"),
                """
                {
                  "pack": { "name": "Pack Quebec" },
                  "settings": { "sun": { "description": "Luminosite du soleil" } }
                }
                """.trimIndent(),
            )
            val manifest = manifest()
            val localized = PackLocalization.localize(manifest, PackLocalization.load(root, "fr-CA"))

            assertEquals("Pack Quebec", localized.name)
            assertEquals("Soleil", localized.settings[0].label)
            assertEquals("Eclairage", localized.settings[0].category)
            assertEquals("Luminosite du soleil", localized.settings[0].description)
            assertEquals(listOf("AgX FR", "aces"), localized.settings[0].optionLabels)
            assertEquals("Fog", localized.settings[1].label)
            assertEquals("Raw description", localized.settings[1].description)
            assertEquals("Eleve", localized.presets[0].label)
            assertEquals("High EN desc", localized.presets[0].description)
        } finally {
            PackLocalization.clearCache()
            root.toFile().deleteRecursively()
        }
    }

    @Test
    fun loadsTranslationsFromZipArchiveCaseInsensitively() {
        val tempDir = Files.createTempDirectory("vsp-zip-locale-test")
        try {
            PackLocalization.clearCache()
            val archive = tempDir.resolve("sample.vsp")
            ZipOutputStream(Files.newOutputStream(archive)).use { zos ->
                zos.putNextEntry(ZipEntry("Lang/RU_RU.JSON"))
                zos.write("""{"pack":{"name":"Русский пак"},"categories":{"lighting":"Освещение"}}""".toByteArray())
                zos.closeEntry()
            }
            val localized = PackLocalization.localize(manifest(), PackLocalization.load(archive, "ru_ru"))
            assertEquals("Русский пак", localized.name)
            assertEquals("Освещение", localized.settings[0].category)
        } finally {
            PackLocalization.clearCache()
            tempDir.toFile().deleteRecursively()
        }
    }

    @Test
    fun inheritsParentPackTranslationsAcrossExtendsChain() {
        val tempDir = Files.createTempDirectory("vsp-extends-locale-test")
        try {
            PackLocalization.clearCache()
            val parentDir = Files.createDirectories(tempDir.resolve("parent"))
            val childDir = Files.createDirectories(tempDir.resolve("child"))
            Files.createDirectories(parentDir.resolve("lang"))
            Files.createDirectories(childDir.resolve("lang"))

            Files.writeString(
                parentDir.resolve("lang/ru_ru.json"),
                """{"pack":{"name":"Родитель"},"settings":{"sun":{"label":"Солнце"}}}""",
            )
            Files.writeString(
                childDir.resolve("lang/ru_ru.json"),
                """{"pack":{"name":"Дочерний пак"}}""",
            )

            val parentPack = LoadedShaderPack(
                manifest = manifest(id = "parent-pack"),
                rootPath = parentDir,
                isBuiltin = false,
                resolvedHooks = emptyMap(),
            )
            val childPack = LoadedShaderPack(
                manifest = manifest(id = "child-pack", extendsPack = "parent-pack"),
                rootPath = childDir,
                isBuiltin = false,
                resolvedHooks = emptyMap(),
            )

            val localized = PackLocalization.localizePack(childPack, "ru_ru", listOf(parentPack, childPack))
            assertEquals("Дочерний пак", localized.name)
            assertEquals("Солнце", localized.settings[0].label)
        } finally {
            PackLocalization.clearCache()
            tempDir.toFile().deleteRecursively()
        }
    }

    private fun manifest(id: String = "test-pack", extendsPack: String? = null) = ShaderPackManifest(
        id = id,
        name = "Raw pack name",
        version = "1.0.0",
        author = "Test",
        description = "Raw pack description",
        supportedTiers = listOf(0),
        settings = listOf(
            PackSettingDefinition(
                key = "sun",
                label = "Raw sun label",
                type = SettingType.ENUM,
                defaultValue = "agx",
                options = listOf("agx", "aces"),
                category = "Lighting",
                description = "Raw sun description",
            ),
            PackSettingDefinition(
                key = "fog",
                label = "Raw fog label",
                type = SettingType.FLOAT,
                defaultValue = "1.0",
                description = "Raw description",
            ),
        ),
        presets = listOf(
            PackPresetDefinition(
                id = "high",
                label = "High",
                description = "Raw preset description",
            ),
        ),
        resources = emptyList(),
        hooks = emptyMap(),
        extendsPackId = extendsPack,
    )
}
