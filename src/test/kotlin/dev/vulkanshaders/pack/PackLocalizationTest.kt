package dev.vulkanshaders.pack

import java.nio.file.Files
import kotlin.test.Test
import kotlin.test.assertEquals

class PackLocalizationTest {
    @Test
    fun requestedLocaleOverridesEnglishAndFallsBackToManifestText() {
        val root = Files.createTempDirectory("vsp-locale-test")
        try {
            val lang = Files.createDirectories(root.resolve("lang"))
            Files.writeString(
                lang.resolve("en_us.json"),
                """{"pack":{"name":"English Pack"},"settings":{"sun":{"label":"Sun"},"fog":{"label":"Fog"}}}""",
            )
            Files.writeString(
                lang.resolve("fr_fr.json"),
                """{"pack":{"name":"Localized Pack"},"settings":{"sun":{"description":"Sun brightness"}}}""",
            )
            val manifest = manifest()
            val localized = PackLocalization.localize(manifest, PackLocalization.load(root, "fr-FR"))

            assertEquals("Localized Pack", localized.name)
            assertEquals("Sun", localized.settings[0].label)
            assertEquals("Sun brightness", localized.settings[0].description)
            assertEquals("Fog", localized.settings[1].label)
            assertEquals("Raw description", localized.settings[1].description)
        } finally {
            root.toFile().deleteRecursively()
        }
    }

    private fun manifest() = ShaderPackManifest(
        id = "test-pack",
        name = "Raw pack name",
        version = "1.0.0",
        author = "Test",
        description = "Raw pack description",
        supportedTiers = listOf(0),
        settings = listOf(
            PackSettingDefinition(
                key = "sun",
                label = "Raw sun label",
                type = SettingType.FLOAT,
                defaultValue = "1.0",
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
        resources = emptyList(),
        hooks = emptyMap(),
    )
}
