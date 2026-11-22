package dev.vulkanshaders.lang

import com.google.gson.JsonObject
import com.google.gson.JsonParser
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertFalse
import org.junit.jupiter.api.Assertions.assertNotNull
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test

class LocalizationParityTest {

    private fun loadLangResource(fileName: String): JsonObject {
        val resourcePath = "/assets/vulkanshaders/lang/$fileName"
        val stream = LocalizationParityTest::class.java.getResourceAsStream(resourcePath)
        assertNotNull(stream, "Missing localization resource on classpath: $resourcePath")
        val text = stream!!.bufferedReader(Charsets.UTF_8).use { it.readText() }
        val parsed = JsonParser.parseString(text)
        assertTrue(parsed.isJsonObject, "$resourcePath must contain a top-level JSON object")
        return parsed.asJsonObject
    }

    @Test
    fun `en_us and ru_ru localization files exist and have identical non-empty key sets`() {
        val en = loadLangResource("en_us.json")
        val ru = loadLangResource("ru_ru.json")

        val enKeys = en.keySet()
        val ruKeys = ru.keySet()

        assertFalse(enKeys.isEmpty(), "en_us.json must not be empty")
        assertEquals(
            enKeys,
            ruKeys,
            "en_us.json and ru_ru.json must have identical key sets " +
                "(missing in ru_ru: ${enKeys - ruKeys}, extra in ru_ru: ${ruKeys - enKeys})",
        )

        for (key in enKeys) {
            val enVal = en.get(key)
            val ruVal = ru.get(key)
            assertTrue(enVal.isJsonPrimitive && enVal.asJsonPrimitive.isString, "en_us.json key '$key' must be a string")
            assertTrue(ruVal.isJsonPrimitive && ruVal.asJsonPrimitive.isString, "ru_ru.json key '$key' must be a string")
            assertTrue(enVal.asString.isNotBlank(), "en_us.json key '$key' must not be blank")
            assertTrue(ruVal.asString.isNotBlank(), "ru_ru.json key '$key' must not be blank")
        }
    }
}
