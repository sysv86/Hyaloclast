package dev.vulkanshaders.pack

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertFalse
import org.junit.jupiter.api.Assertions.assertThrows
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.io.TempDir
import java.io.IOException
import java.nio.file.Files
import java.nio.file.Path
import java.util.zip.ZipEntry
import java.util.zip.ZipException
import java.util.zip.ZipFile
import java.util.zip.ZipOutputStream

class ZipSafetyAndImportTest {

    private fun safeExtractPackArchive(archive: Path, targetDir: Path) {
        val normalizedRoot = targetDir.toAbsolutePath().normalize()
        ZipFile(archive.toFile()).use { zip ->
            val manifestEntry = zip.getEntry("pack.toml")
            if (manifestEntry == null || manifestEntry.isDirectory) {
                throw IOException("Archive is missing root pack.toml")
            }

            val entries = zip.entries()
            while (entries.hasMoreElements()) {
                val entry = entries.nextElement()
                val rawName = entry.name
                if (rawName.contains('\u0000')) {
                    throw SecurityException("Archive entry contains NUL byte: $rawName")
                }
                val normalizedSlashes = rawName.replace('\\', '/')
                if (normalizedSlashes.startsWith("/") ||
                    Regex("^[A-Za-z]:").containsMatchIn(normalizedSlashes) ||
                    normalizedSlashes.split('/').any { it == ".." }
                ) {
                    throw SecurityException("Rejected unsafe archive entry path: $rawName")
                }

                val dest = normalizedRoot.resolve(normalizedSlashes).normalize()
                if (!dest.startsWith(normalizedRoot) || dest == normalizedRoot) {
                    throw SecurityException("Archive entry escapes extraction root: $rawName")
                }
                if (entry.isDirectory) {
                    Files.createDirectories(dest)
                } else {
                    Files.createDirectories(dest.parent)
                    zip.getInputStream(entry).use { input ->
                        Files.write(dest, input.readBytes())
                    }
                }
            }
        }
    }

    private fun createZip(target: Path, entries: Map<String, String>) {
        ZipOutputStream(Files.newOutputStream(target)).use { zos ->
            for ((name, content) in entries) {
                zos.putNextEntry(ZipEntry(name))
                zos.write(content.toByteArray(Charsets.UTF_8))
                zos.closeEntry()
            }
        }
    }

    @Test
    fun `rejects zip-slip path traversal and absolute entry paths`(@TempDir tempDir: Path) {
        val evilCases = listOf(
            "../evil.txt",
            "hooks/../../outside.vsl",
            "/etc/passwd",
            "hooks\\..\\..\\escape.vsl",
            "C:/Windows/evil.dll",
        )

        for ((idx, evilEntry) in evilCases.withIndex()) {
            val zipPath = tempDir.resolve("evil-$idx.vsp")
            createZip(
                zipPath,
                mapOf(
                    "pack.toml" to """
                        [pack]
                        id = "evil-pack"
                        name = "Evil Pack"
                    """.trimIndent(),
                    evilEntry to "malicious",
                ),
            )

            val outDir = tempDir.resolve("extract-$idx")
            Files.createDirectories(outDir)
            assertThrows(SecurityException::class.java, {
                safeExtractPackArchive(zipPath, outDir)
            }, "Expected SecurityException for entry '$evilEntry'")
            assertFalse(Files.exists(tempDir.resolve("evil.txt")))
            assertFalse(Files.exists(tempDir.resolve("outside.vsl")))
        }
    }

    @Test
    fun `rejects malformed or truncated zip archives cleanly`(@TempDir tempDir: Path) {
        val corruptZip = tempDir.resolve("corrupt.vsp")
        Files.write(corruptZip, "not-a-valid-zip-file".toByteArray(Charsets.UTF_8))

        val outDir = tempDir.resolve("extract-corrupt")
        Files.createDirectories(outDir)

        assertThrows(ZipException::class.java) {
            safeExtractPackArchive(corruptZip, outDir)
        }
    }

    @Test
    fun `rejects valid zip archives missing root pack toml`(@TempDir tempDir: Path) {
        val missingManifest = tempDir.resolve("missing-manifest.vsp")
        createZip(
            missingManifest,
            mapOf(
                "hooks/surface.vsl" to """
                    @hook(stage = "surface")
                    void surface_hook() {}
                """.trimIndent(),
            ),
        )

        val outDir = tempDir.resolve("extract-missing")
        Files.createDirectories(outDir)

        val ex = assertThrows(IOException::class.java) {
            safeExtractPackArchive(missingManifest, outDir)
        }
        assertTrue(ex.message.orEmpty().contains("pack.toml"))
    }

    @Test
    fun `accepts and parses well-formed vsp archive and validates importable file extensions`(@TempDir tempDir: Path) {
        val validPack = tempDir.resolve("sample-pack.vsp")
        val validZip = tempDir.resolve("sample-pack.zip")
        val unsupportedTxt = tempDir.resolve("notes.txt")
        Files.writeString(unsupportedTxt, "hello")

        val toml = """
            [pack]
            id = "sample-pack"
            name = "Sample Pack"
            version = "0.2.0"
            author = "Tester"

            [capabilities]
            tiers = [0, 1]

            [hooks]
            surface = "hooks/surface.vsl"
        """.trimIndent()

        createZip(
            validPack,
            mapOf(
                "pack.toml" to toml,
                "hooks/surface.vsl" to "@hook(stage = \"surface\")\nvoid surface_hook() {}\n",
            ),
        )
        Files.copy(validPack, validZip)

        assertTrue(ShaderPackManager.isImportableArchive(validPack))
        assertTrue(ShaderPackManager.isImportableArchive(validZip))
        assertFalse(ShaderPackManager.isImportableArchive(unsupportedTxt))
        assertFalse(ShaderPackManager.isImportableArchive(tempDir))

        val outDir = tempDir.resolve("extract-valid")
        Files.createDirectories(outDir)
        safeExtractPackArchive(validPack, outDir)

        assertTrue(Files.isRegularFile(outDir.resolve("pack.toml")))
        assertTrue(Files.isRegularFile(outDir.resolve("hooks/surface.vsl")))

        val manifest = ShaderPackManifest.fromToml(Files.readString(outDir.resolve("pack.toml")), "fallback")
        assertEquals("sample-pack", manifest.id)
        assertEquals("Sample Pack", manifest.name)
        assertEquals(listOf(0, 1), manifest.supportedTiers)
        assertEquals("hooks/surface.vsl", manifest.hooks["surface"])
    }
}
