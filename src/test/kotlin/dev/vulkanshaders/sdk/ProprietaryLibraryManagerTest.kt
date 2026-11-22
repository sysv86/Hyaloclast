package dev.vulkanshaders.sdk

import dev.vulkanshaders.backend.NativeCoreBridge
import dev.vulkanshaders.config.DenoiserMode
import dev.vulkanshaders.config.FrameGenMode
import dev.vulkanshaders.config.UpscalerMode
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
import java.util.zip.ZipOutputStream

class ProprietaryLibraryManagerTest {

    @Test
    fun detectsPlatformsAccurately() {
        assertEquals(
            NativePlatform(OsFamily.WINDOWS, CpuArch.X86_64),
            NativePlatform.detect("Windows 11", "amd64"),
        )
        assertEquals(
            NativePlatform(OsFamily.LINUX, CpuArch.X86_64),
            NativePlatform.detect("Linux", "x86_64"),
        )
        assertEquals(
            NativePlatform(OsFamily.LINUX, CpuArch.AARCH64),
            NativePlatform.detect("Linux", "aarch64"),
        )
        assertEquals(
            NativePlatform(OsFamily.MACOS, CpuArch.AARCH64),
            NativePlatform.detect("Mac OS X", "aarch64"),
        )
    }

    @Test
    fun enforcesCrossPlatformAcquisitionPolicies() {
        val winX64 = NativePlatform(OsFamily.WINDOWS, CpuArch.X86_64)
        val linuxX64 = NativePlatform(OsFamily.LINUX, CpuArch.X86_64)
        val macArm64 = NativePlatform(OsFamily.MACOS, CpuArch.AARCH64)

        assertEquals(
            AcquisitionPolicy.SYSTEM_PATH,
            ProprietaryLibraryCatalog.resolve(ProprietaryLibraryId.NVIDIA_NGX_CORE, winX64, DistributionChannel.MODRINTH).policy,
        )
        assertEquals(
            AcquisitionPolicy.DOWNLOADABLE,
            ProprietaryLibraryCatalog.resolve(ProprietaryLibraryId.AMD_FSR3, winX64, DistributionChannel.MODRINTH).policy,
        )
        assertEquals(
            AcquisitionPolicy.DOWNLOADABLE,
            ProprietaryLibraryCatalog.resolve(ProprietaryLibraryId.INTEL_XESS, winX64, DistributionChannel.MODRINTH).policy,
        )
        assertEquals(
            AcquisitionPolicy.MANUAL_INSTALL,
            ProprietaryLibraryCatalog.resolve(ProprietaryLibraryId.NVIDIA_DLSS_SR, winX64, DistributionChannel.MODRINTH).policy,
        )
        assertEquals(
            AcquisitionPolicy.MANUAL_INSTALL,
            ProprietaryLibraryCatalog.resolve(ProprietaryLibraryId.NVIDIA_NRD, winX64, DistributionChannel.MODRINTH).policy,
        )

        assertEquals(
            AcquisitionPolicy.SYSTEM_PATH,
            ProprietaryLibraryCatalog.resolve(ProprietaryLibraryId.NVIDIA_NGX_CORE, linuxX64, DistributionChannel.MODRINTH).policy,
        )
        assertEquals(
            AcquisitionPolicy.MANUAL_INSTALL,
            ProprietaryLibraryCatalog.resolve(ProprietaryLibraryId.AMD_FSR3, linuxX64, DistributionChannel.MODRINTH).policy,
        )
        assertEquals(
            AcquisitionPolicy.MANUAL_INSTALL,
            ProprietaryLibraryCatalog.resolve(ProprietaryLibraryId.INTEL_XESS, linuxX64, DistributionChannel.MODRINTH).policy,
        )

        assertEquals(
            AcquisitionPolicy.UNSUPPORTED,
            ProprietaryLibraryCatalog.resolve(ProprietaryLibraryId.NVIDIA_NGX_CORE, macArm64, DistributionChannel.MODRINTH).policy,
        )
        assertEquals(
            AcquisitionPolicy.UNSUPPORTED,
            ProprietaryLibraryCatalog.resolve(ProprietaryLibraryId.INTEL_XESS, macArm64, DistributionChannel.MODRINTH).policy,
        )
        assertEquals(
            AcquisitionPolicy.MANUAL_INSTALL,
            ProprietaryLibraryCatalog.resolve(ProprietaryLibraryId.AMD_FSR3, macArm64, DistributionChannel.MODRINTH).policy,
        )
    }

    @Test
    fun curseForgeChannelDisablesExternalDownloads(@TempDir tempDir: Path) {
        val winX64 = NativePlatform(OsFamily.WINDOWS, CpuArch.X86_64)
        val fsr3 = ProprietaryLibraryCatalog.resolve(ProprietaryLibraryId.AMD_FSR3, winX64, DistributionChannel.CURSEFORGE)
        val xess = ProprietaryLibraryCatalog.resolve(ProprietaryLibraryId.INTEL_XESS, winX64, DistributionChannel.CURSEFORGE)

        assertEquals(AcquisitionPolicy.MANUAL_INSTALL, fsr3.policy)
        assertEquals(AcquisitionPolicy.MANUAL_INSTALL, xess.policy)
        assertEquals(null, fsr3.download)
        assertEquals(null, xess.download)

        ProprietaryLibraryManager.initialize(
            gameDirectory = tempDir,
            detectedPlatform = winX64,
            distributionChannel = DistributionChannel.CURSEFORGE,
            systemDirs = emptyList(),
            capabilitiesSupplier = { 0 },
        )
        assertFalse(ProprietaryLibraryManager.startDownload(ProprietaryLibraryId.AMD_FSR3))
        assertFalse(ProprietaryLibraryManager.startDownload(ProprietaryLibraryId.INTEL_XESS))
    }

    @Test
    fun tracksAndClearsRestartRequiredState(@TempDir tempDir: Path) {
        val winX64 = NativePlatform(OsFamily.WINDOWS, CpuArch.X86_64)
        val sysDir = tempDir.resolve("sys32")
        Files.createDirectories(sysDir)
        Files.write(sysDir.resolve("_nvngx.dll"), fakePeBinary())

        ProprietaryLibraryManager.initialize(
            gameDirectory = tempDir,
            detectedPlatform = winX64,
            distributionChannel = DistributionChannel.MODRINTH,
            systemDirs = listOf(sysDir),
            capabilitiesSupplier = { 0 },
        )

        assertEquals(LibraryInstallState.INSTALLED, ProprietaryLibraryManager.entryFor(ProprietaryLibraryId.NVIDIA_NGX_CORE).state)
        assertEquals(LibraryInstallState.AVAILABLE_TO_DOWNLOAD, ProprietaryLibraryManager.entryFor(ProprietaryLibraryId.AMD_FSR3).state)
        assertFalse(ProprietaryLibraryManager.restartRequired)

        val nativesDir = ProprietaryLibraryManager.ensureNativesDir()
        val fsr3File = nativesDir.resolve("amd_fidelityfx_vk.dll")
        Files.write(fsr3File, fakePeBinary())
        ProprietaryLibraryManager.refresh()

        assertEquals(LibraryInstallState.RESTART_REQUIRED, ProprietaryLibraryManager.entryFor(ProprietaryLibraryId.AMD_FSR3).state)
        assertTrue(ProprietaryLibraryManager.restartRequired)
        assertTrue(ProprietaryLibraryManager.isPendingRestartForUpscaler(UpscalerMode.FSR3))
        assertTrue(ProprietaryLibraryManager.isPendingRestartForFrameGen(FrameGenMode.FSR3_FG))
        assertFalse(ProprietaryLibraryManager.isPendingRestartForDenoiser(DenoiserMode.NRD))

        // Deleting the staged file before restart clears pendingRestartIds and restartRequired
        Files.delete(fsr3File)
        ProprietaryLibraryManager.refresh()
        assertEquals(LibraryInstallState.AVAILABLE_TO_DOWNLOAD, ProprietaryLibraryManager.entryFor(ProprietaryLibraryId.AMD_FSR3).state)
        assertFalse(ProprietaryLibraryManager.restartRequired)

        // Re-adding and simulating next launch after restart where NativeCoreBridge reports CAP_FSR3 active
        Files.write(fsr3File, fakePeBinary())
        ProprietaryLibraryManager.initialize(
            gameDirectory = tempDir,
            detectedPlatform = winX64,
            distributionChannel = DistributionChannel.MODRINTH,
            systemDirs = listOf(sysDir),
            capabilitiesSupplier = { NativeCoreBridge.CAP_FSR3 },
        )
        assertEquals(LibraryInstallState.ACTIVE, ProprietaryLibraryManager.entryFor(ProprietaryLibraryId.AMD_FSR3).state)
        assertFalse(ProprietaryLibraryManager.restartRequired)
    }

    @Test
    fun validatesOfficialUrlsBinaryHeadersAndSafeZipExtraction(@TempDir tempDir: Path) {
        assertTrue(DistributionChannel.isAllowedOfficialHttpsUrl("https://github.com/intel/xess/releases/download/v3.0.2/XeSS_SDK_3.0.2.zip"))
        assertFalse(DistributionChannel.isAllowedOfficialHttpsUrl("http://github.com/intel/xess/releases/download/v3.0.2/XeSS_SDK_3.0.2.zip"))
        assertFalse(DistributionChannel.isAllowedOfficialHttpsUrl("https://example.com/libxess.dll"))
        assertFalse(DistributionChannel.isAllowedOfficialHttpsUrl("https://attacker@github.com/intel/xess/releases/download/v3.0.2/XeSS_SDK_3.0.2.zip"))
        assertFalse(DistributionChannel.isAllowedOfficialHttpsUrl("https://github.com:8443/intel/xess/releases/download/v3.0.2/XeSS_SDK_3.0.2.zip"))

        val fakeHtml = tempDir.resolve("bad.dll")
        Files.write(fakeHtml, "<html>404 Not Found</html>".repeat(1000).toByteArray())
        assertThrows(IOException::class.java) {
            ProprietaryLibraryManager.validateNativeBinary(fakeHtml, "amd_fidelityfx_vk.dll")
        }

        val zipPath = tempDir.resolve("xess.zip")
        ZipOutputStream(Files.newOutputStream(zipPath)).use { zos ->
            zos.putNextEntry(ZipEntry("XeSS_SDK/bin/libxess.dll"))
            zos.write(fakePeBinary())
            zos.closeEntry()
            zos.putNextEntry(ZipEntry("XeSS_SDK/LICENSE.txt"))
            zos.write("Intel Simplified Software License".toByteArray())
            zos.closeEntry()
        }
        val outDir = tempDir.resolve("out_natives")
        Files.createDirectories(outDir)
        val installed = ProprietaryLibraryManager.installFromZipArchive(
            zipPath = zipPath,
            targetDir = outDir,
            binaryEntryName = "libxess.dll",
            targetFileName = "libxess.dll",
            licenseEntryName = "LICENSE.txt",
            licenseFileName = "intel_xess_LICENSE.txt",
        )
        assertTrue(Files.isRegularFile(installed))
        assertTrue(Files.isRegularFile(outDir.resolve("licenses").resolve("intel_xess_LICENSE.txt")))

        assertThrows(IOException::class.java) {
            ProprietaryLibraryManager.installFromZipArchive(
                zipPath = zipPath,
                targetDir = outDir,
                binaryEntryName = "libxess.dll",
                targetFileName = "missing-license.dll",
                licenseEntryName = "missing.txt",
                licenseFileName = "missing.txt",
            )
        }
        assertFalse(Files.exists(outDir.resolve("missing-license.dll")))

        // Uncompressed size limit enforcement (zip-bomb protection)
        assertThrows(IOException::class.java) {
            ProprietaryLibraryManager.installFromZipArchive(
                zipPath = zipPath,
                targetDir = outDir,
                binaryEntryName = "libxess.dll",
                targetFileName = "libxess.dll",
                licenseEntryName = null,
                licenseFileName = "intel_xess_LICENSE.txt",
                maxBinaryBytes = 4096L,
            )
        }

        // Zip-slip rejection
        val evilZip = tempDir.resolve("evil.zip")
        ZipOutputStream(Files.newOutputStream(evilZip)).use { zos ->
            zos.putNextEntry(ZipEntry("../../escape/libxess.dll"))
            zos.write(fakePeBinary())
            zos.closeEntry()
        }
        assertThrows(IOException::class.java) {
            ProprietaryLibraryManager.installFromZipArchive(
                zipPath = evilZip,
                targetDir = outDir,
                binaryEntryName = "libxess.dll",
                targetFileName = "libxess.dll",
                licenseEntryName = null,
                licenseFileName = "intel_xess_LICENSE.txt",
            )
        }
    }

    private fun fakePeBinary(): ByteArray {
        val bytes = ByteArray(20 * 1024)
        bytes[0] = 'M'.code.toByte()
        bytes[1] = 'Z'.code.toByte()
        return bytes
    }
}
