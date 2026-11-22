package dev.vulkanshaders.sdk

import java.nio.file.Files
import java.nio.file.Path

enum class OsFamily(val id: String, val displayName: String) {
    WINDOWS("windows", "Windows"),
    LINUX("linux", "Linux"),
    MACOS("macos", "macOS"),
    UNKNOWN("unknown", "Unknown OS"),
}

enum class CpuArch(val id: String, val displayName: String) {
    X86_64("x86_64", "x86_64"),
    AARCH64("aarch64", "ARM64"),
    UNKNOWN("unknown", "Unknown"),
}

data class NativePlatform(
    val os: OsFamily,
    val arch: CpuArch,
) {
    val displayLabel: String
        get() = "${os.displayName} (${arch.displayName})"

    val sharedLibraryExtension: String
        get() = when (os) {
            OsFamily.WINDOWS -> ".dll"
            OsFamily.MACOS -> ".dylib"
            OsFamily.LINUX, OsFamily.UNKNOWN -> ".so"
        }

    fun systemLibraryDirs(
        systemRootEnv: String? = System.getenv("SystemRoot"),
        extraRoots: List<Path> = emptyList(),
    ): List<Path> {
        val dirs = LinkedHashSet<Path>()
        dirs.addAll(extraRoots)
        when (os) {
            OsFamily.WINDOWS -> {
                val root = systemRootEnv?.takeIf { it.isNotBlank() } ?: "C:\\Windows"
                val sys32 = Path.of(root, "System32")
                dirs.add(sys32)
                val driverRepo = sys32.resolve("DriverStore").resolve("FileRepository")
                if (Files.isDirectory(driverRepo)) {
                    runCatching {
                        Files.list(driverRepo).use { stream ->
                            stream
                                .filter { Files.isDirectory(it) }
                                .filter { p ->
                                    val name = p.fileName.toString().lowercase()
                                    name.startsWith("nv_dispi.inf_amd64_") ||
                                        name.startsWith("nvmdi.inf_amd64_")
                                }
                                .sorted { a, b ->
                                    val ma = runCatching { Files.getLastModifiedTime(a).toMillis() }.getOrDefault(0L)
                                    val mb = runCatching { Files.getLastModifiedTime(b).toMillis() }.getOrDefault(0L)
                                    mb.compareTo(ma)
                                }
                                .forEach { dirs.add(it) }
                        }
                    }
                }
            }
            OsFamily.LINUX -> {
                dirs.add(Path.of("/usr/lib"))
                dirs.add(Path.of("/usr/lib64"))
                dirs.add(Path.of("/usr/lib/x86_64-linux-gnu"))
                dirs.add(Path.of("/usr/lib/aarch64-linux-gnu"))
                dirs.add(Path.of("/usr/local/lib"))
            }
            OsFamily.MACOS -> {
                dirs.add(Path.of("/usr/local/lib"))
                dirs.add(Path.of("/opt/homebrew/lib"))
            }
            OsFamily.UNKNOWN -> Unit
        }
        return dirs.toList()
    }

    companion object {
        @JvmStatic
        fun detect(
            osName: String = System.getProperty("os.name", ""),
            osArch: String = System.getProperty("os.arch", ""),
        ): NativePlatform {
            val normalizedOs = osName.lowercase()
            val normalizedArch = osArch.lowercase()
            val os = when {
                normalizedOs.contains("win") -> OsFamily.WINDOWS
                normalizedOs.contains("mac") || normalizedOs.contains("darwin") -> OsFamily.MACOS
                normalizedOs.contains("nux") || normalizedOs.contains("nix") -> OsFamily.LINUX
                else -> OsFamily.UNKNOWN
            }
            val arch = when {
                normalizedArch.contains("aarch64") || normalizedArch.contains("arm64") -> CpuArch.AARCH64
                normalizedArch.contains("64") || normalizedArch.contains("amd64") || normalizedArch.contains("x86_64") ->
                    CpuArch.X86_64
                else -> CpuArch.UNKNOWN
            }
            return NativePlatform(os = os, arch = arch)
        }
    }
}
