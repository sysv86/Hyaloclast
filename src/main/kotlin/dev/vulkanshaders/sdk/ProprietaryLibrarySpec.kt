package dev.vulkanshaders.sdk

/**
 * How a vendor rendering library may be obtained on the current platform while respecting vendor
 * EULAs, GPL-3.0 + Section 7 Additional Permissions, and Modrinth/CurseForge distribution rules.
 */
enum class AcquisitionPolicy {
    /** Installed by the OS or GPU driver; never downloaded separately, discovered via system path. */
    SYSTEM_PATH,

    /** Official vendor HTTPS release permits user-initiated direct download with license preservation. */
    DOWNLOADABLE,

    /** Vendor EULA restricts automated redistribution/download or no official prebuilt binary exists. */
    MANUAL_INSTALL,

    /** Vendor provides no driver or Vulkan runtime for this operating system / CPU architecture. */
    UNSUPPORTED,
}

enum class LibraryInstallState {
    ACTIVE,
    INSTALLED,
    RESTART_REQUIRED,
    SYSTEM_MISSING,
    AVAILABLE_TO_DOWNLOAD,
    DOWNLOADING,
    MANUAL_REQUIRED,
    UNSUPPORTED,
    ERROR,
}

enum class ProprietaryLibraryId(
    val id: String,
    val displayName: String,
    val translationKey: String,
    val licenseName: String,
    val officialPageUrl: String,
) {
    NVIDIA_NGX_CORE(
        id = "nvidia_ngx_core",
        displayName = "NVIDIA NGX Core",
        translationKey = "vulkanshaders.libraries.item.nvidia_ngx_core",
        licenseName = "NVIDIA Driver EULA (System Library)",
        officialPageUrl = "https://www.nvidia.com/en-us/drivers/",
    ),
    AMD_FSR3(
        id = "amd_fsr3",
        displayName = "AMD FidelityFX FSR 3",
        translationKey = "vulkanshaders.libraries.item.amd_fsr3",
        licenseName = "MIT License",
        officialPageUrl = "https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK/releases",
    ),
    INTEL_XESS(
        id = "intel_xess",
        displayName = "Intel XeSS",
        translationKey = "vulkanshaders.libraries.item.intel_xess",
        licenseName = "Intel Simplified Software License",
        officialPageUrl = "https://github.com/intel/xess/releases",
    ),
    NVIDIA_DLSS_SR(
        id = "nvidia_dlss_sr",
        displayName = "NVIDIA DLSS Super Resolution",
        translationKey = "vulkanshaders.libraries.item.nvidia_dlss_sr",
        licenseName = "NVIDIA RTX SDKs License",
        officialPageUrl = "https://github.com/NVIDIA/DLSS",
    ),
    NVIDIA_DLSS_RR(
        id = "nvidia_dlss_rr",
        displayName = "NVIDIA DLSS Ray Reconstruction",
        translationKey = "vulkanshaders.libraries.item.nvidia_dlss_rr",
        licenseName = "NVIDIA RTX SDKs License",
        officialPageUrl = "https://github.com/NVIDIA/DLSS",
    ),
    NVIDIA_DLSS_FG(
        id = "nvidia_dlss_fg",
        displayName = "NVIDIA DLSS Frame Generation",
        translationKey = "vulkanshaders.libraries.item.nvidia_dlss_fg",
        licenseName = "NVIDIA RTX SDKs License",
        officialPageUrl = "https://github.com/NVIDIA-RTX/Streamline",
    ),
    NVIDIA_NRD(
        id = "nvidia_nrd",
        displayName = "NVIDIA NRD Denoiser",
        translationKey = "vulkanshaders.libraries.item.nvidia_nrd",
        licenseName = "NVIDIA RTX SDKs License",
        officialPageUrl = "https://github.com/NVIDIA-RTX/NRD",
    ),
}

data class DownloadDescriptor(
    val url: String,
    val targetFileName: String,
    val licenseUrl: String? = null,
    val licenseFileName: String,
    val expectedSha256: String? = null,
    val archiveBinaryEntryName: String? = null,
    val archiveLicenseEntryName: String? = null,
    val maxBytes: Long = 128L * 1024L * 1024L,
)

data class ResolvedLibrarySpec(
    val id: ProprietaryLibraryId,
    val policy: AcquisitionPolicy,
    val filePrefixes: List<String>,
    val expectedFileName: String?,
    val download: DownloadDescriptor?,
    val notesKey: String,
)

object ProprietaryLibraryCatalog {
    private const val AMD_FSR3_WIN_URL =
        "https://raw.githubusercontent.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK/v1.1.4/PrebuiltSignedDLL/amd_fidelityfx_vk.dll"
    private const val AMD_FSR3_WIN_SHA256 =
        "a1624cc4238fef046f30c4d80ce3f47be63fc5f5373f49e3ee9edb9960f54c78"
    private const val AMD_FSR3_LICENSE_URL =
        "https://raw.githubusercontent.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK/v1.1.4/LICENSE.txt"

    private const val INTEL_XESS_WIN_ZIP_URL =
        "https://github.com/intel/xess/releases/download/v3.0.2/XeSS_SDK_3.0.2.zip"
    private const val INTEL_XESS_WIN_ZIP_SHA256 =
        "88b8a373f30e33f3558a77a93e634f11b8132fc3047ea1a8edeead32b8471990"

    @JvmStatic
    fun resolveAll(
        platform: NativePlatform,
        channel: DistributionChannel,
    ): List<ResolvedLibrarySpec> =
        ProprietaryLibraryId.entries.map { resolve(it, platform, channel) }

    @JvmStatic
    fun resolve(
        id: ProprietaryLibraryId,
        platform: NativePlatform,
        channel: DistributionChannel,
    ): ResolvedLibrarySpec {
        val base = resolveForPlatform(id, platform)
        if (base.policy == AcquisitionPolicy.DOWNLOADABLE && !channel.externalDownloadsAllowed) {
            return base.copy(
                policy = AcquisitionPolicy.MANUAL_INSTALL,
                download = null,
                notesKey = "vulkanshaders.libraries.note.channel_manual",
            )
        }
        return base
    }

    private fun resolveForPlatform(
        id: ProprietaryLibraryId,
        platform: NativePlatform,
    ): ResolvedLibrarySpec {
        val os = platform.os
        val isX64 = platform.arch == CpuArch.X86_64
        return when (id) {
            ProprietaryLibraryId.NVIDIA_NGX_CORE -> when {
                os == OsFamily.WINDOWS && isX64 -> ResolvedLibrarySpec(
                    id = id,
                    policy = AcquisitionPolicy.SYSTEM_PATH,
                    filePrefixes = listOf("_nvngx", "nvngx"),
                    expectedFileName = "_nvngx.dll",
                    download = null,
                    notesKey = "vulkanshaders.libraries.note.system_driver",
                )
                os == OsFamily.LINUX && isX64 -> ResolvedLibrarySpec(
                    id = id,
                    policy = AcquisitionPolicy.SYSTEM_PATH,
                    filePrefixes = listOf("libnvidia-ngx"),
                    expectedFileName = "libnvidia-ngx.so.1",
                    download = null,
                    notesKey = "vulkanshaders.libraries.note.system_driver",
                )
                else -> unsupported(id)
            }

            ProprietaryLibraryId.AMD_FSR3 -> when {
                os == OsFamily.WINDOWS && isX64 -> ResolvedLibrarySpec(
                    id = id,
                    policy = AcquisitionPolicy.DOWNLOADABLE,
                    filePrefixes = listOf(
                        "amd_fidelityfx_vk",
                        "amd_fidelityfx_upscaler_vk",
                        "amd_fidelityfx_loader_vk",
                        "ffx_fsr3_x64",
                    ),
                    expectedFileName = "amd_fidelityfx_vk.dll",
                    download = DownloadDescriptor(
                        url = AMD_FSR3_WIN_URL,
                        targetFileName = "amd_fidelityfx_vk.dll",
                        licenseUrl = AMD_FSR3_LICENSE_URL,
                        expectedSha256 = AMD_FSR3_WIN_SHA256,
                        licenseFileName = "amd_fsr3_LICENSE.txt",
                        maxBytes = 32L * 1024L * 1024L,
                    ),
                    notesKey = "vulkanshaders.libraries.note.downloadable_mit",
                )
                os == OsFamily.LINUX -> ResolvedLibrarySpec(
                    id = id,
                    policy = AcquisitionPolicy.MANUAL_INSTALL,
                    filePrefixes = listOf(
                        "libffx_fsr3_vk_x64",
                        "libamd_fidelityfx_vk",
                        "libamd_fidelityfx_upscaler_vk",
                        "libffx_fsr3upscaler_vk",
                    ),
                    expectedFileName = "libffx_fsr3_vk_x64.so",
                    download = null,
                    notesKey = "vulkanshaders.libraries.note.manual_no_prebuilt",
                )
                os == OsFamily.MACOS -> ResolvedLibrarySpec(
                    id = id,
                    policy = AcquisitionPolicy.MANUAL_INSTALL,
                    filePrefixes = listOf(
                        "libffx_fsr3_vk",
                        "libamd_fidelityfx_vk",
                        "libamd_fidelityfx_upscaler_vk",
                    ),
                    expectedFileName = "libffx_fsr3_vk.dylib",
                    download = null,
                    notesKey = "vulkanshaders.libraries.note.manual_no_prebuilt",
                )
                else -> unsupported(id)
            }

            ProprietaryLibraryId.INTEL_XESS -> when {
                os == OsFamily.WINDOWS && isX64 -> ResolvedLibrarySpec(
                    id = id,
                    policy = AcquisitionPolicy.DOWNLOADABLE,
                    filePrefixes = listOf("libxess", "xess"),
                    expectedFileName = "libxess.dll",
                    download = DownloadDescriptor(
                        url = INTEL_XESS_WIN_ZIP_URL,
                        targetFileName = "libxess.dll",
                        licenseFileName = "intel_xess_LICENSE.txt",
                        expectedSha256 = INTEL_XESS_WIN_ZIP_SHA256,
                        archiveBinaryEntryName = "libxess.dll",
                        archiveLicenseEntryName = "LICENSE.txt",
                        maxBytes = 128L * 1024L * 1024L,
                    ),
                    notesKey = "vulkanshaders.libraries.note.downloadable_official",
                )
                os == OsFamily.LINUX && isX64 -> ResolvedLibrarySpec(
                    id = id,
                    policy = AcquisitionPolicy.MANUAL_INSTALL,
                    filePrefixes = listOf("libxess"),
                    expectedFileName = "libxess.so",
                    download = null,
                    notesKey = "vulkanshaders.libraries.note.manual_no_prebuilt",
                )
                else -> unsupported(id)
            }

            ProprietaryLibraryId.NVIDIA_DLSS_SR -> when {
                os == OsFamily.WINDOWS && isX64 -> ResolvedLibrarySpec(
                    id = id,
                    policy = AcquisitionPolicy.MANUAL_INSTALL,
                    filePrefixes = listOf("nvngx_dlss"),
                    expectedFileName = "nvngx_dlss.dll",
                    download = null,
                    notesKey = "vulkanshaders.libraries.note.manual_eula",
                )
                os == OsFamily.LINUX && isX64 -> ResolvedLibrarySpec(
                    id = id,
                    policy = AcquisitionPolicy.MANUAL_INSTALL,
                    filePrefixes = listOf("libnvidia-ngx-dlss"),
                    expectedFileName = "libnvidia-ngx-dlss.so",
                    download = null,
                    notesKey = "vulkanshaders.libraries.note.manual_eula",
                )
                else -> unsupported(id)
            }

            ProprietaryLibraryId.NVIDIA_DLSS_RR -> when {
                os == OsFamily.WINDOWS && isX64 -> ResolvedLibrarySpec(
                    id = id,
                    policy = AcquisitionPolicy.MANUAL_INSTALL,
                    filePrefixes = listOf("nvngx_dlssd"),
                    expectedFileName = "nvngx_dlssd.dll",
                    download = null,
                    notesKey = "vulkanshaders.libraries.note.manual_eula",
                )
                os == OsFamily.LINUX && isX64 -> ResolvedLibrarySpec(
                    id = id,
                    policy = AcquisitionPolicy.MANUAL_INSTALL,
                    filePrefixes = listOf("libnvidia-ngx-dlssd", "libnvidia-ngx-builtin-dlssd"),
                    expectedFileName = "libnvidia-ngx-dlssd.so",
                    download = null,
                    notesKey = "vulkanshaders.libraries.note.manual_eula",
                )
                else -> unsupported(id)
            }

            ProprietaryLibraryId.NVIDIA_DLSS_FG -> when {
                os == OsFamily.WINDOWS && isX64 -> ResolvedLibrarySpec(
                    id = id,
                    policy = AcquisitionPolicy.MANUAL_INSTALL,
                    filePrefixes = listOf("nvngx_dlssg"),
                    expectedFileName = "nvngx_dlssg.dll",
                    download = null,
                    notesKey = "vulkanshaders.libraries.note.manual_eula",
                )
                os == OsFamily.LINUX && isX64 -> ResolvedLibrarySpec(
                    id = id,
                    policy = AcquisitionPolicy.MANUAL_INSTALL,
                    filePrefixes = listOf("libnvidia-ngx-dlssg"),
                    expectedFileName = "libnvidia-ngx-dlssg.so",
                    download = null,
                    notesKey = "vulkanshaders.libraries.note.manual_eula",
                )
                else -> unsupported(id)
            }

            ProprietaryLibraryId.NVIDIA_NRD -> when {
                os == OsFamily.WINDOWS && isX64 -> ResolvedLibrarySpec(
                    id = id,
                    policy = AcquisitionPolicy.MANUAL_INSTALL,
                    filePrefixes = listOf("nrd", "libnrd"),
                    expectedFileName = "NRD.dll",
                    download = null,
                    notesKey = "vulkanshaders.libraries.note.manual_eula",
                )
                os == OsFamily.LINUX && isX64 -> ResolvedLibrarySpec(
                    id = id,
                    policy = AcquisitionPolicy.MANUAL_INSTALL,
                    filePrefixes = listOf("libnrd"),
                    expectedFileName = "libNRD.so",
                    download = null,
                    notesKey = "vulkanshaders.libraries.note.manual_eula",
                )
                else -> unsupported(id)
            }
        }
    }

    private fun unsupported(id: ProprietaryLibraryId): ResolvedLibrarySpec = ResolvedLibrarySpec(
        id = id,
        policy = AcquisitionPolicy.UNSUPPORTED,
        filePrefixes = emptyList(),
        expectedFileName = null,
        download = null,
        notesKey = "vulkanshaders.libraries.note.unsupported_os",
    )
}
