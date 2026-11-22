package dev.vulkanshaders.sdk

import dev.vulkanshaders.VulkanShaders
import dev.vulkanshaders.backend.NativeCoreBridge
import dev.vulkanshaders.config.DenoiserMode
import dev.vulkanshaders.config.FrameGenMode
import dev.vulkanshaders.config.UpscalerMode
import java.io.IOException
import java.io.InputStream
import java.net.URI
import java.net.http.HttpClient
import java.net.http.HttpRequest
import java.net.http.HttpResponse
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.StandardCopyOption
import java.security.MessageDigest
import java.time.Duration
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicLong
import java.util.zip.ZipFile

data class LibraryStatusEntry(
    val spec: ResolvedLibrarySpec,
    val state: LibraryInstallState,
    val detectedPath: Path? = null,
    val progressPercent: Int = 0,
    val errorMessage: String? = null,
)

/**
 * Manages discovery, ToS-compliant user-initiated downloading, license notice preservation, and
 * restart-required tracking for optional proprietary rendering libraries.
 */
object ProprietaryLibraryManager {
    private const val MAX_REDIRECTS = 5
    private const val MIN_BINARY_BYTES = 16L * 1024L
    private const val MAX_LICENSE_BYTES = 1024L * 1024L
    private const val CONNECT_TIMEOUT_SECONDS = 15L
    private const val REQUEST_TIMEOUT_SECONDS = 180L

    private val stateLock = Any()
    private val stateVersionCounter = AtomicLong(0L)

    private val executor = Executors.newSingleThreadExecutor { runnable ->
        Thread(runnable, "hyaloclast-sdk-downloader").apply { isDaemon = true }
    }

    private val httpClient: HttpClient = HttpClient.newBuilder()
        .followRedirects(HttpClient.Redirect.NEVER)
        .connectTimeout(Duration.ofSeconds(CONNECT_TIMEOUT_SECONDS))
        .build()

    @Volatile
    var platform: NativePlatform = NativePlatform.detect()
        private set

    @Volatile
    var channel: DistributionChannel = DistributionChannel.detect()
        private set

    @Volatile
    private var gameDir: Path? = null

    @Volatile
    private var systemDirsOverride: List<Path>? = null

    @Volatile
    private var capabilitiesProvider: () -> Int = { NativeCoreBridge.queryCapabilities() }

    private val startupPresentIds = ConcurrentHashMap.newKeySet<ProprietaryLibraryId>()
    private val pendingRestartIds = ConcurrentHashMap.newKeySet<ProprietaryLibraryId>()
    private val activeDownloads = ConcurrentHashMap<ProprietaryLibraryId, Int>()
    private val downloadErrors = ConcurrentHashMap<ProprietaryLibraryId, String>()
    private val cachedEntries = ConcurrentHashMap<ProprietaryLibraryId, LibraryStatusEntry>()

    @Volatile
    var restartRequired: Boolean = false
        private set

    val stateVersion: Long
        get() = stateVersionCounter.get()

    @JvmStatic
    fun nativesDir(): Path {
        val base = gameDir ?: NativeCoreBridge.resolveGameDir() ?: Path.of(".")
        return base.resolve("natives").toAbsolutePath().normalize()
    }

    @JvmStatic
    fun ensureNativesDir(): Path {
        val dir = nativesDir()
        runCatching { Files.createDirectories(dir) }
        return dir
    }

    @JvmStatic
    fun initialize(
        gameDirectory: Path?,
        detectedPlatform: NativePlatform = NativePlatform.detect(),
        distributionChannel: DistributionChannel = DistributionChannel.detect(),
        systemDirs: List<Path>? = null,
        capabilitiesSupplier: () -> Int = { NativeCoreBridge.queryCapabilities() },
    ) {
        synchronized(stateLock) {
            gameDir = gameDirectory?.toAbsolutePath()?.normalize()
            platform = detectedPlatform
            channel = distributionChannel
            systemDirsOverride = systemDirs
            capabilitiesProvider = capabilitiesSupplier

            startupPresentIds.clear()
            pendingRestartIds.clear()
            activeDownloads.clear()
            downloadErrors.clear()
            cachedEntries.clear()
            restartRequired = false

            val specs = ProprietaryLibraryCatalog.resolveAll(platform, channel)
            val caps = runCatching { capabilitiesProvider() }.getOrDefault(0)
            for (spec in specs) {
                if (isActiveByCapability(spec.id, caps) || locateBinary(spec) != null) {
                    startupPresentIds.add(spec.id)
                }
            }
            refreshLocked()
        }
    }

    @JvmStatic
    fun refresh(): List<LibraryStatusEntry> = synchronized(stateLock) {
        refreshLocked()
    }

    private fun refreshLocked(): List<LibraryStatusEntry> {
        val specs = ProprietaryLibraryCatalog.resolveAll(platform, channel)
        val caps = runCatching { capabilitiesProvider() }.getOrDefault(0)
        val result = ArrayList<LibraryStatusEntry>(specs.size)

        for (spec in specs) {
            val activeProgress = activeDownloads[spec.id]
            val error = downloadErrors[spec.id]
            val foundPath = locateBinary(spec)
            val capActive = isActiveByCapability(spec.id, caps)

            val state = when {
                spec.policy == AcquisitionPolicy.UNSUPPORTED -> {
                    pendingRestartIds.remove(spec.id)
                    LibraryInstallState.UNSUPPORTED
                }
                activeProgress != null -> LibraryInstallState.DOWNLOADING
                capActive -> {
                    pendingRestartIds.remove(spec.id)
                    LibraryInstallState.ACTIVE
                }
                spec.id in pendingRestartIds && foundPath != null -> LibraryInstallState.RESTART_REQUIRED
                foundPath != null -> {
                    if (spec.id in startupPresentIds) {
                        LibraryInstallState.INSTALLED
                    } else {
                        // Newly placed into the folder after startup — requires Minecraft restart to load safely.
                        pendingRestartIds.add(spec.id)
                        LibraryInstallState.RESTART_REQUIRED
                    }
                }
                else -> {
                    pendingRestartIds.remove(spec.id)
                    when {
                        error != null -> LibraryInstallState.ERROR
                        spec.policy == AcquisitionPolicy.SYSTEM_PATH -> LibraryInstallState.SYSTEM_MISSING
                        spec.policy == AcquisitionPolicy.DOWNLOADABLE -> LibraryInstallState.AVAILABLE_TO_DOWNLOAD
                        else -> LibraryInstallState.MANUAL_REQUIRED
                    }
                }
            }

            val entry = LibraryStatusEntry(
                spec = spec,
                state = state,
                detectedPath = foundPath,
                progressPercent = activeProgress ?: 0,
                errorMessage = error,
            )
            cachedEntries[spec.id] = entry
            result.add(entry)
        }

        restartRequired = pendingRestartIds.isNotEmpty()
        stateVersionCounter.incrementAndGet()
        return result
    }

    private fun updateDownloadProgress(id: ProprietaryLibraryId, pct: Int) {
        synchronized(stateLock) {
            if (activeDownloads[id] == null) return
            val clamped = pct.coerceIn(0, 100)
            activeDownloads[id] = clamped
            val prev = cachedEntries[id]
            if (prev != null) {
                cachedEntries[id] = prev.copy(
                    state = LibraryInstallState.DOWNLOADING,
                    progressPercent = clamped,
                    errorMessage = null,
                )
            }
            stateVersionCounter.incrementAndGet()
        }
    }

    @JvmStatic
    fun entries(): List<LibraryStatusEntry> {
        if (cachedEntries.isEmpty()) return refresh()
        return ProprietaryLibraryId.entries.mapNotNull { cachedEntries[it] }
    }

    @JvmStatic
    fun entryFor(id: ProprietaryLibraryId): LibraryStatusEntry {
        if (cachedEntries.isEmpty()) refresh()
        return cachedEntries[id] ?: refresh().first { it.spec.id == id }
    }

    @JvmStatic
    fun isPendingRestart(id: ProprietaryLibraryId): Boolean = id in pendingRestartIds

    @JvmStatic
    fun isPendingRestartForUpscaler(mode: UpscalerMode): Boolean = when (mode) {
        UpscalerMode.FSR3 -> isPendingRestart(ProprietaryLibraryId.AMD_FSR3)
        UpscalerMode.XESS -> isPendingRestart(ProprietaryLibraryId.INTEL_XESS)
        UpscalerMode.DLSS -> isPendingRestart(ProprietaryLibraryId.NVIDIA_DLSS_SR)
        UpscalerMode.NONE, UpscalerMode.TSR -> false
    }

    @JvmStatic
    fun isPendingRestartForDenoiser(mode: DenoiserMode): Boolean = when (mode) {
        DenoiserMode.NRD -> isPendingRestart(ProprietaryLibraryId.NVIDIA_NRD)
        DenoiserMode.DLSS_RR -> isPendingRestart(ProprietaryLibraryId.NVIDIA_DLSS_RR)
        DenoiserMode.NONE, DenoiserMode.SVGF -> false
    }

    @JvmStatic
    fun isPendingRestartForFrameGen(mode: FrameGenMode): Boolean = when (mode) {
        FrameGenMode.FSR3_FG -> isPendingRestart(ProprietaryLibraryId.AMD_FSR3)
        FrameGenMode.DLSS_FG -> isPendingRestart(ProprietaryLibraryId.NVIDIA_DLSS_FG)
        FrameGenMode.NONE, FrameGenMode.OPTICAL_FLOW -> false
    }

    /**
     * Initiates a user-requested download for a library whose resolved policy is [AcquisitionPolicy.DOWNLOADABLE].
     * Never called automatically on startup.
     */
    @JvmStatic
    fun startDownload(
        id: ProprietaryLibraryId,
        onComplete: (Boolean, String?) -> Unit = { _, _ -> },
    ): Boolean {
        val spec: ResolvedLibrarySpec
        val descriptor: DownloadDescriptor
        synchronized(stateLock) {
            spec = ProprietaryLibraryCatalog.resolve(id, platform, channel)
            val desc = spec.download
            if (spec.policy != AcquisitionPolicy.DOWNLOADABLE || desc == null || !channel.externalDownloadsAllowed) {
                return false
            }
            if (activeDownloads.putIfAbsent(id, 0) != null) {
                return false
            }
            descriptor = desc
            downloadErrors.remove(id)
            refreshLocked()
        }

        executor.submit {
            try {
                downloadAndInstall(spec, descriptor) { pct ->
                    updateDownloadProgress(id, pct)
                }
                synchronized(stateLock) {
                    pendingRestartIds.add(id)
                    activeDownloads.remove(id)
                    downloadErrors.remove(id)
                    refreshLocked()
                }
                onComplete(true, null)
            } catch (t: Throwable) {
                val msg = t.message?.takeIf { it.isNotBlank() } ?: t.javaClass.simpleName
                VulkanShaders.log.warn("Failed to download {}: {}", id.displayName, msg)
                synchronized(stateLock) {
                    activeDownloads.remove(id)
                    downloadErrors[id] = msg
                    refreshLocked()
                }
                onComplete(false, msg)
            }
        }
        return true
    }

    internal fun downloadAndInstall(
        spec: ResolvedLibrarySpec,
        descriptor: DownloadDescriptor,
        onProgress: (Int) -> Unit = {},
    ): Path {
        require(DistributionChannel.isAllowedOfficialHttpsUrl(descriptor.url)) {
            "Download URL is not an allowlisted official HTTPS vendor endpoint."
        }
        val targetDir = ensureNativesDir()
        val stagingDir = targetDir.resolve(".staging")
        Files.createDirectories(stagingDir)
        val partFile = stagingDir.resolve("${spec.id.id}.part")
        Files.deleteIfExists(partFile)

        try {
            val sha256 = fetchToFile(descriptor.url, partFile, descriptor.maxBytes, onProgress)
            if (descriptor.expectedSha256 != null && !sha256.equals(descriptor.expectedSha256, ignoreCase = true)) {
                throw IOException("SHA-256 verification failed for ${spec.id.displayName}.")
            }

            val installedBinary = if (descriptor.archiveBinaryEntryName != null) {
                installFromZipArchive(
                    zipPath = partFile,
                    targetDir = targetDir,
                    binaryEntryName = descriptor.archiveBinaryEntryName,
                    targetFileName = descriptor.targetFileName,
                    licenseEntryName = descriptor.archiveLicenseEntryName,
                    licenseFileName = descriptor.licenseFileName,
                    maxBinaryBytes = descriptor.maxBytes,
                )
            } else {
                validateNativeBinary(partFile, descriptor.targetFileName)
                val finalPath = targetDir.resolve(descriptor.targetFileName)
                descriptor.licenseUrl?.let { licUrl ->
                    val licPart = stagingDir.resolve("${spec.id.id}_license.part")
                    try {
                        val licDir = targetDir.resolve("licenses")
                        Files.createDirectories(licDir)
                        fetchToFile(licUrl, licPart, MAX_LICENSE_BYTES) {}
                        atomicMove(licPart, licDir.resolve(descriptor.licenseFileName))
                    } finally {
                        runCatching { Files.deleteIfExists(licPart) }
                    }
                }
                atomicMove(partFile, finalPath)
                finalPath
            }

            onProgress(100)
            return installedBinary
        } finally {
            runCatching { Files.deleteIfExists(partFile) }
        }
    }

    internal fun installFromZipArchive(
        zipPath: Path,
        targetDir: Path,
        binaryEntryName: String,
        targetFileName: String,
        licenseEntryName: String?,
        licenseFileName: String,
        maxBinaryBytes: Long = 128L * 1024L * 1024L,
    ): Path {
        val stagingDir = targetDir.resolve(".staging")
        Files.createDirectories(stagingDir)
        val extractedPart = stagingDir.resolve("$targetFileName.part")
        Files.deleteIfExists(extractedPart)

        try {
            ZipFile(zipPath.toFile()).use { zip ->
                var binaryFound = false
                var licenseSaved = false
                val entries = zip.entries()
                while (entries.hasMoreElements()) {
                    val entry = entries.nextElement()
                    if (entry.isDirectory) continue
                    val normalizedName = entry.name.replace('\\', '/')
                    if (normalizedName.contains("..") || normalizedName.startsWith("/")) {
                        throw IOException("Rejected unsafe archive entry path: ${entry.name}")
                    }
                    val baseName = normalizedName.substringAfterLast('/')
                    if (!binaryFound && baseName.equals(binaryEntryName, ignoreCase = true)) {
                        zip.getInputStream(entry).use { input ->
                            copyBounded(input, extractedPart, maxBinaryBytes)
                        }
                        binaryFound = true
                    } else if (!licenseSaved && licenseEntryName != null &&
                        baseName.equals(licenseEntryName, ignoreCase = true)
                    ) {
                        val licDir = targetDir.resolve("licenses")
                        Files.createDirectories(licDir)
                        val licPart = stagingDir.resolve("${licenseFileName}.part")
                        try {
                            zip.getInputStream(entry).use { input ->
                                copyBounded(input, licPart, MAX_LICENSE_BYTES)
                            }
                            atomicMove(licPart, licDir.resolve(licenseFileName))
                            licenseSaved = true
                        } finally {
                            runCatching { Files.deleteIfExists(licPart) }
                        }
                    }
                }
                if (!binaryFound) {
                    throw IOException("Archive did not contain expected library '$binaryEntryName'.")
                }
                if (licenseEntryName != null && !licenseSaved) {
                    throw IOException("Archive did not contain expected license '$licenseEntryName'.")
                }
            }

            validateNativeBinary(extractedPart, targetFileName)
            val finalPath = targetDir.resolve(targetFileName)
            atomicMove(extractedPart, finalPath)
            return finalPath
        } finally {
            runCatching { Files.deleteIfExists(extractedPart) }
        }
    }

    private fun copyBounded(input: InputStream, dest: Path, maxBytes: Long) {
        val buf = ByteArray(64 * 1024)
        var total = 0L
        Files.newOutputStream(dest).use { out ->
            while (true) {
                val n = input.read(buf)
                if (n < 0) break
                total += n
                if (total > maxBytes) {
                    throw IOException("Extracted archive entry exceeds maximum allowed size ($maxBytes bytes).")
                }
                out.write(buf, 0, n)
            }
        }
    }

    internal fun validateNativeBinary(file: Path, targetFileName: String) {
        val size = Files.size(file)
        if (size < MIN_BINARY_BYTES) {
            throw IOException("Downloaded file '$targetFileName' is too small ($size bytes).")
        }
        val header = ByteArray(4)
        Files.newInputStream(file).use { input ->
            val read = input.read(header)
            if (read < 4) throw IOException("Downloaded file '$targetFileName' is truncated.")
        }
        val lower = targetFileName.lowercase()
        val validMagic = when {
            lower.endsWith(".dll") -> header[0] == 'M'.code.toByte() && header[1] == 'Z'.code.toByte()
            lower.endsWith(".so") || lower.contains(".so.") ->
                header[0] == 0x7F.toByte() &&
                    header[1] == 'E'.code.toByte() &&
                    header[2] == 'L'.code.toByte() &&
                    header[3] == 'F'.code.toByte()
            lower.endsWith(".dylib") -> isMachOMagic(header)
            else -> false
        }
        if (!validMagic) {
            throw IOException("Downloaded file '$targetFileName' failed native binary header verification.")
        }
    }

    private fun isMachOMagic(h: ByteArray): Boolean {
        val b0 = h[0].toInt() and 0xFF
        val b1 = h[1].toInt() and 0xFF
        val b2 = h[2].toInt() and 0xFF
        val b3 = h[3].toInt() and 0xFF
        val be = (b0 shl 24) or (b1 shl 16) or (b2 shl 8) or b3
        return be == 0xFEEDFACE.toInt() ||
            be == 0xCEFAEDFE.toInt() ||
            be == 0xFEEDFACF.toInt() ||
            be == 0xCFFAEDFE.toInt() ||
            be == 0xCAFEBABE.toInt() ||
            be == 0xBEBAFECA.toInt()
    }

    private fun fetchToFile(
        initialUrl: String,
        dest: Path,
        maxBytes: Long,
        onProgress: (Int) -> Unit,
    ): String {
        var currentUrl = initialUrl
        var redirects = 0
        while (true) {
            if (!DistributionChannel.isAllowedOfficialHttpsUrl(currentUrl)) {
                throw IOException("Blocked non-allowlisted download URL: $currentUrl")
            }
            val req = HttpRequest.newBuilder(URI.create(currentUrl))
                .GET()
                .timeout(Duration.ofSeconds(REQUEST_TIMEOUT_SECONDS))
                .header("User-Agent", "Hyaloclast-VulkanShaders/1.0")
                .build()
            val resp = httpClient.send(req, HttpResponse.BodyHandlers.ofInputStream())
            val code = resp.statusCode()
            if (code in setOf(301, 302, 303, 307, 308)) {
                resp.body().close()
                if (++redirects > MAX_REDIRECTS) {
                    throw IOException("Too many redirects while downloading library.")
                }
                val loc = resp.headers().firstValue("Location").orElseThrow {
                    IOException("Redirect response missing Location header.")
                }
                currentUrl = URI.create(currentUrl).resolve(loc).toString()
                continue
            }
            if (code != 200) {
                resp.body().close()
                throw IOException("Official download server returned HTTP $code.")
            }

            val totalBytes = resp.headers().firstValueAsLong("Content-Length").orElse(-1L)
            if (totalBytes > maxBytes) {
                resp.body().close()
                throw IOException("Download exceeds maximum allowed size ($maxBytes bytes).")
            }
            return resp.body().use { input ->
                streamWithDigest(input, dest, totalBytes, maxBytes, onProgress)
            }
        }
    }

    private fun streamWithDigest(
        input: InputStream,
        dest: Path,
        totalBytes: Long,
        maxBytes: Long,
        onProgress: (Int) -> Unit,
    ): String {
        val digest = MessageDigest.getInstance("SHA-256")
        val buf = ByteArray(64 * 1024)
        var downloaded = 0L
        var lastPct = -1
        Files.newOutputStream(dest).use { out ->
            while (true) {
                val n = input.read(buf)
                if (n < 0) break
                downloaded += n
                if (downloaded > maxBytes) {
                    throw IOException("Download exceeds maximum allowed size ($maxBytes bytes).")
                }
                digest.update(buf, 0, n)
                out.write(buf, 0, n)
                if (totalBytes > 0L) {
                    val pct = ((downloaded * 100L) / totalBytes).toInt().coerceIn(0, 99)
                    if (pct != lastPct) {
                        lastPct = pct
                        onProgress(pct)
                    }
                }
            }
        }
        return digest.digest().joinToString("") { "%02x".format(it) }
    }

    private fun atomicMove(src: Path, dst: Path) {
        try {
            Files.move(src, dst, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE)
        } catch (_: Throwable) {
            Files.move(src, dst, StandardCopyOption.REPLACE_EXISTING)
        }
    }

    private fun locateBinary(spec: ResolvedLibrarySpec): Path? {
        if (spec.filePrefixes.isEmpty()) return null
        val dirs = buildList {
            if (spec.policy != AcquisitionPolicy.SYSTEM_PATH) {
                add(nativesDir())
            }
            addAll(systemDirsOverride ?: platform.systemLibraryDirs())
            if (spec.policy == AcquisitionPolicy.SYSTEM_PATH) {
                add(nativesDir())
            }
        }
        for (dir in dirs) {
            if (!Files.isDirectory(dir)) continue
            val match = runCatching {
                Files.list(dir).use { stream ->
                    stream
                        .filter { Files.isRegularFile(it) }
                        .filter { path -> matchesLibraryName(path.fileName.toString(), spec.filePrefixes, platform.os) }
                        .findFirst()
                        .orElse(null)
                }
            }.getOrNull()
            if (match != null) return match
        }
        return null
    }

    internal fun matchesLibraryName(fileName: String, prefixes: List<String>, os: OsFamily): Boolean {
        val lower = fileName.lowercase()
        val extOk = when (os) {
            OsFamily.WINDOWS -> lower.endsWith(".dll")
            OsFamily.MACOS -> lower.endsWith(".dylib")
            OsFamily.LINUX -> lower.endsWith(".so") || lower.contains(".so.")
            OsFamily.UNKNOWN -> lower.endsWith(".so") || lower.contains(".so.") || lower.endsWith(".dll")
        }
        if (!extOk) return false
        return prefixes.any { prefix ->
            val p = prefix.lowercase()
            lower == "$p.dll" ||
                lower == "$p.so" ||
                lower == "$p.dylib" ||
                lower.startsWith("$p.so.")
        }
    }

    private fun isActiveByCapability(id: ProprietaryLibraryId, caps: Int): Boolean = when (id) {
        ProprietaryLibraryId.NVIDIA_NGX_CORE ->
            (caps and (NativeCoreBridge.CAP_DLSS_SR or NativeCoreBridge.CAP_DLSS_RR or NativeCoreBridge.CAP_DLSS_FG)) != 0
        ProprietaryLibraryId.AMD_FSR3 -> (caps and NativeCoreBridge.CAP_FSR3) != 0
        ProprietaryLibraryId.INTEL_XESS -> (caps and NativeCoreBridge.CAP_XESS) != 0
        ProprietaryLibraryId.NVIDIA_DLSS_SR -> (caps and NativeCoreBridge.CAP_DLSS_SR) != 0
        ProprietaryLibraryId.NVIDIA_DLSS_RR -> (caps and NativeCoreBridge.CAP_DLSS_RR) != 0
        ProprietaryLibraryId.NVIDIA_DLSS_FG -> (caps and NativeCoreBridge.CAP_DLSS_FG) != 0
        ProprietaryLibraryId.NVIDIA_NRD -> (caps and NativeCoreBridge.CAP_NRD) != 0
    }
}
