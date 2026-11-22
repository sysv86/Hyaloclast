package dev.vulkanshaders.diagnostics

import dev.vulkanshaders.VulkanShaders
import dev.vulkanshaders.backend.NativeCoreBridge
import dev.vulkanshaders.bridge.StandaloneRenderPearlBridge
import dev.vulkanshaders.gui.ShaderNotices
import dev.vulkanshaders.pack.ShaderPackManager
import net.fabricmc.loader.api.FabricLoader
import java.io.PrintWriter
import java.io.StringWriter
import java.nio.charset.StandardCharsets
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.StandardCopyOption
import java.nio.file.StandardOpenOption
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Structured in-memory ring buffer, persistent session log, shader compile diagnostic parser,
 * and crash-log dumper for Hyaloclast (VulkanShaders).
 *
 * When Pyroclast (`VulkanTurbo`) is installed ([StandaloneRenderPearlBridge.vulkanTurboLoaded]),
 * general log entries delegate to `dev.vulkanturbo.diagnostics.VulkanDiagnosticLog` with
 * `modId = "Hyaloclast"` so both mods share a unified session log and crash dump, while
 * structured shader compile diagnostics remain directly inspectable via [latestShaderDiagnostic].
 */
object HyaloclastErrorLog {
    private const val MAX_ENTRIES: Int = 256
    private const val CRASH_SECTION_TAIL_ENTRIES: Int = 40

    private val entryTimestampFormatter: DateTimeFormatter = DateTimeFormatter.ofPattern("HH:mm:ss.SSS")
    private val fullTimestampFormatter: DateTimeFormatter = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss")
    private val crashFileTimestampFormatter: DateTimeFormatter = DateTimeFormatter.ofPattern("yyyy-MM-dd_HH-mm-ss")

    // Matches paths like "hooks/surface.vsl:42:10: error: ..." or "shaders/rt/pathtrace.rgen:128: ..."
    private val FILE_LINE_COL_REGEX = Regex(
        """([A-Za-z0-9_./\\-]+\.(?:vsl|glsl|comp|vert|frag|vsh|fsh|rgen|rchit|rahit|rmiss|rint|rcall)):(\d+)(?::\d+)?\s*:?\s*(.*)""",
        RegexOption.IGNORE_CASE,
    )

    // Matches glslang-style "ERROR: 0:42: '...'" or "ERROR: hooks/sky.vsl:42: ..."
    private val GLSLANG_ERROR_REGEX = Regex(
        """(?:ERROR|WARNING)\s*:\s*([^:\s]+):(\d+)\s*:\s*(.*)""",
        RegexOption.IGNORE_CASE,
    )

    // Matches "line 42" fallback
    private val LINE_ONLY_REGEX = Regex("""\bline\s+(\d+)\b""", RegexOption.IGNORE_CASE)

    enum class Level {
        INFO,
        WARN,
        ERROR,
    }

    /**
     * Structured diagnostic extracted from a `.vsl` / GLSL / `vsp-compiler` error message.
     */
    data class ShaderCompileDiagnostic(
        val sourceFile: String?,
        val lineNumber: Int?,
        val message: String,
        val rawOutput: String? = null,
    ) {
        fun formatLocation(): String = when {
            !sourceFile.isNullOrBlank() && lineNumber != null -> "$sourceFile:$lineNumber"
            !sourceFile.isNullOrBlank() -> sourceFile
            lineNumber != null -> "line $lineNumber"
            else -> "shader"
        }

        fun formatShort(): String {
            val loc = formatLocation()
            val cleanMsg = message.trim().ifEmpty { "Shader compilation failed" }
            return if (sourceFile != null || lineNumber != null) "$loc — $cleanMsg" else cleanMsg
        }
    }

    data class Entry(
        val timestamp: String,
        val level: Level,
        val source: String,
        val summary: String,
        val details: String? = null,
        val sourceFile: String? = null,
        val lineNumber: Int? = null,
    ) {
        fun hasShaderLocation(): Boolean = !sourceFile.isNullOrBlank() || lineNumber != null

        fun formatLocation(): String? = when {
            !sourceFile.isNullOrBlank() && lineNumber != null -> "$sourceFile:$lineNumber"
            !sourceFile.isNullOrBlank() -> sourceFile
            lineNumber != null -> "line $lineNumber"
            else -> null
        }
    }

    private object PyroclastLogDelegate {
        private fun toEntry(e: dev.vulkanturbo.diagnostics.VulkanDiagnosticLog.LogEntry): Entry =
            Entry(
                timestamp = e.timestamp,
                level = when (e.level) {
                    dev.vulkanturbo.diagnostics.VulkanDiagnosticLog.LogLevel.INFO -> Level.INFO
                    dev.vulkanturbo.diagnostics.VulkanDiagnosticLog.LogLevel.WARN -> Level.WARN
                    dev.vulkanturbo.diagnostics.VulkanDiagnosticLog.LogLevel.ERROR -> Level.ERROR
                },
                source = e.source,
                summary = e.summary,
                details = e.details,
            )

        fun hasUnreadErrors(): Boolean = dev.vulkanturbo.diagnostics.VulkanDiagnosticLog.hasUnreadErrors
        fun latestError(): Entry? = dev.vulkanturbo.diagnostics.VulkanDiagnosticLog.latestError?.let(::toEntry)
        fun lastCrashLogPath(): Path? = dev.vulkanturbo.diagnostics.VulkanDiagnosticLog.lastCrashLogPath
        fun logsDir(): Path = dev.vulkanturbo.diagnostics.VulkanDiagnosticLog.logsDir()
        fun crashLogsDir(): Path = dev.vulkanturbo.diagnostics.VulkanDiagnosticLog.crashLogsDir()
        fun latestLogFile(): Path = dev.vulkanturbo.diagnostics.VulkanDiagnosticLog.latestLogFile()
        fun markAllRead() = dev.vulkanturbo.diagnostics.VulkanDiagnosticLog.markAllRead()
        fun markCleanShutdown() = dev.vulkanturbo.diagnostics.VulkanDiagnosticLog.markCleanShutdown()
        fun clear() = dev.vulkanturbo.diagnostics.VulkanDiagnosticLog.clear()
        fun errorCount(): Int = dev.vulkanturbo.diagnostics.VulkanDiagnosticLog.errorCount()
        fun entriesSnapshot(): List<Entry> =
            dev.vulkanturbo.diagnostics.VulkanDiagnosticLog.entriesSnapshot().map(::toEntry)

        fun recordError(
            source: String,
            summary: String,
            details: String?,
            throwable: Throwable?,
            showToast: Boolean,
        ): Entry = toEntry(
            dev.vulkanturbo.diagnostics.VulkanDiagnosticLog.recordError(
                source = source,
                summary = summary,
                details = details,
                throwable = throwable,
                modId = VulkanShaders.PUBLIC_NAME,
                showToast = showToast,
            ),
        )

        fun recordWarning(
            source: String,
            summary: String,
            details: String?,
            throwable: Throwable?,
        ): Entry = toEntry(
            dev.vulkanturbo.diagnostics.VulkanDiagnosticLog.recordWarning(
                source = source,
                summary = summary,
                details = details,
                throwable = throwable,
                modId = VulkanShaders.PUBLIC_NAME,
            ),
        )

        fun recordInfo(
            source: String,
            summary: String,
            details: String?,
        ): Entry = toEntry(
            dev.vulkanturbo.diagnostics.VulkanDiagnosticLog.recordInfo(
                source = source,
                summary = summary,
                details = details,
                modId = VulkanShaders.PUBLIC_NAME,
            ),
        )

        fun formatFullLogText(): String = dev.vulkanturbo.diagnostics.VulkanDiagnosticLog.formatLogText()

        fun dumpCrashLog(reason: String, crashReportText: String?, throwable: Throwable?): Path? =
            dev.vulkanturbo.diagnostics.VulkanDiagnosticLog.dumpCrashLog(reason, crashReportText, throwable)

        fun installCrashHooks() = dev.vulkanturbo.diagnostics.VulkanDiagnosticLog.installCrashHooks()
    }

    private val lock = Any()
    private val entries = ArrayDeque<Entry>(MAX_ENTRIES)
    private var sessionLogInitialized: Boolean = false
    private val hooksInstalled = AtomicBoolean(false)
    private val cleanShutdown = AtomicBoolean(false)
    private val crashDumpWritten = AtomicBoolean(false)

    @Volatile
    private var localHasUnreadErrors: Boolean = false

    @Volatile
    private var localLatestError: Entry? = null

    @Volatile
    private var localLastCrashLogPath: Path? = null

    @Volatile
    var latestShaderDiagnostic: ShaderCompileDiagnostic? = null
        private set

    val hasUnreadErrors: Boolean
        get() = if (StandaloneRenderPearlBridge.vulkanTurboLoaded) {
            PyroclastLogDelegate.hasUnreadErrors()
        } else {
            localHasUnreadErrors
        }

    val latestError: Entry?
        get() = if (StandaloneRenderPearlBridge.vulkanTurboLoaded) {
            PyroclastLogDelegate.latestError()
        } else {
            localLatestError
        }

    val lastCrashLogPath: Path?
        get() = if (StandaloneRenderPearlBridge.vulkanTurboLoaded) {
            PyroclastLogDelegate.lastCrashLogPath()
        } else {
            localLastCrashLogPath
        }

    @JvmStatic
    fun resolveGameDir(): Path =
        runCatching { FabricLoader.getInstance().gameDir }
            .getOrElse { Path.of(System.getProperty("user.dir") ?: ".") }

    @JvmStatic
    fun logsDir(): Path =
        if (StandaloneRenderPearlBridge.vulkanTurboLoaded) {
            PyroclastLogDelegate.logsDir()
        } else {
            resolveGameDir().resolve("hyaloclast").resolve("logs")
        }

    @JvmStatic
    fun crashLogsDir(): Path =
        if (StandaloneRenderPearlBridge.vulkanTurboLoaded) {
            PyroclastLogDelegate.crashLogsDir()
        } else {
            resolveGameDir().resolve("hyaloclast").resolve("crash-logs")
        }

    @JvmStatic
    fun latestLogFile(): Path =
        if (StandaloneRenderPearlBridge.vulkanTurboLoaded) {
            PyroclastLogDelegate.latestLogFile()
        } else {
            logsDir().resolve("latest.log")
        }

    @JvmStatic
    fun markAllRead() {
        if (StandaloneRenderPearlBridge.vulkanTurboLoaded) {
            PyroclastLogDelegate.markAllRead()
            return
        }
        localHasUnreadErrors = false
    }

    @JvmStatic
    fun markCleanShutdown() {
        if (StandaloneRenderPearlBridge.vulkanTurboLoaded) {
            PyroclastLogDelegate.markCleanShutdown()
            return
        }
        cleanShutdown.set(true)
    }

    @JvmStatic
    fun clear() {
        latestShaderDiagnostic = null
        if (StandaloneRenderPearlBridge.vulkanTurboLoaded) {
            PyroclastLogDelegate.clear()
            return
        }
        synchronized(lock) {
            entries.clear()
        }
        localHasUnreadErrors = false
        localLatestError = null
    }

    @JvmStatic
    fun errorCount(): Int {
        if (StandaloneRenderPearlBridge.vulkanTurboLoaded) {
            return PyroclastLogDelegate.errorCount()
        }
        return synchronized(lock) {
            entries.count { it.level == Level.ERROR }
        }
    }

    @JvmStatic
    fun entriesSnapshot(): List<Entry> {
        if (StandaloneRenderPearlBridge.vulkanTurboLoaded) {
            return PyroclastLogDelegate.entriesSnapshot()
        }
        return synchronized(lock) {
            entries.toList()
        }
    }

    /**
     * Parses a raw compiler diagnostic from `vsp-compiler`, `shaderc`, `glslang`, or `naga`
     * into a structured [ShaderCompileDiagnostic] with exact source file and line number when present.
     */
    @JvmStatic
    fun parseShaderCompileDiagnostic(rawError: String): ShaderCompileDiagnostic {
        val cleaned = rawError.trim()
        if (cleaned.isEmpty()) {
            return ShaderCompileDiagnostic(
                sourceFile = null,
                lineNumber = null,
                message = "Unknown shader compilation error",
                rawOutput = null,
            )
        }

        for (line in cleaned.lineSequence()) {
            val trimmed = line.trim()
            if (trimmed.isEmpty()) continue

            FILE_LINE_COL_REGEX.find(trimmed)?.let { match ->
                val file = match.groupValues[1].replace('\\', '/')
                val lineNo = match.groupValues[2].toIntOrNull()
                val msg = match.groupValues[3].trim().ifEmpty { trimmed }
                return ShaderCompileDiagnostic(
                    sourceFile = file,
                    lineNumber = lineNo,
                    message = msg,
                    rawOutput = cleaned,
                )
            }

            GLSLANG_ERROR_REGEX.find(trimmed)?.let { match ->
                val rawFile = match.groupValues[1].trim()
                val file = rawFile.takeIf { it != "0" && it.isNotEmpty() }?.replace('\\', '/')
                val lineNo = match.groupValues[2].toIntOrNull()
                val msg = match.groupValues[3].trim().ifEmpty { trimmed }
                return ShaderCompileDiagnostic(
                    sourceFile = file,
                    lineNumber = lineNo,
                    message = msg,
                    rawOutput = cleaned,
                )
            }
        }

        val firstLine = cleaned.lineSequence().firstOrNull { it.isNotBlank() }?.trim() ?: cleaned
        val fallbackLineNo = LINE_ONLY_REGEX.find(cleaned)?.groupValues?.getOrNull(1)?.toIntOrNull()
        return ShaderCompileDiagnostic(
            sourceFile = null,
            lineNumber = fallbackLineNo,
            message = firstLine,
            rawOutput = cleaned,
        )
    }

    /**
     * Records a structured shader compilation error with extracted `.vsl` / GLSL file and line number,
     * and optionally displays a clickable toast showing the exact location.
     */
    @JvmStatic
    @JvmOverloads
    fun recordShaderCompileError(
        packName: String,
        rawError: String,
        statusCode: Int = -1,
        showToast: Boolean = true,
    ): ShaderCompileDiagnostic {
        val diagnostic = parseShaderCompileDiagnostic(rawError)
        latestShaderDiagnostic = diagnostic
        val location = diagnostic.formatLocation()
        val summary = buildString {
            append("Shader compile error in '").append(packName).append("'")
            if (diagnostic.sourceFile != null || diagnostic.lineNumber != null) {
                append(" at ").append(location)
            }
            if (statusCode < 0) {
                append(" (status ").append(statusCode).append(')')
            }
            append(": ").append(diagnostic.message)
        }

        if (StandaloneRenderPearlBridge.vulkanTurboLoaded) {
            PyroclastLogDelegate.recordError(
                source = "ShaderCompile",
                summary = summary,
                details = diagnostic.rawOutput ?: rawError,
                throwable = null,
                showToast = false,
            )
        } else {
            val entry = appendEntry(
                level = Level.ERROR,
                source = "ShaderCompile",
                summary = summary,
                details = diagnostic.rawOutput ?: rawError,
                throwable = null,
                sourceFile = diagnostic.sourceFile,
                lineNumber = diagnostic.lineNumber,
            )
            localHasUnreadErrors = true
            localLatestError = entry
            VulkanShaders.log.error("[ShaderCompile] {} — {}", summary, diagnostic.rawOutput ?: rawError)
        }

        if (showToast) {
            runCatching { ShaderNotices.showShaderCompileErrorToast(packName, diagnostic) }
        }
        return diagnostic
    }

    /**
     * Records a Vulkan `VK_ERROR_DEVICE_LOST` / GPU driver TDR safe-fallback recovery event
     * and notifies the player via [ShaderNotices.showDeviceLostNotice].
     */
    @JvmStatic
    @JvmOverloads
    fun recordDeviceLostRecovery(
        reason: String,
        details: String? = null,
        showToast: Boolean = true,
    ): Entry {
        val cleanReason = reason.trim().ifEmpty { "VK_ERROR_DEVICE_LOST" }
        val summary = "Vulkan device lost / driver TDR ($cleanReason) — reverted to safe raster fallback."
        val entry = recordError(
            source = "DeviceLost",
            summary = summary,
            details = details,
            throwable = null,
            showToast = false,
        )
        if (showToast) {
            runCatching { ShaderNotices.showDeviceLostNotice(cleanReason) }
        }
        return entry
    }

    @JvmStatic
    @JvmOverloads
    fun recordError(
        source: String,
        summary: String,
        details: String? = null,
        throwable: Throwable? = null,
        showToast: Boolean = true,
        sourceFile: String? = null,
        lineNumber: Int? = null,
    ): Entry {
        if (StandaloneRenderPearlBridge.vulkanTurboLoaded) {
            val formattedSummary = if (!sourceFile.isNullOrBlank() || lineNumber != null) {
                val loc = listOfNotNull(sourceFile?.takeIf { it.isNotBlank() }, lineNumber?.toString()).joinToString(":")
                "($loc) $summary"
            } else {
                summary
            }
            return PyroclastLogDelegate.recordError(source, formattedSummary, details, throwable, showToast)
        }
        val entry = appendEntry(Level.ERROR, source, summary, details, throwable, sourceFile, lineNumber)
        localHasUnreadErrors = true
        localLatestError = entry
        if (throwable != null) {
            VulkanShaders.log.error("[{}] {}{}", source, summary, detailsSuffix(details), throwable)
        } else {
            VulkanShaders.log.error("[{}] {}{}", source, summary, detailsSuffix(details))
        }
        if (showToast) {
            runCatching { ShaderNotices.showClickableErrorToast(summary) }
        }
        return entry
    }

    @JvmStatic
    @JvmOverloads
    fun recordWarning(
        source: String,
        summary: String,
        details: String? = null,
        throwable: Throwable? = null,
    ): Entry {
        if (StandaloneRenderPearlBridge.vulkanTurboLoaded) {
            return PyroclastLogDelegate.recordWarning(source, summary, details, throwable)
        }
        val entry = appendEntry(Level.WARN, source, summary, details, throwable)
        if (throwable != null) {
            VulkanShaders.log.warn("[{}] {}{}", source, summary, detailsSuffix(details), throwable)
        } else {
            VulkanShaders.log.warn("[{}] {}{}", source, summary, detailsSuffix(details))
        }
        return entry
    }

    @JvmStatic
    @JvmOverloads
    fun recordInfo(
        source: String,
        summary: String,
        details: String? = null,
    ): Entry {
        if (StandaloneRenderPearlBridge.vulkanTurboLoaded) {
            return PyroclastLogDelegate.recordInfo(source, summary, details)
        }
        val entry = appendEntry(Level.INFO, source, summary, details, null)
        VulkanShaders.log.info("[{}] {}{}", source, summary, detailsSuffix(details))
        return entry
    }

    @JvmStatic
    fun formatFullLogText(): String {
        if (StandaloneRenderPearlBridge.vulkanTurboLoaded) {
            return PyroclastLogDelegate.formatFullLogText()
        }
        val snapshot = entriesSnapshot()
        if (snapshot.isEmpty()) {
            return "No Hyaloclast errors or warnings recorded in this session."
        }
        val sb = StringBuilder(snapshot.size * 128)
        for (i in snapshot.indices) {
            if (i > 0) sb.append('\n')
            appendFormattedEntry(sb, snapshot[i])
        }
        return sb.toString()
    }

    @JvmStatic
    fun formatLatestErrorLog(): String {
        val latest = latestError ?: return formatFullLogText()
        val sb = StringBuilder(256)
        appendFormattedEntry(sb, latest)
        return sb.toString()
    }

    /**
     * Builds the Hyaloclast-specific diagnostic summary block without trailing log entries so
     * Pyroclast's unified crash report can include it without duplicating the session log tail.
     */
    @JvmStatic
    fun buildHyaloclastDiagnosticBlock(): String {
        val sb = StringBuilder(1024)
        sb.append("-- Hyaloclast (VulkanShaders) Diagnostics --\n")

        val modVersion = runCatching {
            FabricLoader.getInstance()
                .getModContainer(VulkanShaders.MOD_ID)
                .map { it.metadata.version.friendlyString }
                .orElse("development")
        }.getOrElse { "unknown" }

        val pyroclastLoaded = runCatching { StandaloneRenderPearlBridge.vulkanTurboLoaded }.getOrDefault(false)
        val backendAllowed = runCatching { StandaloneRenderPearlBridge.isBackendAllowed() }.getOrDefault(false)
        val vulkanBackendActive = runCatching { StandaloneRenderPearlBridge.isVulkanBackendActive() }.getOrDefault(false)
        val vulkanReady = runCatching { VulkanShaders.vulkanReady }.getOrDefault(false)
        val shaderPackActive = runCatching { VulkanShaders.shaderPackActive }.getOrDefault(false)
        val nativeAvailable = runCatching { NativeCoreBridge.nativeAvailable }.getOrDefault(false)
        val nativeInitialized = runCatching { NativeCoreBridge.initialized }.getOrDefault(false)

        val cfg = runCatching { VulkanShaders.config }.getOrNull()
        val activePack = runCatching { ShaderPackManager.activePack }.getOrNull()
        val lastPackError = runCatching { ShaderPackManager.lastPackError }.getOrNull()
        val lastCompileError = runCatching { NativeCoreBridge.lastCompilationError }.getOrNull()

        sb.append("Mod Version: ").append(modVersion)
            .append(" (").append(VulkanShaders.PUBLIC_NAME)
            .append(" / ").append(VulkanShaders.CODENAME).append(")\n")
        sb.append("Pyroclast Loaded: ").append(pyroclastLoaded)
            .append(" | Backend Allowed: ").append(backendAllowed)
            .append(" | Vulkan Backend Active: ").append(vulkanBackendActive)
            .append(" | Vulkan Ready: ").append(vulkanReady).append('\n')
        sb.append("Native Core: available=").append(nativeAvailable)
            .append(", initialized=").append(nativeInitialized)
            .append(", shaderPackActive=").append(shaderPackActive).append('\n')

        val packSummary = if (activePack != null) {
            "${activePack.manifest.id} (${activePack.manifest.name} v${activePack.manifest.version})"
        } else {
            "none (selected='${cfg?.selectedPackId ?: "unknown"}', enabled=${cfg?.enabled ?: false})"
        }
        sb.append("Active Pack: ").append(packSummary).append('\n')

        if (cfg != null) {
            sb.append("Render Tier: ").append(cfg.renderTier.name).append('\n')
            sb.append("Upscaler Mode: ").append(cfg.upscalerMode.name)
                .append(" (quality=").append(cfg.upscalerQuality.name).append(")\n")
            sb.append("Denoiser Mode: ").append(cfg.denoiserMode.name).append('\n')
            sb.append("FrameGen Mode: ").append(cfg.frameGenMode.name).append('\n')
        }

        latestShaderDiagnostic?.let { diag ->
            sb.append("Last Shader Diagnostic: ").append(diag.formatShort()).append('\n')
        }
        sb.append("Last Pack Error: ").append(lastPackError ?: "<none>").append('\n')
        sb.append("Last Compilation Error: ").append(lastCompileError ?: "<none>")
        return sb.toString()
    }

    @JvmStatic
    fun buildCrashDiagnosticSection(): String {
        val sb = StringBuilder(1280)
        sb.append(buildHyaloclastDiagnosticBlock()).append('\n')

        val recent = entriesSnapshot().takeLast(CRASH_SECTION_TAIL_ENTRIES)
        sb.append("Recent Log Entries (last ").append(recent.size).append("):\n")
        if (recent.isEmpty()) {
            sb.append("  <none>\n")
        } else {
            for (entry in recent) {
                appendFormattedEntry(sb, entry, indent = "  ")
                sb.append('\n')
            }
        }
        return sb.toString().trimEnd()
    }

    @JvmStatic
    @JvmOverloads
    fun dumpCrashLog(
        reason: String,
        crashReportText: String? = null,
        throwable: Throwable? = null,
    ): Path? {
        if (StandaloneRenderPearlBridge.vulkanTurboLoaded) {
            return PyroclastLogDelegate.dumpCrashLog(reason, crashReportText, throwable)
        }
        if (!crashDumpWritten.compareAndSet(false, true)) {
            return localLastCrashLogPath
        }
        return try {
            val dir = crashLogsDir()
            Files.createDirectories(dir)
            val now = LocalDateTime.now()
            val fileName = "crash-${crashFileTimestampFormatter.format(now)}.log"
            val target = dir.resolve(fileName)
            val tmp = dir.resolve("$fileName.tmp")

            val sb = StringBuilder(4096)
            sb.append("================================================================================\n")
            sb.append("Hyaloclast Crash / Diagnostic Log\n")
            sb.append("Timestamp: ").append(fullTimestampFormatter.format(now)).append('\n')
            sb.append("Reason: ").append(reason).append('\n')
            sb.append("================================================================================\n\n")
            sb.append(buildCrashDiagnosticSection()).append("\n\n")

            if (throwable != null) {
                sb.append("-- Throwable Stacktrace --\n")
                sb.append(formatThrowable(throwable)).append("\n\n")
            }

            sb.append("-- Full Session Error Log --\n")
            sb.append(formatFullLogText()).append("\n")

            if (!crashReportText.isNullOrBlank()) {
                sb.append("\n-- Minecraft CrashReport Details --\n")
                sb.append(crashReportText).append('\n')
            }

            Files.writeString(
                tmp,
                sb.toString(),
                StandardCharsets.UTF_8,
                StandardOpenOption.CREATE,
                StandardOpenOption.TRUNCATE_EXISTING,
                StandardOpenOption.WRITE,
            )
            try {
                Files.move(
                    tmp,
                    target,
                    StandardCopyOption.REPLACE_EXISTING,
                    StandardCopyOption.ATOMIC_MOVE,
                )
            } catch (_: Throwable) {
                Files.move(tmp, target, StandardCopyOption.REPLACE_EXISTING)
            }
            localLastCrashLogPath = target
            target
        } catch (t: Throwable) {
            crashDumpWritten.set(false)
            runCatching { VulkanShaders.log.error("Failed to write Hyaloclast crash log", t) }
            null
        }
    }

    @JvmStatic
    fun installCrashHooks() {
        if (StandaloneRenderPearlBridge.vulkanTurboLoaded) {
            PyroclastLogDelegate.installCrashHooks()
            return
        }
        if (!hooksInstalled.compareAndSet(false, true)) return

        val previousHandler = Thread.getDefaultUncaughtExceptionHandler()
        Thread.setDefaultUncaughtExceptionHandler { thread, throwable ->
            runCatching {
                recordError(
                    source = "UncaughtException",
                    summary = "Uncaught exception in thread '${thread.name}': ${throwable.message ?: throwable.javaClass.simpleName}",
                    throwable = throwable,
                    showToast = false,
                )
                dumpCrashLog("Uncaught exception in thread ${thread.name}", throwable = throwable)
            }
            previousHandler?.uncaughtException(thread, throwable)
        }

        runCatching {
            Runtime.getRuntime().addShutdownHook(
                Thread(
                    {
                        val hasErrors = errorCount() > 0
                        val unclean = !cleanShutdown.get()
                        if (hasErrors || unclean) {
                            val reason = when {
                                unclean && hasErrors -> "Abnormal JVM shutdown with recorded Hyaloclast errors"
                                unclean -> "Abnormal JVM shutdown (clean shutdown not marked)"
                                else -> "Session ended with recorded Hyaloclast errors"
                            }
                            dumpCrashLog(reason)
                        }
                    },
                    "Hyaloclast-CrashLog-ShutdownHook",
                ),
            )
        }
    }

    private fun appendEntry(
        level: Level,
        source: String,
        summary: String,
        details: String?,
        throwable: Throwable?,
        sourceFile: String? = null,
        lineNumber: Int? = null,
    ): Entry {
        val timestamp = entryTimestampFormatter.format(LocalDateTime.now())
        val combinedDetails = combineDetails(details, throwable)
        val entry = Entry(
            timestamp = timestamp,
            level = level,
            source = source.ifBlank { "Hyaloclast" },
            summary = summary.trim().ifEmpty { throwable?.javaClass?.simpleName ?: "Unknown error" },
            details = combinedDetails,
            sourceFile = sourceFile?.takeIf { it.isNotBlank() },
            lineNumber = lineNumber,
        )
        synchronized(lock) {
            if (entries.size >= MAX_ENTRIES) {
                entries.removeFirst()
            }
            entries.addLast(entry)
            appendToLatestLogLocked(entry)
        }
        return entry
    }

    private fun appendToLatestLogLocked(entry: Entry) {
        runCatching {
            val dir = logsDir()
            Files.createDirectories(dir)
            val file = latestLogFile()
            val sb = StringBuilder(256)
            if (!sessionLogInitialized) {
                sessionLogInitialized = true
                sb.append("=== Hyaloclast Session Log (started ")
                    .append(fullTimestampFormatter.format(LocalDateTime.now()))
                    .append(") ===\n")
                appendFormattedEntry(sb, entry)
                sb.append('\n')
                Files.writeString(
                    file,
                    sb.toString(),
                    StandardCharsets.UTF_8,
                    StandardOpenOption.CREATE,
                    StandardOpenOption.TRUNCATE_EXISTING,
                    StandardOpenOption.WRITE,
                )
            } else {
                appendFormattedEntry(sb, entry)
                sb.append('\n')
                Files.writeString(
                    file,
                    sb.toString(),
                    StandardCharsets.UTF_8,
                    StandardOpenOption.CREATE,
                    StandardOpenOption.APPEND,
                    StandardOpenOption.WRITE,
                )
            }
        }
    }

    private fun appendFormattedEntry(sb: StringBuilder, entry: Entry, indent: String = "") {
        sb.append(indent)
            .append('[').append(entry.timestamp).append("] ")
            .append('[').append(entry.level.name).append("] ")
            .append('[').append(entry.source).append("] ")
        val loc = entry.formatLocation()
        if (loc != null) {
            sb.append('(').append(loc).append(") ")
        }
        sb.append(entry.summary)
        val details = entry.details
        if (!details.isNullOrBlank()) {
            for (line in details.lineSequence()) {
                sb.append('\n').append(indent).append("    ").append(line)
            }
        }
    }

    private fun combineDetails(details: String?, throwable: Throwable?): String? {
        val cleanDetails = details?.trim()?.takeIf { it.isNotEmpty() }
        val trace = throwable?.let { formatThrowable(it).trim() }?.takeIf { it.isNotEmpty() }
        return when {
            cleanDetails != null && trace != null -> "$cleanDetails\n$trace"
            cleanDetails != null -> cleanDetails
            else -> trace
        }
    }

    private fun formatThrowable(throwable: Throwable): String {
        val sw = StringWriter()
        throwable.printStackTrace(PrintWriter(sw))
        return sw.toString()
    }

    private fun detailsSuffix(details: String?): String =
        if (details.isNullOrBlank()) "" else " — $details"
}
