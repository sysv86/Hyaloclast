package dev.vulkanshaders.backend

import com.google.gson.JsonParser
import dev.vulkanshaders.VulkanShaders
import dev.vulkanshaders.bridge.StandaloneRenderPearlBridge
import dev.vulkanshaders.config.VulkanShadersConfig
import dev.vulkanshaders.diagnostics.HyaloclastDebugHud
import dev.vulkanshaders.diagnostics.HyaloclastErrorLog
import dev.vulkanshaders.gui.ShaderNotices
import net.fabricmc.loader.api.FabricLoader
import org.lwjgl.vulkan.VK
import java.lang.foreign.Arena
import java.lang.foreign.FunctionDescriptor
import java.lang.foreign.Linker
import java.lang.foreign.MemorySegment
import java.lang.foreign.SymbolLookup
import java.lang.foreign.ValueLayout
import java.lang.invoke.MethodHandle
import java.nio.ByteBuffer
import java.nio.charset.StandardCharsets
import java.nio.file.Files
import java.nio.file.Path

/**
 * Java 25 Foreign Function & Memory (FFM) bridge to the native Rust `vulkanshaders_core`
 * (`crates/vulkanshaders-core`) shared library. Operates in pure Java/Kotlin mode gracefully
 * when the native binary is absent.
 */
object NativeCoreBridge {
    const val REFLEX_MARKER_SIMULATION_START: Int = 0
    const val REFLEX_MARKER_SIMULATION_END: Int = 1
    const val REFLEX_MARKER_RENDERSUBMIT_START: Int = 2
    const val REFLEX_MARKER_RENDERSUBMIT_END: Int = 3
    const val REFLEX_MARKER_PRESENT_START: Int = 4
    const val REFLEX_MARKER_PRESENT_END: Int = 5
    const val REFLEX_MARKER_INPUT_SAMPLE: Int = 6

    // Bit layout of `vs_core_query_capabilities()` (mirrors vulkanshaders-core FFI contract).
    const val CAP_RAY_TRACING_AS: Int = 1 shl 0
    const val CAP_RAY_QUERY: Int = 1 shl 1
    const val CAP_RAY_TRACING_PIPELINE: Int = 1 shl 2
    const val CAP_BINDLESS: Int = 1 shl 3
    const val CAP_REFLEX: Int = 1 shl 4
    const val CAP_DLSS_SR: Int = 1 shl 5
    const val CAP_DLSS_RR: Int = 1 shl 6
    const val CAP_DLSS_FG: Int = 1 shl 7
    const val CAP_FSR3: Int = 1 shl 8
    const val CAP_XESS: Int = 1 shl 9
    const val CAP_NRD: Int = 1 shl 10
    const val CAP_HDR_SWAPCHAIN: Int = 1 shl 11
    const val CAP_BUFFER_DEVICE_ADDRESS: Int = 1 shl 12
    const val CAP_SER_NV: Int = 1 shl 13
    const val CAP_SER_EXT: Int = 1 shl 14
    const val CAP_OPACITY_MICROMAP: Int = 1 shl 15
    const val CAP_HIGH_END_GPU: Int = 1 shl 16
    const val CAP_ADA_OR_NEWER: Int = 1 shl 17

    data class HardwareProfile(
        val gpuName: String = "",
        val apiVersion: String = "",
        val driver: String = "",
        val driverVersion: String = "",
        val vendorId: Int = 0,
        val deviceId: Int = 0,
        val deviceType: String = "other",
        val vramBytes: Long = 0L,
        val vramMb: Long = 0L,
        val gpuArch: String = "unknown",
        val gpuTierClass: String = "low",
        val serNv: Boolean = false,
        val serExt: Boolean = false,
        val serActive: Boolean = false,
        val opacityMicromap: Boolean = false,
        val lastFrameDtMs: Double = 0.0,
        val capabilitiesMask: Int = 0,
    )

    data class PrewarmProgress(
        val completed: Int = 0,
        val total: Int = 0,
        val ready: Boolean = true,
    )

    data class VramBudgetSnapshot(
        val usedMb: Long,
        val budgetMb: Long,
        val pressure: Int,
    )

    private val linker: Linker = Linker.nativeLinker()
    private val arena: Arena = Arena.ofShared()

    @JvmField @Volatile var nativeAvailable: Boolean = false
    @JvmField @Volatile var initialized: Boolean = false
    @JvmField @Volatile var lastCompilationError: String? = null
    @Volatile private var cachedGpuTierClass: String? = null

    private var fnSetInstanceProcAddr: MethodHandle? = null
    private var fnInit: MethodHandle? = null
    private var fnSetQueueFamilies: MethodHandle? = null
    private var fnBindTimelineSemaphore: MethodHandle? = null
    private var fnShutdown: MethodHandle? = null
    private var fnCreateMergedInstance: MethodHandle? = null
    private var fnCreateMergedDevice: MethodHandle? = null
    private var fnGetCapabilitiesMask: MethodHandle? = null
    private var fnLockQueue: MethodHandle? = null
    private var fnUnlockQueue: MethodHandle? = null
    private var fnClearSectionSlices: MethodHandle? = null
    private var fnRecordSubmission: MethodHandle? = null
    private var fnNotifyTimelineCompleted: MethodHandle? = null
    private var fnOnWorldUnloaded: MethodHandle? = null
    private var fnSetRenderScale: MethodHandle? = null
    private var fnSetHdrParams: MethodHandle? = null
    private var fnOnSwapchainCreated: MethodHandle? = null
    private var fnSetGameDir: MethodHandle? = null
    private var fnMarkHistoryReset: MethodHandle? = null
    private var fnSetJitter: MethodHandle? = null
    private var fnSetCameraPosition: MethodHandle? = null
    private var fnUpdateWorldUbo: MethodHandle? = null
    private var fnResetTemporalHistory: MethodHandle? = null
    private var fnRegisterSlice: MethodHandle? = null
    private var fnRegisterSliceEx: MethodHandle? = null
    private var fnUnregisterSlice: MethodHandle? = null
    private var fnRegisterTexture: MethodHandle? = null
    private var fnRegisterTextureEx: MethodHandle? = null
    private var fnRegisterTextureWithSampler: MethodHandle? = null
    private var fnLinkPbrCompanions: MethodHandle? = null
    private var fnUnregisterTexture: MethodHandle? = null
    private var fnConfigurePipeline: MethodHandle? = null
    private var fnSetShadowConfig: MethodHandle? = null
    private var fnInterceptSurfaceBlit: MethodHandle? = null
    private var fnInterceptAcquireNextImage: MethodHandle? = null
    private var fnInterceptQueuePresent: MethodHandle? = null
    private var fnSetOverlayDepthParams: MethodHandle? = null
    private var fnReflexSetSwapchain: MethodHandle? = null
    private var fnReflexMarker: MethodHandle? = null
    private var fnReflexSleepAndSampleInput: MethodHandle? = null
    private var fnSubmitEntityVertices: MethodHandle? = null
    private var fnRecordFrameRtx: MethodHandle? = null
    private var fnWaitPresentIdle: MethodHandle? = null
    private var fnCanEnableFrameGen: MethodHandle? = null
    private var fnTakeFrameGenFailure: MethodHandle? = null
    private var fnCompileActivePack: MethodHandle? = null
    private var fnPrepareSpecConstants: MethodHandle? = null
    private var fnUpdateSpecConstants: MethodHandle? = null
    private var fnGetLastCompileError: MethodHandle? = null
    private var fnBindGbufferImages: MethodHandle? = null
    private var fnBindGbufferViews: MethodHandle? = null
    private var fnBindPostTargets: MethodHandle? = null
    private var fnExecuteRenderGraph: MethodHandle? = null
    private var fnExecuteAfterGui: MethodHandle? = null

    private var fnQueryCapabilities: MethodHandle? = null
    private var fnGetCapabilityDetails: MethodHandle? = null
    private var fnIsHdrSwapchainSupported: MethodHandle? = null
    private var fnUploadPackTexture: MethodHandle? = null
    private var fnGetCustomDescriptorSetLayouts: MethodHandle? = null
    private var fnGetCustomDescriptorSets: MethodHandle? = null
    private var fnBindCustomDescriptorSets: MethodHandle? = null
    private var fnRegisterSectionSliceWithPbr: MethodHandle? = null
    private var fnSubmitEmissiveLights: MethodHandle? = null
    private var fnRegisterLodSlice: MethodHandle? = null
    private var fnUploadLodMeshCpu: MethodHandle? = null
    private var fnUnregisterLodSlice: MethodHandle? = null
    private var fnClearLodSlices: MethodHandle? = null
    private var fnSetLodCutoffDistance: MethodHandle? = null
    private var fnBindDhTargets: MethodHandle? = null
    private var fnPreprocessRasterShader: MethodHandle? = null
    private var fnGetNativeErrorCount: MethodHandle? = null
    private var fnDrainNativeErrors: MethodHandle? = null

    // Wave 2 optional telemetry, recovery, prewarm, LOD, and dimension FFI handles.
    private var fnGetVramBudget: MethodHandle? = null
    private var fnGetRtStats: MethodHandle? = null
    private var fnSetGpuProfilerEnabled: MethodHandle? = null
    private var fnGetGpuPassTimings: MethodHandle? = null
    private var fnGetAverageGpuFrameTimeMs: MethodHandle? = null
    private var fnGetRecoveryState: MethodHandle? = null
    private var fnResetRecoveryState: MethodHandle? = null
    private var fnStartPackPrewarm: MethodHandle? = null
    private var fnGetPrewarmProgress: MethodHandle? = null
    private var fnSetLodEnabled: MethodHandle? = null
    private var fnSetDimension: MethodHandle? = null

    // Wave 3 optional OMM, categorized entity vertex, display resize, and resource reload FFI handles.
    private var fnAttachCutoutOmm: MethodHandle? = null
    private var fnSubmitEntityVerticesCategorized: MethodHandle? = null
    private var fnOnDisplayResized: MethodHandle? = null
    private var fnOnResourcesReloaded: MethodHandle? = null

    /**
     * Resizable persistent off-heap scratch segment backed by [Arena.ofAuto] so hot-path FFI
     * calls reuse native memory without per-frame [Arena.ofConfined] churn or leaks on growth.
     */
    private class PersistentScratchBuffer(
        initialCapacityBytes: Long,
        private val alignment: Long = 8L,
    ) {
        private var segment: MemorySegment = Arena.ofAuto().allocate(initialCapacityBytes.coerceAtLeast(64L), alignment)

        fun ensureCapacity(minBytes: Long): MemorySegment {
            val required = minBytes.coerceAtLeast(64L)
            if (segment.byteSize() < required) {
                val nextCap = maxOf(required, segment.byteSize() * 2L)
                segment = Arena.ofAuto().allocate(nextCap, alignment)
            }
            return segment
        }
    }

    private val loggedFfiErrors = java.util.concurrent.ConcurrentHashMap.newKeySet<String>()
    private val dhInvViewProjSegment: MemorySegment = Arena.ofAuto().allocate(ValueLayout.JAVA_FLOAT, 16L)

    // Pre-allocated persistent scratch MemorySegments for zero-allocation hot paths and telemetry queries.
    private val telemetryLock = Any()
    private val vramUsedSeg: MemorySegment = arena.allocate(ValueLayout.JAVA_LONG)
    private val vramBudgetSeg: MemorySegment = arena.allocate(ValueLayout.JAVA_LONG)
    private val vramPressureSeg: MemorySegment = arena.allocate(ValueLayout.JAVA_INT)
    private val averageGpuFrameTimeSeg: MemorySegment = arena.allocate(ValueLayout.JAVA_DOUBLE)
    private val rtBlasCountSeg: MemorySegment = arena.allocate(ValueLayout.JAVA_INT)
    private val rtTlasInstancesSeg: MemorySegment = arena.allocate(ValueLayout.JAVA_INT)
    private val recoveryDisabledSeg: MemorySegment = arena.allocate(ValueLayout.JAVA_INT)
    private val prewarmCompletedSeg: MemorySegment = arena.allocate(ValueLayout.JAVA_INT)
    private val prewarmTotalSeg: MemorySegment = arena.allocate(ValueLayout.JAVA_INT)
    private val prewarmReadySeg: MemorySegment = arena.allocate(ValueLayout.JAVA_INT)
    private val gpuTimingsScratch = PersistentScratchBuffer(4096L, 8L)
    private val recoveryNoticeScratch = PersistentScratchBuffer(1024L, 8L)

    private val descriptorScratchLock = Any()
    private val customDescBindlessSeg: MemorySegment = arena.allocate(ValueLayout.JAVA_LONG)
    private val customDescPackSeg: MemorySegment = arena.allocate(ValueLayout.JAVA_LONG)
    private val customDescLayoutsScratch = PersistentScratchBuffer(128L, 8L)

    private class IoScratchBuffers {
        val primary = PersistentScratchBuffer(8192L, 8L)
        val secondary = PersistentScratchBuffer(4096L, 8L)
        val output = PersistentScratchBuffer(8192L, 8L)
    }

    private class LodUploadScratchBuffers {
        val vertices = PersistentScratchBuffer(65536L, 8L)
        val indices = PersistentScratchBuffer(16384L, 8L)
    }

    private val ioScratch = ThreadLocal.withInitial { IoScratchBuffers() }
    private val lodUploadScratch = ThreadLocal.withInitial { LodUploadScratchBuffers() }
    private val ommDataScratch = ThreadLocal.withInitial { PersistentScratchBuffer(16384L, 8L) }
    private val entityVertexScratch = ThreadLocal.withInitial { PersistentScratchBuffer(65536L, 8L) }

    @Volatile
    private var deviceLostNotifiedForIncident: Boolean = false

    private fun logNativeCallError(op: String, t: Throwable) {
        if (loggedFfiErrors.add(op)) {
            HyaloclastErrorLog.recordError(
                source = "NativeCoreBridge",
                summary = "NativeCoreBridge.$op failed: ${t.message ?: t.javaClass.simpleName}",
                throwable = t,
            )
        }
    }

    init {
        loadNativeLibrary()
    }

    private fun loadNativeLibrary() {
        val os = System.getProperty("os.name").lowercase()
        val arch = System.getProperty("os.arch").lowercase()
        val isArm64 = arch.contains("aarch64") || arch.contains("arm64")
        val platformDir = when {
            os.contains("win") -> "win32-x86-64"
            os.contains("mac") -> if (isArm64) "macos-aarch64" else "macos-x86-64"
            isArm64 -> "linux-aarch64"
            else -> "linux-x86-64"
        }
        val libName = when {
            os.contains("win") -> "vulkanshaders_core.dll"
            os.contains("mac") -> "libvulkanshaders_core.dylib"
            else -> "libvulkanshaders_core.so"
        }

        preloadVendorNatives()

        val hasBundledNative =
            NativeCoreBridge::class.java.getResource("/natives/$platformDir/$libName") != null ||
                NativeCoreBridge::class.java.getResource("/natives/$libName") != null

        val lookup =
            tryExtractOrLookup("/natives/$platformDir/$libName")
                ?: tryExtractOrLookup("/natives/$libName")
                ?: tryGameDirLibrary(platformDir, libName)
                ?: runCatching { SymbolLookup.libraryLookup(libName, arena) }.getOrNull()
        if (lookup == null) {
            if (!hasBundledNative) {
                HyaloclastErrorLog.recordWarning(
                    source = "NativeCoreBridge",
                    summary = "Unsupported OS/architecture '$platformDir' (os='$os', arch='$arch'): native library '$libName' is not bundled in this JAR; running in JVM-only fallback mode.",
                )
            } else {
                VulkanShaders.log.info("Native library '{}' not present or failed verification; running in JVM-only fallback mode.", libName)
            }
            return
        }

        try {
            fnSetInstanceProcAddr = downcall(
                lookup,
                "vs_core_set_instance_proc_addr",
                FunctionDescriptor.ofVoid(ValueLayout.JAVA_LONG),
            )
            fnInit = downcall(
                lookup,
                "vs_core_init",
                FunctionDescriptor.of(
                    ValueLayout.JAVA_INT,
                    ValueLayout.JAVA_LONG,
                    ValueLayout.JAVA_LONG,
                    ValueLayout.JAVA_LONG,
                    ValueLayout.JAVA_LONG,
                    ValueLayout.JAVA_LONG,
                ),
            )
            fnSetQueueFamilies = downcall(
                lookup,
                "vs_core_set_queue_families",
                FunctionDescriptor.ofVoid(ValueLayout.JAVA_INT, ValueLayout.JAVA_INT),
            )
            fnBindTimelineSemaphore = downcall(
                lookup,
                "vs_core_bind_timeline_semaphore",
                FunctionDescriptor.ofVoid(ValueLayout.JAVA_LONG),
            )
            fnShutdown = downcall(lookup, "vs_core_shutdown", FunctionDescriptor.ofVoid())
            fnCreateMergedInstance = downcall(
                lookup,
                "vs_core_create_merged_instance",
                FunctionDescriptor.of(
                    ValueLayout.JAVA_INT,
                    ValueLayout.JAVA_LONG,
                    ValueLayout.JAVA_LONG,
                    ValueLayout.JAVA_LONG,
                ),
            )
            fnCreateMergedDevice = downcall(
                lookup,
                "vs_core_create_merged_device",
                FunctionDescriptor.of(
                    ValueLayout.JAVA_INT,
                    ValueLayout.JAVA_LONG,
                    ValueLayout.JAVA_LONG,
                    ValueLayout.JAVA_LONG,
                    ValueLayout.JAVA_LONG,
                ),
            )
            fnGetCapabilitiesMask = downcall(
                lookup,
                "vs_core_get_capabilities_mask",
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.JAVA_LONG),
            )
            fnLockQueue = downcall(lookup, "vs_core_lock_queue", FunctionDescriptor.ofVoid())
            fnUnlockQueue = downcall(lookup, "vs_core_unlock_queue", FunctionDescriptor.ofVoid())
            fnClearSectionSlices = downcall(lookup, "vs_core_clear_section_slices", FunctionDescriptor.ofVoid())
            fnRecordSubmission = downcall(lookup, "vs_core_record_submission", FunctionDescriptor.ofVoid(ValueLayout.JAVA_LONG))
            fnNotifyTimelineCompleted = downcall(lookup, "vs_core_notify_timeline_completed", FunctionDescriptor.ofVoid(ValueLayout.JAVA_LONG))
            fnOnWorldUnloaded = downcall(lookup, "vs_core_on_world_unloaded", FunctionDescriptor.ofVoid())
            fnSetRenderScale = downcall(lookup, "vs_core_set_render_scale", FunctionDescriptor.ofVoid(ValueLayout.JAVA_FLOAT))
            fnSetHdrParams = downcall(lookup, "vs_core_set_hdr_params", FunctionDescriptor.ofVoid(ValueLayout.JAVA_FLOAT, ValueLayout.JAVA_FLOAT))
            fnOnSwapchainCreated = downcall(
                lookup,
                "vs_core_on_swapchain_created",
                FunctionDescriptor.ofVoid(
                    ValueLayout.JAVA_INT,
                    ValueLayout.JAVA_INT,
                    ValueLayout.JAVA_INT,
                    ValueLayout.JAVA_INT,
                ),
            )
            fnSetGameDir = downcall(lookup, "vs_core_set_game_dir", FunctionDescriptor.ofVoid(ValueLayout.ADDRESS, ValueLayout.JAVA_INT))
            fnMarkHistoryReset = downcall(lookup, "vs_core_mark_history_reset", FunctionDescriptor.ofVoid())
            fnSetJitter = downcall(
                lookup,
                "vs_core_set_jitter",
                FunctionDescriptor.ofVoid(ValueLayout.JAVA_FLOAT, ValueLayout.JAVA_FLOAT),
            )
            fnSetCameraPosition = downcall(
                lookup,
                "vs_core_set_camera_position",
                FunctionDescriptor.ofVoid(
                    ValueLayout.JAVA_DOUBLE,
                    ValueLayout.JAVA_DOUBLE,
                    ValueLayout.JAVA_DOUBLE,
                ),
            )
            fnUpdateWorldUbo = downcall(
                lookup,
                "vs_core_update_world_ubo",
                FunctionDescriptor.ofVoid(ValueLayout.JAVA_LONG, ValueLayout.JAVA_INT),
            )
            fnResetTemporalHistory = downcall(
                lookup,
                "vs_core_reset_temporal_history",
                FunctionDescriptor.ofVoid(),
            )
            fnRegisterSlice = downcall(
                lookup,
                "vs_core_register_section_slice",
                FunctionDescriptor.ofVoid(
                    ValueLayout.JAVA_LONG,
                    ValueLayout.JAVA_INT,
                    ValueLayout.JAVA_LONG,
                    ValueLayout.JAVA_LONG,
                    ValueLayout.JAVA_LONG,
                ),
            )
            fnRegisterSliceEx = downcall(
                lookup,
                "vs_core_register_section_slice_ex",
                FunctionDescriptor.ofVoid(
                    ValueLayout.JAVA_LONG,
                    ValueLayout.JAVA_INT,
                    ValueLayout.JAVA_LONG,
                    ValueLayout.JAVA_LONG,
                    ValueLayout.JAVA_LONG,
                    ValueLayout.JAVA_LONG,
                    ValueLayout.JAVA_INT,
                    ValueLayout.JAVA_INT,
                    ValueLayout.JAVA_INT,
                ),
            )
            fnUnregisterSlice = downcall(
                lookup,
                "vs_core_unregister_section_slice",
                FunctionDescriptor.ofVoid(ValueLayout.JAVA_LONG, ValueLayout.JAVA_INT),
            )
            fnRegisterTexture = downcall(
                lookup,
                "vs_core_register_texture",
                FunctionDescriptor.ofVoid(
                    ValueLayout.JAVA_LONG,
                    ValueLayout.JAVA_INT,
                    ValueLayout.JAVA_INT,
                    ValueLayout.JAVA_INT,
                ),
            )
            fnRegisterTextureEx = downcall(
                lookup,
                "vs_core_register_texture_ex",
                FunctionDescriptor.of(
                    ValueLayout.JAVA_INT,
                    ValueLayout.JAVA_LONG,
                    ValueLayout.JAVA_LONG,
                    ValueLayout.JAVA_INT,
                    ValueLayout.JAVA_INT,
                    ValueLayout.JAVA_INT,
                    ValueLayout.JAVA_INT,
                ),
            )
            fnRegisterTextureWithSampler = downcall(
                lookup,
                "vs_core_register_texture_with_sampler",
                FunctionDescriptor.of(
                    ValueLayout.JAVA_INT,
                    ValueLayout.JAVA_LONG,
                    ValueLayout.JAVA_LONG,
                    ValueLayout.JAVA_LONG,
                    ValueLayout.JAVA_INT,
                    ValueLayout.JAVA_INT,
                    ValueLayout.JAVA_INT,
                    ValueLayout.JAVA_INT,
                    ValueLayout.JAVA_INT,
                ),
            )
            fnLinkPbrCompanions = downcall(
                lookup,
                "vs_core_link_pbr_companions",
                FunctionDescriptor.ofVoid(
                    ValueLayout.JAVA_INT,
                    ValueLayout.JAVA_INT,
                    ValueLayout.JAVA_INT,
                ),
            )
            fnUnregisterTexture = downcall(
                lookup,
                "vs_core_unregister_texture",
                FunctionDescriptor.ofVoid(ValueLayout.JAVA_LONG),
            )
            fnConfigurePipeline = downcall(
                lookup,
                "vs_core_configure_pipeline",
                FunctionDescriptor.of(
                    ValueLayout.JAVA_INT,
                    ValueLayout.JAVA_INT,
                    ValueLayout.JAVA_INT,
                    ValueLayout.JAVA_INT,
                    ValueLayout.JAVA_INT,
                    ValueLayout.JAVA_INT,
                    ValueLayout.JAVA_INT,
                    ValueLayout.JAVA_INT,
                ),
            )
            fnSetShadowConfig = downcall(
                lookup,
                "vs_core_set_shadow_config",
                FunctionDescriptor.ofVoid(
                    ValueLayout.JAVA_INT,
                    ValueLayout.JAVA_INT,
                ),
            )
            fnInterceptSurfaceBlit = downcall(
                lookup,
                "vs_core_intercept_surface_blit",
                FunctionDescriptor.of(
                    ValueLayout.JAVA_INT,
                    ValueLayout.JAVA_LONG,
                    ValueLayout.JAVA_LONG,
                    ValueLayout.JAVA_INT,
                    ValueLayout.JAVA_LONG,
                    ValueLayout.JAVA_INT,
                    ValueLayout.JAVA_INT,
                    ValueLayout.JAVA_INT,
                ),
            )
            fnInterceptAcquireNextImage = downcall(
                lookup,
                "vs_core_intercept_acquire_next_image",
                FunctionDescriptor.of(
                    ValueLayout.JAVA_INT,
                    ValueLayout.JAVA_LONG,
                    ValueLayout.JAVA_LONG,
                    ValueLayout.JAVA_LONG,
                    ValueLayout.JAVA_LONG,
                ),
            )
            fnInterceptQueuePresent = downcall(
                lookup,
                "vs_core_intercept_queue_present",
                FunctionDescriptor.of(
                    ValueLayout.JAVA_INT,
                    ValueLayout.JAVA_LONG,
                    ValueLayout.JAVA_LONG,
                    ValueLayout.JAVA_INT,
                    ValueLayout.JAVA_LONG,
                ),
            )
            fnSetOverlayDepthParams = downcall(
                lookup,
                "vs_core_set_overlay_depth_params",
                FunctionDescriptor.of(
                    ValueLayout.JAVA_FLOAT,
                    ValueLayout.JAVA_FLOAT,
                    ValueLayout.JAVA_FLOAT,
                    ValueLayout.JAVA_FLOAT,
                    ValueLayout.JAVA_FLOAT,
                    ValueLayout.JAVA_INT,
                    ValueLayout.JAVA_INT,
                    ValueLayout.JAVA_INT,
                    ValueLayout.JAVA_INT,
                ),
            )
            fnReflexSetSwapchain = downcall(
                lookup,
                "vs_core_reflex_set_swapchain",
                FunctionDescriptor.ofVoid(ValueLayout.JAVA_LONG),
            )
            fnReflexMarker = downcall(
                lookup,
                "vs_core_reflex_marker",
                FunctionDescriptor.ofVoid(ValueLayout.JAVA_INT),
            )
            fnReflexSleepAndSampleInput = downcall(
                lookup,
                "vs_core_reflex_sleep_and_sample_input",
                FunctionDescriptor.ofVoid(),
            )
            fnSubmitEntityVertices = downcall(
                lookup,
                "vs_core_submit_entity_vertices",
                FunctionDescriptor.of(
                    ValueLayout.JAVA_INT,
                    ValueLayout.JAVA_LONG,
                    ValueLayout.JAVA_INT,
                    ValueLayout.JAVA_INT,
                    ValueLayout.JAVA_INT,
                    ValueLayout.JAVA_INT,
                ),
            )
            fnRecordFrameRtx = downcall(
                lookup,
                "vs_core_record_frame_rtx",
                FunctionDescriptor.of(
                    ValueLayout.JAVA_LONG,
                    ValueLayout.JAVA_LONG,
                    ValueLayout.JAVA_DOUBLE,
                    ValueLayout.JAVA_DOUBLE,
                    ValueLayout.JAVA_DOUBLE,
                    ValueLayout.JAVA_DOUBLE,
                    ValueLayout.JAVA_DOUBLE,
                    ValueLayout.JAVA_DOUBLE,
                    ValueLayout.JAVA_INT,
                    ValueLayout.JAVA_INT,
                ),
            )
            fnWaitPresentIdle = downcall(
                lookup,
                "vs_core_wait_present_idle",
                FunctionDescriptor.ofVoid(),
            )
            fnCanEnableFrameGen = downcall(
                lookup,
                "vs_core_can_enable_framegen",
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.JAVA_INT),
            )
            fnTakeFrameGenFailure = downcall(
                lookup,
                "vs_core_take_framegen_failure",
                FunctionDescriptor.of(ValueLayout.JAVA_INT),
            )
            fnCompileActivePack = downcall(
                lookup,
                "vs_core_compile_active_pack",
                FunctionDescriptor.of(
                    ValueLayout.JAVA_INT,
                    ValueLayout.JAVA_LONG,
                    ValueLayout.JAVA_INT,
                ),
            )
            fnPrepareSpecConstants = downcall(
                lookup,
                "vs_core_prepare_spec_constants",
                FunctionDescriptor.of(
                    ValueLayout.JAVA_INT,
                    ValueLayout.JAVA_LONG,
                    ValueLayout.JAVA_INT,
                ),
            )
            fnUpdateSpecConstants = downcall(
                lookup,
                "vs_core_update_spec_constants",
                FunctionDescriptor.of(
                    ValueLayout.JAVA_INT,
                    ValueLayout.JAVA_LONG,
                    ValueLayout.JAVA_INT,
                ),
            )
            fnGetLastCompileError = downcall(
                lookup,
                "vs_core_get_last_compile_error",
                FunctionDescriptor.of(
                    ValueLayout.JAVA_INT,
                    ValueLayout.JAVA_LONG,
                    ValueLayout.JAVA_INT,
                ),
            )
            fnBindGbufferImages = downcall(
                lookup,
                "vs_core_bind_gbuffer_images",
                FunctionDescriptor.ofVoid(
                    ValueLayout.JAVA_LONG,
                    ValueLayout.JAVA_LONG,
                    ValueLayout.JAVA_LONG,
                    ValueLayout.JAVA_LONG,
                    ValueLayout.JAVA_LONG,
                    ValueLayout.JAVA_LONG,
                ),
            )
            fnBindGbufferViews = downcall(
                lookup,
                "vs_core_bind_gbuffer_views",
                FunctionDescriptor.ofVoid(
                    ValueLayout.JAVA_LONG,
                    ValueLayout.JAVA_LONG,
                    ValueLayout.JAVA_LONG,
                    ValueLayout.JAVA_LONG,
                    ValueLayout.JAVA_LONG,
                    ValueLayout.JAVA_LONG,
                    ValueLayout.JAVA_LONG,
                    ValueLayout.JAVA_LONG,
                    ValueLayout.JAVA_LONG,
                    ValueLayout.JAVA_LONG,
                    ValueLayout.JAVA_LONG,
                    ValueLayout.JAVA_LONG,
                ),
            )
            fnBindPostTargets = downcall(
                lookup,
                "vs_core_bind_post_targets",
                FunctionDescriptor.ofVoid(
                    ValueLayout.JAVA_LONG,
                    ValueLayout.JAVA_LONG,
                    ValueLayout.JAVA_LONG,
                    ValueLayout.JAVA_LONG,
                    ValueLayout.JAVA_LONG,
                    ValueLayout.JAVA_LONG,
                    ValueLayout.JAVA_LONG,
                    ValueLayout.JAVA_LONG,
                ),
            )
            fnExecuteRenderGraph = downcall(
                lookup,
                "vs_core_execute_render_graph",
                FunctionDescriptor.of(
                    ValueLayout.JAVA_INT,
                    ValueLayout.JAVA_LONG,
                    ValueLayout.JAVA_INT,
                    ValueLayout.JAVA_INT,
                ),
            )
            fnExecuteAfterGui = downcall(
                lookup,
                "vs_core_execute_after_gui",
                FunctionDescriptor.of(
                    ValueLayout.JAVA_INT,
                    ValueLayout.JAVA_LONG,
                    ValueLayout.JAVA_LONG,
                    ValueLayout.JAVA_LONG,
                    ValueLayout.JAVA_INT,
                    ValueLayout.JAVA_INT,
                ),
            )

            fnQueryCapabilities = downcall(
                lookup,
                "vs_core_query_capabilities",
                FunctionDescriptor.of(ValueLayout.JAVA_INT),
            )
            fnGetCapabilityDetails = downcall(
                lookup,
                "vs_core_get_capability_details",
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.JAVA_LONG, ValueLayout.JAVA_INT),
            )
            fnIsHdrSwapchainSupported = downcall(
                lookup,
                "vs_core_is_hdr_swapchain_supported",
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.JAVA_LONG),
            )
            fnUploadPackTexture = downcall(
                lookup,
                "vs_core_upload_pack_texture",
                FunctionDescriptor.of(
                    ValueLayout.JAVA_INT,
                    ValueLayout.JAVA_INT,
                    ValueLayout.JAVA_INT,
                    ValueLayout.JAVA_INT,
                    ValueLayout.JAVA_LONG,
                    ValueLayout.JAVA_INT,
                    ValueLayout.JAVA_INT,
                    ValueLayout.JAVA_INT,
                ),
            )
            fnGetCustomDescriptorSetLayouts = downcall(
                lookup,
                "vs_core_get_custom_descriptor_set_layouts",
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.JAVA_LONG, ValueLayout.JAVA_INT),
            )
            fnGetCustomDescriptorSets = downcall(
                lookup,
                "vs_core_get_custom_descriptor_sets",
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS),
            )
            fnBindCustomDescriptorSets = downcall(
                lookup,
                "vs_core_bind_custom_descriptor_sets",
                FunctionDescriptor.ofVoid(
                    ValueLayout.JAVA_LONG,
                    ValueLayout.JAVA_LONG,
                    ValueLayout.JAVA_INT,
                    ValueLayout.JAVA_INT,
                ),
            )
            fnRegisterSectionSliceWithPbr = downcall(
                lookup,
                "vs_core_register_section_slice_with_pbr",
                FunctionDescriptor.ofVoid(
                    ValueLayout.JAVA_LONG, // key_id: u64
                    ValueLayout.JAVA_INT,  // layer_ordinal: i32
                    ValueLayout.JAVA_LONG, // section_pos_packed: i64
                    ValueLayout.JAVA_LONG, // vk_buffer: u64
                    ValueLayout.JAVA_LONG, // byte_offset: u64
                    ValueLayout.JAVA_LONG, // byte_size: u64
                    ValueLayout.JAVA_INT,  // vertex_stride: i32
                    ValueLayout.JAVA_INT,  // vertex_count: i32
                    ValueLayout.JAVA_INT,  // is_region_shifted: i32
                    ValueLayout.JAVA_INT,  // albedo_slot: i32
                    ValueLayout.JAVA_INT,  // normal_slot: i32
                    ValueLayout.JAVA_INT,  // specular_slot: i32
                    ValueLayout.JAVA_INT,  // material_flags: u32 (passed as i32 bits)
                ),
            )
            fnSubmitEmissiveLights = downcall(
                lookup,
                "vs_core_submit_emissive_lights",
                FunctionDescriptor.of(
                    ValueLayout.JAVA_INT,
                    ValueLayout.JAVA_LONG,
                    ValueLayout.JAVA_INT,
                    ValueLayout.JAVA_DOUBLE,
                    ValueLayout.JAVA_DOUBLE,
                    ValueLayout.JAVA_DOUBLE,
                ),
            )
            fnRegisterLodSlice = downcall(
                lookup,
                "vs_core_register_lod_slice",
                FunctionDescriptor.ofVoid(
                    ValueLayout.JAVA_LONG,
                    ValueLayout.JAVA_INT,
                    ValueLayout.JAVA_INT,
                    ValueLayout.JAVA_INT,
                    ValueLayout.JAVA_INT,
                    ValueLayout.JAVA_FLOAT,
                    ValueLayout.JAVA_LONG,
                    ValueLayout.JAVA_LONG,
                    ValueLayout.JAVA_LONG,
                    ValueLayout.JAVA_INT,
                    ValueLayout.JAVA_INT,
                    ValueLayout.JAVA_LONG,
                    ValueLayout.JAVA_INT,
                    ValueLayout.JAVA_INT,
                ),
            )
            fnUploadLodMeshCpu = downcall(
                lookup,
                "vs_core_upload_lod_mesh_cpu",
                FunctionDescriptor.of(
                    ValueLayout.JAVA_INT,
                    ValueLayout.JAVA_LONG,
                    ValueLayout.JAVA_INT,
                    ValueLayout.JAVA_INT,
                    ValueLayout.JAVA_INT,
                    ValueLayout.JAVA_INT,
                    ValueLayout.JAVA_FLOAT,
                    ValueLayout.JAVA_LONG,
                    ValueLayout.JAVA_INT,
                    ValueLayout.JAVA_LONG,
                    ValueLayout.JAVA_INT,
                    ValueLayout.JAVA_INT,
                ),
            )
            fnUnregisterLodSlice = downcall(
                lookup,
                "vs_core_unregister_lod_slice",
                FunctionDescriptor.ofVoid(ValueLayout.JAVA_LONG),
            )
            fnClearLodSlices = downcall(
                lookup,
                "vs_core_clear_lod_slices",
                FunctionDescriptor.ofVoid(),
            )
            fnSetLodCutoffDistance = downcall(
                lookup,
                "vs_core_set_lod_cutoff_distance",
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.JAVA_FLOAT),
            )
            fnBindDhTargets = downcall(
                lookup,
                "vs_core_bind_dh_targets",
                FunctionDescriptor.of(
                    ValueLayout.JAVA_INT,
                    ValueLayout.JAVA_LONG, // color_img
                    ValueLayout.JAVA_LONG, // color_view
                    ValueLayout.JAVA_LONG, // normal_img
                    ValueLayout.JAVA_LONG, // normal_view
                    ValueLayout.JAVA_LONG, // material_img
                    ValueLayout.JAVA_LONG, // material_view
                    ValueLayout.JAVA_LONG, // depth_img
                    ValueLayout.JAVA_LONG, // depth_view
                    ValueLayout.JAVA_INT,  // width
                    ValueLayout.JAVA_INT,  // height
                    ValueLayout.ADDRESS,   // inv_view_proj_ptr (16 floats or null)
                    ValueLayout.JAVA_FLOAT, // fade_start
                    ValueLayout.JAVA_FLOAT, // fade_end
                    ValueLayout.JAVA_FLOAT, // dh_far_clip
                    ValueLayout.JAVA_INT,  // active (0 or 1)
                ),
            )
            fnPreprocessRasterShader = downcall(
                lookup,
                "vs_core_preprocess_raster_shader",
                FunctionDescriptor.of(
                    ValueLayout.JAVA_INT,
                    ValueLayout.JAVA_LONG,
                    ValueLayout.JAVA_INT,
                    ValueLayout.JAVA_LONG,
                    ValueLayout.JAVA_INT,
                    ValueLayout.JAVA_INT,
                    ValueLayout.JAVA_LONG,
                    ValueLayout.JAVA_INT,
                ),
            )
            fnGetNativeErrorCount = downcall(
                lookup,
                "vs_core_get_native_error_count",
                FunctionDescriptor.of(ValueLayout.JAVA_INT),
            )
            fnDrainNativeErrors = downcall(
                lookup,
                "vs_core_drain_native_errors",
                FunctionDescriptor.of(
                    ValueLayout.JAVA_INT,
                    ValueLayout.JAVA_LONG,
                    ValueLayout.JAVA_INT,
                ),
            )

            // Wave 2 optional FFI bindings (gracefully null when an older native binary lacks a symbol).
            fnGetVramBudget = downcall(
                lookup,
                "vs_core_get_vram_budget",
                FunctionDescriptor.of(
                    ValueLayout.JAVA_INT,
                    ValueLayout.JAVA_LONG,
                    ValueLayout.JAVA_LONG,
                    ValueLayout.JAVA_LONG,
                ),
            )
            fnGetRtStats = downcall(
                lookup,
                "vs_core_get_rt_stats",
                FunctionDescriptor.of(
                    ValueLayout.JAVA_INT,
                    ValueLayout.JAVA_LONG,
                    ValueLayout.JAVA_LONG,
                ),
            )
            fnSetGpuProfilerEnabled = downcall(
                lookup,
                "vs_core_set_gpu_profiler_enabled",
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.JAVA_INT),
            )
            fnGetGpuPassTimings = downcall(
                lookup,
                "vs_core_get_gpu_pass_timings",
                FunctionDescriptor.of(
                    ValueLayout.JAVA_INT,
                    ValueLayout.JAVA_LONG,
                    ValueLayout.JAVA_INT,
                ),
            )
            fnGetAverageGpuFrameTimeMs = downcall(
                lookup,
                "vs_core_get_average_gpu_frame_time_ms",
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.JAVA_LONG),
            )
            fnGetRecoveryState = downcall(
                lookup,
                "vs_core_get_recovery_state",
                FunctionDescriptor.of(
                    ValueLayout.JAVA_INT,
                    ValueLayout.JAVA_LONG,
                    ValueLayout.JAVA_LONG,
                    ValueLayout.JAVA_INT,
                ),
            )
            fnResetRecoveryState = downcall(
                lookup,
                "vs_core_reset_recovery_state",
                FunctionDescriptor.of(ValueLayout.JAVA_INT),
            )
            fnStartPackPrewarm = downcall(
                lookup,
                "vs_core_start_pack_prewarm",
                FunctionDescriptor.of(
                    ValueLayout.JAVA_INT,
                    ValueLayout.JAVA_LONG,
                    ValueLayout.JAVA_INT,
                ),
            )
            fnGetPrewarmProgress = downcall(
                lookup,
                "vs_core_get_prewarm_progress",
                FunctionDescriptor.of(
                    ValueLayout.JAVA_INT,
                    ValueLayout.JAVA_LONG,
                    ValueLayout.JAVA_LONG,
                    ValueLayout.JAVA_LONG,
                ),
            )
            fnSetLodEnabled = downcall(
                lookup,
                "vs_core_set_lod_enabled",
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.JAVA_INT),
            )
            fnSetDimension = downcall(
                lookup,
                "vs_core_set_dimension",
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.JAVA_INT),
            )

            // Wave 3 optional FFI bindings (gracefully null when an older native binary lacks a symbol).
            fnAttachCutoutOmm = downcall(
                lookup,
                "vs_core_attach_cutout_omm",
                FunctionDescriptor.of(
                    ValueLayout.JAVA_INT,
                    ValueLayout.JAVA_LONG, // allocationKeyId
                    ValueLayout.JAVA_INT,  // layerOrdinal
                    ValueLayout.JAVA_INT,  // triangleCount
                    ValueLayout.JAVA_INT,  // subdivLevel
                    ValueLayout.JAVA_INT,  // use4State
                    ValueLayout.JAVA_LONG, // encodedDataPtr
                    ValueLayout.JAVA_INT,  // encodedDataLen
                ),
            )
            fnSubmitEntityVerticesCategorized = downcall(
                lookup,
                "vs_core_submit_entity_vertices_categorized",
                FunctionDescriptor.ofVoid(
                    ValueLayout.JAVA_LONG, // vertexDataPtr
                    ValueLayout.JAVA_INT,  // vertexCount
                    ValueLayout.JAVA_INT,  // vertexStride
                    ValueLayout.JAVA_INT,  // isQuads
                    ValueLayout.JAVA_INT,  // textureId
                    ValueLayout.JAVA_INT,  // normalTexId
                    ValueLayout.JAVA_INT,  // specularTexId
                    ValueLayout.JAVA_INT,  // kind
                    ValueLayout.JAVA_INT,  // category
                    ValueLayout.JAVA_INT,  // entityFlags
                ),
            )
            fnOnDisplayResized = downcall(
                lookup,
                "vs_core_on_display_resized",
                FunctionDescriptor.of(ValueLayout.JAVA_INT),
            )
            fnOnResourcesReloaded = downcall(
                lookup,
                "vs_core_on_resources_reloaded",
                FunctionDescriptor.of(ValueLayout.JAVA_INT),
            )

            nativeAvailable = fnInit != null
            if (nativeAvailable) {
                VulkanShaders.log.info("Loaded native Rust core '{}'", libName)
            }
        } catch (t: Throwable) {
            VulkanShaders.log.warn("Failed to bind native FFM symbols from '{}': {}", libName, t.message)
            nativeAvailable = false
        }
    }

    private fun sha256Hex(bytes: ByteArray): String {
        val digest = java.security.MessageDigest.getInstance("SHA-256").digest(bytes)
        val sb = StringBuilder(digest.size * 2)
        for (b in digest) {
            sb.append("%02x".format(b))
        }
        return sb.toString()
    }

    private fun sha256Hex(path: Path): String? = runCatching {
        val md = java.security.MessageDigest.getInstance("SHA-256")
        val buf = ByteArray(16 * 1024)
        Files.newInputStream(path).use { input ->
            while (true) {
                val n = input.read(buf)
                if (n < 0) break
                md.update(buf, 0, n)
            }
        }
        val digest = md.digest()
        val sb = StringBuilder(digest.size * 2)
        for (b in digest) {
            sb.append("%02x".format(b))
        }
        sb.toString()
    }.getOrNull()

    private fun readBundledSha256Sidecar(resourcePath: String): String? {
        val sidecarStream = NativeCoreBridge::class.java.getResourceAsStream("$resourcePath.sha256") ?: return null
        return runCatching {
            val raw = sidecarStream.use { String(it.readAllBytes(), StandardCharsets.UTF_8) }.trim()
            val token = raw.split(Regex("\\s+")).firstOrNull()?.trim()?.lowercase()
            token?.takeIf { it.length == 64 && it.all { c -> c in '0'..'9' || c in 'a'..'f' } }
        }.getOrNull()
    }

    private fun tryExtractOrLookup(resourcePath: String): SymbolLookup? {
        val stream = NativeCoreBridge::class.java.getResourceAsStream(resourcePath) ?: return null
        val fileName = resourcePath.substringAfterLast('/')
        return try {
            val bytes = stream.use { it.readAllBytes() }
            val bundledDigest = sha256Hex(bytes)
            val sidecarDigest = readBundledSha256Sidecar(resourcePath)
            if (sidecarDigest != null && !sidecarDigest.equals(bundledDigest, ignoreCase = true)) {
                HyaloclastErrorLog.recordError(
                    source = "NativeCoreBridge",
                    summary = "Packaged native '$fileName' failed SHA-256 verification against bundled sidecar (expected $sidecarDigest, actual $bundledDigest).",
                    showToast = false,
                )
                return null
            }
            val expectedDigest = sidecarDigest ?: bundledDigest
            val hashPrefix = expectedDigest.take(16)
            val cacheDir = resolveConfigDir()?.resolve("vulkanshaders")?.resolve("natives")
                ?: Path.of(System.getProperty("java.io.tmpdir"), "vulkanshaders-natives")
            Files.createDirectories(cacheDir)
            val cachedFile = cacheDir.resolve("${hashPrefix}_$fileName")
            val cacheValid = Files.isRegularFile(cachedFile) &&
                Files.size(cachedFile) == bytes.size.toLong() &&
                sha256Hex(cachedFile).equals(expectedDigest, ignoreCase = true)
            if (!cacheValid) {
                val tmp = cacheDir.resolve("${hashPrefix}_${fileName}.tmp")
                Files.write(tmp, bytes)
                try {
                    Files.move(
                        tmp,
                        cachedFile,
                        java.nio.file.StandardCopyOption.REPLACE_EXISTING,
                        java.nio.file.StandardCopyOption.ATOMIC_MOVE,
                    )
                } catch (_: Throwable) {
                    Files.move(tmp, cachedFile, java.nio.file.StandardCopyOption.REPLACE_EXISTING)
                }
                val verifiedExtracted = sha256Hex(cachedFile)
                if (!verifiedExtracted.equals(expectedDigest, ignoreCase = true)) {
                    HyaloclastErrorLog.recordError(
                        source = "NativeCoreBridge",
                        summary = "Extracted native '$fileName' failed post-write SHA-256 verification (expected $expectedDigest, actual ${verifiedExtracted ?: "unreadable"}).",
                        showToast = false,
                    )
                    runCatching { Files.deleteIfExists(cachedFile) }
                    return null
                }
            }
            SymbolLookup.libraryLookup(cachedFile, arena)
        } catch (t: Throwable) {
            VulkanShaders.log.debug("Could not load packaged native {}: {}", resourcePath, t.message)
            null
        }
    }

    internal fun resolveGameDir(): Path? =
        runCatching { FabricLoader.getInstance().gameDir }.getOrNull()

    private fun resolveConfigDir(): Path? =
        runCatching { FabricLoader.getInstance().configDir }.getOrNull()

    private fun gameDirNatives(): Path? = resolveGameDir()?.resolve("natives")

    private fun tryGameDirLibrary(platformDir: String, libName: String): SymbolLookup? {
        val dir = gameDirNatives() ?: return null
        for (candidate in listOf(dir.resolve(platformDir).resolve(libName), dir.resolve(libName))) {
            if (Files.isRegularFile(candidate)) {
                val found = runCatching { SymbolLookup.libraryLookup(candidate, arena) }.getOrNull()
                if (found != null) return found
            }
        }
        return null
    }

    private fun preloadVendorNatives() {
        val dir = gameDirNatives() ?: return
        if (!Files.isDirectory(dir)) return
        val patterns = listOf(
            "libnvidia-ngx", "nvngx", "libnrd", "nrd", "amd_fidelityfx", "libffx", "ffx", "libxess", "xess",
        )
        runCatching {
            Files.list(dir).use { stream ->
                stream.filter { Files.isRegularFile(it) }
                    .filter { f ->
                        val n = f.fileName.toString().lowercase()
                        patterns.any { p -> n.contains(p.lowercase()) } &&
                            (n.endsWith(".so") || n.contains(".so.") || n.endsWith(".dll") || n.endsWith(".dylib"))
                    }
                    .forEach { path ->
                        runCatching { System.load(path.toAbsolutePath().toString()) }
                            .onFailure { t -> VulkanShaders.log.debug("Could not preload vendor native {}: {}", path.fileName, t.message) }
                    }
            }
        }.onFailure { t -> VulkanShaders.log.debug("Could not scan <gameDir>/natives: {}", t.message) }
    }

    private fun downcall(lookup: SymbolLookup, symbol: String, desc: FunctionDescriptor): MethodHandle? {
        val addr = lookup.find(symbol).orElse(null) ?: return null
        return linker.downcallHandle(addr, desc)
    }

    @JvmStatic
    fun syncLwjglInstanceProcAddr() {
        val fn = fnSetInstanceProcAddr ?: return
        try {
            val provider = VK.getFunctionProvider()
            val pfn = provider?.getFunctionAddress("vkGetInstanceProcAddr") ?: 0L
            if (pfn != 0L) {
                fn.invokeExact(pfn)
            }
        } catch (t: Throwable) {
            logNativeCallError("syncLwjglInstanceProcAddr", t)
        }
    }

    @JvmStatic
    fun createMergedInstance(createInfoAddress: Long, allocatorAddress: Long, outInstanceAddress: Long): Int {
        val fn = fnCreateMergedInstance ?: return -1
        syncLwjglInstanceProcAddr()
        return runCatching {
            fn.invokeExact(createInfoAddress, allocatorAddress, outInstanceAddress) as Int
        }.getOrDefault(-1)
    }

    @JvmStatic
    fun createMergedDevice(
        physicalDeviceAddress: Long,
        createInfoAddress: Long,
        allocatorAddress: Long,
        outDeviceAddress: Long,
    ): Int {
        val fn = fnCreateMergedDevice ?: return -1
        syncLwjglInstanceProcAddr()
        val result = runCatching {
            fn.invokeExact(physicalDeviceAddress, createInfoAddress, allocatorAddress, outDeviceAddress) as Int
        }.getOrDefault(-1)
        if (result == 0 && StandaloneRenderPearlBridge.vulkanTurboLoaded) {
            runCatching {
                val vkDevice = MemorySegment.ofAddress(outDeviceAddress)
                    .reinterpret(ValueLayout.ADDRESS.byteSize())
                    .get(ValueLayout.ADDRESS, 0)
                    .address()
                val capsMask = getCapabilitiesMask(vkDevice)
                val bda = (capsMask and CAP_BUFFER_DEVICE_ADDRESS) != 0
                dev.vulkanturbo.api.shader.TurboShadersBridge.bdaSupported = bda
                dev.vulkanturbo.api.shader.TurboShadersBridge.bdaCapabilityResolvedByExtension = true
                VulkanShaders.log.info("Early Vulkan device capabilities: mask=0x{}, bufferDeviceAddress={}", capsMask.toString(16), bda)
            }.onFailure { VulkanShaders.log.warn("Failed to propagate early Vulkan device capabilities to VulkanTurbo", it) }
        }
        return result
    }

    @JvmStatic
    fun getCapabilitiesMask(vkDeviceAddress: Long): Int {
        val fn = fnGetCapabilitiesMask ?: return 0
        return runCatching {
            fn.invokeExact(vkDeviceAddress) as Int
        }.getOrDefault(0)
    }

    @JvmStatic
    fun lockQueue(): Boolean {
        if (!initialized) return false
        val fn = fnLockQueue ?: return false
        try {
            fn.invokeExact()
            return true
        } catch (t: Throwable) {
            logNativeCallError("lockQueue", t)
            return false
        }
    }

    @JvmStatic
    fun unlockQueue() {
        // Always release the native QUEUE_MUTEX even if shutdown() toggled `initialized = false`
        // between lockQueue() and the caller's finally block.
        val fn = fnUnlockQueue ?: return
        try {
            fn.invokeExact()
        } catch (t: Throwable) {
            logNativeCallError("unlockQueue", t)
        }
    }

    fun tryInitialize(
        vkInstance: Long,
        vkPhysicalDevice: Long,
        vkDevice: Long,
        graphicsQueue: Long,
        computeQueue: Long,
        graphicsQueueFamilyIndex: Int = 0,
        computeQueueFamilyIndex: Int = 0,
        timelineSemaphore: Long = 0L,
    ) {
        val fn = fnInit ?: return
        syncLwjglInstanceProcAddr()
        try {
            val rc = fn.invokeExact(vkInstance, vkPhysicalDevice, vkDevice, graphicsQueue, computeQueue) as Int
            initialized = rc == 0
            if (initialized) {
                setQueueFamilies(graphicsQueueFamilyIndex, computeQueueFamilyIndex)
                if (timelineSemaphore != 0L) {
                    bindTimelineSemaphore(timelineSemaphore)
                }
                configurePipeline(VulkanShaders.config)
                syncCustomDescriptorSetsToBridge()
            } else {
                VulkanShaders.log.error("vs_core_init returned error code {}", rc)
            }
        } catch (t: Throwable) {
            VulkanShaders.log.error("Failed to invoke vs_core_init", t)
            initialized = false
        }
    }

    fun setQueueFamilies(graphicsFamilyIndex: Int, computeFamilyIndex: Int) {
        if (!initialized) return
        val fn = fnSetQueueFamilies ?: return
        try {
            fn.invokeExact(graphicsFamilyIndex, computeFamilyIndex)
        } catch (t: Throwable) {
            logNativeCallError("setQueueFamilies", t)
        }
    }

    fun bindTimelineSemaphore(semaphoreRaw: Long) {
        if (!initialized || semaphoreRaw == 0L) return
        val fn = fnBindTimelineSemaphore ?: return
        try {
            fn.invokeExact(semaphoreRaw)
        } catch (t: Throwable) {
            logNativeCallError("bindTimelineSemaphore", t)
        }
    }

    fun configurePipeline(cfg: VulkanShadersConfig): Int {
        if (!initialized) return -1
        val fn = fnConfigurePipeline ?: return -1
        val rc = try {
            fn.invokeExact(
                cfg.renderTier.tierIndex,
                cfg.denoiserMode.ordinal,
                cfg.upscalerMode.ordinal,
                cfg.frameGenMode.ordinal,
                if (cfg.hdrEnabled) 1 else 0,
                if (cfg.enableOverlayPass) 1 else 0,
                if (cfg.enableSer) 1 else 0,
            ) as Int
        } catch (t: Throwable) {
            logNativeCallError("configurePipeline", t)
            -1
        }
        if (rc < 0) {
            VulkanShaders.log.warn("vs_core_configure_pipeline returned negative error status: {}", rc)
        }
        setShadowConfig(cfg.shadowResolution, cfg.shadowDistanceChunks)
        return rc
    }

    fun canEnableFrameGen(mode: Int): Boolean {
        if (!initialized) return false
        val fn = fnCanEnableFrameGen ?: return false
        return runCatching { (fn.invokeExact(mode) as Int) == 1 }
            .getOrElse { logNativeCallError("canEnableFrameGen", it); false }
    }

    fun takeFrameGenFailure(): Int {
        if (!initialized) return 0
        val fn = fnTakeFrameGenFailure ?: return 0
        return runCatching { fn.invokeExact() as Int }
            .getOrElse { logNativeCallError("takeFrameGenFailure", it); 0 }
    }

    @JvmStatic
    fun setShadowConfig(shadowResolution: Int, shadowDistanceChunks: Int) {
        if (!initialized) return
        val fn = fnSetShadowConfig ?: return
        try {
            val maxShadowRes = if (isHighEndGpu()) 8192 else 4096
            fn.invokeExact(
                shadowResolution.coerceIn(512, maxShadowRes),
                shadowDistanceChunks.coerceIn(4, 64),
            )
        } catch (t: Throwable) {
            logNativeCallError("setShadowConfig", t)
        }
    }

    @JvmStatic
    fun setJitter(jitterX: Float, jitterY: Float) {
        if (!initialized) return
        val fn = fnSetJitter ?: return
        try {
            fn.invokeExact(jitterX, jitterY)
        } catch (t: Throwable) {
            logNativeCallError("setJitter", t)
        }
    }

    @JvmStatic
    fun setCameraPosition(x: Double, y: Double, z: Double) {
        if (!initialized) return
        val fn = fnSetCameraPosition ?: return
        try {
            fn.invokeExact(x, y, z)
        } catch (t: Throwable) {
            logNativeCallError("setCameraPosition", t)
        }
    }

    fun updateWorldUbo(address: Long, byteSize: Int) {
        if (!initialized) return
        val fn = fnUpdateWorldUbo ?: return
        try {
            fn.invokeExact(address, byteSize)
        } catch (t: Throwable) {
            logNativeCallError("updateWorldUbo", t)
        }
    }

    @JvmStatic
    fun resetTemporalHistory() {
        if (!initialized) return
        val fn = fnResetTemporalHistory ?: return
        try {
            fn.invokeExact()
        } catch (t: Throwable) {
            logNativeCallError("resetTemporalHistory", t)
        }
    }

    fun registerSectionSlice(keyId: Long, layerOrdinal: Int, vkBuffer: Long, byteOffset: Long, byteSize: Long) {
        if (!initialized) return
        val fn = fnRegisterSlice ?: return
        try {
            fn.invokeExact(keyId, layerOrdinal, vkBuffer, byteOffset, byteSize)
        } catch (t: Throwable) {
            logNativeCallError("registerSectionSlice", t)
        }
    }

    fun registerSectionSliceEx(
        keyId: Long,
        layerOrdinal: Int,
        sectionPosPacked: Long,
        vkBuffer: Long,
        byteOffset: Long,
        byteSize: Long,
        vertexStride: Int,
        vertexCount: Int,
        isRegionShifted: Boolean,
    ) {
        if (!initialized) return
        val ex = fnRegisterSliceEx
        if (ex != null) {
            try {
                ex.invokeExact(
                    keyId,
                    layerOrdinal,
                    sectionPosPacked,
                    vkBuffer,
                    byteOffset,
                    byteSize,
                    vertexStride,
                    vertexCount,
                    if (isRegionShifted) 1 else 0,
                )
            } catch (t: Throwable) {
                logNativeCallError("registerSectionSliceEx", t)
            }
        } else {
            registerSectionSlice(keyId, layerOrdinal, vkBuffer, byteOffset, byteSize)
        }
    }

    fun unregisterSectionSlice(keyId: Long, layerOrdinal: Int) {
        if (!initialized) return
        val fn = fnUnregisterSlice ?: return
        try {
            fn.invokeExact(keyId, layerOrdinal)
        } catch (t: Throwable) {
            logNativeCallError("unregisterSectionSlice", t)
        }
    }

    fun registerTexture(vkImage: Long, width: Int, height: Int, mipLevels: Int) {
        if (!initialized) return
        val fn = fnRegisterTexture ?: return
        try {
            fn.invokeExact(vkImage, width, height, mipLevels)
        } catch (t: Throwable) {
            logNativeCallError("registerTexture", t)
        }
    }

    fun registerTextureEx(
        vkImage: Long,
        vkImageView: Long,
        width: Int,
        height: Int,
        mipLevels: Int,
        pbrRoleOrdinal: Int,
    ): Int {
        return registerTextureWithSampler(
            vkImage = vkImage,
            vkImageView = vkImageView,
            vkSampler = 0L,
            vkFormat = 37,
            width = width,
            height = height,
            mipLevels = mipLevels,
            pbrRoleOrdinal = pbrRoleOrdinal,
        )
    }

    fun registerTextureWithSampler(
        vkImage: Long,
        vkImageView: Long,
        vkSampler: Long,
        vkFormat: Int,
        width: Int,
        height: Int,
        mipLevels: Int,
        pbrRoleOrdinal: Int,
    ): Int {
        if (!initialized) return -1
        val withSampler = fnRegisterTextureWithSampler
        if (withSampler != null) {
            val slot = runCatching {
                withSampler.invokeExact(
                    vkImage,
                    vkImageView,
                    vkSampler,
                    vkFormat,
                    width,
                    height,
                    mipLevels,
                    pbrRoleOrdinal,
                ) as Int
            }.getOrDefault(-1)
            if (slot < 0) {
                VulkanShaders.log.debug("Bindless registration returned {} for vkImage=0x{}", slot, java.lang.Long.toHexString(vkImage))
            }
            return slot
        }
        val ex = fnRegisterTextureEx
        if (ex != null) {
            return runCatching {
                ex.invokeExact(vkImage, vkImageView, width, height, mipLevels, pbrRoleOrdinal) as Int
            }.getOrDefault(-1)
        }
        registerTexture(vkImage, width, height, mipLevels)
        return 0
    }

    @JvmStatic
    fun linkPbrCompanions(albedoSlot: Int, normalSlot: Int, specularSlot: Int) {
        if (!initialized || albedoSlot < 0) return
        val fn = fnLinkPbrCompanions ?: return
        try {
            fn.invokeExact(albedoSlot, normalSlot, specularSlot)
        } catch (t: Throwable) {
            logNativeCallError("linkPbrCompanions", t)
        }
    }

    fun unregisterTexture(vkImage: Long) {
        if (!initialized) return
        val fn = fnUnregisterTexture ?: return
        try {
            fn.invokeExact(vkImage)
        } catch (t: Throwable) {
            logNativeCallError("unregisterTexture", t)
        }
    }

    @JvmStatic
    fun interceptSurfaceBlit(
        cmdAddress: Long,
        srcImage: Long,
        srcLayout: Int,
        dstImage: Long,
        dstLayout: Int,
        width: Int,
        height: Int,
    ): Boolean {
        if (!initialized) return false
        val fn = fnInterceptSurfaceBlit ?: return false
        return try {
            (fn.invokeExact(cmdAddress, srcImage, srcLayout, dstImage, dstLayout, width, height) as Int) > 0
        } catch (t: Throwable) {
            logNativeCallError("interceptSurfaceBlit", t)
            false
        }
    }

    @JvmStatic
    fun setOverlayDepthParams(
        zScale: Float,
        zOffset: Float,
        wScale: Float,
        wOffset: Float,
        srcW: Int,
        srcH: Int,
        dstW: Int,
        dstH: Int,
    ): Float {
        if (!initialized) return 1.0f
        val fn = fnSetOverlayDepthParams ?: return 1.0f
        return try {
            fn.invokeExact(zScale, zOffset, wScale, wOffset, srcW, srcH, dstW, dstH) as Float
        } catch (t: Throwable) {
            logNativeCallError("setOverlayDepthParams", t)
            1.0f
        }
    }

    @JvmStatic
    fun reflexSetSwapchain(swapchain: Long) {
        if (!initialized) return
        val fn = fnReflexSetSwapchain ?: return
        try {
            fn.invokeExact(swapchain)
        } catch (t: Throwable) {
            logNativeCallError("reflexSetSwapchain", t)
        }
    }

    @JvmStatic
    fun reflexMarker(markerId: Int) {
        if (!initialized) return
        val fn = fnReflexMarker ?: return
        try {
            fn.invokeExact(markerId)
        } catch (t: Throwable) {
            logNativeCallError("reflexMarker", t)
        }
    }

    @JvmStatic
    fun reflexSleepAndSampleInput() {
        if (!initialized) return
        val fn = fnReflexSleepAndSampleInput ?: return
        try {
            fn.invokeExact()
        } catch (t: Throwable) {
            logNativeCallError("reflexSleepAndSampleInput", t)
        }
    }

    @JvmStatic
    fun submitEntityVertices(
        basePtr: Long,
        vertexCount: Int,
        vertexStride: Int,
        albedoSlot: Int,
        crumbling: Boolean,
        firstPersonAvatar: Boolean,
    ): Int {
        if (!initialized || basePtr == 0L || vertexCount <= 0 || albedoSlot < 0) return 0
        val fn = fnSubmitEntityVertices ?: return 0
        val geometryKind = when {
            crumbling -> 1
            firstPersonAvatar -> 2
            else -> 0
        }
        return runCatching {
            fn.invokeExact(basePtr, vertexCount, vertexStride, albedoSlot, geometryKind) as Int
        }.getOrDefault(0)
    }

    @JvmStatic
    fun recordFrameRtx(
        cmdAddress: Long,
        cameraX: Double,
        cameraY: Double,
        cameraZ: Double,
        prevCameraX: Double,
        prevCameraY: Double,
        prevCameraZ: Double,
        frameIndex: Int,
        maxBlasBuilds: Int = 64,
    ): Long {
        if (!initialized || cmdAddress == 0L) return 0L
        val fn = fnRecordFrameRtx ?: return 0L
        return runCatching {
            fn.invokeExact(
                cmdAddress,
                cameraX,
                cameraY,
                cameraZ,
                prevCameraX,
                prevCameraY,
                prevCameraZ,
                frameIndex,
                maxBlasBuilds,
            ) as Long
        }.getOrDefault(0L)
    }

    @JvmStatic
    fun compileActivePack(packPath: Path): Int {
        if (!initialized) return 0
        val fn = fnCompileActivePack ?: return 0
        val rc = runCatching {
            val utf8 = packPath.toAbsolutePath().toString().toByteArray(StandardCharsets.UTF_8)
            val seg = ioScratch.get().primary.ensureCapacity(utf8.size.toLong())
            MemorySegment.copy(utf8, 0, seg, ValueLayout.JAVA_BYTE, 0L, utf8.size)
            fn.invokeExact(seg.address(), utf8.size) as Int
        }.getOrDefault(-1)

        if (rc < 0) {
            val errText = fetchLastCompileError() ?: "Native shader compilation failed (code $rc)"
            lastCompilationError = errText
        } else {
            val diagText = fetchLastCompileError()?.takeIf { it.isNotBlank() }
            lastCompilationError = diagText
            syncCustomDescriptorSetsToBridge()
        }
        return rc
    }

    @JvmStatic
    fun prepareSpecConstants(specPairs: List<Pair<Int, Int>>): Int =
        invokeSpecConstantsHandle(fnPrepareSpecConstants, specPairs)

    @JvmStatic
    fun updateSpecConstants(specPairs: List<Pair<Int, Int>>): Int =
        invokeSpecConstantsHandle(fnUpdateSpecConstants, specPairs)

    private fun invokeSpecConstantsHandle(fn: MethodHandle?, specPairs: List<Pair<Int, Int>>): Int {
        if (!initialized) return 0
        val handle = fn ?: return 0
        if (specPairs.isEmpty()) {
            return runCatching { handle.invokeExact(0L, 0) as Int }.getOrDefault(0)
        }
        return runCatching {
            val flat = IntArray(specPairs.size * 2)
            for (i in specPairs.indices) {
                flat[i * 2] = specPairs[i].first
                flat[i * 2 + 1] = specPairs[i].second
            }
            val seg = ioScratch.get().primary.ensureCapacity(flat.size.toLong() * ValueLayout.JAVA_INT.byteSize())
            MemorySegment.copy(flat, 0, seg, ValueLayout.JAVA_INT, 0L, flat.size)
            handle.invokeExact(seg.address(), specPairs.size) as Int
        }.getOrDefault(-1)
    }

    @JvmStatic
    fun fetchLastCompileError(): String? {
        if (!initialized) return lastCompilationError
        val fn = fnGetLastCompileError ?: return lastCompilationError
        return runCatching {
            val len = fn.invokeExact(0L, 0) as Int
            if (len <= 0) return@runCatching null
            val buf = ioScratch.get().output.ensureCapacity(len.toLong())
            val copied = (fn.invokeExact(buf.address(), len) as Int).coerceIn(0, len)
            if (copied <= 0) return@runCatching null
            val bytes = ByteArray(copied)
            MemorySegment.copy(buf, ValueLayout.JAVA_BYTE, 0L, bytes, 0, copied)
            String(bytes, StandardCharsets.UTF_8)
        }.getOrNull()
    }

    @JvmStatic
    fun bindGbufferImages(
        albedoImg: Long,
        normalImg: Long,
        materialImg: Long,
        velocityImg: Long,
        depthImg: Long,
        radianceImg: Long,
    ) {
        if (!initialized) return
        val fn = fnBindGbufferImages ?: return
        try {
            fn.invokeExact(
                albedoImg,
                normalImg,
                materialImg,
                velocityImg,
                depthImg,
                radianceImg,
            )
        } catch (t: Throwable) {
            logNativeCallError("bindGbufferImages", t)
        }
    }

    @JvmStatic
    fun bindGbufferViews(
        albedoImg: Long,
        albedoView: Long,
        normalImg: Long,
        normalView: Long,
        materialImg: Long,
        materialView: Long,
        velocityImg: Long,
        velocityView: Long,
        depthImg: Long,
        depthView: Long,
        radianceImg: Long,
        radianceView: Long,
    ) {
        if (!initialized) return
        val fn = fnBindGbufferViews
        if (fn != null) {
            try {
                fn.invokeExact(
                    albedoImg,
                    albedoView,
                    normalImg,
                    normalView,
                    materialImg,
                    materialView,
                    velocityImg,
                    velocityView,
                    depthImg,
                    depthView,
                    radianceImg,
                    radianceView,
                )
            } catch (t: Throwable) {
                logNativeCallError("bindGbufferViews", t)
            }
        } else {
            bindGbufferImages(albedoImg, normalImg, materialImg, velocityImg, depthImg, radianceImg)
        }
    }

    @JvmStatic
    fun bindPostTargets(
        hudlessImg: Long,
        hudlessView: Long,
        prevHudlessImg: Long,
        prevHudlessView: Long,
        uiImg: Long,
        uiView: Long,
        interpolatedImg: Long,
        interpolatedView: Long,
    ) {
        if (!initialized) return
        val fn = fnBindPostTargets ?: return
        try {
            fn.invokeExact(
                hudlessImg,
                hudlessView,
                prevHudlessImg,
                prevHudlessView,
                uiImg,
                uiView,
                interpolatedImg,
                interpolatedView,
            )
        } catch (t: Throwable) {
            logNativeCallError("bindPostTargets", t)
        }
    }

    @JvmStatic
    fun executeRenderGraph(cmdAddress: Long, width: Int, height: Int): Int {
        if (!initialized || cmdAddress == 0L) return 0
        val fn = fnExecuteRenderGraph ?: return 0
        val rc = try {
            fn.invokeExact(cmdAddress, width, height) as Int
        } catch (t: Throwable) {
            logNativeCallError("executeRenderGraph", t)
            -1
        }
        if (rc < 0) {
            VulkanShaders.log.warn("vs_core_execute_render_graph returned negative status: {}", rc)
        }
        return rc
    }

    @JvmStatic
    fun executeAfterGui(cmdAddress: Long, uiImage: Long, uiView: Long, width: Int, height: Int): Int {
        if (!initialized || cmdAddress == 0L) return 0
        val fn = fnExecuteAfterGui ?: return 0
        return try {
            fn.invokeExact(cmdAddress, uiImage, uiView, width, height) as Int
        } catch (t: Throwable) {
            logNativeCallError("executeAfterGui", t)
            0
        }
    }

    @JvmStatic
    fun interceptAcquireNextImage(
        swapchainHandle: Long,
        timeoutNs: Long,
        semaphore: Long,
        fence: Long,
    ): Int {
        if (!initialized) return -1
        val fn = fnInterceptAcquireNextImage ?: return -1
        return try {
            fn.invokeExact(swapchainHandle, timeoutNs, semaphore, fence) as Int
        } catch (t: Throwable) {
            logNativeCallError("interceptAcquireNextImage", t)
            -1
        }
    }

    @JvmStatic
    fun interceptQueuePresent(
        queueHandle: Long,
        swapchainHandle: Long,
        imageIndex: Int,
        waitSemaphore: Long,
    ): Boolean {
        if (!initialized) return false
        val fn = fnInterceptQueuePresent ?: return false
        return try {
            (fn.invokeExact(queueHandle, swapchainHandle, imageIndex, waitSemaphore) as Int) != 0
        } catch (t: Throwable) {
            logNativeCallError("interceptQueuePresent", t)
            false
        }
    }

    @JvmStatic
    fun waitPresentIdle() {
        if (!initialized) return
        val fn = fnWaitPresentIdle ?: return
        try {
            fn.invokeExact()
        } catch (t: Throwable) {
            logNativeCallError("waitPresentIdle", t)
        }
    }

    @JvmStatic
    fun queryCapabilities(): Int {
        fnQueryCapabilities?.let { fn ->
            return runCatching { fn.invokeExact() as Int }
                .getOrElse { t -> logNativeCallError("queryCapabilities", t); 0 }
        }
        if (!initialized) return 0
        return runCatching { getCapabilitiesMask(0L) }.getOrDefault(0)
    }

    @JvmStatic
    fun hasCapability(bit: Int): Boolean = (queryCapabilities() and bit) != 0

    @JvmStatic
    fun getCapabilityDetails(): String {
        val fn = fnGetCapabilityDetails ?: return ""
        return runCatching {
            val needed = fn.invokeExact(0L, 0) as Int
            if (needed <= 0) return ""
            val outSeg = ioScratch.get().output.ensureCapacity(needed.toLong())
            val written = (fn.invokeExact(outSeg.address(), needed) as Int).coerceIn(0, needed)
            if (written <= 0) return@runCatching ""
            val bytes = ByteArray(written)
            MemorySegment.copy(outSeg, ValueLayout.JAVA_BYTE, 0L, bytes, 0, written)
            String(bytes, StandardCharsets.UTF_8)
        }.getOrElse { t -> logNativeCallError("getCapabilityDetails", t); "" }
    }

    @JvmStatic
    fun getHardwareProfile(): HardwareProfile {
        val capsMask = queryCapabilities()
        val detailsJson = getCapabilityDetails()
        val profile = parseHardwareProfile(detailsJson, capsMask)
        if (initialized && profile.gpuTierClass.isNotBlank()) {
            cachedGpuTierClass = profile.gpuTierClass
        }
        return profile
    }

    internal fun parseHardwareProfile(detailsJson: String, capabilitiesMask: Int): HardwareProfile {
        if (detailsJson.isBlank()) {
            val fallbackTier = if ((capabilitiesMask and CAP_HIGH_END_GPU) != 0) "high" else "low"
            return HardwareProfile(
                gpuTierClass = fallbackTier,
                serNv = (capabilitiesMask and CAP_SER_NV) != 0,
                serExt = (capabilitiesMask and CAP_SER_EXT) != 0,
                serActive = (capabilitiesMask and (CAP_SER_NV or CAP_SER_EXT)) != 0,
                opacityMicromap = (capabilitiesMask and CAP_OPACITY_MICROMAP) != 0,
                capabilitiesMask = capabilitiesMask,
            )
        }
        return runCatching {
            val root = JsonParser.parseString(detailsJson).asJsonObject
            val gpu = root.getAsJsonObject("gpu")
            val lastFrameDtMs = root.get("lastFrameDtMs")?.takeIf { !it.isJsonNull }?.asDouble ?: 0.0
            if (gpu == null) {
                HardwareProfile(lastFrameDtMs = lastFrameDtMs, capabilitiesMask = capabilitiesMask)
            } else {
                fun str(key: String, def: String = ""): String =
                    gpu.get(key)?.takeIf { !it.isJsonNull }?.asString ?: def
                fun intVal(key: String, def: Int = 0): Int =
                    gpu.get(key)?.takeIf { !it.isJsonNull }?.asInt ?: def
                fun longVal(key: String, def: Long = 0L): Long =
                    gpu.get(key)?.takeIf { !it.isJsonNull }?.asLong ?: def
                fun boolVal(key: String, def: Boolean = false): Boolean =
                    gpu.get(key)?.takeIf { !it.isJsonNull }?.asBoolean ?: def

                HardwareProfile(
                    gpuName = str("name"),
                    apiVersion = str("apiVersion"),
                    driver = str("driver"),
                    driverVersion = str("driverVersion"),
                    vendorId = intVal("vendorId"),
                    deviceId = intVal("deviceId"),
                    deviceType = str("deviceType", "other"),
                    vramBytes = longVal("vramBytes"),
                    vramMb = longVal("vramMb"),
                    gpuArch = str("gpuArch", "unknown"),
                    gpuTierClass = str("gpuTierClass", "low"),
                    serNv = boolVal("serNv", (capabilitiesMask and CAP_SER_NV) != 0),
                    serExt = boolVal("serExt", (capabilitiesMask and CAP_SER_EXT) != 0),
                    serActive = boolVal("serActive"),
                    opacityMicromap = boolVal("opacityMicromap", (capabilitiesMask and CAP_OPACITY_MICROMAP) != 0),
                    lastFrameDtMs = lastFrameDtMs,
                    capabilitiesMask = capabilitiesMask,
                )
            }
        }.getOrElse {
            HardwareProfile(capabilitiesMask = capabilitiesMask)
        }
    }

    @JvmStatic
    fun getVramBudgetSnapshot(): VramBudgetSnapshot? {
        if (!initialized) return null
        val fn = fnGetVramBudget ?: return null
        return synchronized(telemetryLock) {
            runCatching {
                vramUsedSeg.set(ValueLayout.JAVA_LONG, 0L, 0L)
                vramBudgetSeg.set(ValueLayout.JAVA_LONG, 0L, 0L)
                vramPressureSeg.set(ValueLayout.JAVA_INT, 0L, 0)
                val rc = fn.invokeExact(
                    vramUsedSeg.address(),
                    vramBudgetSeg.address(),
                    vramPressureSeg.address(),
                ) as Int
                if (rc < 0) return@synchronized null
                VramBudgetSnapshot(
                    usedMb = vramUsedSeg.get(ValueLayout.JAVA_LONG, 0L).coerceAtLeast(0L) / (1024L * 1024L),
                    budgetMb = vramBudgetSeg.get(ValueLayout.JAVA_LONG, 0L).coerceAtLeast(0L) / (1024L * 1024L),
                    pressure = vramPressureSeg.get(ValueLayout.JAVA_INT, 0L).coerceAtLeast(0),
                )
            }.getOrNull()
        }
    }

    @JvmStatic
    fun getAverageGpuFrameTimeMs(): Double? {
        if (!initialized) return null
        val fn = fnGetAverageGpuFrameTimeMs ?: return null
        return synchronized(telemetryLock) {
            runCatching {
                averageGpuFrameTimeSeg.set(ValueLayout.JAVA_DOUBLE, 0L, 0.0)
                if ((fn.invokeExact(averageGpuFrameTimeSeg.address()) as Int) != 1) {
                    return@synchronized null
                }
                averageGpuFrameTimeSeg.get(ValueLayout.JAVA_DOUBLE, 0L)
                    .takeIf { it.isFinite() && it > 0.0 }
            }.getOrNull()
        }
    }

    @JvmStatic
    fun getGpuTierClass(): String {
        cachedGpuTierClass?.let { return it }
        if (!initialized) {
            return if (hasCapability(CAP_HIGH_END_GPU)) "high" else "low"
        }
        val tier = getHardwareProfile().gpuTierClass.lowercase()
        cachedGpuTierClass = tier
        return tier
    }

    @JvmStatic
    fun isHighEndGpu(): Boolean {
        if (hasCapability(CAP_HIGH_END_GPU)) return true
        val tier = cachedGpuTierClass ?: return false
        return tier.equals("high", ignoreCase = true) || tier.equals("enthusiast", ignoreCase = true)
    }

    @JvmStatic
    fun isEnthusiastGpu(): Boolean {
        if (!hasCapability(CAP_HIGH_END_GPU)) return false
        return getGpuTierClass().equals("enthusiast", ignoreCase = true)
    }

    @JvmStatic
    fun isAdaOrNewer(): Boolean = hasCapability(CAP_ADA_OR_NEWER)

    @JvmStatic
    fun isSerActive(): Boolean {
        if (!hasCapability(CAP_SER_NV) && !hasCapability(CAP_SER_EXT)) return false
        return getHardwareProfile().serActive
    }

    @JvmStatic
    fun isHdrSwapchainSupported(surfaceHandle: Long): Boolean {
        fnIsHdrSwapchainSupported?.let { fn ->
            if (surfaceHandle == 0L) return false
            return runCatching { (fn.invokeExact(surfaceHandle) as Int) > 0 }
                .getOrElse { t -> logNativeCallError("isHdrSwapchainSupported", t); false }
        }
        return hasCapability(CAP_HDR_SWAPCHAIN)
    }

    @JvmStatic
    fun uploadPackTexture(
        slot: Int,
        width: Int,
        height: Int,
        rgba: ByteArray,
        filterLinear: Boolean,
        wrapRepeat: Boolean,
    ): Int {
        if (!initialized) return -1
        val fn = fnUploadPackTexture ?: return -1
        if (width <= 0 || height <= 0 || rgba.isEmpty()) return -1
        return runCatching {
            val pixels = ioScratch.get().primary.ensureCapacity(rgba.size.toLong())
            MemorySegment.copy(rgba, 0, pixels, ValueLayout.JAVA_BYTE, 0L, rgba.size)
            fn.invokeExact(
                slot,
                width,
                height,
                pixels.address(),
                rgba.size,
                if (filterLinear) 1 else 0,
                if (wrapRepeat) 1 else 0,
            ) as Int
        }.getOrElse { t -> logNativeCallError("uploadPackTexture", t); -1 }
    }

    @JvmStatic
    fun getCustomDescriptorSetLayouts(maxCount: Int = 8): LongArray {
        if (!initialized) return LongArray(0)
        val fn = fnGetCustomDescriptorSetLayouts ?: return LongArray(0)
        if (maxCount <= 0) return LongArray(0)
        return runCatching {
            synchronized(descriptorScratchLock) {
                val out = customDescLayoutsScratch.ensureCapacity(maxCount * 8L)
                val count = fn.invokeExact(out.address(), maxCount) as Int
                if (count <= 0) {
                    LongArray(0)
                } else {
                    val n = count.coerceAtMost(maxCount)
                    LongArray(n) { i -> out.get(ValueLayout.JAVA_LONG, i * 8L) }
                }
            }
        }.getOrElse { t -> logNativeCallError("getCustomDescriptorSetLayouts", t); LongArray(0) }
    }

    @JvmStatic
    fun getCustomDescriptorSets(): LongArray {
        if (!initialized) return longArrayOf(0L, 0L)
        val fn = fnGetCustomDescriptorSets ?: return longArrayOf(0L, 0L)
        return runCatching {
            synchronized(descriptorScratchLock) {
                customDescBindlessSeg.set(ValueLayout.JAVA_LONG, 0L, 0L)
                customDescPackSeg.set(ValueLayout.JAVA_LONG, 0L, 0L)
                val rc = fn.invokeExact(customDescBindlessSeg, customDescPackSeg) as Int
                if (rc >= 0) {
                    longArrayOf(
                        customDescBindlessSeg.get(ValueLayout.JAVA_LONG, 0L),
                        customDescPackSeg.get(ValueLayout.JAVA_LONG, 0L),
                    )
                } else {
                    longArrayOf(0L, 0L)
                }
            }
        }.getOrElse { t ->
            logNativeCallError("getCustomDescriptorSets", t)
            longArrayOf(0L, 0L)
        }
    }

    /** Caches the bindless and pack VkDescriptorSet handles in Pyroclast's bridge to avoid per-draw FFI calls. */
    @JvmStatic
    fun syncCustomDescriptorSetsToBridge() {
        if (!StandaloneRenderPearlBridge.vulkanTurboLoaded) return
        val sets = getCustomDescriptorSets()
        runCatching {
            dev.vulkanturbo.api.shader.TurboShadersBridge.updateCachedCustomDescriptorSets(sets[0], sets[1])
        }
    }

    @JvmStatic
    fun bindCustomDescriptorSets(
        vkCommandBuffer: Long,
        vkPipelineLayout: Long,
        pipelineBindPoint: Int,
        firstSet: Int,
    ) {
        if (!initialized) return
        val fn = fnBindCustomDescriptorSets ?: return
        try {
            fn.invokeExact(vkCommandBuffer, vkPipelineLayout, pipelineBindPoint, firstSet)
        } catch (t: Throwable) {
            logNativeCallError("bindCustomDescriptorSets", t)
        }
    }

    /**
     * Registers a section slice in a single atomic call with its geometry metadata and LabPBR
     * albedo/normal/specular bindless slots + material flags.
     */
    @JvmStatic
    fun registerSectionSliceWithPbr(
        keyId: Long,
        layerOrdinal: Int,
        sectionPosPacked: Long,
        vkBuffer: Long,
        byteOffset: Long,
        byteSize: Long,
        vertexStride: Int,
        vertexCount: Int,
        isRegionShifted: Boolean,
        albedoSlot: Int,
        normalSlot: Int,
        specularSlot: Int,
        materialFlags: Int,
    ) {
        if (!initialized) return
        val fn = fnRegisterSectionSliceWithPbr
        if (fn != null) {
            try {
                fn.invokeExact(
                    keyId,
                    layerOrdinal,
                    sectionPosPacked,
                    vkBuffer,
                    byteOffset,
                    byteSize,
                    vertexStride,
                    vertexCount,
                    if (isRegionShifted) 1 else 0,
                    albedoSlot,
                    normalSlot,
                    specularSlot,
                    materialFlags,
                )
            } catch (t: Throwable) {
                logNativeCallError("registerSectionSliceWithPbr", t)
            }
            return
        }
        registerSectionSliceEx(
            keyId = keyId,
            layerOrdinal = layerOrdinal,
            sectionPosPacked = sectionPosPacked,
            vkBuffer = vkBuffer,
            byteOffset = byteOffset,
            byteSize = byteSize,
            vertexStride = vertexStride,
            vertexCount = vertexCount,
            isRegionShifted = isRegionShifted,
        )
    }

    @JvmStatic
    fun submitEmissiveLights(
        lightsPtr: Long,
        lightCount: Int,
        camX: Double,
        camY: Double,
        camZ: Double,
    ): Int {
        if (!initialized) return 0
        val fn = fnSubmitEmissiveLights ?: return 0
        return runCatching {
            fn.invokeExact(lightsPtr, lightCount, camX, camY, camZ) as Int
        }.getOrElse { t -> logNativeCallError("submitEmissiveLights", t); 0 }
    }

    @JvmStatic
    fun registerLodSlice(
        lodKeyId: Long,
        level: Int,
        originBlockX: Int,
        originBlockY: Int,
        originBlockZ: Int,
        scaleBlocks: Float,
        vkBuffer: Long,
        byteOffset: Long,
        byteSize: Long,
        vertexStride: Int,
        vertexCount: Int,
        indexBuffer: Long,
        indexCount: Int,
        albedoColorAbgr: Int,
    ) {
        if (!initialized) return
        val fn = fnRegisterLodSlice ?: return
        try {
            fn.invokeExact(
                lodKeyId,
                level,
                originBlockX,
                originBlockY,
                originBlockZ,
                scaleBlocks,
                vkBuffer,
                byteOffset,
                byteSize,
                vertexStride,
                vertexCount,
                indexBuffer,
                indexCount,
                albedoColorAbgr,
            )
        } catch (t: Throwable) {
            logNativeCallError("registerLodSlice", t)
        }
    }

    @JvmStatic
    @JvmOverloads
    fun uploadLodMeshCpu(
        lodKeyId: Long,
        level: Int,
        originBlockX: Int,
        originBlockY: Int,
        originBlockZ: Int,
        scaleBlocks: Float,
        vertices: ByteBuffer,
        vertexCount: Int,
        indices: ByteBuffer?,
        indexCount: Int,
        materialFlags: Int = 0,
    ): Boolean {
        if (!initialized || vertices.remaining() <= 0) return false
        val hasIndices = indices != null && indexCount > 0 && indices.remaining() > 0
        if (hasIndices) {
            if (vertexCount < 3 || indexCount < 3) return false
        } else {
            if (indices != null || indexCount != 0 || vertexCount < 4) return false
        }
        val fn = fnUploadLodMeshCpu ?: return false
        return runCatching {
            val scratch = lodUploadScratch.get()
            fun bufferAddress(buf: ByteBuffer, buffer: PersistentScratchBuffer): Long {
                val view = buf.slice()
                return if (view.isDirect) {
                    MemorySegment.ofBuffer(view).address()
                } else {
                    val remaining = view.remaining()
                    val seg = buffer.ensureCapacity(remaining.toLong())
                    seg.asSlice(0L, remaining.toLong()).asByteBuffer().put(view)
                    seg.address()
                }
            }
            val vertPtr = bufferAddress(vertices, scratch.vertices)
            val idxPtr = if (hasIndices) bufferAddress(indices!!, scratch.indices) else 0L
            val effIndexCount = if (hasIndices) indexCount else 0
            val res = fn.invokeExact(
                lodKeyId,
                level,
                originBlockX,
                originBlockY,
                originBlockZ,
                scaleBlocks,
                vertPtr,
                vertexCount,
                idxPtr,
                effIndexCount,
                materialFlags,
            ) as Int
            res > 0
        }.getOrElse { t -> logNativeCallError("uploadLodMeshCpu", t); false }
    }

    @JvmStatic
    fun unregisterLodSlice(lodKeyId: Long) {
        if (!initialized) return
        val fn = fnUnregisterLodSlice ?: return
        try {
            fn.invokeExact(lodKeyId)
        } catch (t: Throwable) {
            logNativeCallError("unregisterLodSlice", t)
        }
    }

    @JvmStatic
    fun clearLodSlices() {
        if (!initialized) return
        val fn = fnClearLodSlices ?: return
        try {
            fn.invokeExact()
        } catch (t: Throwable) {
            logNativeCallError("clearLodSlices", t)
        }
    }

    @JvmStatic
    fun setLodCutoffDistance(cutoffBlocks: Float): Int {
        if (!initialized) return 0
        val fn = fnSetLodCutoffDistance ?: return 0
        return try {
            fn.invokeExact(cutoffBlocks) as Int
        } catch (t: Throwable) {
            logNativeCallError("setLodCutoffDistance", t)
            -1
        }
    }

    @JvmStatic
    fun bindDhTargets(
        colorImg: Long,
        colorView: Long,
        normalImg: Long,
        normalView: Long,
        materialImg: Long,
        materialView: Long,
        depthImg: Long,
        depthView: Long,
        width: Int,
        height: Int,
        invViewProj: FloatArray?,
        fadeStart: Float,
        fadeEnd: Float,
        dhFarClip: Float,
        active: Boolean,
    ): Int {
        if (!initialized) return 0
        val fn = fnBindDhTargets ?: return 0
        return try {
            val matSeg = if (active && invViewProj != null && invViewProj.size >= 16) {
                MemorySegment.copy(invViewProj, 0, dhInvViewProjSegment, ValueLayout.JAVA_FLOAT, 0L, 16)
                dhInvViewProjSegment
            } else {
                MemorySegment.NULL
            }
            fn.invokeExact(
                colorImg,
                colorView,
                normalImg,
                normalView,
                materialImg,
                materialView,
                depthImg,
                depthView,
                width,
                height,
                matSeg,
                fadeStart,
                fadeEnd,
                dhFarClip,
                if (active) 1 else 0,
            ) as Int
        } catch (t: Throwable) {
            logNativeCallError("bindDhTargets", t)
            -1
        }
    }

    @JvmStatic
    @JvmOverloads
    fun preprocessRasterShader(source: String, packDir: Path?, stageIsFragment: Boolean = false): String? {
        val fn = fnPreprocessRasterShader ?: return null
        if (source.isEmpty()) return source
        val stageFlag = if (stageIsFragment) 1 else 0
        return runCatching {
            val srcBytes = source.toByteArray(StandardCharsets.UTF_8)
            val dirBytes = packDir?.toAbsolutePath()?.toString()?.toByteArray(StandardCharsets.UTF_8) ?: ByteArray(0)
            val scratch = ioScratch.get()
            val srcSeg = scratch.primary.ensureCapacity(srcBytes.size.toLong())
            MemorySegment.copy(srcBytes, 0, srcSeg, ValueLayout.JAVA_BYTE, 0L, srcBytes.size)
            val dirSeg = if (dirBytes.isNotEmpty()) {
                val seg = scratch.secondary.ensureCapacity(dirBytes.size.toLong())
                MemorySegment.copy(dirBytes, 0, seg, ValueLayout.JAVA_BYTE, 0L, dirBytes.size)
                seg
            } else {
                MemorySegment.NULL
            }
            val needed = fn.invokeExact(
                srcSeg.address(),
                srcBytes.size,
                dirSeg.address(),
                dirBytes.size,
                stageFlag,
                0L,
                0,
            ) as Int
            if (needed <= 0) return@runCatching null
            val outSeg = scratch.output.ensureCapacity(needed.toLong())
            val written = fn.invokeExact(
                srcSeg.address(),
                srcBytes.size,
                dirSeg.address(),
                dirBytes.size,
                stageFlag,
                outSeg.address(),
                needed,
            ) as Int
            val clamped = written.coerceIn(0, needed)
            if (clamped <= 0) return@runCatching null
            val outBytes = ByteArray(clamped)
            MemorySegment.copy(outSeg, ValueLayout.JAVA_BYTE, 0L, outBytes, 0, clamped)
            String(outBytes, StandardCharsets.UTF_8)
        }.getOrNull()
    }

    fun shutdown() {
        if (!initialized) return
        val fn = fnShutdown
        if (fn != null) {
            try {
                fn.invokeExact()
            } catch (t: Throwable) {
                logNativeCallError("shutdown", t)
            }
        }
        initialized = false
        cachedGpuTierClass = null
        deviceLostNotifiedForIncident = false
        HyaloclastDebugHud.clearNativeMetrics()
        if (StandaloneRenderPearlBridge.vulkanTurboLoaded) {
            runCatching {
                dev.vulkanturbo.api.shader.TurboShadersBridge.updateCachedCustomDescriptorSets(0L, 0L)
            }
        }
    }

    @JvmStatic
    fun clearSectionSlices() {
        if (!initialized) return
        val fn = fnClearSectionSlices ?: return
        runCatching { fn.invokeExact(); Unit }.onFailure { logNativeCallError("clearSectionSlices", it) }
    }

    @JvmStatic
    fun recordSubmission(timelineValue: Long) {
        if (!initialized || timelineValue <= 0L) return
        val fn = fnRecordSubmission ?: return
        runCatching { fn.invokeExact(timelineValue); Unit }.onFailure { logNativeCallError("recordSubmission", it) }
    }

    @JvmStatic
    fun notifyTimelineCompleted(completedValue: Long) {
        if (!initialized || completedValue <= 0L) return
        val fn = fnNotifyTimelineCompleted ?: return
        runCatching { fn.invokeExact(completedValue); Unit }.onFailure { logNativeCallError("notifyTimelineCompleted", it) }
    }

    @JvmStatic
    fun onWorldUnloaded() {
        if (!initialized) return
        val fn = fnOnWorldUnloaded ?: return
        runCatching { fn.invokeExact(); Unit }.onFailure { logNativeCallError("onWorldUnloaded", it) }
    }

    @JvmStatic
    fun setRenderScale(scale: Float) {
        val fn = fnSetRenderScale ?: return
        runCatching { fn.invokeExact(scale.coerceIn(0.33f, 1.0f)); Unit }.onFailure { logNativeCallError("setRenderScale", it) }
    }

    @JvmStatic
    fun setHdrParams(peakBrightnessNits: Float, paperWhiteNits: Float) {
        val fn = fnSetHdrParams ?: return
        runCatching { fn.invokeExact(peakBrightnessNits, paperWhiteNits); Unit }.onFailure { logNativeCallError("setHdrParams", it) }
    }

    @JvmStatic
    fun onSwapchainCreated(imageFormat: Int, colorSpace: Int, width: Int, height: Int) {
        val fn = fnOnSwapchainCreated ?: return
        runCatching { fn.invokeExact(imageFormat, colorSpace, width, height); Unit }.onFailure { logNativeCallError("onSwapchainCreated", it) }
    }

    @JvmStatic
    fun setGameDir(path: String) {
        val fn = fnSetGameDir ?: return
        val bytes = path.toByteArray(Charsets.UTF_8)
        if (bytes.isEmpty()) return
        runCatching {
            val seg = ioScratch.get().primary.ensureCapacity(bytes.size.toLong())
            MemorySegment.copy(bytes, 0, seg, ValueLayout.JAVA_BYTE, 0L, bytes.size)
            fn.invokeExact(seg, bytes.size)
            Unit
        }.onFailure { logNativeCallError("setGameDir", it) }
    }

    @JvmStatic
    fun markHistoryReset() {
        if (!initialized) return
        val fn = fnMarkHistoryReset ?: return
        runCatching { fn.invokeExact(); Unit }.onFailure { logNativeCallError("markHistoryReset", it) }
    }

    @JvmStatic
    fun drainNativeErrorsToLog() {
        if (!nativeAvailable) return
        val drainFn = fnDrainNativeErrors ?: return
        val countFn = fnGetNativeErrorCount
        if (countFn != null) {
            val pending = runCatching { countFn.invokeExact() as Int }.getOrDefault(0)
            if (pending <= 0) return
        }
        val drainedText = runCatching {
            val needed = drainFn.invokeExact(0L, 0) as Int
            if (needed <= 0) return@runCatching null
            val capacity = needed + 1024
            val outSeg = ioScratch.get().output.ensureCapacity(capacity.toLong())
            val written = (drainFn.invokeExact(outSeg.address(), capacity) as Int).coerceIn(0, capacity)
            if (written <= 0) return@runCatching null
            val bytes = ByteArray(written)
            MemorySegment.copy(outSeg, ValueLayout.JAVA_BYTE, 0L, bytes, 0, written)
            String(bytes, StandardCharsets.UTF_8)
        }.getOrElse { t ->
            logNativeCallError("drainNativeErrorsToLog", t)
            null
        } ?: return

        val lines = drainedText.lineSequence()
            .map { it.trim() }
            .filter { it.isNotEmpty() }
            .toList()
        if (lines.isEmpty()) return

        for (line in lines) {
            val parts = line.split('|', limit = 3)
            if (parts.size == 3) {
                val level = parts[0].trim().uppercase()
                val source = parts[1].trim().ifEmpty { "NativeCore" }
                val msg = parts[2].trim()
                if (msg.isEmpty()) continue
                when (level) {
                    "WARN", "WARNING" -> HyaloclastErrorLog.recordWarning(
                        source = source,
                        summary = msg,
                    )
                    else -> HyaloclastErrorLog.recordError(
                        source = source,
                        summary = msg,
                    )
                }
            } else {
                HyaloclastErrorLog.recordError(
                    source = "NativeCore",
                    summary = line,
                )
            }
        }
    }

    /**
     * Queries native VRAM budget, RT BLAS/TLAS counts, and (when GPU profiler is enabled)
     * per-pass GPU timings using pre-allocated persistent scratch segments, then pushes them
     * into [HyaloclastDebugHud.updateNativeMetrics].
     */
    @JvmStatic
    fun pollTelemetryToDebugHud() {
        if (!initialized) return
        val vramFn = fnGetVramBudget
        val rtFn = fnGetRtStats
        val timingsFn = fnGetGpuPassTimings
        if (vramFn == null && rtFn == null && timingsFn == null) return

        synchronized(telemetryLock) {
            var vramUsedMb = 0L
            var vramBudgetMb = 0L
            var hasMetrics = false

            if (vramFn != null) {
                runCatching {
                    vramUsedSeg.set(ValueLayout.JAVA_LONG, 0L, 0L)
                    vramBudgetSeg.set(ValueLayout.JAVA_LONG, 0L, 0L)
                    vramPressureSeg.set(ValueLayout.JAVA_INT, 0L, 0)
                    val rc = vramFn.invokeExact(
                        vramUsedSeg.address(),
                        vramBudgetSeg.address(),
                        vramPressureSeg.address(),
                    ) as Int
                    if (rc >= 0) {
                        val usedBytes = vramUsedSeg.get(ValueLayout.JAVA_LONG, 0L).coerceAtLeast(0L)
                        val budgetBytes = vramBudgetSeg.get(ValueLayout.JAVA_LONG, 0L).coerceAtLeast(0L)
                        vramUsedMb = usedBytes / (1024L * 1024L)
                        vramBudgetMb = budgetBytes / (1024L * 1024L)
                        hasMetrics = true
                    }
                }.onFailure { logNativeCallError("pollTelemetryToDebugHud.vram", it) }
            }

            var blasCount = 0
            var tlasInstances = 0
            if (rtFn != null) {
                runCatching {
                    rtBlasCountSeg.set(ValueLayout.JAVA_INT, 0L, 0)
                    rtTlasInstancesSeg.set(ValueLayout.JAVA_INT, 0L, 0)
                    val rc = rtFn.invokeExact(
                        rtBlasCountSeg.address(),
                        rtTlasInstancesSeg.address(),
                    ) as Int
                    if (rc >= 0) {
                        blasCount = rtBlasCountSeg.get(ValueLayout.JAVA_INT, 0L).coerceAtLeast(0)
                        tlasInstances = rtTlasInstancesSeg.get(ValueLayout.JAVA_INT, 0L).coerceAtLeast(0)
                        hasMetrics = true
                    }
                }.onFailure { logNativeCallError("pollTelemetryToDebugHud.rtStats", it) }
            }

            var passTimings: List<HyaloclastDebugHud.PassTiming> = emptyList()
            if (timingsFn != null && VulkanShaders.config.enableGpuProfiler) {
                runCatching {
                    var seg = gpuTimingsScratch.ensureCapacity(4096L)
                    var maxLen = seg.byteSize().toInt()
                    var written = timingsFn.invokeExact(seg.address(), maxLen) as Int
                    if (written > maxLen) {
                        seg = gpuTimingsScratch.ensureCapacity(written.toLong() + 256L)
                        maxLen = seg.byteSize().toInt()
                        written = timingsFn.invokeExact(seg.address(), maxLen) as Int
                    }
                    if (written > 0) {
                        val clamped = written.coerceAtMost(maxLen)
                        val bytes = ByteArray(clamped)
                        MemorySegment.copy(seg, ValueLayout.JAVA_BYTE, 0L, bytes, 0, clamped)
                        val raw = String(bytes, StandardCharsets.UTF_8).trimEnd('\u0000')
                        passTimings = parseGpuPassTimings(raw)
                        hasMetrics = true
                    }
                }.onFailure { logNativeCallError("pollTelemetryToDebugHud.gpuTimings", it) }
            }

            if (hasMetrics) {
                HyaloclastDebugHud.updateNativeMetrics(
                    blasCount = blasCount,
                    tlasInstanceCount = tlasInstances,
                    vramUsedMb = vramUsedMb,
                    vramBudgetMb = vramBudgetMb,
                    passTimings = passTimings,
                )
            }
        }
    }

    internal fun parseGpuPassTimings(raw: String): List<HyaloclastDebugHud.PassTiming> {
        if (raw.isBlank()) return emptyList()
        val result = ArrayList<HyaloclastDebugHud.PassTiming>()
        for (entry in raw.split(';')) {
            val trimmed = entry.trim()
            if (trimmed.isEmpty()) continue
            val eqIdx = trimmed.indexOf('=')
            if (eqIdx <= 0 || eqIdx >= trimmed.lastIndex) continue
            val name = trimmed.substring(0, eqIdx).trim()
            val ms = trimmed.substring(eqIdx + 1).trim().toDoubleOrNull()
            if (name.isNotEmpty() && ms != null && ms.isFinite() && ms >= 0.0) {
                result.add(HyaloclastDebugHud.PassTiming(name = name, durationMs = ms))
            }
        }
        return result
    }

    /**
     * Checks `vs_core_get_recovery_state` for native Vulkan device-lost / TDR fallback.
     * Records the incident via [HyaloclastErrorLog.recordDeviceLostRecovery] and displays
     * [ShaderNotices.showDeviceLostNotice] once per incident.
     */
    @JvmStatic
    fun checkDeviceLostRecovery(): Boolean {
        if (!initialized) return false
        val fn = fnGetRecoveryState ?: return false
        return synchronized(telemetryLock) {
            runCatching {
                recoveryDisabledSeg.set(ValueLayout.JAVA_INT, 0L, 0)
                val noticeSeg = recoveryNoticeScratch.ensureCapacity(1024L)
                val maxLen = noticeSeg.byteSize().toInt()
                val written = fn.invokeExact(
                    recoveryDisabledSeg.address(),
                    noticeSeg.address(),
                    maxLen,
                ) as Int
                val disabled = recoveryDisabledSeg.get(ValueLayout.JAVA_INT, 0L) != 0
                if (disabled) {
                    if (!deviceLostNotifiedForIncident) {
                        deviceLostNotifiedForIncident = true
                        val notice = if (written > 0) {
                            val clamped = written.coerceAtMost(maxLen)
                            val bytes = ByteArray(clamped)
                            MemorySegment.copy(noticeSeg, ValueLayout.JAVA_BYTE, 0L, bytes, 0, clamped)
                            String(bytes, StandardCharsets.UTF_8).trimEnd('\u0000').trim()
                        } else {
                            ""
                        }.ifEmpty { "VK_ERROR_DEVICE_LOST" }
                        HyaloclastErrorLog.recordDeviceLostRecovery(
                            reason = notice,
                            details = "Native Vulkan backend entered safe recovery state.",
                            showToast = false,
                        )
                        ShaderNotices.showDeviceLostNotice(notice)
                    }
                } else {
                    deviceLostNotifiedForIncident = false
                }
                disabled
            }.getOrElse { t ->
                logNativeCallError("checkDeviceLostRecovery", t)
                false
            }
        }
    }

    @JvmStatic
    fun resetRecoveryState(): Int {
        deviceLostNotifiedForIncident = false
        if (!initialized) return 0
        val fn = fnResetRecoveryState ?: return 0
        return runCatching { fn.invokeExact() as Int }
            .getOrElse { t -> logNativeCallError("resetRecoveryState", t); -1 }
    }

    @JvmStatic
    fun setGpuProfilerEnabled(enabled: Boolean): Int {
        if (!initialized) return 0
        val fn = fnSetGpuProfilerEnabled ?: return 0
        return runCatching { fn.invokeExact(if (enabled) 1 else 0) as Int }
            .getOrElse { t -> logNativeCallError("setGpuProfilerEnabled", t); -1 }
    }

    @JvmStatic
    fun startPackPrewarm(packPath: Path): Int {
        if (!initialized) return 0
        val fn = fnStartPackPrewarm ?: return 0
        val utf8 = packPath.toAbsolutePath().toString().toByteArray(StandardCharsets.UTF_8)
        if (utf8.isEmpty()) return 0
        return runCatching {
            val seg = ioScratch.get().primary.ensureCapacity(utf8.size.toLong())
            MemorySegment.copy(utf8, 0, seg, ValueLayout.JAVA_BYTE, 0L, utf8.size)
            fn.invokeExact(seg.address(), utf8.size) as Int
        }.getOrElse { t -> logNativeCallError("startPackPrewarm", t); -1 }
    }

    @JvmStatic
    fun getPrewarmProgress(): PrewarmProgress {
        if (!initialized) return PrewarmProgress()
        val fn = fnGetPrewarmProgress ?: return PrewarmProgress()
        return synchronized(telemetryLock) {
            runCatching {
                prewarmCompletedSeg.set(ValueLayout.JAVA_INT, 0L, 0)
                prewarmTotalSeg.set(ValueLayout.JAVA_INT, 0L, 0)
                prewarmReadySeg.set(ValueLayout.JAVA_INT, 0L, 1)
                val rc = fn.invokeExact(
                    prewarmCompletedSeg.address(),
                    prewarmTotalSeg.address(),
                    prewarmReadySeg.address(),
                ) as Int
                if (rc >= 0) {
                    PrewarmProgress(
                        completed = prewarmCompletedSeg.get(ValueLayout.JAVA_INT, 0L).coerceAtLeast(0),
                        total = prewarmTotalSeg.get(ValueLayout.JAVA_INT, 0L).coerceAtLeast(0),
                        ready = prewarmReadySeg.get(ValueLayout.JAVA_INT, 0L) != 0,
                    )
                } else {
                    PrewarmProgress()
                }
            }.getOrElse { t ->
                logNativeCallError("getPrewarmProgress", t)
                PrewarmProgress()
            }
        }
    }

    @JvmStatic
    fun setLodEnabled(enabled: Boolean): Int {
        if (!initialized) return 0
        val fn = fnSetLodEnabled ?: return 0
        return runCatching { fn.invokeExact(if (enabled) 1 else 0) as Int }
            .getOrElse { t -> logNativeCallError("setLodEnabled", t); -1 }
    }

    @JvmStatic
    fun setDimension(dimensionId: Int): Int {
        if (!initialized) return 0
        val fn = fnSetDimension ?: return 0
        return runCatching { fn.invokeExact(dimensionId) as Int }
            .getOrElse { t -> logNativeCallError("setDimension", t); -1 }
    }

    @JvmStatic
    fun attachCutoutOmm(
        allocationKeyId: Long,
        layerOrdinal: Int,
        triangleCount: Int,
        subdivLevel: Int,
        use4State: Boolean,
        encodedData: ByteArray,
    ): Boolean {
        if (!initialized || triangleCount <= 0 || encodedData.isEmpty()) return false
        val fn = fnAttachCutoutOmm ?: return false
        return runCatching {
            val seg = ommDataScratch.get().ensureCapacity(encodedData.size.toLong())
            MemorySegment.copy(encodedData, 0, seg, ValueLayout.JAVA_BYTE, 0L, encodedData.size)
            val rc = fn.invokeExact(
                allocationKeyId,
                layerOrdinal,
                triangleCount,
                subdivLevel,
                if (use4State) 1 else 0,
                seg.address(),
                encodedData.size,
            ) as Int
            rc >= 0
        }.getOrElse { t ->
            logNativeCallError("attachCutoutOmm", t)
            false
        }
    }

    @JvmStatic
    fun submitEntityVerticesCategorized(
        vertexBytes: ByteArray,
        vertexCount: Int,
        vertexStride: Int,
        isQuads: Boolean,
        textureId: Int,
        normalTexId: Int,
        specularTexId: Int,
        kind: Int,
        category: Int,
        entityFlags: Int,
    ) {
        if (!initialized || vertexBytes.isEmpty() || vertexCount <= 0 || vertexStride <= 0) return
        if (vertexBytes.size.toLong() < vertexCount.toLong() * vertexStride.toLong()) return
        val categorizedFn = fnSubmitEntityVerticesCategorized
        val legacyFn = if (categorizedFn == null) fnSubmitEntityVertices else null
        if (categorizedFn == null && (legacyFn == null || textureId < 0)) return
        try {
            val seg = entityVertexScratch.get().ensureCapacity(vertexBytes.size.toLong())
            MemorySegment.copy(vertexBytes, 0, seg, ValueLayout.JAVA_BYTE, 0L, vertexBytes.size)
            if (categorizedFn != null) {
                categorizedFn.invokeExact(
                    seg.address(),
                    vertexCount,
                    vertexStride,
                    if (isQuads) 1 else 0,
                    textureId,
                    normalTexId,
                    specularTexId,
                    kind,
                    category,
                    entityFlags,
                )
            } else if (legacyFn != null) {
                @Suppress("UNUSED_VARIABLE")
                val ignored = legacyFn.invokeExact(
                    seg.address(),
                    vertexCount,
                    vertexStride,
                    textureId,
                    kind,
                ) as Int
            }
        } catch (t: Throwable) {
            logNativeCallError("submitEntityVerticesCategorized", t)
        }
    }

    @JvmStatic
    fun onDisplayResized() {
        if (!initialized) return
        val fn = fnOnDisplayResized ?: return
        try {
            @Suppress("UNUSED_VARIABLE")
            val ignored = fn.invokeExact() as Int
        } catch (t: Throwable) {
            logNativeCallError("onDisplayResized", t)
        }
    }

    @JvmStatic
    fun onResourcePacksReloaded() {
        if (!initialized) return
        val fn = fnOnResourcesReloaded ?: return
        try {
            @Suppress("UNUSED_VARIABLE")
            val ignored = fn.invokeExact() as Int
        } catch (t: Throwable) {
            logNativeCallError("onResourcePacksReloaded", t)
        }
    }
}
