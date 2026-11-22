package dev.vulkanshaders.backend

import com.mojang.renderpearl.api.device.GpuDevice
import com.mojang.renderpearl.backend.vulkan.VulkanDevice
import dev.vulkanshaders.VulkanShaders
import dev.vulkanshaders.bridge.StandaloneRenderPearlBridge
import dev.vulkanshaders.mixin.backend.FrontendGpuDeviceAccessor
import dev.vulkanshaders.mixin.backend.VulkanCommandEncoderAccessor

/**
 * Extracts and holds raw Vulkan handles, queue family indices, timeline semaphore handles,
 * and hardware RT/BDA capability flags from RenderPearl's `VulkanDevice`.
 */
object VulkanDeviceContext {
    @Volatile
    var vulkanDevice: VulkanDevice? = null
        private set

    @JvmField @Volatile var vkInstance: Long = 0L
    @JvmField @Volatile var vkPhysicalDevice: Long = 0L
    @JvmField @Volatile var vkDevice: Long = 0L
    @JvmField @Volatile var vmaAllocator: Long = 0L
    @JvmField @Volatile var graphicsQueue: Long = 0L
    @JvmField @Volatile var computeQueue: Long = 0L
    @JvmField @Volatile var graphicsQueueFamilyIndex: Int = 0
    @JvmField @Volatile var computeQueueFamilyIndex: Int = 0
    @JvmField @Volatile var encoderTimelineSemaphore: Long = 0L
    @JvmField @Volatile var deviceName: String = ""
    @JvmField @Volatile var enabledExtensions: Set<String> = emptySet()

    @JvmField @Volatile var supportsAccelerationStructure: Boolean = false
    @JvmField @Volatile var supportsRayQuery: Boolean = false
    @JvmField @Volatile var supportsRayTracingPipeline: Boolean = false
    @JvmField @Volatile var supportsDescriptorIndexing: Boolean = false
    @JvmField @Volatile var supportsLowLatency2Nv: Boolean = false

    @JvmStatic
    fun initialize(device: GpuDevice?) {
        if (device == null) return
        val backend = (device as? FrontendGpuDeviceAccessor)?.`vulkanshaders$getBackend`()
        val vkDev = backend as? VulkanDevice
        if (vkDev == null) {
            vulkanDevice = null
            vkDevice = 0L
            return
        }

        vulkanDevice = vkDev
        vkInstance = vkDev.instance().vkInstance().address()
        vkPhysicalDevice = vkDev.vkDevice().physicalDevice.address()
        vkDevice = vkDev.vkDevice().address()
        vmaAllocator = vkDev.vma()
        graphicsQueue = vkDev.graphicsQueue().vkQueue().address()
        computeQueue = vkDev.computeQueue().vkQueue().address()
        graphicsQueueFamilyIndex = vkDev.graphicsQueue().queueFamilyIndex()
        computeQueueFamilyIndex = vkDev.computeQueue().queueFamilyIndex()
        encoderTimelineSemaphore =
            (vkDev.createCommandEncoder() as? VulkanCommandEncoderAccessor)?.`vulkanshaders$getSubmitSemaphore`() ?: 0L
        deviceName = vkDev.getDeviceInfo().name()
        val exts = vkDev.getDeviceInfo().underlyingExtensions()
        enabledExtensions = exts

        val capsMask = NativeCoreBridge.getCapabilitiesMask(vkDevice)
        supportsAccelerationStructure =
            ((capsMask and NativeCoreBridge.CAP_RAY_TRACING_AS) != 0) || exts.contains("VK_KHR_acceleration_structure")
        supportsRayQuery =
            ((capsMask and NativeCoreBridge.CAP_RAY_QUERY) != 0) || exts.contains("VK_KHR_ray_query")
        supportsRayTracingPipeline =
            ((capsMask and NativeCoreBridge.CAP_RAY_TRACING_PIPELINE) != 0) || exts.contains("VK_KHR_ray_tracing_pipeline")
        supportsDescriptorIndexing =
            ((capsMask and NativeCoreBridge.CAP_BINDLESS) != 0) || exts.contains("VK_EXT_descriptor_indexing")
        supportsLowLatency2Nv =
            ((capsMask and NativeCoreBridge.CAP_REFLEX) != 0) || exts.contains("VK_NV_low_latency2")
        if (StandaloneRenderPearlBridge.vulkanTurboLoaded) {
            runCatching {
                val bda = (capsMask and NativeCoreBridge.CAP_BUFFER_DEVICE_ADDRESS) != 0
                dev.vulkanturbo.api.shader.TurboShadersBridge.bdaSupported = bda
                dev.vulkanturbo.api.shader.TurboShadersBridge.bdaCapabilityResolvedByExtension = true
                dev.vulkanturbo.api.shader.TurboShadersBridge.accelStructSupported = supportsAccelerationStructure
                dev.vulkanturbo.api.shader.TurboShadersBridge.rayTracingPipelineSupported = supportsRayTracingPipeline
            }
        }

        VulkanShaders.log.info(
            "VulkanDeviceContext bound: '{}' (vkDevice=0x{}, gfxFamily={}, compFamily={}, timelineSem=0x{}, AS={}, RQ={}, RTPipe={}, Bindless={}, Reflex={})",
            deviceName,
            java.lang.Long.toHexString(vkDevice),
            graphicsQueueFamilyIndex,
            computeQueueFamilyIndex,
            java.lang.Long.toHexString(encoderTimelineSemaphore),
            supportsAccelerationStructure,
            supportsRayQuery,
            supportsRayTracingPipeline,
            supportsDescriptorIndexing,
            supportsLowLatency2Nv,
        )

        NativeCoreBridge.tryInitialize(
            vkInstance = vkInstance,
            vkPhysicalDevice = vkPhysicalDevice,
            vkDevice = vkDevice,
            graphicsQueue = graphicsQueue,
            computeQueue = computeQueue,
            vmaAllocator = vmaAllocator,
            graphicsQueueFamilyIndex = graphicsQueueFamilyIndex,
            computeQueueFamilyIndex = computeQueueFamilyIndex,
            timelineSemaphore = encoderTimelineSemaphore,
        )
        dev.vulkanshaders.bridge.VulkanShadersTurboExtension.syncCapabilitiesToBridge()
    }

    @JvmStatic
    fun shutdown() {
        runCatching { dev.vulkanshaders.pipeline.GbufferTargetManager.closeAll() }
        runCatching { dev.vulkanshaders.texture.TextureRegistry.closeCompanionAtlases() }
        dev.vulkanshaders.bridge.VulkanShadersTurboExtension.onDeviceShutdown()
        NativeCoreBridge.shutdown()
        vulkanDevice = null
        vkInstance = 0L
        vkPhysicalDevice = 0L
        vkDevice = 0L
        vmaAllocator = 0L
        graphicsQueue = 0L
        computeQueue = 0L
        encoderTimelineSemaphore = 0L
    }

    @JvmStatic
    fun isReady(): Boolean = vkDevice != 0L
}
