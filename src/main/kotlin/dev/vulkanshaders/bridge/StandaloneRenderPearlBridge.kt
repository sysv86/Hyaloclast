package dev.vulkanshaders.bridge

import com.mojang.blaze3d.systems.RenderSystem
import dev.vulkanshaders.VulkanShaders
import dev.vulkanshaders.pipeline.GbufferTargetManager
import dev.vulkanshaders.pipeline.SectionSliceRegistry
import dev.vulkanshaders.texture.TextureRegistry
import net.fabricmc.loader.api.FabricLoader
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Fabric mod-presence helper and standalone RenderPearl runtime guard for `VulkanTurbo`
 * (`Pyroclast / TEPHRA`) integration.
 *
 * When `vulkanturbo` (Pyroclast) is loaded, Hyaloclast shares its `TurboShadersBridge` hooks for
 * 40-byte `IGNIS_TERRAIN_40B` vertices, 4-MRT G-Buffer attachment overrides, and `UberGpuBuffer`
 * zero-copy BLAS/TLAS slices.
 *
 * When Pyroclast is absent (standalone Fabric mode or unit tests), this bridge keeps classloading
 * safe without throwing `NoClassDefFoundError` and ensures `TextureAtlasPbrMixin`
 * ([TextureRegistry]), [SectionSliceRegistry], and [GbufferTargetManager] initialize cleanly.
 */
object StandaloneRenderPearlBridge {
    private val standaloneInitialized = AtomicBoolean(false)

    @JvmStatic
    fun isModLoaded(modId: String): Boolean =
        runCatching { FabricLoader.getInstance().isModLoaded(modId) }
            .getOrDefault(false)

    @JvmField
    val vulkanTurboLoaded: Boolean = runCatching {
        isModLoaded("vulkanturbo") &&
            Class.forName("dev.vulkanturbo.VulkanTurbo") != null
    }.getOrDefault(false)

    /**
     * Returns `true` when Hyaloclast is running on Fabric without Pyroclast (`VulkanTurbo`) loaded.
     */
    @JvmStatic
    fun isStandaloneMode(): Boolean = !vulkanTurboLoaded

    @JvmStatic
    fun isBackendAllowed(): Boolean {
        if (vulkanTurboLoaded) {
            return runCatching { dev.vulkanturbo.VulkanTurbo.isBackendAllowed() }.getOrDefault(true)
        }
        return true
    }

    @JvmStatic
    fun isVulkanBackendActive(): Boolean {
        if (vulkanTurboLoaded) {
            return runCatching {
                !dev.vulkanturbo.VulkanTurbo.backendDetected || dev.vulkanturbo.VulkanTurbo.isVulkanBackend
            }.getOrDefault(true)
        }
        return runCatching {
            val device = RenderSystem.getDevice()
            val name = device?.getDeviceInfo()?.backendName()
            name != null && name.equals("Vulkan", ignoreCase = true)
        }.getOrDefault(false)
    }

    /**
     * Ensures standalone-mode subsystems (`TextureAtlasPbrMixin` / [TextureRegistry],
     * [SectionSliceRegistry], and [GbufferTargetManager]) are initialized safely without
     * null or reflection hazards when Pyroclast is absent.
     */
    @JvmStatic
    fun ensureStandaloneInitialized(): Boolean {
        if (vulkanTurboLoaded) return true
        if (!standaloneInitialized.compareAndSet(false, true)) return true
        return runCatching {
            // Touch singletons cleanly so TextureAtlasPbrMixin, SectionSliceRegistry, and
            // GbufferTargetManager are primed identically in standalone mode.
            TextureRegistry.setSkipScratchUploads(false)
            SectionSliceRegistry.activeCount()
            GbufferTargetManager.closeDhTargets()
            VulkanShaders.log.info(
                "Initialized standalone RenderPearl bridge (TextureRegistry, SectionSliceRegistry, GbufferTargetManager).",
            )
            true
        }.getOrElse { t ->
            standaloneInitialized.set(false)
            VulkanShaders.log.warn("Standalone RenderPearl bridge initialization failed: {}", t.message)
            false
        }
    }

    /**
     * Resets standalone-mode transient GPU state on world unload without requiring Pyroclast.
     */
    @JvmStatic
    fun resetStandaloneWorldState() {
        if (vulkanTurboLoaded) return
        runCatching { SectionSliceRegistry.clear() }
        runCatching { GbufferTargetManager.closeAll() }
    }

    @JvmStatic
    fun refreshBridgeState() {
        if (vulkanTurboLoaded) {
            runCatching { dev.vulkanturbo.api.shader.TurboShadersBridge.refreshState() }
                .onFailure { t -> VulkanShaders.log.debug("TurboShadersBridge refresh skipped: {}", t.message) }
        } else {
            ensureStandaloneInitialized()
        }
    }
}
