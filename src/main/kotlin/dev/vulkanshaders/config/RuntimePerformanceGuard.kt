package dev.vulkanshaders.config

import dev.vulkanshaders.VulkanShaders
import dev.vulkanshaders.backend.NativeCoreBridge
import dev.vulkanshaders.diagnostics.HyaloclastErrorLog
import dev.vulkanshaders.gui.ShaderNotices
import dev.vulkanshaders.pack.ShaderPackManager
import net.minecraft.client.Minecraft
import net.minecraft.network.chat.Component

/**
 * Monitors frame/tick cadence and GPU frame times during live rendering, stepping down
 * [LightingMode] automatically when sustained overload or a multi-second render stall occurs.
 */
object RuntimePerformanceGuard {
    private const val WARMUP_TICKS = 20
    private const val CONSECUTIVE_OVERLOAD_TICKS = 6
    private const val OVERLOAD_TICK_MS = 120.0
    private const val OVERLOAD_GPU_MS = 90.0
    private const val CRITICAL_STALL_MS = 1400.0
    private const val COOLDOWN_MS = 10_000L

    private var warmupRemaining = WARMUP_TICKS
    private var consecutiveOverloadTicks = 0
    private var lastTickNanos = 0L
    private var lastDowngradeAtMs = 0L

    fun resetForStateChange() {
        warmupRemaining = WARMUP_TICKS
        consecutiveOverloadTicks = 0
        lastTickNanos = 0L
    }

    /**
     * Returns the next lower [LightingMode] supported by [supportedTiers], or `null` when already
     * at the lowest supported tier.
     */
    @JvmStatic
    fun nextLowerTier(current: LightingMode, supportedTiers: List<Int>): LightingMode? {
        val candidates = LightingMode.entries
            .filter { it.tierIndex < current.tierIndex }
            .sortedByDescending { it.tierIndex }
        if (candidates.isEmpty()) return null
        if (supportedTiers.isEmpty()) return candidates.first()
        return candidates.firstOrNull { it.tierIndex in supportedTiers } ?: candidates.first()
    }

    fun onClientTick(mc: Minecraft) {
        val cfg = VulkanShaders.config
        val pack = ShaderPackManager.activePack
        if (!VulkanShaders.shaderPackActive || !cfg.enabled || pack == null) {
            resetForStateChange()
            return
        }
        if (mc.level == null || mc.isPaused) {
            lastTickNanos = 0L
            consecutiveOverloadTicks = 0
            return
        }

        val nowNanos = System.nanoTime()
        if (lastTickNanos == 0L) {
            lastTickNanos = nowNanos
            return
        }
        val deltaMs = (nowNanos - lastTickNanos) / 1_000_000.0
        lastTickNanos = nowNanos

        if (warmupRemaining > 0) {
            warmupRemaining--
            return
        }

        val nowMs = System.currentTimeMillis()
        if (nowMs - lastDowngradeAtMs < COOLDOWN_MS) {
            return
        }

        val avgGpuMs = NativeCoreBridge.getAverageGpuFrameTimeMs()
        // VRAM pressure is a leading indicator: by the time the render thread blocks in submit,
        // the tick-driven stall check is already too late to prevent the 5 s semaphore timeout.
        val vramPressure = NativeCoreBridge.getVramBudgetSnapshot()?.pressure ?: 0
        val overloaded = deltaMs >= OVERLOAD_TICK_MS ||
            (avgGpuMs != null && avgGpuMs >= OVERLOAD_GPU_MS) || vramPressure >= 1
        val criticalStall = deltaMs >= CRITICAL_STALL_MS || vramPressure >= 2

        if (overloaded) {
            consecutiveOverloadTicks++
        } else {
            consecutiveOverloadTicks = 0
        }

        if (!criticalStall && consecutiveOverloadTicks < CONSECUTIVE_OVERLOAD_TICKS) {
            return
        }

        consecutiveOverloadTicks = 0
        lastDowngradeAtMs = nowMs

        val reason = if (vramPressure > 0) {
            "VRAM pressure $vramPressure"
        } else {
            "${"%.0f".format(avgGpuMs?.takeIf { it >= OVERLOAD_GPU_MS } ?: deltaMs)} ms/frame"
        }
        if (cfg.renderTier == LightingMode.PATH) {
            // Full-resolution path tracing is the dominant cost on a 6 GB GPU. Reducing the
            // internal render resolution is reversible in the GUI and avoids a hard GPU stall
            // before touching bounce count or lighting mode.
            if (cfg.upscalerMode == UpscalerMode.NONE) {
                VulkanShaders.updateConfig(
                    cfg.copy(upscalerMode = UpscalerMode.TSR, upscalerQuality = UpscalerQuality.BALANCED),
                )
                val msg = "$reason — enabled TSR Balanced (58% render scale) for path tracing."
                HyaloclastErrorLog.recordError("PerfGuard", msg)
                notifyPlayer(mc, msg)
                return
            }
            val activeOverrides = cfg.activePackOverrides(pack.manifest.id, cfg.renderTier)
            val currentBounces = activeOverrides["pt_max_bounces"]?.toIntOrNull()
                ?: pack.manifest.settings.firstOrNull { it.key == "pt_max_bounces" }?.defaultValue?.toIntOrNull()
                ?: 2
            if (currentBounces > 2) {
                val updatedMap = activeOverrides.toMutableMap()
                updatedMap["pt_max_bounces"] = "2"
                val tierMap = cfg.packOverridesByTier[pack.manifest.id]?.toMutableMap() ?: mutableMapOf()
                tierMap[cfg.renderTier.tierIndex] = updatedMap
                val byTier = cfg.packOverridesByTier.toMutableMap()
                byTier[pack.manifest.id] = tierMap
                VulkanShaders.updateConfig(cfg.copy(packOverridesByTier = byTier))
                val msg = "$reason — lowered path-tracing bounces from $currentBounces to 2."
                HyaloclastErrorLog.recordError("PerfGuard", msg)
                notifyPlayer(mc, msg)
                return
            }
        }
        val lowerTier = nextLowerTier(cfg.renderTier, pack.manifest.supportedTiers)
        if (lowerTier != null) {
            VulkanShaders.updateConfig(cfg.copy(renderTier = lowerTier))
            val msg = "$reason — lowered lighting mode to ${lowerTier.shortName}."
            HyaloclastErrorLog.recordError("PerfGuard", msg)
            notifyPlayer(mc, msg)
        } else if (criticalStall) {
            VulkanShaders.updateConfig(cfg.copy(enabled = false))
            val msg = "$reason at Classic mode — shaders disabled."
            HyaloclastErrorLog.recordError("PerfGuard", msg)
            notifyPlayer(mc, msg)
        }
    }

    private fun notifyPlayer(mc: Minecraft, message: String) {
        ShaderNotices.showStatusNotice(
            Component.literal("Hyaloclast Auto-Quality"),
            Component.literal(message),
        )
        runCatching {
            val chat = VulkanShaders.buildGradientMessage(": $message")
            mc.gui.hud.getChat().addClientSystemMessage(chat)
        }
    }
}
