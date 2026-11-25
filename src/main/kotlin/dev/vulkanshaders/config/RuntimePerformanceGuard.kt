package dev.vulkanshaders.config

import dev.vulkanshaders.VulkanShaders
import dev.vulkanshaders.backend.NativeCoreBridge
import dev.vulkanshaders.diagnostics.HyaloclastErrorLog
import dev.vulkanshaders.gui.ShaderNotices
import dev.vulkanshaders.pack.ShaderPackManager
import net.minecraft.ChatFormatting
import net.minecraft.client.Minecraft
import net.minecraft.network.chat.Component
import net.minecraft.network.chat.Style

/**
 * Monitors frame/tick cadence and GPU frame times during live rendering, stepping down
 * [RenderTier] automatically when sustained overload or a multi-second render stall occurs.
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
     * Returns the next lower [RenderTier] supported by [supportedTiers], or `null` when already
     * at the lowest supported tier.
     */
    @JvmStatic
    fun nextLowerTier(current: RenderTier, supportedTiers: List<Int>): RenderTier? {
        val candidates = RenderTier.entries
            .filter { it.tierIndex < current.tierIndex }
            // TIER_2 ray-query stalls the GPU on first submission on the affected driver,
            // so automatic downgrades go straight from TIER_3 to TIER_1.
            .filter { it != RenderTier.TIER_2_RAY_QUERY }
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
        val overloaded = deltaMs >= OVERLOAD_TICK_MS || (avgGpuMs != null && avgGpuMs >= OVERLOAD_GPU_MS)
        val criticalStall = deltaMs >= CRITICAL_STALL_MS

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

        val lowerTier = nextLowerTier(cfg.renderTier, pack.manifest.supportedTiers)
        val reasonMs = avgGpuMs?.takeIf { it >= OVERLOAD_GPU_MS } ?: deltaMs
        if (lowerTier != null) {
            VulkanShaders.updateConfig(cfg.copy(renderTier = lowerTier))
            val msg = "GPU overload (${"%.0f".format(reasonMs)} ms/frame) — lowered render tier to ${lowerTier.name}."
            HyaloclastErrorLog.recordError("PerfGuard", msg)
            notifyPlayer(mc, msg)
        } else if (criticalStall) {
            VulkanShaders.updateConfig(cfg.copy(enabled = false))
            val msg = "Severe GPU stall (${"%.0f".format(reasonMs)} ms) at lowest tier — shaders disabled."
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
            val chat = VulkanShaders.buildGradientBrandComponent()
                .append(Component.literal(": $message").withStyle(Style.EMPTY.withColor(ChatFormatting.YELLOW)))
            mc.gui.hud.getChat().addClientSystemMessage(chat)
        }
    }
}
