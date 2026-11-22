package dev.vulkanshaders.input

import com.mojang.blaze3d.platform.InputConstants
import dev.vulkanshaders.VulkanShaders
import dev.vulkanshaders.gui.PackSettingsScreen
import dev.vulkanshaders.gui.ShaderNotices
import dev.vulkanshaders.gui.ShaderPacksScreen
import dev.vulkanshaders.pack.ShaderPackManager
import net.minecraft.client.KeyMapping
import net.minecraft.client.Minecraft
import net.minecraft.network.chat.Component
import net.minecraft.resources.Identifier
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Registers and handles Hyaloclast's Fabric client keybinds:
 * - Toggle Shaders On/Off (default: `K`)
 * - Reload Active Shader Pack (default: `R`)
 * - Open Shader Packs / Pack Settings Screen (default: `O`)
 *
 * Wave 2 wires [register] into `VulkanShaders.onInitializeClient()` and [tick] into
 * `VulkanShaders.onClientTick(mc)`.
 */
object HyaloclastKeybinds {
    const val CATEGORY_KEY: String = "key.categories.vulkanshaders"
    const val KEY_TOGGLE_SHADERS: String = "key.vulkanshaders.toggle_shaders"
    const val KEY_RELOAD_SHADERS: String = "key.vulkanshaders.reload_shaders"
    const val KEY_OPEN_SHADER_MENU: String = "key.vulkanshaders.open_shader_menu"

    private val registered = AtomicBoolean(false)

    @Volatile
    var toggleShadersKey: KeyMapping? = null
        private set

    @Volatile
    var reloadShadersKey: KeyMapping? = null
        private set

    @Volatile
    var openShaderMenuKey: KeyMapping? = null
        private set

    /**
     * Creates and registers all Hyaloclast [KeyMapping]s via Fabric's `KeyBindingHelper` when
     * `fabric-key-binding-api-v1` is present at runtime. Safe to call multiple times (idempotent).
     */
    @JvmStatic
    fun register() {
        if (!registered.compareAndSet(false, true)) return

        val toggle = createKeyMapping(KEY_TOGGLE_SHADERS, InputConstants.KEY_K)
        val reload = createKeyMapping(KEY_RELOAD_SHADERS, InputConstants.KEY_R)
        val openMenu = createKeyMapping(KEY_OPEN_SHADER_MENU, InputConstants.KEY_O)

        toggleShadersKey = toggle?.let(::registerWithFabricIfAvailable)
        reloadShadersKey = reload?.let(::registerWithFabricIfAvailable)
        openShaderMenuKey = openMenu?.let(::registerWithFabricIfAvailable)

        VulkanShaders.log.info(
            "Registered {} Hyaloclast keybind(s)",
            allMappings().size,
        )
    }

    /**
     * Returns all instantiated Hyaloclast [KeyMapping]s (for optional `Options.keyMappings` mixin wiring
     * when `fabric-key-binding-api-v1` is not bundled).
     */
    @JvmStatic
    fun allMappings(): List<KeyMapping> =
        listOfNotNull(toggleShadersKey, reloadShadersKey, openShaderMenuKey)

    /**
     * Polls and dispatches keybind presses on the client tick thread.
     */
    @JvmStatic
    fun tick(client: Minecraft) {
        if (client.gui.screen() != null) return

        toggleShadersKey?.let { key ->
            while (key.consumeClick()) {
                handleToggleShaders()
            }
        }

        reloadShadersKey?.let { key ->
            while (key.consumeClick()) {
                handleReloadActivePack()
            }
        }

        openShaderMenuKey?.let { key ->
            while (key.consumeClick()) {
                handleOpenShaderMenu(client)
            }
        }
    }

    private fun handleToggleShaders() {
        val cfg = VulkanShaders.config
        val nextEnabled = !cfg.enabled
        val targetPackId = if (nextEnabled && (cfg.selectedPackId.isBlank() || cfg.selectedPackId.equals("off", ignoreCase = true))) {
            ShaderPackManager.discoveredPacks.firstOrNull()?.manifest?.id ?: ShaderPackManager.BUILTIN_PACK_ID
        } else {
            cfg.selectedPackId
        }

        val ok = VulkanShaders.updateConfig(
            cfg.copy(
                enabled = nextEnabled,
                selectedPackId = targetPackId,
            ),
        )
        if (!ok) return

        val statusKey = if (VulkanShaders.shaderPackActive) {
            "vulkanshaders.keybind.toast.shaders_enabled"
        } else {
            "vulkanshaders.keybind.toast.shaders_disabled"
        }
        val packName = ShaderPackManager.activePack?.manifest?.name ?: targetPackId
        ShaderNotices.showStatusNotice(
            Component.translatable("vulkanshaders.screen.title"),
            Component.translatable(statusKey, packName),
        )
    }

    private fun handleReloadActivePack() {
        val result = ShaderPackManager.reloadActivePackInWorld()
        if (result.success && result.packName == null) {
            ShaderNotices.showStatusNotice(
                Component.translatable("vulkanshaders.screen.title"),
                Component.translatable("vulkanshaders.keybind.toast.no_active_pack"),
            )
        }
    }

    private fun handleOpenShaderMenu(client: Minecraft) {
        val parent = client.gui.screen()
        val activePack = ShaderPackManager.activePack.takeIf { VulkanShaders.shaderPackActive }
        if (activePack != null && activePack.manifest.settings.isNotEmpty()) {
            client.gui.setScreen(PackSettingsScreen(parent, client.options, activePack))
        } else {
            client.gui.setScreen(ShaderPacksScreen(parent, client.options))
        }
    }

    /**
     * Instantiates a vanilla [KeyMapping] supporting both MC 26.3's `KeyMapping.Category` constructor
     * and the legacy `String` category constructor cleanly.
     */
    private fun createKeyMapping(translationKey: String, defaultKeyCode: Int): KeyMapping? = runCatching {
        val categoryClass = runCatching { Class.forName("net.minecraft.client.KeyMapping\$Category") }.getOrNull()
        if (categoryClass != null) {
            val categoryObj = resolveOrRegisterCategory(categoryClass)
            if (categoryObj != null) {
                val ctor = KeyMapping::class.java.getConstructor(
                    String::class.java,
                    InputConstants.Type::class.java,
                    Int::class.javaPrimitiveType,
                    categoryClass,
                )
                return@runCatching ctor.newInstance(translationKey, InputConstants.Type.KEYBOARD, defaultKeyCode, categoryObj)
            }
        }
        val legacyCtor = KeyMapping::class.java.getConstructor(
            String::class.java,
            InputConstants.Type::class.java,
            Int::class.javaPrimitiveType,
            String::class.java,
        )
        legacyCtor.newInstance(translationKey, InputConstants.Type.KEYBOARD, defaultKeyCode, CATEGORY_KEY)
    }.onFailure { t ->
        VulkanShaders.log.warn("Could not create KeyMapping '{}': {}", translationKey, t.message)
    }.getOrNull()

    private fun resolveOrRegisterCategory(categoryClass: Class<*>): Any? {
        val id = Identifier.fromNamespaceAndPath(VulkanShaders.MOD_ID, "hyaloclast")
        runCatching {
            val registerMethod = categoryClass.getMethod("register", Identifier::class.java)
            return registerMethod.invoke(null, id)
        }
        runCatching {
            val ctor = categoryClass.getConstructor(Identifier::class.java)
            return ctor.newInstance(id)
        }
        return runCatching { categoryClass.getField("MISC").get(null) }.getOrNull()
    }

    private fun registerWithFabricIfAvailable(mapping: KeyMapping): KeyMapping {
        runCatching {
            val helperCls = Class.forName("net.fabricmc.fabric.api.client.keybinding.v1.KeyBindingHelper")
            val registerMethod = helperCls.getMethod("registerKeyBinding", KeyMapping::class.java)
            registerMethod.invoke(null, mapping)
        }.onFailure { t ->
            VulkanShaders.log.debug("Fabric KeyBindingHelper unavailable ({}); using direct KeyMapping list.", t.message)
        }
        return mapping
    }
}
