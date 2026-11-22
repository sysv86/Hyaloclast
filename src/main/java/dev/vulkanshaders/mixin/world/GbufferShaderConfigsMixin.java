package dev.vulkanshaders.mixin.world;

import com.mojang.renderpearl.api.pipeline.ShaderType;
import dev.vulkanshaders.pipeline.GbufferPipelineRegistry;
import net.minecraft.client.renderer.ShaderManager;
import net.minecraft.resources.Identifier;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfoReturnable;

/**
 * Resolves bundled `vulkanshaders:*` region shader sources for RenderPearl's `ShaderManager`.
 */
@Mixin(value = ShaderManager.Configs.class, priority = 1100)
public class GbufferShaderConfigsMixin {
    @Inject(method = "getShader", at = @At("HEAD"), cancellable = true)
    private void vulkanshaders$resolveBundledShader(
        Identifier id,
        ShaderType type,
        CallbackInfoReturnable<String> cir
    ) {
        String source = GbufferPipelineRegistry.resolveShader(id, type);
        if (source != null) {
            cir.setReturnValue(source);
        }
    }
}
