package dev.vulkanshaders.mixin.compat;

import dev.vulkanshaders.bridge.DistantHorizonsBridge;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Pseudo;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Coerce;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

/**
 * Clean-room {@code @Pseudo} hook on Distant Horizons' {@code BlazeDhMetaRenderer} to suppress
 * {@code copyToMcTexture} when Hyaloclast supplies the DH composite.
 */
@Pseudo
@Mixin(targets = "com.seibel.distanthorizons.common.render.blaze.BlazeDhMetaRenderer", remap = false)
public abstract class DhMetaRendererMixin {
    @Inject(
        method = {"copyToMcTexture(Lcom/seibel/distanthorizons/core/render/RenderParams;)V", "copyToMcTexture(Lcom/seibel/distanthorizons/api/methods/events/sharedParameterObjects/DhApiRenderParam;)V"},
        at = @At("HEAD"),
        cancellable = true,
        require = 0
    )
    private void vulkanshaders$suppressDhCopyToMcTexture(@Coerce Object renderParams, CallbackInfo ci) {
        if (DistantHorizonsBridge.shouldSuppressDhRasterComposite()) {
            ci.cancel();
        }
    }
}
