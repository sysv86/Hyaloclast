package dev.vulkanshaders.mixin.compat;

import dev.vulkanshaders.bridge.DistantHorizonsBridge;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Pseudo;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Coerce;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

/**
 * Clean-room {@code @Pseudo} hook on Distant Horizons' {@code BlazeVanillaFadeRenderer} to
 * suppress DH's vanilla fade when Hyaloclast supplies the DH composite.
 */
@Pseudo
@Mixin(targets = "com.seibel.distanthorizons.common.render.blaze.postProcessing.BlazeVanillaFadeRenderer", remap = false)
public abstract class DhVanillaFadeRendererMixin {
    @Inject(
        method = {"render(Lcom/seibel/distanthorizons/core/render/RenderParams;)V", "render(Lcom/seibel/distanthorizons/api/methods/events/sharedParameterObjects/DhApiRenderParam;)V"},
        at = @At("HEAD"),
        cancellable = true,
        require = 0
    )
    private void vulkanshaders$suppressDhVanillaFade(@Coerce Object renderParams, CallbackInfo ci) {
        if (DistantHorizonsBridge.shouldSuppressDhRasterComposite()) {
            ci.cancel();
        }
    }
}
