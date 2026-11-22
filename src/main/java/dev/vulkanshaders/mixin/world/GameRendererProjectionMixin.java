package dev.vulkanshaders.mixin.world;

import com.mojang.blaze3d.systems.RenderSystem;
import dev.vulkanshaders.VulkanShaders;
import dev.vulkanshaders.bridge.VulkanShadersTurboExtension;
import dev.vulkanshaders.state.WorldStateCollector;
import java.util.Optional;
import java.util.OptionalDouble;
import net.minecraft.client.Minecraft;
import net.minecraft.client.renderer.GameRenderer;
import net.minecraft.client.renderer.SubmitNodeStorage;
import net.minecraft.client.renderer.feature.FeatureRenderDispatcher;
import net.minecraft.client.renderer.state.GameRenderState;
import net.minecraft.client.renderer.state.level.CameraRenderState;
import org.joml.Matrix4f;
import org.spongepowered.asm.mixin.Final;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Shadow;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.ModifyArg;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

@Mixin(GameRenderer.class)
public abstract class GameRendererProjectionMixin {
    @Shadow @Final private GameRenderState gameRenderState;

    @ModifyArg(
        method = "renderLevel",
        at = @At(value = "INVOKE",
            target = "Lnet/minecraft/client/renderer/ProjectionMatrixBuffer;getBuffer(Lorg/joml/Matrix4f;)Lcom/mojang/renderpearl/api/buffers/GpuBufferSlice;")
    )
    private Matrix4f vulkanshaders$prepareWorldProjection(Matrix4f projection) {
        if (!VulkanShaders.shaderPackActive) return projection;
        CameraRenderState camera = gameRenderState.levelRenderState.cameraRenderState;
        return WorldStateCollector.beginFrame(
            camera, projection, gameRenderState.levelRenderState.skyRenderState.sunAngle, camera.fogData.color
        );
    }

    @Inject(method = "render3dHud", at = @At("HEAD"))
    private void vulkanshaders$renderPathTracedWeather(CallbackInfo ci) {
        if (!VulkanShadersTurboExtension.INSTANCE.shouldRenderPathTracedWeather()) {
            return;
        }
        var world = this.gameRenderState.levelRenderState;
        var weather = world.weatherRenderState;
        var particles = world.particlesRenderState;
        boolean hasWeather = !weather.rainColumns.isEmpty() || !weather.snowColumns.isEmpty();
        boolean hasParticles = particles != null && !particles.particles.isEmpty();
        if (!hasWeather && !hasParticles) {
            return;
        }

        var renderer = (GameRenderer) (Object) this;
        var target = renderer.mainRenderTarget();
        FeatureRenderDispatcher.PreparedFrame preparedParticles = null;
        var modelView = RenderSystem.getModelViewStack();
        modelView.pushMatrix();
        try {
            modelView.mul(world.cameraRenderState.viewRotationMatrix);
            if (hasParticles) {
                SubmitNodeStorage particleSubmits = new SubmitNodeStorage();
                particles.submit(particleSubmits, world.cameraRenderState);
                preparedParticles = renderer.featureRenderDispatcher().prepareFrame(particleSubmits);
            }
            // Keep the path-traced depth attachment intact for precipitation and particle occlusion.
            try (var pass = RenderSystem.getDevice().createCommandEncoder().createRenderPass(
                () -> "Path-traced weather & particles", target.getColorTextureView(), Optional.empty(),
                target.getDepthTextureView(), OptionalDouble.empty()
            )) {
                RenderSystem.bindDefaultUniforms(pass);
                if (preparedParticles != null) {
                    FeatureRenderDispatcher.renderAllFeatures(pass, preparedParticles);
                }
                if (hasWeather) {
                    Minecraft.getInstance().levelRenderer.weatherEffectRenderer().render(weather, pass);
                }
            }
        } finally {
            modelView.popMatrix();
            if (preparedParticles != null) {
                preparedParticles.close();
            }
        }
    }
}
