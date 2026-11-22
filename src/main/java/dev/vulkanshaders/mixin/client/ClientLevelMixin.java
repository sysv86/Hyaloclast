package dev.vulkanshaders.mixin.client;

import dev.vulkanshaders.VulkanShaders;
import dev.vulkanshaders.state.WorldStateCollector;
import net.minecraft.client.multiplayer.ClientLevel;
import net.minecraft.core.BlockPos;
import net.minecraft.world.level.block.state.BlockState;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

@Mixin(ClientLevel.class)
public class ClientLevelMixin {
    @Inject(method = "sendBlockUpdated", at = @At("RETURN"))
    private void vulkanshaders$refreshChangedLighting(
        BlockPos pos, BlockState oldState, BlockState newState, int flags, CallbackInfo ci
    ) {
        // Refresh explicit light sources when block emission changes.
        if (VulkanShaders.shaderPackActive && oldState.getLightEmission() != newState.getLightEmission()) {
            WorldStateCollector.onBlockLightingChanged(pos);
        }
    }
}
