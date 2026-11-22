package dev.vulkanshaders.mixin.world;

import com.llamalad7.mixinextras.sugar.Local;
import dev.vulkanshaders.state.WorldStateCollector;
import net.minecraft.client.Minecraft;
import net.minecraft.client.renderer.FirstPersonHandsAndItemsRenderer;
import net.minecraft.client.renderer.entity.layers.ItemInHandLayer;
import net.minecraft.client.renderer.entity.state.ArmedEntityRenderState;
import net.minecraft.client.renderer.entity.state.AvatarRenderState;
import net.minecraft.world.item.ItemStack;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.ModifyVariable;

public final class HeldItemLightingMixins {
    private HeldItemLightingMixins() {}

    @Mixin(FirstPersonHandsAndItemsRenderer.class)
    public static class FirstPerson {
        @ModifyVariable(method = "submitArmWithItem", at = @At("HEAD"), argsOnly = true, ordinal = 0)
        private int vulkanshaders$lightHeldItem(int light, @Local(argsOnly = true) ItemStack item) {
            var player = Minecraft.getInstance().player;
            return player != null && !player.isSpectator()
                ? WorldStateCollector.heldItemLight(item, light)
                : light;
        }
    }

    @Mixin(ItemInHandLayer.class)
    public static class ThirdPerson {
        @ModifyVariable(method = "submitArmWithItem", at = @At("HEAD"), argsOnly = true, ordinal = 0)
        private int vulkanshaders$lightHeldItem(
            int light,
            @Local(argsOnly = true) ItemStack item,
            @Local(argsOnly = true) ArmedEntityRenderState state
        ) {
            var player = Minecraft.getInstance().player;
            if (player != null && state instanceof AvatarRenderState avatar &&
                avatar.id == player.getId() && player.isSpectator()
            ) {
                return light;
            }
            return WorldStateCollector.heldItemLight(item, light);
        }
    }
}
