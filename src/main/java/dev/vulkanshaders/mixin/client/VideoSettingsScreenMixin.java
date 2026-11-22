package dev.vulkanshaders.mixin.client;

import dev.vulkanshaders.gui.ShaderPacksScreen;
import net.minecraft.client.Options;
import net.minecraft.client.gui.components.Button;
import net.minecraft.client.gui.screens.Screen;
import net.minecraft.client.gui.screens.options.OptionsSubScreen;
import net.minecraft.client.gui.screens.options.VideoSettingsScreen;
import net.minecraft.network.chat.Component;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

/**
 * Adds the Hyaloclast shader pack configuration button to Video Settings.
 */
@Mixin(VideoSettingsScreen.class)
public abstract class VideoSettingsScreenMixin extends OptionsSubScreen {
    private VideoSettingsScreenMixin(Screen lastScreen, Options options, Component title) {
        super(lastScreen, options, title);
    }

    @Inject(method = "addOptions", at = @At("HEAD"))
    private void vulkanshaders$addShadersButton(CallbackInfo ci) {
        if (this.list == null) {
            return;
        }
        Button button = Button.builder(
            Component.translatable("vulkanshaders.options.button"),
            ignored -> {
                this.list.applyUnsavedChanges();
                this.minecraft.gui.setScreen(new ShaderPacksScreen(this, this.options));
            }
        ).build();

        this.list.addBig(button);
    }
}
