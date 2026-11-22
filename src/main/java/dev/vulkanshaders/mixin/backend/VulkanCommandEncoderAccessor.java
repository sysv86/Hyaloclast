package dev.vulkanshaders.mixin.backend;

import com.mojang.renderpearl.backend.vulkan.VulkanCommandEncoder;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.gen.Accessor;

@Mixin(VulkanCommandEncoder.class)
public interface VulkanCommandEncoderAccessor {
    @Accessor("submitSemaphore")
    long vulkanshaders$getSubmitSemaphore();
}
