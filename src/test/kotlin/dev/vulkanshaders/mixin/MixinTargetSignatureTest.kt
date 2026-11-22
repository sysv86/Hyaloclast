package dev.vulkanshaders.mixin

import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test
import java.nio.file.Files
import java.nio.file.Paths

class MixinTargetSignatureTest {
    @Test
    fun `all mixins declared in vulkanshaders mixins json exist in source tree`() {
        val jsonPath = Paths.get("src/main/resources/vulkanshaders.mixins.json")
        assertTrue(Files.isRegularFile(jsonPath), "vulkanshaders.mixins.json must exist")
        val content = Files.readString(jsonPath)
        val expectedEntries = listOf(
            "backend.FrontendGpuDeviceAccessor",
            "backend.RenderSystemMixin",
            "backend.VulkanBackendMixin",
            "backend.VulkanCommandEncoderAccessor",
            "backend.VulkanDeviceMixin",
            "backend.VulkanGpuTextureMixin",
            "backend.VulkanGpuTextureMixin\$TextureViewMixin",
            "backend.VulkanInstanceMixin",
            "backend.VulkanQueueLockMixins\$EncoderWaitMixin",
            "backend.VulkanQueueLockMixins\$QueueMixin",
            "backend.VulkanQueueLockMixins\$SubmissionMixin",
            "backend.VulkanQueueLockMixins\$SurfaceMixin",
            "client.CrashReportMixin",
            "client.DebugScreenOverlayMixin",
            "client.MinecraftClientMixin",
            "client.VideoSettingsScreenMixin",
            "framegen.ReflexSurfaceMixin",
            "texture.TextureAtlasPbrMixin",
            "world.GbufferRenderPipelinesMixin",
            "world.GbufferShaderConfigsMixin",
            "world.LevelRendererMixin",
        )
        for (entry in expectedEntries) {
            assertTrue(content.contains(entry), "Missing mixin entry '$entry' in vulkanshaders.mixins.json")
            val outerClass = entry.substringBefore('$').replace('.', '/')
            val javaFile = Paths.get("src/main/java/dev/vulkanshaders/mixin/$outerClass.java")
            assertTrue(Files.isRegularFile(javaFile), "Missing Java source file $javaFile for mixin $entry")
        }
    }
}
