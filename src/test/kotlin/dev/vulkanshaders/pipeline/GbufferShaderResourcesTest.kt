package dev.vulkanshaders.pipeline

import com.mojang.renderpearl.api.pipeline.ShaderType
import net.minecraft.resources.Identifier
import org.junit.jupiter.api.Assertions.assertNotNull
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test

/** The bundled G-Buffer shaders must exist for every 4-MRT pipeline variant. */
class GbufferShaderResourcesTest {
    @Test
    fun `terrain entity and particle shaders resolve with 4 MRT outputs`() {
        val ids = listOf(
            Identifier.fromNamespaceAndPath("vulkanshaders", "core/gbuffer_terrain_region"),
            Identifier.fromNamespaceAndPath("vulkanshaders", "core/gbuffer_entity"),
            Identifier.fromNamespaceAndPath("vulkanshaders", "core/gbuffer_particle"),
        )
        for (id in ids) {
            val vsh = GbufferPipelineRegistry.resolveShader(id, ShaderType.VERTEX)
            val fsh = GbufferPipelineRegistry.resolveShader(id, ShaderType.FRAGMENT)
            assertNotNull(vsh, "vertex shader missing for $id")
            assertNotNull(fsh, "fragment shader missing for $id")
            assertTrue(vsh!!.contains("prevClipPos"), "$id vertex shader must compute prevClipPos")
            assertTrue(fsh!!.contains("outVelocity"), "$id fragment shader must write velocity")
            assertTrue(fsh.contains("outNormalRoughness"), "$id must write normal MRT")
            assertTrue(fsh.contains("outMaterialPbr"), "$id must write material MRT")
        }
    }

    @Test
    fun `terrain shader uses real motion vectors and LabPBR atlases`() {
        val terrain = Identifier.fromNamespaceAndPath("vulkanshaders", "core/gbuffer_terrain_region")
        val vsh = GbufferPipelineRegistry.resolveShader(terrain, ShaderType.VERTEX)!!
        val fsh = GbufferPipelineRegistry.resolveShader(terrain, ShaderType.FRAGMENT)!!
        assertTrue(vsh.contains("vsMotion.prevViewProj"), "no previous view-projection matrix")
        assertTrue(vsh.contains("cameraDeltaTime.xyz"), "no camera delta reprojection")
        assertTrue(vsh.contains("// @hook vertex_displace"))
        assertTrue(vsh.contains("// @hook water_waves"))
        assertTrue(fsh.contains("VsNormalAtlas") && fsh.contains("VsSpecularAtlas"))
        assertTrue(fsh.contains("255.0 / 254.0"), "LabPBR emission scaling missing")
        assertTrue(fsh.contains("// @hook surface"), "surface hook marker missing")
    }
}
