package dev.vulkanshaders.mixin.texture;

import com.llamalad7.mixinextras.injector.wrapmethod.WrapMethod;
import com.llamalad7.mixinextras.injector.wrapoperation.Operation;
import com.mojang.blaze3d.platform.NativeImage;
import com.mojang.blaze3d.systems.RenderSystem;
import com.mojang.renderpearl.api.GpuFormat;
import com.mojang.renderpearl.api.commands.CommandEncoder;
import com.mojang.renderpearl.api.device.GpuDevice;
import com.mojang.renderpearl.api.textures.AddressMode;
import com.mojang.renderpearl.api.textures.FilterMode;
import com.mojang.renderpearl.api.textures.GpuSampler;
import com.mojang.renderpearl.api.textures.GpuTexture;
import com.mojang.renderpearl.api.textures.GpuTextureView;
import com.mojang.renderpearl.backend.vulkan.VulkanGpuSampler;
import com.mojang.renderpearl.backend.vulkan.VulkanGpuTexture;
import com.mojang.renderpearl.backend.vulkan.VulkanGpuTextureView;
import dev.vulkanshaders.VulkanShaders;
import dev.vulkanshaders.backend.NativeCoreBridge;
import dev.vulkanshaders.texture.TextureRegistry;
import net.minecraft.client.Minecraft;
import net.minecraft.client.renderer.texture.AbstractTexture;
import net.minecraft.client.renderer.texture.SpriteLoader;
import net.minecraft.client.renderer.texture.TextureAtlas;
import net.minecraft.client.renderer.texture.TextureAtlasSprite;
import net.minecraft.resources.Identifier;
import net.minecraft.server.packs.resources.Resource;
import net.minecraft.server.packs.resources.ResourceManager;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Unique;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

import java.io.IOException;
import java.io.InputStream;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.OptionalDouble;

/**
 * 1. Suppresses bindless registration of transient scratch textures created during
 *    {@code TextureAtlas.uploadInitialContents} with exception-safe {@code @WrapMethod}.
 * 2. Registers the completed atlas {@code GpuTexture}, {@code GpuTextureView}, and {@code GpuSampler}
 *    into {@link TextureRegistry} at the end of {@code TextureAtlas.upload}.
 * 3. Builds mipmapped LabPBR 1.3 companion atlases (normal XY renormalization and threshold-preserving
 *    specular downsampling) only when {@code _n} or {@code _s} sprites exist, keeping
 *    {@code FLAT_SPECULAR_ARGB = 0xFF000000} as an explicit "no _s sprite" sentinel.
 */
@Mixin(TextureAtlas.class)
public abstract class TextureAtlasPbrMixin extends AbstractTexture {
    private static final int FLAT_NORMAL_ARGB = 0xFF8080FF; // RGBA (128,128,255,255)
    private static final int FLAT_SPECULAR_ARGB = 0xFF000000; // Explicit "no _s sprite" sentinel
    private static final int MAX_PBR_MIP_LEVELS = 4;

    @Unique
    private SpriteLoader.Preparations vulkanshaders$lastPreparations;

    @Unique
    private long vulkanshaders$albedoVkImage = 0L;

    @WrapMethod(method = "uploadInitialContents")
    private void vulkanshaders$wrapUploadInitialContents(Operation<Void> original) {
        TextureRegistry.setSkipScratchUploads(true);
        try {
            original.call();
        } finally {
            TextureRegistry.setSkipScratchUploads(false);
        }
    }

    @Inject(method = "upload", at = @At("TAIL"))
    private void vulkanshaders$onAtlasUploaded(SpriteLoader.Preparations preparations, CallbackInfo ci) {
        TextureAtlas self = (TextureAtlas) (Object) this;
        boolean blocksAtlas = TextureAtlas.LOCATION_BLOCKS.equals(self.location());
        TextureRegistry.onTextureViewAndSamplerBound(this.texture, this.textureView, this.sampler, blocksAtlas);

        long albedoVk = (this.texture instanceof VulkanGpuTexture vkTex) ? vkTex.vkImage() : 0L;
        this.vulkanshaders$albedoVkImage = albedoVk;
        this.vulkanshaders$lastPreparations = preparations;

        String atlasKey = self.location().toString();
        Runnable builder = () -> {
            SpriteLoader.Preparations p = this.vulkanshaders$lastPreparations;
            if (p != null) {
                this.buildCompanionAtlases(p);
            }
        };
        if (blocksAtlas) {
            TextureRegistry.setCompanionAtlasBuilder(builder);
        } else {
            TextureRegistry.registerCompanionAtlasBuilder(atlasKey, builder);
        }

        if (VulkanShaders.shaderPackActive) {
            buildCompanionAtlases(preparations);
        }
    }

    @Inject(method = "close", at = @At("HEAD"))
    private void vulkanshaders$onAtlasClosed(CallbackInfo ci) {
        TextureAtlas self = (TextureAtlas) (Object) this;
        boolean blocksAtlas = TextureAtlas.LOCATION_BLOCKS.equals(self.location());
        this.vulkanshaders$lastPreparations = null;
        if (blocksAtlas) {
            TextureRegistry.setCompanionAtlasBuilder(null);
            TextureRegistry.closeCompanionAtlases();
        } else {
            TextureRegistry.registerCompanionAtlasBuilder(self.location().toString(), null);
            TextureRegistry.closeCompanionForAlbedo(this.vulkanshaders$albedoVkImage, false);
        }
        this.vulkanshaders$albedoVkImage = 0L;
    }

    @Unique
    private void buildCompanionAtlases(SpriteLoader.Preparations preparations) {
        if (!VulkanShaders.shaderPackActive) return;
        TextureAtlas self = (TextureAtlas) (Object) this;
        boolean blocksAtlas = TextureAtlas.LOCATION_BLOCKS.equals(self.location());

        int atlasW = preparations.width();
        int atlasH = preparations.height();
        if (atlasW <= 0 || atlasH <= 0) return;

        long albedoVk = (this.texture instanceof VulkanGpuTexture vkTex) ? vkTex.vkImage() : this.vulkanshaders$albedoVkImage;
        int albedoSlot = blocksAtlas
            ? TextureRegistry.albedoAtlasBindlessSlot
            : TextureRegistry.getBindlessSlotForImage(albedoVk);

        ResourceManager resources = Minecraft.getInstance().getResourceManager();
        NativeImage normalAtlas = null;
        NativeImage specularAtlas = null;

        int normalPairs = 0;
        int specularPairs = 0;
        try {
            for (Map.Entry<Identifier, TextureAtlasSprite> entry : preparations.regions().entrySet()) {
                TextureAtlasSprite sprite = entry.getValue();
                Identifier spriteName = sprite.contents().name();
                int x = sprite.getX();
                int y = sprite.getY();
                int w = sprite.contents().width();
                int h = sprite.contents().height();
                if (x < 0 || y < 0 || w <= 0 || h <= 0 || x + w > atlasW || y + h > atlasH) continue;

                NativeImage normal = readPbrLayer(resources, spriteName, "_n");
                if (normal != null) {
                    if (normalAtlas == null) {
                        normalAtlas = new NativeImage(atlasW, atlasH, false);
                        normalAtlas.fillRect(0, 0, atlasW, atlasH, FLAT_NORMAL_ARGB);
                    }
                    blitScaledFrame0(normal, normalAtlas, x, y, w, h);
                    normalPairs++;
                    normal.close();
                }
                NativeImage specular = readPbrLayer(resources, spriteName, "_s");
                if (specular != null) {
                    if (specularAtlas == null) {
                        specularAtlas = new NativeImage(atlasW, atlasH, false);
                        specularAtlas.fillRect(0, 0, atlasW, atlasH, FLAT_SPECULAR_ARGB);
                    }
                    blitScaledFrame0(specular, specularAtlas, x, y, w, h);
                    specularPairs++;
                    specular.close();
                }
            }
        } catch (Throwable t) {
            if (normalAtlas != null) normalAtlas.close();
            if (specularAtlas != null) specularAtlas.close();
            VulkanShaders.INSTANCE.getLog().warn("[Hyaloclast] Failed scanning LabPBR sprites for {}: {}", self.location(), t.toString());
            return;
        }

        if (normalPairs == 0 && specularPairs == 0) {
            if (blocksAtlas) {
                TextureRegistry.closeCompanionForAlbedo(albedoVk, true);
                installFlatFallbackAtlases(albedoVk, albedoSlot);
                VulkanShaders.INSTANCE.getLog().info(
                    "[Hyaloclast] LabPBR companions: no _n/_s sprite pairs found for blocks atlas; using flat defaults"
                );
            } else {
                TextureRegistry.closeCompanionForAlbedo(albedoVk, false);
            }
            return;
        }

        int minDim = Math.max(1, Math.min(atlasW, atlasH));
        int mipLevels = Math.min(MAX_PBR_MIP_LEVELS, 32 - Integer.numberOfLeadingZeros(minDim));
        NativeImage[] normalMips = (normalPairs > 0) ? new NativeImage[mipLevels] : null;
        NativeImage[] specMips = (specularPairs > 0) ? new NativeImage[mipLevels] : null;
        if (normalMips != null) normalMips[0] = normalAtlas;
        if (specMips != null) specMips[0] = specularAtlas;

        GpuTexture normalTex = null;
        GpuTextureView normalView = null;
        GpuSampler normalSampler = null;
        GpuTexture specTex = null;
        GpuTextureView specView = null;
        GpuSampler specSampler = null;
        boolean registered = false;

        try {
            if (normalMips != null) {
                for (int level = 1; level < mipLevels; level++) {
                    normalMips[level] = downsampleNormalMip(normalMips[level - 1]);
                }
            }
            if (specMips != null) {
                for (int level = 1; level < mipLevels; level++) {
                    specMips[level] = downsampleSpecularMip(specMips[level - 1]);
                }
            }

            GpuDevice device = RenderSystem.getDevice();
            int usage = GpuTexture.USAGE_TEXTURE_BINDING | GpuTexture.USAGE_COPY_DST | GpuTexture.USAGE_COPY_SRC;
            String atlasTag = self.location().getPath();
            CommandEncoder encoder = device.createCommandEncoder();

            int normalSlot = -1;
            if (normalMips != null) {
                normalTex = device.createTexture(
                    () -> "Hyaloclast LabPBR Normal Atlas (" + atlasTag + ")",
                    usage, GpuFormat.RGBA8_UNORM, atlasW, atlasH, 1, mipLevels);
                normalView = device.createTextureView(normalTex);
                normalSampler = device.createSampler(
                    AddressMode.REPEAT, AddressMode.REPEAT, FilterMode.NEAREST, FilterMode.NEAREST, 1, OptionalDouble.empty());
                uploadMipChain(encoder, normalTex, normalMips);
                normalSlot = NativeCoreBridge.INSTANCE.registerTextureWithSampler(
                    ((VulkanGpuTexture) normalTex).vkImage(),
                    ((VulkanGpuTextureView) normalView).vkImageView(),
                    ((VulkanGpuSampler) normalSampler).vkSampler(),
                    37, atlasW, atlasH, mipLevels, TextureRegistry.PbrRole.NORMAL.ordinal());
            } else if (blocksAtlas) {
                normalTex = device.createTexture(
                    () -> "Hyaloclast LabPBR Flat Normal", usage, GpuFormat.RGBA8_UNORM, 1, 1, 1, 1);
                normalView = device.createTextureView(normalTex);
                normalSampler = device.createSampler(
                    AddressMode.REPEAT, AddressMode.REPEAT, FilterMode.NEAREST, FilterMode.NEAREST, 1, OptionalDouble.empty());
                try (NativeImage flatNormal = new NativeImage(1, 1, false)) {
                    flatNormal.fillRect(0, 0, 1, 1, FLAT_NORMAL_ARGB);
                    encoder.writeToTexture(normalTex, flatNormal);
                }
            }

            int specSlot = -1;
            if (specMips != null) {
                specTex = device.createTexture(
                    () -> "Hyaloclast LabPBR Specular Atlas (" + atlasTag + ")",
                    usage, GpuFormat.RGBA8_UNORM, atlasW, atlasH, 1, mipLevels);
                specView = device.createTextureView(specTex);
                specSampler = device.createSampler(
                    AddressMode.REPEAT, AddressMode.REPEAT, FilterMode.NEAREST, FilterMode.NEAREST, 1, OptionalDouble.empty());
                uploadMipChain(encoder, specTex, specMips);
                specSlot = NativeCoreBridge.INSTANCE.registerTextureWithSampler(
                    ((VulkanGpuTexture) specTex).vkImage(),
                    ((VulkanGpuTextureView) specView).vkImageView(),
                    ((VulkanGpuSampler) specSampler).vkSampler(),
                    37, atlasW, atlasH, mipLevels, TextureRegistry.PbrRole.SPECULAR.ordinal());
            } else if (blocksAtlas) {
                specTex = device.createTexture(
                    () -> "Hyaloclast LabPBR Flat Specular", usage, GpuFormat.RGBA8_UNORM, 1, 1, 1, 1);
                specView = device.createTextureView(specTex);
                specSampler = device.createSampler(
                    AddressMode.REPEAT, AddressMode.REPEAT, FilterMode.NEAREST, FilterMode.NEAREST, 1, OptionalDouble.empty());
                try (NativeImage flatSpec = new NativeImage(1, 1, false)) {
                    flatSpec.fillRect(0, 0, 1, 1, FLAT_SPECULAR_ARGB);
                    encoder.writeToTexture(specTex, flatSpec);
                }
            }

            TextureRegistry.registerCompanionPair(
                albedoVk,
                albedoSlot,
                normalTex, normalView, normalSampler, normalSlot,
                specTex, specView, specSampler, specSlot,
                blocksAtlas,
                false
            );
            registered = true;
            VulkanShaders.INSTANCE.getLog().info(
                "[Hyaloclast] LabPBR companions ready ({}): {} _n / {} _s sprites, atlas {}x{} ({} mips), slots albedo={} n={} s={}",
                self.location(), normalPairs, specularPairs,
                atlasW, atlasH, mipLevels, albedoSlot, normalSlot, specSlot
            );
        } catch (Throwable t) {
            if (!registered) {
                safeClose(normalView, normalTex, normalSampler, specView, specTex, specSampler);
            }
            TextureRegistry.closeCompanionForAlbedo(albedoVk, blocksAtlas);
            VulkanShaders.INSTANCE.getLog().warn("[Hyaloclast] Could not create LabPBR companion atlases for {}: {}", self.location(), t.toString());
        } finally {
            if (normalMips != null) {
                for (NativeImage img : normalMips) {
                    if (img != null) img.close();
                }
            }
            if (specMips != null) {
                for (NativeImage img : specMips) {
                    if (img != null) img.close();
                }
            }
        }
    }

    @Unique
    private static void safeClose(AutoCloseable... resources) {
        for (AutoCloseable res : resources) {
            if (res != null) {
                try {
                    res.close();
                } catch (Throwable ignored) {
                }
            }
        }
    }

    @Unique
    private static void uploadMipChain(CommandEncoder encoder, GpuTexture tex, NativeImage[] mips) {
        encoder.writeToTexture(tex, mips[0]);
        for (int level = 1; level < mips.length; level++) {
            NativeImage img = mips[level];
            encoder.writeToTexture(tex, img, level, 0, 0, 0);
        }
    }

    @Unique
    private static NativeImage downsampleNormalMip(NativeImage src) {
        int sw = src.getWidth();
        int sh = src.getHeight();
        int dw = Math.max(1, sw >> 1);
        int dh = Math.max(1, sh >> 1);
        NativeImage dst = new NativeImage(dw, dh, false);
        for (int y = 0; y < dh; y++) {
            int sy0 = Math.min(sh - 1, y * 2);
            int sy1 = Math.min(sh - 1, sy0 + 1);
            for (int x = 0; x < dw; x++) {
                int sx0 = Math.min(sw - 1, x * 2);
                int sx1 = Math.min(sw - 1, sx0 + 1);
                int p00 = src.getPixel(sx0, sy0);
                int p10 = src.getPixel(sx1, sy0);
                int p01 = src.getPixel(sx0, sy1);
                int p11 = src.getPixel(sx1, sy1);

                float nx = 0f, ny = 0f, nz = 0f;
                int aoSum = 0;
                int hSum = 0;
                int[] pixels = { p00, p10, p01, p11 };
                for (int p : pixels) {
                    // NativeImage.getPixel returns ARGB (A<<24 | R<<16 | G<<8 | B)
                    int r = (p >>> 16) & 0xFF;
                    int g = (p >>> 8) & 0xFF;
                    int b = p & 0xFF;
                    int a = (p >>> 24) & 0xFF;
                    float vx = (r / 255.0f) * 2.0f - 1.0f;
                    float vy = (g / 255.0f) * 2.0f - 1.0f;
                    float vzSq = Math.max(0.0f, 1.0f - vx * vx - vy * vy);
                    nx += vx;
                    ny += vy;
                    nz += (float) Math.sqrt(vzSq);
                    aoSum += b;
                    hSum += a;
                }
                float len = (float) Math.sqrt(nx * nx + ny * ny + nz * nz);
                if (len > 1e-5f) {
                    nx /= len;
                    ny /= len;
                } else {
                    nx = 0.0f;
                    ny = 0.0f;
                }
                int outR = Math.max(0, Math.min(255, Math.round((nx * 0.5f + 0.5f) * 255.0f)));
                int outG = Math.max(0, Math.min(255, Math.round((ny * 0.5f + 0.5f) * 255.0f)));
                int outB = (aoSum + 2) >> 2;
                int outA = (hSum + 2) >> 2;
                dst.setPixel(x, y, (outA << 24) | (outR << 16) | (outG << 8) | outB);
            }
        }
        return dst;
    }

    /**
     * Downsamples a LabPBR 1.3 specular mip while preserving discrete channel thresholds:
     * - G channel: dielectric F0 (0..229) vs hardcoded metal (230..255).
     * - B channel: porosity (0..64) vs subsurface scattering (65..255).
     * - A channel: 255 = no emission, 0..254 = emission strength.
     */
    @Unique
    private static NativeImage downsampleSpecularMip(NativeImage src) {
        int sw = src.getWidth();
        int sh = src.getHeight();
        int dw = Math.max(1, sw >> 1);
        int dh = Math.max(1, sh >> 1);
        NativeImage dst = new NativeImage(dw, dh, false);
        for (int y = 0; y < dh; y++) {
            int sy0 = Math.min(sh - 1, y * 2);
            int sy1 = Math.min(sh - 1, sy0 + 1);
            for (int x = 0; x < dw; x++) {
                int sx0 = Math.min(sw - 1, x * 2);
                int sx1 = Math.min(sw - 1, sx0 + 1);
                int p00 = src.getPixel(sx0, sy0);
                int p10 = src.getPixel(sx1, sy0);
                int p01 = src.getPixel(sx0, sy1);
                int p11 = src.getPixel(sx1, sy1);

                if (p00 == FLAT_SPECULAR_ARGB && p10 == FLAT_SPECULAR_ARGB &&
                    p01 == FLAT_SPECULAR_ARGB && p11 == FLAT_SPECULAR_ARGB) {
                    dst.setPixel(x, y, FLAT_SPECULAR_ARGB);
                    continue;
                }

                int r = (((p00 >>> 16) & 0xFF) + ((p10 >>> 16) & 0xFF) + ((p01 >>> 16) & 0xFF) + ((p11 >>> 16) & 0xFF) + 2) >> 2;

                int g0 = (p00 >>> 8) & 0xFF;
                int g1 = (p10 >>> 8) & 0xFF;
                int g2 = (p01 >>> 8) & 0xFF;
                int g3 = (p11 >>> 8) & 0xFF;
                int outG = downsampleThresholdChannel(g0, g1, g2, g3, 230);

                int b0 = p00 & 0xFF;
                int b1 = p10 & 0xFF;
                int b2 = p01 & 0xFF;
                int b3 = p11 & 0xFF;
                int outB = downsampleThresholdChannel(b0, b1, b2, b3, 65);

                int a0 = (p00 >>> 24) & 0xFF;
                int a1 = (p10 >>> 24) & 0xFF;
                int a2 = (p01 >>> 24) & 0xFF;
                int a3 = (p11 >>> 24) & 0xFF;
                int emSum = (a0 == 255 ? 0 : a0) + (a1 == 255 ? 0 : a1) + (a2 == 255 ? 0 : a2) + (a3 == 255 ? 0 : a3);
                int outA = (emSum == 0) ? 255 : Math.min(254, (emSum + 2) >> 2);

                dst.setPixel(x, y, (outA << 24) | (r << 16) | (outG << 8) | outB);
            }
        }
        return dst;
    }

    /** Averages 2x2 samples only within the majority regime split by {@code highThreshold} (tie -> nearest p00). */
    @Unique
    private static int downsampleThresholdChannel(int c0, int c1, int c2, int c3, int highThreshold) {
        int hiCount = 0;
        int hiSum = 0;
        int loCount = 0;
        int loSum = 0;
        if (c0 >= highThreshold) { hiCount++; hiSum += c0; } else { loCount++; loSum += c0; }
        if (c1 >= highThreshold) { hiCount++; hiSum += c1; } else { loCount++; loSum += c1; }
        if (c2 >= highThreshold) { hiCount++; hiSum += c2; } else { loCount++; loSum += c2; }
        if (c3 >= highThreshold) { hiCount++; hiSum += c3; } else { loCount++; loSum += c3; }

        if (hiCount == 4) return (hiSum + 2) >> 2;
        if (loCount == 4) return (loSum + 2) >> 2;
        boolean pickHigh = (hiCount > loCount) || (hiCount == loCount && c0 >= highThreshold);
        if (pickHigh) {
            return (hiSum + (hiCount >> 1)) / hiCount;
        }
        return (loSum + (loCount >> 1)) / loCount;
    }

    @Unique
    private void installFlatFallbackAtlases(long albedoVk, int albedoSlot) {
        GpuTexture normalTex = null;
        GpuTextureView normalView = null;
        GpuSampler sampler = null;
        GpuTexture specTex = null;
        GpuTextureView specView = null;
        GpuSampler specSampler = null;
        boolean registered = false;
        try {
            GpuDevice device = RenderSystem.getDevice();
            int usage = GpuTexture.USAGE_TEXTURE_BINDING | GpuTexture.USAGE_COPY_DST | GpuTexture.USAGE_COPY_SRC;

            normalTex = device.createTexture(
                () -> "Hyaloclast LabPBR Flat Normal", usage, GpuFormat.RGBA8_UNORM, 1, 1, 1, 1);
            normalView = device.createTextureView(normalTex);
            sampler = device.createSampler(
                AddressMode.REPEAT, AddressMode.REPEAT, FilterMode.NEAREST, FilterMode.NEAREST, 1, OptionalDouble.empty());

            specTex = device.createTexture(
                () -> "Hyaloclast LabPBR Flat Specular", usage, GpuFormat.RGBA8_UNORM, 1, 1, 1, 1);
            specView = device.createTextureView(specTex);
            specSampler = device.createSampler(
                AddressMode.REPEAT, AddressMode.REPEAT, FilterMode.NEAREST, FilterMode.NEAREST, 1, OptionalDouble.empty());

            try (NativeImage flatNormal = new NativeImage(1, 1, false);
                 NativeImage flatSpecular = new NativeImage(1, 1, false)) {
                flatNormal.fillRect(0, 0, 1, 1, FLAT_NORMAL_ARGB);
                flatSpecular.fillRect(0, 0, 1, 1, FLAT_SPECULAR_ARGB);
                device.createCommandEncoder().writeToTexture(normalTex, flatNormal);
                device.createCommandEncoder().writeToTexture(specTex, flatSpecular);
            }

            TextureRegistry.registerCompanionPair(
                albedoVk,
                albedoSlot,
                normalTex, normalView, sampler, -1,
                specTex, specView, specSampler, -1,
                true,
                true
            );
            registered = true;
        } catch (Throwable t) {
            if (!registered) {
                safeClose(normalView, normalTex, sampler, specView, specTex, specSampler);
            }
            TextureRegistry.closeCompanionForAlbedo(albedoVk, true);
            VulkanShaders.INSTANCE.getLog().warn("[Hyaloclast] Could not create flat LabPBR fallback atlases: {}", t.toString());
        }
    }

    @Unique
    private NativeImage readPbrLayer(ResourceManager resources, Identifier spriteName, String suffix) {
        String ns = spriteName.getNamespace();
        String path = spriteName.getPath().toLowerCase(Locale.ROOT);
        Identifier[] candidates = new Identifier[]{
            Identifier.fromNamespaceAndPath(ns, path + suffix),
            Identifier.fromNamespaceAndPath(ns, "textures/" + path + suffix + ".png"),
            Identifier.fromNamespaceAndPath(ns, path + suffix + ".png"),
        };
        for (Identifier id : candidates) {
            Optional<Resource> found = resources.getResource(id);
            if (found.isEmpty()) continue;
            try (InputStream in = found.get().open()) {
                return NativeImage.read(in);
            } catch (IOException | RuntimeException e) {
                return null;
            }
        }
        return null;
    }

    @Unique
    private static void blitScaledFrame0(NativeImage src, NativeImage dst, int ox, int oy, int w, int h) {
        int sw = src.getWidth();
        int rawSh = src.getHeight();
        int sh = (rawSh > sw && sw > 0 && (rawSh % sw) == 0) ? sw : rawSh;
        for (int y = 0; y < h; y++) {
            int sy = Math.min(sh - 1, y * sh / h);
            for (int x = 0; x < w; x++) {
                int sx = Math.min(sw - 1, x * sw / w);
                dst.setPixel(ox + x, oy + y, src.getPixel(sx, sy));
            }
        }
    }
}
