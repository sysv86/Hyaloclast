use crate::gpu::{
    VulkanContext,
    buffer::{GpuBufferAllocation, PooledImageAllocation},
    retire::RetireQueue,
    with_queue_lock,
};
use ash::vk::{self, Handle};
use gpu_allocator::MemoryLocation;
use parking_lot::Mutex;
use std::{collections::HashMap, sync::Arc};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum TexturePbrRole {
    Albedo = 0,
    Normal = 1,
    Specular = 2,
    Other = 3,
}

impl TexturePbrRole {
    pub const fn from_u32(val: u32) -> Self {
        match val {
            0 => Self::Albedo,
            1 => Self::Normal,
            2 => Self::Specular,
            _ => Self::Other,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct BindlessTextureEntry {
    pub slot: u32,
    pub vk_image: u64,
    pub vk_image_view: u64,
    pub vk_sampler: u64,
    pub width: u32,
    pub height: u32,
    pub mip_levels: u32,
    pub role: TexturePbrRole,
}

/// Manages the zero-copy bindless texture descriptor table (`VK_EXT_descriptor_indexing`,
/// `PARTIALLY_BOUND_BIT | UPDATE_AFTER_BIND_BIT`) directly over `RenderPearl`'s `VulkanGpuTexture`,
/// `VulkanGpuTextureView`, and `VulkanGpuSampler` handles without duplicating texture memory on the
/// CPU or GPU.
pub struct BindlessTextureTable {
    max_slots: u32,
    free_slots: Arc<Mutex<Vec<u32>>>,
    next_fresh_slot: u32,
    by_image: HashMap<u64, BindlessTextureEntry>,
    slot_entries: Vec<Option<(u64, u64)>>,
    owned_views: HashMap<u64, (vk::ImageView, u32)>,
    pbr_companions: HashMap<u32, (u32, u32)>,
    primary_albedo_atlas_slot: Option<u32>,
    primary_albedo_atlas_dims: (u32, u32),
    primary_normal_atlas_slot: Option<u32>,
    primary_normal_atlas_dims: (u32, u32),
    primary_specular_atlas_slot: Option<u32>,
    primary_specular_atlas_dims: (u32, u32),
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    pub descriptor_pool: vk::DescriptorPool,
    pub descriptor_set: vk::DescriptorSet,
    pub nearest_sampler: vk::Sampler,
    pub linear_sampler: vk::Sampler,
    pub pbr_linear_sampler: vk::Sampler,
}

impl Default for BindlessTextureTable {
    fn default() -> Self {
        Self::new(4096)
    }
}

impl BindlessTextureTable {
    pub fn new(max_slots: u32) -> Self {
        Self {
            max_slots,
            free_slots: Arc::new(Mutex::new(Vec::with_capacity(256))),
            next_fresh_slot: 1, // Slot 0 reserved for fallback texture
            by_image: HashMap::with_capacity(256),
            slot_entries: vec![None; max_slots as usize],
            owned_views: HashMap::new(),
            pbr_companions: HashMap::new(),
            primary_albedo_atlas_slot: None,
            primary_albedo_atlas_dims: (0, 0),
            primary_normal_atlas_slot: None,
            primary_normal_atlas_dims: (0, 0),
            primary_specular_atlas_slot: None,
            primary_specular_atlas_dims: (0, 0),
            descriptor_set_layout: vk::DescriptorSetLayout::null(),
            descriptor_pool: vk::DescriptorPool::null(),
            descriptor_set: vk::DescriptorSet::null(),
            nearest_sampler: vk::Sampler::null(),
            linear_sampler: vk::Sampler::null(),
            pbr_linear_sampler: vk::Sampler::null(),
        }
    }

    /// Allocates the hardware `VkDescriptorSetLayout`, `VkDescriptorPool`, `VkDescriptorSet`,
    /// and default samplers when `VK_EXT_descriptor_indexing` is supported by the `VkDevice`.
    pub fn init_gpu_resources(&mut self, ctx: &VulkanContext) -> Result<(), vk::Result> {
        if !ctx.capabilities.descriptor_indexing || self.descriptor_set != vk::DescriptorSet::null()
        {
            return Ok(());
        }

        unsafe {
            let nearest_info = vk::SamplerCreateInfo::default()
                .mag_filter(vk::Filter::NEAREST)
                .min_filter(vk::Filter::NEAREST)
                .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
                .address_mode_u(vk::SamplerAddressMode::REPEAT)
                .address_mode_v(vk::SamplerAddressMode::REPEAT)
                .address_mode_w(vk::SamplerAddressMode::REPEAT)
                .max_lod(16.0);
            self.nearest_sampler = ctx.device.create_sampler(&nearest_info, None)?;

            let linear_info = vk::SamplerCreateInfo::default()
                .mag_filter(vk::Filter::LINEAR)
                .min_filter(vk::Filter::LINEAR)
                .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
                .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .max_lod(16.0);
            self.linear_sampler = ctx.device.create_sampler(&linear_info, None)?;

            let pbr_linear_info = vk::SamplerCreateInfo::default()
                .mag_filter(vk::Filter::LINEAR)
                .min_filter(vk::Filter::LINEAR)
                .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
                .address_mode_u(vk::SamplerAddressMode::REPEAT)
                .address_mode_v(vk::SamplerAddressMode::REPEAT)
                .address_mode_w(vk::SamplerAddressMode::REPEAT)
                .max_lod(16.0);
            self.pbr_linear_sampler = ctx.device.create_sampler(&pbr_linear_info, None)?;

            let binding_flags = [vk::DescriptorBindingFlags::PARTIALLY_BOUND
                | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND
                | vk::DescriptorBindingFlags::UPDATE_UNUSED_WHILE_PENDING];
            let mut flags_info = vk::DescriptorSetLayoutBindingFlagsCreateInfo::default()
                .binding_flags(&binding_flags);

            let mut stage_flags = vk::ShaderStageFlags::VERTEX
                | vk::ShaderStageFlags::FRAGMENT
                | vk::ShaderStageFlags::COMPUTE;
            if ctx.capabilities.ray_tracing_pipeline {
                stage_flags |= vk::ShaderStageFlags::RAYGEN_KHR
                    | vk::ShaderStageFlags::CLOSEST_HIT_KHR
                    | vk::ShaderStageFlags::ANY_HIT_KHR
                    | vk::ShaderStageFlags::MISS_KHR;
            }
            let binding = [vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(self.max_slots)
                .stage_flags(stage_flags)];

            let layout_info = vk::DescriptorSetLayoutCreateInfo::default()
                .flags(vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL)
                .bindings(&binding)
                .push_next(&mut flags_info);
            self.descriptor_set_layout = ctx
                .device
                .create_descriptor_set_layout(&layout_info, None)?;

            let pool_sizes = [vk::DescriptorPoolSize {
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: self.max_slots,
            }];
            let pool_info = vk::DescriptorPoolCreateInfo::default()
                .flags(vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND)
                .max_sets(1)
                .pool_sizes(&pool_sizes);
            self.descriptor_pool = ctx.device.create_descriptor_pool(&pool_info, None)?;

            let set_layouts = [self.descriptor_set_layout];
            let alloc_info = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(self.descriptor_pool)
                .set_layouts(&set_layouts);
            let sets = ctx.device.allocate_descriptor_sets(&alloc_info)?;
            self.descriptor_set = sets[0];
        }

        Ok(())
    }

    // Mirrors the fields in a pack texture descriptor.
    #[allow(clippy::too_many_arguments)]
    pub fn register_or_update(
        &mut self,
        vk_image: u64,
        vk_image_view: u64,
        width: u32,
        height: u32,
        mip_levels: u32,
        role: TexturePbrRole,
        ctx: Option<&VulkanContext>,
    ) -> Option<u32> {
        self.register_or_update_with_sampler(
            vk_image,
            vk_image_view,
            0,
            vk::Format::R8G8B8A8_UNORM,
            width,
            height,
            mip_levels,
            role,
            ctx,
        )
    }

    // Mirrors format, dimensions, and sampler fields in the renderer texture descriptor.
    #[allow(clippy::too_many_arguments)]
    pub fn register_or_update_with_sampler(
        &mut self,
        vk_image: u64,
        vk_image_view: u64,
        vk_sampler: u64,
        format: vk::Format,
        width: u32,
        height: u32,
        mip_levels: u32,
        role: TexturePbrRole,
        ctx: Option<&VulkanContext>,
    ) -> Option<u32> {
        self.register_or_update_with_sampler_deferred(
            vk_image,
            vk_image_view,
            vk_sampler,
            format,
            width,
            height,
            mip_levels,
            role,
            ctx,
            None,
        )
    }

    // Preserves the deferred texture upload ABI used by the Kotlin bridge.
    #[allow(clippy::too_many_arguments)]
    pub fn register_or_update_with_sampler_deferred(
        &mut self,
        vk_image: u64,
        mut vk_image_view: u64,
        vk_sampler: u64,
        format: vk::Format,
        width: u32,
        height: u32,
        mip_levels: u32,
        role: TexturePbrRole,
        ctx: Option<&VulkanContext>,
        retire_queue: Option<&RetireQueue>,
    ) -> Option<u32> {
        if vk_image == 0 {
            return None;
        }

        // If the caller did not supply a VkImageView handle, create or reuse an engine-owned 2D ImageView
        let desired_mips = mip_levels.max(1);
        if vk_image_view == 0 {
            let cached = self.owned_views.get(&vk_image).copied();
            if let Some((cached_view, cached_mips)) = cached
                && cached_mips == desired_mips
            {
                vk_image_view = cached_view.as_raw();
            } else if let Some(c) = ctx
                && format != vk::Format::UNDEFINED
            {
                let is_depth = matches!(
                    format,
                    vk::Format::D32_SFLOAT
                        | vk::Format::D24_UNORM_S8_UINT
                        | vk::Format::D32_SFLOAT_S8_UINT
                        | vk::Format::D16_UNORM
                );
                let view_info = vk::ImageViewCreateInfo::default()
                    .image(vk::Image::from_raw(vk_image))
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(format)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: if is_depth {
                            vk::ImageAspectFlags::DEPTH
                        } else {
                            vk::ImageAspectFlags::COLOR
                        },
                        base_mip_level: 0,
                        level_count: desired_mips,
                        base_array_layer: 0,
                        layer_count: 1,
                    });
                if let Ok(created_view) = unsafe { c.device.create_image_view(&view_info, None) } {
                    vk_image_view = created_view.as_raw();
                    if let Some((old_view, _)) = self
                        .owned_views
                        .insert(vk_image, (created_view, desired_mips))
                        && old_view != vk::ImageView::null()
                    {
                        if let Some(rq) = retire_queue {
                            rq.retire_image_view(c.device.clone(), old_view);
                        } else {
                            unsafe {
                                c.device.destroy_image_view(old_view, None);
                            }
                        }
                    }
                } else if let Some((cached_view, _)) = cached {
                    vk_image_view = cached_view.as_raw();
                }
            } else if let Some((cached_view, _)) = cached {
                vk_image_view = cached_view.as_raw();
            }
        }

        let slot = if let Some(existing) = self.by_image.get_mut(&vk_image) {
            if vk_image_view != 0 {
                existing.vk_image_view = vk_image_view;
            }
            if vk_sampler != 0 {
                existing.vk_sampler = vk_sampler;
            }
            existing.width = width;
            existing.height = height;
            existing.mip_levels = mip_levels;
            existing.role = role;
            existing.slot
        } else {
            let s = if let Some(reused) = self.free_slots.lock().pop() {
                reused
            } else if self.next_fresh_slot < self.max_slots {
                let fresh = self.next_fresh_slot;
                self.next_fresh_slot += 1;
                fresh
            } else {
                return None;
            };

            let entry = BindlessTextureEntry {
                slot: s,
                vk_image,
                vk_image_view,
                vk_sampler,
                width,
                height,
                mip_levels,
                role,
            };
            self.by_image.insert(vk_image, entry);
            s
        };

        // Track primary block atlas slots (largest registered atlas for each PBR role)
        if width >= 512 && height >= 512 {
            let area = (width as u64) * (height as u64);
            match role {
                TexturePbrRole::Albedo => {
                    let cur_area = (self.primary_albedo_atlas_dims.0 as u64)
                        * (self.primary_albedo_atlas_dims.1 as u64);
                    if self.primary_albedo_atlas_slot.is_none()
                        || self.primary_albedo_atlas_slot == Some(slot)
                        || area > cur_area
                    {
                        self.primary_albedo_atlas_slot = Some(slot);
                        self.primary_albedo_atlas_dims = (width, height);
                    }
                }
                TexturePbrRole::Normal => {
                    let cur_area = (self.primary_normal_atlas_dims.0 as u64)
                        * (self.primary_normal_atlas_dims.1 as u64);
                    if self.primary_normal_atlas_slot.is_none()
                        || self.primary_normal_atlas_slot == Some(slot)
                        || area > cur_area
                    {
                        self.primary_normal_atlas_slot = Some(slot);
                        self.primary_normal_atlas_dims = (width, height);
                    }
                }
                TexturePbrRole::Specular => {
                    let cur_area = (self.primary_specular_atlas_dims.0 as u64)
                        * (self.primary_specular_atlas_dims.1 as u64);
                    if self.primary_specular_atlas_slot.is_none()
                        || self.primary_specular_atlas_slot == Some(slot)
                        || area > cur_area
                    {
                        self.primary_specular_atlas_slot = Some(slot);
                        self.primary_specular_atlas_dims = (width, height);
                    }
                }
                TexturePbrRole::Other => {}
            }
        }

        // Write VkDescriptorImageInfo directly into the bindless descriptor array
        let resolved_view = self
            .by_image
            .get(&vk_image)
            .map(|e| e.vk_image_view)
            .unwrap_or(vk_image_view);
        let resolved_sampler = self
            .by_image
            .get(&vk_image)
            .map(|e| e.vk_sampler)
            .unwrap_or(vk_sampler);
        if (slot as usize) < self.slot_entries.len() {
            self.slot_entries[slot as usize] = Some((resolved_view, resolved_sampler));
        }

        if resolved_view != 0
            && self.descriptor_set != vk::DescriptorSet::null()
            && let Some(c) = ctx
        {
            let sampler_handle = if resolved_sampler != 0 {
                vk::Sampler::from_raw(resolved_sampler)
            } else if matches!(role, TexturePbrRole::Normal | TexturePbrRole::Specular)
                && self.pbr_linear_sampler != vk::Sampler::null()
            {
                self.pbr_linear_sampler
            } else if self.nearest_sampler != vk::Sampler::null() {
                self.nearest_sampler
            } else {
                vk::Sampler::null()
            };

            if sampler_handle != vk::Sampler::null() {
                let img_info = [vk::DescriptorImageInfo {
                    sampler: sampler_handle,
                    image_view: vk::ImageView::from_raw(resolved_view),
                    image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                }];
                let write = [vk::WriteDescriptorSet::default()
                    .dst_set(self.descriptor_set)
                    .dst_binding(0)
                    .dst_array_element(slot)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&img_info)];
                unsafe {
                    c.device.update_descriptor_sets(&write, &[]);
                }

                // Also populate slot 0 as fallback if this is the primary albedo atlas
                if self.primary_albedo_atlas_slot == Some(slot) {
                    if !self.slot_entries.is_empty() {
                        self.slot_entries[0] = Some((resolved_view, resolved_sampler));
                    }
                    let write_fallback = [vk::WriteDescriptorSet::default()
                        .dst_set(self.descriptor_set)
                        .dst_binding(0)
                        .dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .image_info(&img_info)];
                    unsafe {
                        c.device.update_descriptor_sets(&write_fallback, &[]);
                    }
                }
            }
        }

        Some(slot)
    }

    pub fn unregister(&mut self, vk_image: u64) -> Option<BindlessTextureEntry> {
        self.unregister_deferred(vk_image, None, None)
    }

    /// Removes `vk_image` from the active table and defers returning its slot index (and destroying
    /// any engine-owned `VkImageView`) until in-flight GPU frames finish via `RetireQueue`.
    pub fn unregister_deferred(
        &mut self,
        vk_image: u64,
        ctx: Option<&VulkanContext>,
        retire_queue: Option<&RetireQueue>,
    ) -> Option<BindlessTextureEntry> {
        let removed = self.by_image.remove(&vk_image)?;
        if (removed.slot as usize) < self.slot_entries.len() {
            self.slot_entries[removed.slot as usize] = None;
        }
        let was_primary_albedo = self.primary_albedo_atlas_slot == Some(removed.slot);
        if was_primary_albedo {
            self.primary_albedo_atlas_slot = None;
            self.primary_albedo_atlas_dims = (0, 0);
            if !self.slot_entries.is_empty() {
                self.slot_entries[0] = None;
            }
            // Re-point slot 0 to another valid texture if one exists so slot 0 never references a destroyed view.
            let replacement = self
                .by_image
                .values()
                .filter(|e| e.vk_image_view != 0)
                .max_by_key(|e| {
                    (
                        e.role == TexturePbrRole::Albedo,
                        (e.width as u64) * (e.height as u64),
                    )
                })
                .copied();
            if let Some(rep) = replacement {
                if rep.role == TexturePbrRole::Albedo && rep.width >= 512 && rep.height >= 512 {
                    self.primary_albedo_atlas_slot = Some(rep.slot);
                    self.primary_albedo_atlas_dims = (rep.width, rep.height);
                }
                if !self.slot_entries.is_empty() {
                    self.slot_entries[0] = Some((rep.vk_image_view, rep.vk_sampler));
                }
                if let Some(c) = ctx
                    && self.descriptor_set != vk::DescriptorSet::null()
                {
                    let sampler_handle = if rep.vk_sampler != 0 {
                        vk::Sampler::from_raw(rep.vk_sampler)
                    } else {
                        self.nearest_sampler
                    };
                    if sampler_handle != vk::Sampler::null() {
                        let img_info = [vk::DescriptorImageInfo {
                            sampler: sampler_handle,
                            image_view: vk::ImageView::from_raw(rep.vk_image_view),
                            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                        }];
                        let write_fallback = [vk::WriteDescriptorSet::default()
                            .dst_set(self.descriptor_set)
                            .dst_binding(0)
                            .dst_array_element(0)
                            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                            .image_info(&img_info)];
                        unsafe {
                            c.device.update_descriptor_sets(&write_fallback, &[]);
                        }
                    }
                }
            }
        }
        if self.primary_normal_atlas_slot == Some(removed.slot) {
            if let Some(rep) = self
                .by_image
                .values()
                .filter(|e| e.role == TexturePbrRole::Normal && e.width >= 512 && e.height >= 512)
                .max_by_key(|e| (e.width as u64) * (e.height as u64))
                .copied()
            {
                self.primary_normal_atlas_slot = Some(rep.slot);
                self.primary_normal_atlas_dims = (rep.width, rep.height);
            } else {
                self.primary_normal_atlas_slot = None;
                self.primary_normal_atlas_dims = (0, 0);
            }
        }
        if self.primary_specular_atlas_slot == Some(removed.slot) {
            if let Some(rep) = self
                .by_image
                .values()
                .filter(|e| e.role == TexturePbrRole::Specular && e.width >= 512 && e.height >= 512)
                .max_by_key(|e| (e.width as u64) * (e.height as u64))
                .copied()
            {
                self.primary_specular_atlas_slot = Some(rep.slot);
                self.primary_specular_atlas_dims = (rep.width, rep.height);
            } else {
                self.primary_specular_atlas_slot = None;
                self.primary_specular_atlas_dims = (0, 0);
            }
        }

        self.pbr_companions.remove(&removed.slot);
        self.pbr_companions.retain(|_, pair| {
            if pair.0 == removed.slot {
                pair.0 = 0;
            }
            if pair.1 == removed.slot {
                pair.1 = 0;
            }
            pair.0 != 0 || pair.1 != 0
        });

        let owned_view = self.owned_views.remove(&vk_image).map(|(view, _)| view);
        let free_slots = Arc::clone(&self.free_slots);
        let slot = removed.slot;

        if let (Some(c), Some(rq)) = (ctx, retire_queue) {
            let dev = c.device.clone();
            rq.retire_later(move || {
                if let Some(view) = owned_view {
                    unsafe {
                        dev.destroy_image_view(view, None);
                    }
                }
                free_slots.lock().push(slot);
            });
        } else {
            if let (Some(c), Some(view)) = (ctx, owned_view) {
                unsafe {
                    c.device.destroy_image_view(view, None);
                }
            }
            free_slots.lock().push(slot);
        }

        Some(removed)
    }

    /// Binds this bindless descriptor set to `cmd` at `set_index` (typically Set 1).
    ///
    /// # Safety
    /// `cmd` must be recording and `pipeline_layout` must include `self.descriptor_set_layout` at `set_index`.
    pub unsafe fn cmd_bind(
        &self,
        ctx: &VulkanContext,
        cmd: vk::CommandBuffer,
        bind_point: vk::PipelineBindPoint,
        pipeline_layout: vk::PipelineLayout,
        set_index: u32,
    ) {
        if self.descriptor_set == vk::DescriptorSet::null()
            || pipeline_layout == vk::PipelineLayout::null()
        {
            return;
        }
        let sets = [self.descriptor_set];
        unsafe {
            ctx.device.cmd_bind_descriptor_sets(
                cmd,
                bind_point,
                pipeline_layout,
                set_index,
                &sets,
                &[],
            );
        }
    }

    pub fn lookup_slot(&self, vk_image: u64) -> Option<u32> {
        self.by_image.get(&vk_image).map(|e| e.slot)
    }

    pub fn lookup_entry(&self, vk_image: u64) -> Option<BindlessTextureEntry> {
        self.by_image.get(&vk_image).copied()
    }

    /// Returns the `(image_view, sampler)` currently occupying `slot`, if any. Used by the CSM
    /// shadow pass to bind the primary block atlas as its sampler2D without inventing a handle.
    pub fn view_sampler_for_slot(&self, slot: u32) -> Option<(vk::ImageView, vk::Sampler)> {
        let (view_raw, sampler_raw) = self.slot_entries.get(slot as usize).copied().flatten()?;
        if view_raw == 0 {
            return None;
        }
        Some((
            vk::ImageView::from_raw(view_raw),
            if sampler_raw != 0 {
                vk::Sampler::from_raw(sampler_raw)
            } else {
                vk::Sampler::null()
            },
        ))
    }

    pub fn active_count(&self) -> usize {
        self.by_image.len()
    }

    pub fn primary_pbr_slots(&self) -> (u32, u32, u32) {
        (
            self.primary_albedo_atlas_slot.unwrap_or(0),
            self.primary_normal_atlas_slot.unwrap_or(0),
            self.primary_specular_atlas_slot.unwrap_or(0),
        )
    }

    /// Links companion LabPBR normal and specular bindless slots to an albedo slot.
    pub fn link_pbr_companions(&mut self, albedo_slot: u32, normal_slot: u32, specular_slot: u32) {
        if albedo_slot == 0 {
            return;
        }
        if normal_slot == 0 && specular_slot == 0 {
            self.pbr_companions.remove(&albedo_slot);
        } else {
            self.pbr_companions
                .insert(albedo_slot, (normal_slot, specular_slot));
        }
    }

    /// Returns `(normal_slot, specular_slot)` linked to `albedo_slot`, or `(0, 0)` if none.
    pub fn companions_for_albedo(&self, albedo_slot: u32) -> (u32, u32) {
        self.pbr_companions
            .get(&albedo_slot)
            .copied()
            .unwrap_or((0, 0))
    }

    pub fn destroy(&mut self, ctx: &VulkanContext) {
        self.pbr_companions.clear();
        unsafe {
            for (_, (view, _)) in self.owned_views.drain() {
                if view != vk::ImageView::null() {
                    ctx.device.destroy_image_view(view, None);
                }
            }
            if self.descriptor_pool != vk::DescriptorPool::null() {
                ctx.device
                    .destroy_descriptor_pool(self.descriptor_pool, None);
                self.descriptor_pool = vk::DescriptorPool::null();
                self.descriptor_set = vk::DescriptorSet::null();
            }
            if self.descriptor_set_layout != vk::DescriptorSetLayout::null() {
                ctx.device
                    .destroy_descriptor_set_layout(self.descriptor_set_layout, None);
                self.descriptor_set_layout = vk::DescriptorSetLayout::null();
            }
            if self.nearest_sampler != vk::Sampler::null() {
                ctx.device.destroy_sampler(self.nearest_sampler, None);
                self.nearest_sampler = vk::Sampler::null();
            }
            if self.linear_sampler != vk::Sampler::null() {
                ctx.device.destroy_sampler(self.linear_sampler, None);
                self.linear_sampler = vk::Sampler::null();
            }
            if self.pbr_linear_sampler != vk::Sampler::null() {
                ctx.device.destroy_sampler(self.pbr_linear_sampler, None);
                self.pbr_linear_sampler = vk::Sampler::null();
            }
        }
    }
}

/// Maximum number of pack-declared `[resources.*]` textures addressable through Set 3.
pub const PACK_RESOURCE_SLOTS: u32 = 16;

struct PackTexture {
    image: vk::Image,
    memory: vk::DeviceMemory,
    pooled: Option<PooledImageAllocation>,
    view: vk::ImageView,
    sampler: vk::Sampler,
}

/// Owns the `VkImage` / `VkImageView` / `VkSampler` objects uploaded from a shader pack's
/// `[resources.*]` PNG files (e.g. `blue_noise_64x64.png`) and exposes the single descriptor
/// set bound at descriptor **Set 3** of every compute / ray-tracing pipeline.
pub struct PackTextureTable {
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    pub descriptor_pool: vk::DescriptorPool,
    pub descriptor_set: vk::DescriptorSet,
    fallback_view: vk::ImageView,
    fallback_image: vk::Image,
    fallback_memory: vk::DeviceMemory,
    fallback_pooled: Option<PooledImageAllocation>,
    fallback_sampler: vk::Sampler,
    textures: HashMap<u32, PackTexture>,
}

impl Default for PackTextureTable {
    fn default() -> Self {
        Self::new()
    }
}

impl PackTextureTable {
    pub fn new() -> Self {
        Self {
            descriptor_set_layout: vk::DescriptorSetLayout::null(),
            descriptor_pool: vk::DescriptorPool::null(),
            descriptor_set: vk::DescriptorSet::null(),
            fallback_view: vk::ImageView::null(),
            fallback_image: vk::Image::null(),
            fallback_memory: vk::DeviceMemory::null(),
            fallback_pooled: None,
            fallback_sampler: vk::Sampler::null(),
            textures: HashMap::new(),
        }
    }

    /// Creates the Set 3 descriptor layout (`uVsPackRes_<name>`, bindings `0..PACK_RESOURCE_SLOTS`)
    /// plus a 1x1 opaque-white fallback image so that every unbound slot is still a valid sampler.
    pub fn init_gpu_resources(&mut self, ctx: &VulkanContext) -> Result<(), vk::Result> {
        if self.descriptor_set != vk::DescriptorSet::null() {
            return Ok(());
        }
        let init_res: Result<(), vk::Result> = (|| unsafe {
            // 1x1 RGBA8 white fallback texture.
            let img_info = vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(vk::Format::R8G8B8A8_UNORM)
                .extent(vk::Extent3D {
                    width: 1,
                    height: 1,
                    depth: 1,
                })
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(
                    vk::ImageUsageFlags::TRANSFER_DST
                        | vk::ImageUsageFlags::SAMPLED
                        | vk::ImageUsageFlags::TRANSFER_SRC,
                )
                .initial_layout(vk::ImageLayout::UNDEFINED);
            if let Some(ref allocator) = ctx.memory_allocator {
                let pooled = allocator.allocate_image(
                    &ctx.device,
                    &img_info,
                    MemoryLocation::GpuOnly,
                    "pack_fallback_1x1",
                )?;
                self.fallback_image = pooled.image;
                self.fallback_memory = vk::DeviceMemory::null();
                self.fallback_pooled = Some(pooled);
            } else {
                self.fallback_image = ctx.device.create_image(&img_info, None)?;
                let mem_req = ctx
                    .device
                    .get_image_memory_requirements(self.fallback_image);
                let props = ctx
                    .instance
                    .get_physical_device_memory_properties(ctx.physical_device);
                let mem_idx = (0..props.memory_type_count).find(|&i| {
                    (mem_req.memory_type_bits & (1 << i)) != 0
                        && props.memory_types[i as usize]
                            .property_flags
                            .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
                });
                let Some(mem_idx) = mem_idx else {
                    return Err(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY);
                };
                let alloc = vk::MemoryAllocateInfo::default()
                    .allocation_size(mem_req.size)
                    .memory_type_index(mem_idx);
                self.fallback_memory = ctx.device.allocate_memory(&alloc, None)?;
                ctx.device
                    .bind_image_memory(self.fallback_image, self.fallback_memory, 0)?;
                self.fallback_pooled = None;
            }
            let view_info = vk::ImageViewCreateInfo::default()
                .image(self.fallback_image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(vk::Format::R8G8B8A8_UNORM)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });
            self.fallback_view = ctx.device.create_image_view(&view_info, None)?;
            let sampler_info = vk::SamplerCreateInfo::default()
                .mag_filter(vk::Filter::NEAREST)
                .min_filter(vk::Filter::NEAREST)
                .address_mode_u(vk::SamplerAddressMode::REPEAT)
                .address_mode_v(vk::SamplerAddressMode::REPEAT);
            self.fallback_sampler = ctx.device.create_sampler(&sampler_info, None)?;

            // White clear of fallback image via a transient transfer command. The image starts
            // UNDEFINED; a clear via cmd_clear_color_image needs TRANSFER_DST + GENERAL/CLEAR layout.
            self.transition_and_clear_fallback(ctx)?;

            let mut stage_flags = vk::ShaderStageFlags::COMPUTE
                | vk::ShaderStageFlags::FRAGMENT
                | vk::ShaderStageFlags::VERTEX;
            if ctx.capabilities.ray_tracing_pipeline {
                stage_flags |= vk::ShaderStageFlags::RAYGEN_KHR
                    | vk::ShaderStageFlags::CLOSEST_HIT_KHR
                    | vk::ShaderStageFlags::ANY_HIT_KHR
                    | vk::ShaderStageFlags::MISS_KHR;
            }
            let bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..PACK_RESOURCE_SLOTS)
                .map(|i| {
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(i)
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .descriptor_count(1)
                        .stage_flags(stage_flags)
                })
                .collect();
            let binding_flags = vec![
                vk::DescriptorBindingFlags::PARTIALLY_BOUND
                    | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND
                    | vk::DescriptorBindingFlags::UPDATE_UNUSED_WHILE_PENDING;
                PACK_RESOURCE_SLOTS as usize
            ];
            let mut flags_info = vk::DescriptorSetLayoutBindingFlagsCreateInfo::default()
                .binding_flags(&binding_flags);
            let mut layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
            if ctx.capabilities.descriptor_indexing {
                layout_info = layout_info
                    .flags(vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL)
                    .push_next(&mut flags_info);
            }
            self.descriptor_set_layout = ctx
                .device
                .create_descriptor_set_layout(&layout_info, None)?;

            let pool_size = [vk::DescriptorPoolSize {
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: PACK_RESOURCE_SLOTS,
            }];
            let mut pool_info = vk::DescriptorPoolCreateInfo::default()
                .max_sets(1)
                .pool_sizes(&pool_size);
            if ctx.capabilities.descriptor_indexing {
                pool_info = pool_info.flags(vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND);
            }
            self.descriptor_pool = ctx.device.create_descriptor_pool(&pool_info, None)?;
            let layouts = [self.descriptor_set_layout];
            let alloc_info = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(self.descriptor_pool)
                .set_layouts(&layouts);
            self.descriptor_set = ctx.device.allocate_descriptor_sets(&alloc_info)?[0];

            // Point every slot at the white fallback until a real pack texture is uploaded.
            let imgs: Vec<vk::DescriptorImageInfo> = (0..PACK_RESOURCE_SLOTS)
                .map(|_| vk::DescriptorImageInfo {
                    sampler: self.fallback_sampler,
                    image_view: self.fallback_view,
                    image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                })
                .collect();
            let writes: Vec<vk::WriteDescriptorSet> = (0..PACK_RESOURCE_SLOTS)
                .map(|i| {
                    vk::WriteDescriptorSet::default()
                        .dst_set(self.descriptor_set)
                        .dst_binding(i)
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .image_info(&imgs[i as usize..i as usize + 1])
                })
                .collect();
            ctx.device.update_descriptor_sets(&writes, &[]);
            Ok(())
        })();
        if let Err(err) = init_res {
            self.destroy(ctx);
            return Err(err);
        }
        Ok(())
    }

    unsafe fn transition_and_clear_fallback(
        &mut self,
        ctx: &VulkanContext,
    ) -> Result<(), vk::Result> {
        let pool_info = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::TRANSIENT)
            .queue_family_index(ctx.graphics_queue_family_index);
        let pool = unsafe { ctx.device.create_command_pool(&pool_info, None)? };
        let alloc = vk::CommandBufferAllocateInfo::default()
            .command_pool(pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let cmd = match unsafe { ctx.device.allocate_command_buffers(&alloc) } {
            Ok(cmds) => cmds[0],
            Err(e) => {
                unsafe { ctx.device.destroy_command_pool(pool, None) };
                return Err(e);
            }
        };
        let fence_info = vk::FenceCreateInfo::default();
        let fence = match unsafe { ctx.device.create_fence(&fence_info, None) } {
            Ok(f) => f,
            Err(e) => {
                unsafe { ctx.device.destroy_command_pool(pool, None) };
                return Err(e);
            }
        };
        with_queue_lock(|| unsafe {
            let begin = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            let _ = ctx.device.begin_command_buffer(cmd, &begin);
            let barrier = [vk::ImageMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .image(self.fallback_image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                })];
            ctx.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &barrier,
            );
            ctx.device.cmd_clear_color_image(
                cmd,
                self.fallback_image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &vk::ClearColorValue {
                    float32: [1.0, 1.0, 1.0, 1.0],
                },
                &[vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                }],
            );
            let barrier2 = [vk::ImageMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ)
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image(self.fallback_image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                })];
            ctx.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &barrier2,
            );
            let _ = ctx.device.end_command_buffer(cmd);
            let cmd_arr = [cmd];
            let submits = [vk::SubmitInfo::default().command_buffers(&cmd_arr)];
            let _ = ctx.device.queue_submit(ctx.graphics_queue, &submits, fence);
        });
        unsafe {
            let _ = ctx.device.wait_for_fences(&[fence], true, 5_000_000_000);
        }
        unsafe {
            ctx.device.destroy_fence(fence, None);
            ctx.device.destroy_command_pool(pool, None);
        }
        Ok(())
    }

    /// Uploads an RGBA8 pack resource into `binding = slot` of Set 3. Returns `0` on success and a
    /// negative Vulkan result code on failure.
    ///
    /// # Safety
    /// `ctx` and its Vulkan device and graphics queue must remain valid for the duration of upload.
    // Preserves the texture upload ABI used by the Kotlin bridge.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn upload_rgba(
        &mut self,
        ctx: &VulkanContext,
        slot: u32,
        width: u32,
        height: u32,
        rgba: &[u8],
        filter_linear: bool,
        wrap_repeat: bool,
    ) -> i32 {
        unsafe {
            self.upload_rgba_deferred(
                ctx,
                None,
                slot,
                width,
                height,
                rgba,
                filter_linear,
                wrap_repeat,
            )
        }
    }

    /// # Safety
    /// `ctx` and `retire` must refer to the same live Vulkan device for the duration of upload.
    // Preserves the deferred texture upload ABI used by the Kotlin bridge.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn upload_rgba_deferred(
        &mut self,
        ctx: &VulkanContext,
        retire: Option<&crate::RetireQueue>,
        slot: u32,
        width: u32,
        height: u32,
        rgba: &[u8],
        filter_linear: bool,
        wrap_repeat: bool,
    ) -> i32 {
        if slot >= PACK_RESOURCE_SLOTS || width == 0 || height == 0 {
            return -1;
        }
        if self.descriptor_set == vk::DescriptorSet::null() && self.init_gpu_resources(ctx).is_err()
        {
            return -2;
        }
        let expected = (width as usize) * (height as usize) * 4;
        if rgba.len() < expected {
            return -3;
        }
        unsafe {
            let mut staging = match GpuBufferAllocation::allocate_host_visible_bda(
                ctx,
                expected as vk::DeviceSize,
                vk::BufferUsageFlags::TRANSFER_SRC,
            ) {
                Ok(s) => s,
                Err(e) => return e.as_raw(),
            };
            staging.write_bytes(0, &rgba[..expected]);

            let img_info = vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(vk::Format::R8G8B8A8_UNORM)
                .extent(vk::Extent3D {
                    width,
                    height,
                    depth: 1,
                })
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(
                    vk::ImageUsageFlags::TRANSFER_DST
                        | vk::ImageUsageFlags::SAMPLED
                        | vk::ImageUsageFlags::TRANSFER_SRC,
                )
                .initial_layout(vk::ImageLayout::UNDEFINED);
            let (image, memory, mut pooled) = if let Some(ref allocator) = ctx.memory_allocator {
                match allocator.allocate_image(
                    &ctx.device,
                    &img_info,
                    MemoryLocation::GpuOnly,
                    "pack_texture",
                ) {
                    Ok(p) => {
                        let img = p.image;
                        (img, vk::DeviceMemory::null(), Some(p))
                    }
                    Err(e) => {
                        staging.destroy(&ctx.device);
                        return e.as_raw();
                    }
                }
            } else {
                let image = match ctx.device.create_image(&img_info, None) {
                    Ok(i) => i,
                    Err(e) => {
                        staging.destroy(&ctx.device);
                        return e.as_raw();
                    }
                };
                let req = ctx.device.get_image_memory_requirements(image);
                let props = ctx
                    .instance
                    .get_physical_device_memory_properties(ctx.physical_device);
                let mem_idx = (0..props.memory_type_count).find(|&i| {
                    (req.memory_type_bits & (1 << i)) != 0
                        && props.memory_types[i as usize]
                            .property_flags
                            .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
                });
                let Some(mem_idx) = mem_idx else {
                    ctx.device.destroy_image(image, None);
                    staging.destroy(&ctx.device);
                    return vk::Result::ERROR_OUT_OF_DEVICE_MEMORY.as_raw();
                };
                let memory = match ctx.device.allocate_memory(
                    &vk::MemoryAllocateInfo::default()
                        .allocation_size(req.size)
                        .memory_type_index(mem_idx),
                    None,
                ) {
                    Ok(m) => m,
                    Err(e) => {
                        ctx.device.destroy_image(image, None);
                        staging.destroy(&ctx.device);
                        return e.as_raw();
                    }
                };
                if ctx.device.bind_image_memory(image, memory, 0).is_err() {
                    ctx.device.free_memory(memory, None);
                    ctx.device.destroy_image(image, None);
                    staging.destroy(&ctx.device);
                    return vk::Result::ERROR_UNKNOWN.as_raw();
                }
                (image, memory, None)
            };

            let cleanup_image = |pooled: &mut Option<PooledImageAllocation>| {
                if let Some(mut p) = pooled.take() {
                    p.destroy(&ctx.device);
                } else {
                    if memory != vk::DeviceMemory::null() {
                        ctx.device.free_memory(memory, None);
                    }
                    if image != vk::Image::null() {
                        ctx.device.destroy_image(image, None);
                    }
                }
            };

            let view_info = vk::ImageViewCreateInfo::default()
                .image(image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(vk::Format::R8G8B8A8_UNORM)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });
            let view = match ctx.device.create_image_view(&view_info, None) {
                Ok(v) => v,
                Err(e) => {
                    cleanup_image(&mut pooled);
                    staging.destroy(&ctx.device);
                    return e.as_raw();
                }
            };
            let address_mode = if wrap_repeat {
                vk::SamplerAddressMode::REPEAT
            } else {
                vk::SamplerAddressMode::CLAMP_TO_EDGE
            };
            let filter = if filter_linear {
                vk::Filter::LINEAR
            } else {
                vk::Filter::NEAREST
            };
            let sampler_info = vk::SamplerCreateInfo::default()
                .mag_filter(filter)
                .min_filter(filter)
                .address_mode_u(address_mode)
                .address_mode_v(address_mode)
                .address_mode_w(address_mode);
            let sampler = match ctx.device.create_sampler(&sampler_info, None) {
                Ok(s) => s,
                Err(e) => {
                    ctx.device.destroy_image_view(view, None);
                    cleanup_image(&mut pooled);
                    staging.destroy(&ctx.device);
                    return e.as_raw();
                }
            };

            let pool_info = vk::CommandPoolCreateInfo::default()
                .flags(vk::CommandPoolCreateFlags::TRANSIENT)
                .queue_family_index(ctx.graphics_queue_family_index);
            let pool = match ctx.device.create_command_pool(&pool_info, None) {
                Ok(p) => p,
                Err(e) => {
                    ctx.device.destroy_sampler(sampler, None);
                    ctx.device.destroy_image_view(view, None);
                    cleanup_image(&mut pooled);
                    staging.destroy(&ctx.device);
                    return e.as_raw();
                }
            };
            let alloc = vk::CommandBufferAllocateInfo::default()
                .command_pool(pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            let cmd = match ctx.device.allocate_command_buffers(&alloc) {
                Ok(c) => c[0],
                Err(e) => {
                    ctx.device.destroy_command_pool(pool, None);
                    ctx.device.destroy_sampler(sampler, None);
                    ctx.device.destroy_image_view(view, None);
                    cleanup_image(&mut pooled);
                    staging.destroy(&ctx.device);
                    return e.as_raw();
                }
            };
            let fence = match ctx
                .device
                .create_fence(&vk::FenceCreateInfo::default(), None)
            {
                Ok(f) => f,
                Err(e) => {
                    ctx.device.destroy_command_pool(pool, None);
                    ctx.device.destroy_sampler(sampler, None);
                    ctx.device.destroy_image_view(view, None);
                    cleanup_image(&mut pooled);
                    staging.destroy(&ctx.device);
                    return e.as_raw();
                }
            };
            with_queue_lock(|| {
                let begin = vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
                let _ = ctx.device.begin_command_buffer(cmd, &begin);
                let subrange = vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                };
                let pre = [vk::ImageMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::empty())
                    .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .image(image)
                    .subresource_range(subrange)];
                let copy = [vk::BufferImageCopy::default()
                    .buffer_offset(0)
                    .buffer_row_length(width)
                    .buffer_image_height(height)
                    .image_subresource(vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: 0,
                        base_array_layer: 0,
                        layer_count: 1,
                    })
                    .image_extent(vk::Extent3D {
                        width,
                        height,
                        depth: 1,
                    })];
                let post = [vk::ImageMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ)
                    .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .image(image)
                    .subresource_range(subrange)];
                ctx.device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &pre,
                );
                ctx.device.cmd_copy_buffer_to_image(
                    cmd,
                    staging.buffer,
                    image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &copy,
                );
                ctx.device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::COMPUTE_SHADER
                        | vk::PipelineStageFlags::FRAGMENT_SHADER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &post,
                );
                let _ = ctx.device.end_command_buffer(cmd);
                let cmd_arr = [cmd];
                let submits = [vk::SubmitInfo::default().command_buffers(&cmd_arr)];
                let _ = ctx.device.queue_submit(ctx.graphics_queue, &submits, fence);
            });
            let _ = ctx.device.wait_for_fences(&[fence], true, 5_000_000_000);
            ctx.device.destroy_fence(fence, None);
            ctx.device.destroy_command_pool(pool, None);
            staging.destroy(&ctx.device);

            // Replace any previous texture in this slot via RetireQueue to avoid UAF on in-flight frames.
            if let Some(mut old) = self.textures.remove(&slot) {
                if let Some(rq) = retire {
                    if let Some(mut old_pooled) = old.pooled.take() {
                        let dev = ctx.device.clone();
                        let old_view = old.view;
                        let old_sampler = old.sampler;
                        rq.retire_later(move || {
                            if old_sampler != vk::Sampler::null() {
                                dev.destroy_sampler(old_sampler, None);
                            }
                            if old_view != vk::ImageView::null() {
                                dev.destroy_image_view(old_view, None);
                            }
                            old_pooled.destroy(&dev);
                        });
                    } else {
                        rq.retire_image_resources(
                            ctx.device.clone(),
                            old.image,
                            old.memory,
                            old.view,
                            old.sampler,
                        );
                    }
                } else {
                    if old.sampler != vk::Sampler::null() {
                        ctx.device.destroy_sampler(old.sampler, None);
                    }
                    if old.view != vk::ImageView::null() {
                        ctx.device.destroy_image_view(old.view, None);
                    }
                    if let Some(mut old_pooled) = old.pooled.take() {
                        old_pooled.destroy(&ctx.device);
                    } else {
                        if old.image != vk::Image::null() {
                            ctx.device.destroy_image(old.image, None);
                        }
                        if old.memory != vk::DeviceMemory::null() {
                            ctx.device.free_memory(old.memory, None);
                        }
                    }
                }
            }
            self.textures.insert(
                slot,
                PackTexture {
                    image,
                    memory,
                    pooled,
                    view,
                    sampler,
                },
            );

            let img_info = [vk::DescriptorImageInfo {
                sampler,
                image_view: view,
                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            }];
            let write = [vk::WriteDescriptorSet::default()
                .dst_set(self.descriptor_set)
                .dst_binding(slot)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&img_info)];
            ctx.device.update_descriptor_sets(&write, &[]);
        }
        0
    }

    pub fn destroy(&mut self, ctx: &VulkanContext) {
        unsafe {
            for (_, mut tex) in self.textures.drain() {
                if tex.sampler != vk::Sampler::null() {
                    ctx.device.destroy_sampler(tex.sampler, None);
                }
                if tex.view != vk::ImageView::null() {
                    ctx.device.destroy_image_view(tex.view, None);
                }
                if let Some(mut pooled) = tex.pooled.take() {
                    pooled.destroy(&ctx.device);
                } else {
                    if tex.image != vk::Image::null() {
                        ctx.device.destroy_image(tex.image, None);
                    }
                    if tex.memory != vk::DeviceMemory::null() {
                        ctx.device.free_memory(tex.memory, None);
                    }
                }
            }
            if self.descriptor_pool != vk::DescriptorPool::null() {
                ctx.device
                    .destroy_descriptor_pool(self.descriptor_pool, None);
                self.descriptor_pool = vk::DescriptorPool::null();
                self.descriptor_set = vk::DescriptorSet::null();
            }
            if self.descriptor_set_layout != vk::DescriptorSetLayout::null() {
                ctx.device
                    .destroy_descriptor_set_layout(self.descriptor_set_layout, None);
                self.descriptor_set_layout = vk::DescriptorSetLayout::null();
            }
            if self.fallback_sampler != vk::Sampler::null() {
                ctx.device.destroy_sampler(self.fallback_sampler, None);
                self.fallback_sampler = vk::Sampler::null();
            }
            if self.fallback_view != vk::ImageView::null() {
                ctx.device.destroy_image_view(self.fallback_view, None);
                self.fallback_view = vk::ImageView::null();
            }
            if let Some(mut pooled) = self.fallback_pooled.take() {
                pooled.destroy(&ctx.device);
                self.fallback_image = vk::Image::null();
                self.fallback_memory = vk::DeviceMemory::null();
            } else {
                if self.fallback_image != vk::Image::null() {
                    ctx.device.destroy_image(self.fallback_image, None);
                    self.fallback_image = vk::Image::null();
                }
                if self.fallback_memory != vk::DeviceMemory::null() {
                    ctx.device.free_memory(self.fallback_memory, None);
                    self.fallback_memory = vk::DeviceMemory::null();
                }
            }
        }
    }
}
