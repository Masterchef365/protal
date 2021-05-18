use nalgebra::{Matrix4, Vector3};
use watertender::prelude::*;
use watertender::defaults::FRAMES_IN_FLIGHT;
mod managed_ubo;
use managed_ubo::ManagedUbo;

use anyhow::Result;

struct Protal {
    rainbow_cube: ManagedMesh,
    //transforms: ManagedBuffer,
    scene_data: ManagedUbo<SceneData>,

    descriptor_sets: Vec<vk::DescriptorSet>,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set_layout: vk::DescriptorSetLayout,

    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,

    camera: MultiPlatformCamera,
    anim: f32,
    starter_kit: StarterKit,
}

fn main() -> Result<()> {
    let info = AppInfo::default().validation(true);
    let vr = std::env::args().count() > 1;
    launch::<Protal>(info, vr)
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct SceneData {
    cameras: [f32; 4 * 4 * 2],
    anim: f32,
}

unsafe impl bytemuck::Zeroable for SceneData {}
unsafe impl bytemuck::Pod for SceneData {}

impl MainLoop for Protal {
    fn new(core: &SharedCore, mut platform: Platform<'_>) -> Result<Self> {
        let mut starter_kit = StarterKit::new(core.clone(), &mut platform)?;

        // Camera
        let camera = MultiPlatformCamera::new(&mut platform);

        // Scene data
        let scene_data = ManagedUbo::new(core.clone(), FRAMES_IN_FLIGHT)?;

        // Create descriptor set layout
        let binding = 0;
        let bindings = [vk::DescriptorSetLayoutBindingBuilder::new()
            .binding(binding)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::ALL_GRAPHICS)];

        let descriptor_set_layout_ci =
            vk::DescriptorSetLayoutCreateInfoBuilder::new().bindings(&bindings);

        let descriptor_set_layout = unsafe {
            core.device
                .create_descriptor_set_layout(&descriptor_set_layout_ci, None, None)
        }
        .result()?;

        // Create descriptor pool
        let pool_sizes = [vk::DescriptorPoolSizeBuilder::new()
            ._type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(FRAMES_IN_FLIGHT as _)];

        let create_info = vk::DescriptorPoolCreateInfoBuilder::new()
            .pool_sizes(&pool_sizes)
            .max_sets(FRAMES_IN_FLIGHT as _);

        let descriptor_pool =
            unsafe { core.device.create_descriptor_pool(&create_info, None, None) }.result()?;

        // Create descriptor sets
        let layouts = vec![descriptor_set_layout; FRAMES_IN_FLIGHT];
        let create_info = vk::DescriptorSetAllocateInfoBuilder::new()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&layouts);

        let descriptor_sets =
            unsafe { core.device.allocate_descriptor_sets(&create_info) }.result()?;

        // Write descriptor sets
        let buffer_infos: Vec<_> = (0..FRAMES_IN_FLIGHT)
            .map(|frame| {
                [scene_data.descriptor_buffer_info(frame)]
            })
            .collect();

        let writes: Vec<_> = buffer_infos
            .iter()
            .zip(descriptor_sets.iter())
            .map(|(info, &descriptor_set)| {
                vk::WriteDescriptorSetBuilder::new()
                    .buffer_info(info)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .dst_set(descriptor_set)
                    .dst_binding(binding)
                    .dst_array_element(0)
            })
            .collect();

        unsafe {
            core.device.update_descriptor_sets(&writes, &[]);
        }

        // Pipeline layout
        let push_constant_ranges = [vk::PushConstantRangeBuilder::new()
            .stage_flags(vk::ShaderStageFlags::VERTEX)
            .offset(0)
            .size(std::mem::size_of::<[f32; 4 * 4]>() as u32)];

        let descriptor_set_layouts = [descriptor_set_layout];
        let create_info = vk::PipelineLayoutCreateInfoBuilder::new()
            .push_constant_ranges(&push_constant_ranges)
            .set_layouts(&descriptor_set_layouts);

        let pipeline_layout =
            unsafe { core.device.create_pipeline_layout(&create_info, None, None) }.result()?;

        // Pipeline
        let pipeline = shader(
            core,
            &std::fs::read("shaders/unlit.vert.spv")?,
            &std::fs::read("shaders/unlit.frag.spv")?,
            vk::PrimitiveTopology::TRIANGLE_LIST,
            starter_kit.render_pass,
            pipeline_layout,
        )?;

        // Mesh uploads
        let (vertices, indices) = rainbow_cube();
        let rainbow_cube = upload_mesh(
            &mut starter_kit.staging_buffer,
            starter_kit.command_buffers[0],
            &vertices,
            &indices,
        )?;

        Ok(Self {
            camera,
            anim: 0.0,
            pipeline_layout,
            descriptor_set_layout,
            descriptor_sets,
            descriptor_pool,
            scene_data,
            rainbow_cube,
            pipeline,
            starter_kit,
        })
    }

    fn frame(
        &mut self,
        frame: Frame,
        core: &SharedCore,
        platform: Platform<'_>,
    ) -> Result<PlatformReturn> {
        let cmd = self.starter_kit.begin_command_buffer(frame)?;
        let command_buffer = cmd.command_buffer;

        unsafe {
            core.device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layout,
                0,
                &[self.descriptor_sets[self.starter_kit.frame]],
                &[],
            );

            // Draw cmds
            core.device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline,
            );

            core.device.cmd_bind_vertex_buffers(
                command_buffer,
                0,
                &[self.rainbow_cube.vertices.instance()],
                &[0],
            );
            core.device.cmd_bind_index_buffer(
                command_buffer,
                self.rainbow_cube.indices.instance(),
                0,
                vk::IndexType::UINT32,
            );

            core.device
                .cmd_draw_indexed(command_buffer, self.rainbow_cube.n_indices, 1, 0, 0, 0);
        }

        let (ret, cameras) = self.camera.get_matrices(platform)?;

        self.scene_data.upload(
            self.starter_kit.frame,
            &SceneData {
                cameras,
                anim: self.anim,
            },
        )?;

        // End draw cmds
        self.starter_kit.end_command_buffer(cmd)?;

        Ok(ret)
    }

    fn swapchain_resize(&mut self, images: Vec<vk::Image>, extent: vk::Extent2D) -> Result<()> {
        self.starter_kit.swapchain_resize(images, extent)
    }

    fn event(
        &mut self,
        mut event: PlatformEvent<'_, '_>,
        _core: &Core,
        mut platform: Platform<'_>,
    ) -> Result<()> {
        self.camera.handle_event(&mut event, &mut platform);
        starter_kit::close_when_asked(event, platform);
        Ok(())
    }
}

impl SyncMainLoop for Protal {
    fn winit_sync(&self) -> (vk::Semaphore, vk::Semaphore) {
        self.starter_kit.winit_sync()
    }
}

struct FrameData {
    positions: Vec<[[f32; 4]; 4]>,
}

impl Protal {
    fn frame_data(&mut self) -> FrameData {
        FrameData {
            positions: vec![
                *Matrix4::new_translation(&Vector3::new(0., -3., 0.)).as_ref(),
                *Matrix4::from_euler_angles(0., self.anim, 0.).as_ref(),
            ],
        }
    }
}

fn rainbow_cube() -> (Vec<Vertex>, Vec<u32>) {
    let vertices = vec![
        Vertex::new([-1.0, -1.0, -1.0], [0.0, 1.0, 1.0]),
        Vertex::new([1.0, -1.0, -1.0], [1.0, 0.0, 1.0]),
        Vertex::new([1.0, 1.0, -1.0], [1.0, 1.0, 0.0]),
        Vertex::new([-1.0, 1.0, -1.0], [0.0, 1.0, 1.0]),
        Vertex::new([-1.0, -1.0, 1.0], [1.0, 0.0, 1.0]),
        Vertex::new([1.0, -1.0, 1.0], [1.0, 1.0, 0.0]),
        Vertex::new([1.0, 1.0, 1.0], [0.0, 1.0, 1.0]),
        Vertex::new([-1.0, 1.0, 1.0], [1.0, 0.0, 1.0]),
    ];

    let indices = vec![
        3, 1, 0, 2, 1, 3, 2, 5, 1, 6, 5, 2, 6, 4, 5, 7, 4, 6, 7, 0, 4, 3, 0, 7, 7, 2, 3, 6, 2, 7,
        0, 5, 4, 1, 5, 0,
    ];

    (vertices, indices)
}
