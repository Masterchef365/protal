use nalgebra::{Matrix4, Vector3, Unit, Quaternion};
use watertender::defaults::FRAMES_IN_FLIGHT;
use watertender::memory;
use watertender::prelude::*;
mod managed_ubo;
use managed_ubo::ManagedUbo;
use watertender::openxr as xr;

const MAX_TRANSFORMS: usize = 2;

use anyhow::Result;

struct Protal {
    rainbow_cube: ManagedMesh,
    transforms: Vec<ManagedBuffer>,
    scene_data: ManagedUbo<SceneData>,

    action_stuff: Option<(xr::ActionSet, xr::Action<xr::Posef>)>,

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

type Transform = [[f32; 4]; 4];

pub struct FrameData {
    pub positions: Vec<Transform>,
}

impl MainLoop for Protal {
    fn new(core: &SharedCore, mut platform: Platform<'_>) -> Result<Self> {
        let mut starter_kit = StarterKit::new(core.clone(), &mut platform)?;

        // Camera
        let camera = MultiPlatformCamera::new(&mut platform);

        // Scene data
        let scene_data = ManagedUbo::new(core.clone(), FRAMES_IN_FLIGHT)?;

        // Transforms data
        let total_size = std::mem::size_of::<Transform>() * MAX_TRANSFORMS;
        let ci = vk::BufferCreateInfoBuilder::new()
            .size(total_size as u64)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER);
        let transforms = (0..FRAMES_IN_FLIGHT)
            .map(|_| ManagedBuffer::new(core.clone(), ci, memory::UsageFlags::UPLOAD))
            .collect::<Result<Vec<_>>>()?;

        // Create descriptor set layout
        const FRAME_DATA_BINDING: u32 = 0;
        const TRANSFORM_BINDING: u32 = 1;
        let bindings = [
            vk::DescriptorSetLayoutBindingBuilder::new()
                .binding(FRAME_DATA_BINDING)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::ALL_GRAPHICS),
            vk::DescriptorSetLayoutBindingBuilder::new()
                .binding(TRANSFORM_BINDING)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::ALL_GRAPHICS),
        ];

        let descriptor_set_layout_ci =
            vk::DescriptorSetLayoutCreateInfoBuilder::new().bindings(&bindings);

        let descriptor_set_layout = unsafe {
            core.device
                .create_descriptor_set_layout(&descriptor_set_layout_ci, None, None)
        }
        .result()?;

        // Create descriptor pool
        let pool_sizes = [
            vk::DescriptorPoolSizeBuilder::new()
                ._type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(FRAMES_IN_FLIGHT as _),
            vk::DescriptorPoolSizeBuilder::new()
                ._type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(FRAMES_IN_FLIGHT as _),
        ];

        let create_info = vk::DescriptorPoolCreateInfoBuilder::new()
            .pool_sizes(&pool_sizes)
            .max_sets((FRAMES_IN_FLIGHT * 2) as _);

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
        for (frame, &descriptor_set) in descriptor_sets.iter().enumerate() {
            let frame_data_bi = [scene_data.descriptor_buffer_info(frame)];
            let transform_bi = [vk::DescriptorBufferInfoBuilder::new()
                .buffer(transforms[frame].instance())
                .offset(0)
                .range(vk::WHOLE_SIZE)];

            let writes = [
                vk::WriteDescriptorSetBuilder::new()
                    .buffer_info(&frame_data_bi)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .dst_set(descriptor_set)
                    .dst_binding(FRAME_DATA_BINDING)
                    .dst_array_element(0),
                vk::WriteDescriptorSetBuilder::new()
                    .buffer_info(&transform_bi)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .dst_set(descriptor_set)
                    .dst_binding(TRANSFORM_BINDING)
                    .dst_array_element(0),
            ];

            unsafe {
                core.device.update_descriptor_sets(&writes, &[]);
            }
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
        let (vertices, indices) = rainbow_cube(0.1);
        let rainbow_cube = upload_mesh(
            &mut starter_kit.staging_buffer,
            starter_kit.command_buffers[0],
            &vertices,
            &indices,
        )?;

        let action_stuff = if let Platform::OpenXr { xr_core, .. } = platform {
            // Create action set
            let action_set = xr_core
                .instance
                .create_action_set("gameplay", "Gameplay", 0)?;
            let action = action_set.create_action::<xr::Posef>(
                "right_pose",
                "Right pose",
                &[],
            )?;

            // Set interaction profile BS
            let right_hand_pose_path = xr_core
                .instance
                .string_to_path("/user/hand/right/input/aim/pose")?;
            let interaction_profile_path = 
            xr_core
                .instance
                .string_to_path("/interaction_profiles/khr/simple_controller")?;

            let bindings = [
                xr::Binding::new(&action, right_hand_pose_path),
            ];
            xr_core.instance.suggest_interaction_profile_bindings(interaction_profile_path, &bindings)?;

            // Attach sets
            xr_core.session.attach_action_sets(&[&action_set])?;
            Some((action_set, action))
        } else {
            None
        };

        Ok(Self {
            action_stuff,
            camera,
            transforms,
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
        // TODO: Rename me! This should be called something like SwapchainStuff
        frame: Frame,
        core: &SharedCore,
        platform: Platform<'_>,
    ) -> Result<PlatformReturn> {
        let frame_data = self.frame_data(&platform)?;

        self.transforms[self.starter_kit.frame]
            .write_bytes(0, bytemuck::cast_slice(frame_data.positions.as_slice()))?;

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

            for idx in 0..frame_data.positions.len() {
                let push_const = [idx as u32];
                // TODO: Make this a shortcut
                core.device.cmd_push_constants(
                    command_buffer,
                    self.pipeline_layout,
                    vk::ShaderStageFlags::VERTEX,
                    0,
                    std::mem::size_of_val(&push_const) as u32,
                    push_const.as_ptr() as _,
                );
                core.device.cmd_draw_indexed(
                    command_buffer,
                    self.rainbow_cube.n_indices,
                    1,
                    0,
                    0,
                    0,
                );
            }
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

impl Protal {
    fn frame_data(&mut self, platform: &Platform<'_>) -> Result<FrameData> {
        self.anim += 0.02;
        if let (Some((set, action)), Platform::OpenXr { xr_core, frame_state }) = (self.action_stuff.as_ref(), platform) {
            let active = xr::ActiveActionSet::new(set);
            xr_core.session.sync_actions(&[active])?;

            let space = action.create_space(xr_core.session.clone(), xr::Path::NULL, xr::Posef::IDENTITY)?;
            let time = frame_state.unwrap().predicted_display_time;
            let space_loc = space.locate(&xr_core.stage, time)?;
            let pose = space_loc.pose;
            let matrix = transform_from_pose(&pose);

            Ok(FrameData { positions: vec![*matrix.as_ref()] })
        } else {
            Ok(FrameData {
                positions: vec![
                    *Matrix4::new_translation(&Vector3::new(0., -3., 0.)).as_ref(),
                    *Matrix4::from_euler_angles(0., self.anim, 0.).as_ref(),
                ],
            })
        }
    }
}

fn rainbow_cube(size: f32) -> (Vec<Vertex>, Vec<u32>) {
    let vertices = vec![
        Vertex::new([-size, -size, -size], [0.0, 1.0, 1.0]),
        Vertex::new([size, -size, -size], [1.0, 0.0, 1.0]),
        Vertex::new([size, size, -size], [1.0, 1.0, 0.0]),
        Vertex::new([-size, size, -size], [0.0, 1.0, 1.0]),
        Vertex::new([-size, -size, size], [1.0, 0.0, 1.0]),
        Vertex::new([size, -size, size], [1.0, 1.0, 0.0]),
        Vertex::new([size, size, size], [0.0, 1.0, 1.0]),
        Vertex::new([-size, size, size], [1.0, 0.0, 1.0]),
    ];

    let indices = vec![
        3, 1, 0, 2, 1, 3, 2, 5, 1, 6, 5, 2, 6, 4, 5, 7, 4, 6, 7, 0, 4, 3, 0, 7, 7, 2, 3, 6, 2, 7,
        0, 5, 4, 1, 5, 0,
    ];

    (vertices, indices)
}

pub fn transform_from_pose(pose: &xr::Posef) -> Matrix4<f32> {
    let quat = pose.orientation;
    let quat = Quaternion::new(quat.w, quat.x, quat.y, quat.z);
    let quat = Unit::try_new(quat, 0.0).expect("Not a unit quaternion");
    let rotation = quat.to_homogeneous();

    let position = pose.position;
    let position = Vector3::new(position.x, position.y, position.z);
    let translation = Matrix4::new_translation(&position);

    translation * rotation
}