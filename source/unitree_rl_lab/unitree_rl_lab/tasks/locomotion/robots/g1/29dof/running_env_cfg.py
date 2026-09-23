import math

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.utils.noise import NoiseModelWithAdditiveBiasCfg

from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_29DOF_CFG as ROBOT_CFG
from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_29DOF_K_EL
from unitree_rl_lab.tasks.locomotion import mdp

GAIT_PERIOD = 0.5
GAIT_OFFSETS = (0.0, 0.5)
GAIT_STANCE_RATIO = 0.30
GAIT_FEET_BODY_NAMES = ["left_ankle_roll_link", "right_ankle_roll_link"]
# Single stand/run rule for the task: a velocity command with norm below this value means "stand
# still", at or above it means "run". Shared by the gait clock observation (masked to (0, 0) while
# standing), the standing rewards/penalties and the run gate of the gait rewards. The deployment
# code reads the same value from the exported deploy.yaml, so sim and robot agree.
STAND_CMD_THRESHOLD = 0.1

# Running-specific neutral pose: keep the elbows bent and the arms close to the
# torso while leaving shoulder pitch available for fore-aft arm swing.
RUNNING_JOINT_POS = dict(ROBOT_CFG.init_state.joint_pos)
RUNNING_JOINT_POS.update(
    {
        # ".*_shoulder_pitch_joint": 0.10,
        # "left_shoulder_roll_joint": 0.10,
        # "right_shoulder_roll_joint": -0.10,
        # ".*_shoulder_yaw_joint": 0.0,
        ".*_elbow_joint": 0.0,
        # "left_wrist_roll_joint": 0.0,
        # "right_wrist_roll_joint": 0.0,
    }
)
RUNNING_ROBOT_CFG = ROBOT_CFG.replace(
    prim_path="{ENV_REGEX_NS}/Robot",
    init_state=ROBOT_CFG.init_state.replace(joint_pos=RUNNING_JOINT_POS),
)

COBBLESTONE_ROAD_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=9,
    num_cols=21,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.75),
        "small_roughness": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.25,
            noise_range=(-0.01, 0.01),
            noise_step=0.005,
            downsampled_scale=0.2,
            border_width=0.25,
        ),
    },
)


@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",  # "plane", "generator"
        terrain_generator=COBBLESTONE_ROAD_CFG,  # None, ROUGH_TERRAINS_CFG
        max_init_terrain_level=COBBLESTONE_ROAD_CFG.num_rows - 1,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = RUNNING_ROBOT_CFG

    # sensors
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True
    )
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 1.2),
            "dynamic_friction_range": (0.5, 0.8),
            "restitution_range": (0.0, 0.05),
            "num_buckets": 64,
        },
    )

    link_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (0.95, 1.05),
            "operation": "scale",
            "recompute_inertia": True,
        },
    )

    torso_center_of_mass = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "com_range": {
                "x": (-0.015, 0.015),
                "y": (-0.015, 0.015),
                "z": (-0.01, 0.01),
            },
        },
    )

    actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.9, 1.1),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    joint_friction = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "friction_distribution_params": (0.0, 0.03),
            "operation": "add",
            "distribution": "uniform",
        },
    )

    joint_armature = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "armature_distribution_params": (0.9, 1.1),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    # reset
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "roll": (-0.05, 0.05),
                "pitch": (-0.05, 0.05),
                "yaw": (-3.14, 3.14),
            },
            "velocity_range": {
                "x": (-0.2, 0.2),
                "y": (-0.2, 0.2),
                "z": (-0.1, 0.1),
                "roll": (-0.2, 0.2),
                "pitch": (-0.2, 0.2),
                "yaw": (-0.2, 0.2),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.95, 1.05),
            "velocity_range": (-0.2, 0.2),
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 9.0),
        params={
            "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-0.3, 0.3)}
        },
    )


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.LoggedUniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        log_tracking_metrics=True,
        # Measurement only -- these metrics never enter the reward, so they stay on in the
        # baseline too. Without them a baseline run has no energy curve to compare against.
        log_energy_metrics=True,
        # Electrical view (copper loss + positive work) next to the mechanical one; k_el is the
        # estimate documented in the robot asset. Measurement only, never in a reward.
        joule_coefficients=UNITREE_G1_29DOF_K_EL,
        contact_sensor_name="contact_forces",
        feet_body_names=GAIT_FEET_BODY_NAMES,
        gait_period=GAIT_PERIOD,
        gait_offsets=GAIT_OFFSETS,
        gait_stance_ratio=GAIT_STANCE_RATIO,
        rel_standing_envs=0.1,
        rel_heading_envs=1.0,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.LoggedUniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.4, 0.4), lin_vel_y=(-0.4, 0.4), ang_vel_z=(-0.4, 0.4)
        ),
        limit_ranges=mdp.LoggedUniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 2.5), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0)
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    JointPositionAction = mdp.RandomizedDelayedJointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.25,
        use_default_offset=True,
        min_delay=0,
        max_delay=1,
        action_bias_range=(-0.02, 0.02),
        action_noise_range=(-0.01, 0.01),
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            scale=0.2,
            noise=NoiseModelWithAdditiveBiasCfg(
                noise_cfg=Unoise(n_min=-0.2, n_max=0.2),
                bias_noise_cfg=Unoise(n_min=-0.03, n_max=0.03, operation="abs"),
            ),
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=NoiseModelWithAdditiveBiasCfg(
                noise_cfg=Unoise(n_min=-0.05, n_max=0.05),
                bias_noise_cfg=Unoise(n_min=-0.01, n_max=0.01, operation="abs"),
            ),
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "base_velocity"}
        )
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=NoiseModelWithAdditiveBiasCfg(
                noise_cfg=Unoise(n_min=-0.01, n_max=0.01),
                bias_noise_cfg=Unoise(n_min=-0.005, n_max=0.005, operation="abs"),
            ),
        )
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel,
            scale=0.05,
            noise=NoiseModelWithAdditiveBiasCfg(
                noise_cfg=Unoise(n_min=-1.5, n_max=1.5),
                bias_noise_cfg=Unoise(n_min=-0.1, n_max=0.1, operation="abs"),
            ),
        )
        last_action = ObsTerm(func=mdp.last_action)
        gait_phase = ObsTerm(
            func=mdp.gait_phase,
            params={
                "period": GAIT_PERIOD,
                "command_name": "base_velocity",
                "command_threshold": STAND_CMD_THRESHOLD,
            },
        )

        def __post_init__(self):
            self.history_length = 5
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "base_velocity"}
        )
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
        last_action = ObsTerm(func=mdp.last_action)
        gait_phase = ObsTerm(
            func=mdp.gait_phase,
            params={
                "period": GAIT_PERIOD,
                "command_name": "base_velocity",
                "command_threshold": STAND_CMD_THRESHOLD,
            },
        )
        # height_scanner = ObsTerm(func=mdp.height_scan,
        #     params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        #     clip=(-1.0, 5.0),
        # )

        def __post_init__(self):
            self.history_length = 5

    # privileged observations
    critic: CriticCfg = CriticCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- task
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.1)},
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.04)},
    )

    alive = RewTerm(func=mdp.is_alive, weight=0.15)

    # -- base
    base_linear_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.05)
    base_angular_velocity = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.001)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.05)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-5.0)

    joint_deviation_arm_swing = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.02,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_pitch_joint",
                    ".*_elbow_joint",
                ],
            )
        },
    )
    joint_deviation_arm_lateral = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_roll_joint",
                    ".*_shoulder_yaw_joint",
                    ".*_wrist_.*",
                ],
            )
        },
    )
    joint_deviation_waists = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "waist.*",
                ],
            )
        },
    )
    joint_deviation_legs = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[".*_hip_roll_joint", ".*_hip_yaw_joint"]
            )
        },
    )

    # -- robot
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-2.0)
    base_height = RewTerm(
        func=mdp.base_height_l2, weight=-10.0, params={"target_height": 0.78}
    )

    # -- feet
    run_contact_pattern = RewTerm(
        func=mdp.run_contact_pattern,
        weight=1.0,
        params={
            "period": GAIT_PERIOD,
            "offset": GAIT_OFFSETS,
            "stance_ratio": GAIT_STANCE_RATIO,
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=GAIT_FEET_BODY_NAMES
            ),
        },
    )
    short_contact = RewTerm(
        func=mdp.short_contact_penalty,
        weight=-0.75,
        params={
            "max_contact_time": 0.15,
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=GAIT_FEET_BODY_NAMES
            ),
        },
    )
    push_off = RewTerm(
        func=mdp.push_off_reward,
        weight=0.5,
        params={
            "period": GAIT_PERIOD,
            "offset": GAIT_OFFSETS,
            "stance_ratio": GAIT_STANCE_RATIO,
            "target_vz": 0.45,
            "max_target_vz": 0.55,
            "target_speed": 2.0,
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=GAIT_FEET_BODY_NAMES
            ),
        },
    )
    stand_feet_contact = RewTerm(
        func=mdp.feet_contact_without_cmd,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "command_threshold": STAND_CMD_THRESHOLD,
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=GAIT_FEET_BODY_NAMES
            ),
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
        },
    )

    # feet_clearance = RewTerm(
    #     func=mdp.foot_clearance_reward,
    #     weight=1.0,
    #     params={
    #         "std": 0.05,
    #         "tanh_mult": 2.0,
    #         "target_height": 0.3,
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
    #     },
    # )

    # -- other
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1,
        params={
            "threshold": 1,
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["(?!.*ankle.*).*"]
            ),
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_height = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": 0.2}
    )
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8})


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    lin_vel_cmd_levels = CurrTerm(mdp.lin_vel_cmd_levels)


@configclass
class RobotEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: RobotSceneCfg = RobotSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False


@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 10
        # Keep play deterministic enough for visual evaluation. Startup dynamics
        # randomization remains enabled, but observation/action noise and pushes do not.
        self.observations.policy.enable_corruption = False
        self.actions.JointPositionAction.min_delay = 0
        self.actions.JointPositionAction.max_delay = 0
        self.actions.JointPositionAction.action_bias_range = (0.0, 0.0)
        self.actions.JointPositionAction.action_noise_range = (0.0, 0.0)
        self.events.push_robot = None
        self.commands.base_velocity.ranges = (
            mdp.LoggedUniformLevelVelocityCommandCfg.Ranges(
                lin_vel_x=(-1.0, 2.5), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0)
            )
        )
