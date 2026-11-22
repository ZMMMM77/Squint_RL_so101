# Squint_RL_so101: pick the cube and drop it into an open box, on the SO101.
#
# Built on top of the SO101 pick-cube vision task (same observations and action
# space, so the deploy stack is shared), with:
#   * a 12 x 12 x 7 cm open box (5 mm walls) next to the cube area, as a mocap
#     body jittered in position/yaw and recolored every episode;
#   * the goal fixed DROP_HEIGHT above the box center (the policy's
#     `goal_position` observation therefore tells it where the box is);
#   * a start pose with the wrist rolled -90 deg: the wrist camera sits on top of
#     the gripper looking straight down and the jaws close left-right;
#   * staged rewards: reach -> grasp -> lift -> carry over the box -> release ->
#     cube in box -> retreat, with the pick terms switched off once the cube is in
#     the box, plus posture terms (gripper pointing down, jaws level) and
#     box/floor contact penalties.
"""SO101 pick-and-place-into-box task configuration."""

import math
from typing import Any, Literal

import mujoco
from mujoco_playground.robots.so101.real import constant as real_constant
from mujoco_playground.robots.so101.sim import constant as sim_constant
from mujoco_playground.tasks.manipulation.pick_cube import mdp
from mujoco_playground.tasks.manipulation.pick_cube import reset
from mujoco_playground.tasks.manipulation.pick_cube import task as pick_task
from mujoco_playground.tasks.manipulation.pick_cube.so101 import task as so101_task
from reaf.simulation import torch as sim_device
from reaf.simulation.torch import flat_scene as config
from reaf.simulation.torch import noise
import torch

# ---- The real box: 12 x 12 x 7 cm outside, 5 mm walls and bottom. ----
BOX_HALF = 0.06
BOX_WALL = 0.005
BOX_HEIGHT = 0.07
BOX_INNER_HALF = BOX_HALF - BOX_WALL
# Nominal box center in the robot base frame (x forward, y to the robot's left).
# The cube spawns at x 0.20..0.35, |y| <= 0.10; the box sits to the left of it.
BOX_CENTER = (0.20, 0.23)
BOX_JITTER = 0.025  # +- per axis, so a hand-placed box is still in distribution
BOX_YAW_DEG = 10.0
# Cube center height to bring the cube to before releasing it over the box.
DROP_HEIGHT = 0.12
# The deploy target (`--target_pos`) for a box placed at BOX_CENTER.
PLACE_TARGET = (BOX_CENTER[0], BOX_CENTER[1], DROP_HEIGHT)

# Gripper opening (rad) used to score open/closed, as in the pick task's grasp term.
GRIPPER_OPEN = 0.80
GRIPPER_CLOSED = 0.15

# Start pose: the pick task's hover pose, pan centered (the camera view is then
# centered on the cube area) and the wrist rolled -90 deg (camera on top).
_h = real_constant.HOVER_POSE_OPEN_GRIPPER
PLACE_HOVER_POSE = (0.0, _h[1], _h[2], _h[3], _h[4] - math.pi / 2, GRIPPER_OPEN)

BOX = "box"
BOX_POS = f"{BOX}/mocap_pos"
BOX_QUAT = f"{BOX}/mocap_quat"
CUBE_POS = "object/xpos"
GRIPPER_QPOS = f"robot_state/{real_constant.JOINT_POS}"


def box_spec() -> mujoco.MjSpec:
  """An open box with its bottom on the table (body origin at the bottom center)."""
  spec = mujoco.MjSpec()
  spec.modelname = BOX
  body = spec.worldbody.add_body(name=BOX, mocap=True)
  t = BOX_WALL / 2
  h = BOX_HEIGHT / 2
  side = BOX_HALF - t
  parts = {
      "bottom": ((BOX_HALF, BOX_HALF, t), (0.0, 0.0, t)),
      "wall_front": ((t, BOX_HALF, h), (side, 0.0, h)),
      "wall_back": ((t, BOX_HALF, h), (-side, 0.0, h)),
      "wall_left": ((BOX_INNER_HALF, t, h), (0.0, side, h)),
      "wall_right": ((BOX_INNER_HALF, t, h), (0.0, -side, h)),
  }
  for name, (size, pos) in parts.items():
    body.add_geom(
        name=name,
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=size,
        pos=pos,
        rgba=(0.72, 0.58, 0.40, 1.0),
        # Collides with the cube (contype 2 / conaffinity 1) and the robot.
        contype=1,
        conaffinity=1,
        condim=3,
        friction=(0.8, 0.005, 0.0001),
        group=0,
    )
  return spec


# ---- MDP helpers ----


def _box_frame_offset(features, cube_key: str = CUBE_POS):
  """Cube position relative to the box: (x, y) in the box's yawed frame, z up."""
  d = features[cube_key] - features[BOX_POS]
  q = features[BOX_QUAT]
  yaw = 2.0 * torch.atan2(q[..., 3], q[..., 0])
  c, s = torch.cos(yaw), torch.sin(yaw)
  lx = c * d[..., 0] + s * d[..., 1]
  ly = -s * d[..., 0] + c * d[..., 1]
  return lx, ly, d[..., 2]


def cube_in_box(features, cube_key: str = CUBE_POS) -> torch.Tensor:
  """True when the cube center is inside the walls and below the rim."""
  lx, ly, z = _box_frame_offset(features, cube_key)
  inside = (lx.abs() < BOX_INNER_HALF) & (ly.abs() < BOX_INNER_HALF)
  return inside & (z < BOX_HEIGHT) & (z > 0.0)


def _open_frac(features) -> torch.Tensor:
  q = features[GRIPPER_QPOS][..., -1]
  return torch.clamp(
      (q - GRIPPER_CLOSED) / (GRIPPER_OPEN - GRIPPER_CLOSED), 0.0, 1.0
  )


def _until_boxed(fn):
  """Wraps a pick reward so it pays only while the cube is not in the box yet."""

  def gated(features, **params):
    return fn(features, **params) * (~cube_in_box(features)).float()

  gated.__name__ = f"{fn.__name__}_until_boxed"
  gated.__doc__ = f"{fn.__doc__} Zero once the cube is in the box."
  return gated


reward_reaching = _until_boxed(mdp.reward_reaching)
reward_grasp = _until_boxed(mdp.reward_gripper_close)
reward_lift = _until_boxed(mdp.reward_cube_lift)
reward_bringing = _until_boxed(mdp.reward_cube_to_target)


def reward_release_over_box(features, margin: float = 0.02) -> torch.Tensor:
  """Gripper opening while the cube is above the box opening (lets it drop in)."""
  lx, ly, z = _box_frame_offset(features)
  over = (
      (lx.abs() < BOX_INNER_HALF - margin)
      & (ly.abs() < BOX_INNER_HALF - margin)
      & (z > BOX_HEIGHT)
  )
  return over.float() * _open_frac(features) * (~cube_in_box(features)).float()


def reward_in_box(features) -> torch.Tensor:
  """The cube is in the box: the largest per-step term, so releasing pays."""
  return cube_in_box(features).float()


def reward_released_in_box(features) -> torch.Tensor:
  """Cube in the box and the gripper open (let go, do not hold it in there)."""
  return cube_in_box(features).float() * _open_frac(features)


def reward_retreat(
    features, ee_xpos_key: str = "ee/xpos", clearance: float = 0.05
) -> torch.Tensor:
  """After placing, lift the gripper clear of the box rim."""
  rim_z = features[BOX_POS][..., 2] + BOX_HEIGHT
  up = torch.clamp(
      (features[ee_xpos_key][..., 2] - rim_z) / clearance, 0.0, 1.0
  )
  return cube_in_box(features).float() * up


def reward_gripper_down(
    features, xmat_key: str = "ee/xmat", full_deg: float = 45.0
) -> torch.Tensor:
  """1 when the gripper points at least `full_deg` below horizontal, 0 when level or up."""
  approach_z = features[xmat_key][..., 6]  # z of the site x axis (approach)
  return torch.clamp(-approach_z / math.sin(math.radians(full_deg)), 0.0, 1.0)


def reward_jaws_level(features, xmat_key: str = "ee/xmat") -> torch.Tensor:
  """1 when the jaws close horizontally (left-right), 0 when they close vertically."""
  jaw_z = features[xmat_key][..., 8]  # z of the site z axis (jaw closing direction)
  return 1.0 - jaw_z.abs()


def reward_box_contact(
    features, contact_key: str = "box_contact/data", penalty_scale: float = -1.0
) -> torch.Tensor:
  """Penalty while the arm or gripper touches the box."""
  return penalty_scale * features[contact_key].any(dim=(-2, -1)).float()


def metric_success(features) -> torch.Tensor:
  """Cube in the box with the gripper open."""
  return (cube_in_box(features) & (_open_frac(features) > 0.5)).float()


def metric_in_box(features) -> torch.Tensor:
  return cube_in_box(features).float()


def metric_box_xy_err(features) -> torch.Tensor:
  """Horizontal distance from the cube to the box center."""
  d = features[CUBE_POS][..., :2] - features[BOX_POS][..., :2]
  return torch.linalg.norm(d, dim=-1)


def metric_gripper_down_deg(features, xmat_key: str = "ee/xmat") -> torch.Tensor:
  """Gripper angle below horizontal, in degrees."""
  return torch.rad2deg(torch.asin(torch.clamp(-features[xmat_key][..., 6], -1.0, 1.0)))


def obs_cube_to_box(features) -> torch.Tensor:
  """Privileged critic input: box center minus cube position."""
  return features[BOX_POS] - features[CUBE_POS]


def obs_cube_in_box(features) -> torch.Tensor:
  return cube_in_box(features).float().unsqueeze(-1)


class PlaceCubeResetter(reset.PickCubeResetter):
  """Pick-cube reset plus a jittered box, with the goal fixed above the box."""

  def __init__(
      self,
      *,
      box_center: tuple[float, float] = BOX_CENTER,
      box_jitter: float = BOX_JITTER,
      box_yaw_deg: float = BOX_YAW_DEG,
      drop_height: float = DROP_HEIGHT,
      **kwargs: Any,
  ):
    super().__init__(**kwargs)
    self._box_center = box_center
    self._box_jitter = box_jitter
    self._box_yaw = math.radians(box_yaw_deg)
    self._drop_height = drop_height

  def on_reset(self, config: Any, env_mask: torch.Tensor) -> None:
    super().on_reset(config, env_mask)
    nworld, dev = self._env.nworld, self._env.xpu_device
    cx, cy = self._box_center
    j = self._box_jitter
    xy = reset._sample_pos_3d(  # pylint: disable=protected-access
        nworld, dev, ((cx - j, cx + j), (cy - j, cy + j), (0.0, 0.0))
    )
    yaw = noise.sample_uniform(
        (nworld, 1), dev, low=-self._box_yaw, high=self._box_yaw
    )
    box = self._env.device_layer.get_device(BOX)
    box.set_pos(env_mask, xy)
    box.set_quat(env_mask, reset._quat_from_yaw(yaw, dev))  # pylint: disable=protected-access
    goal = xy.clone()
    goal[:, 2] = self._drop_height
    self._env.device_layer.get_device(pick_task.TARGET).set_pos(env_mask, goal)

  def on_step(self) -> None:
    """The goal stays above the box for the whole episode."""


def _arm_bodies() -> tuple[str, ...]:
  p = sim_constant.ATTACHMENT_PREFIX
  return tuple(
      f"{p}{b}"
      for b in ("lower_arm", "wrist", "gripper", "moving_jaw_so101_v1", "camera_mount")
  )


def so101_place_cube_pixels_env(
    impl: Literal["cpp", "warp"] = "warp",
) -> config.Environment:
  """SO101 pick-and-place into a box, wrist camera policy."""
  cfg = so101_task.so101_pick_cube_pixels_env(impl=impl)
  devices = cfg.device_layer.devices

  # Start pose: wrist camera on top, jaws closing left-right.
  devices["robot_state"].runtime_params["home_qpos"] = list(PLACE_HOVER_POSE)
  devices["robot"].runtime_params["home_ctrl"] = list(PLACE_HOVER_POSE)
  cfg.task_layer.command_processors["delta_action"].runtime_params[
      "offset"
  ] = PLACE_HOVER_POSE

  # The box, and a contact sensor between it and the arm.
  devices[BOX] = config.Component(
      sim_device.MocapDevice, body_name=f"{BOX}/{BOX}"
  ).attach(box_spec, prefix=BOX, pos=(BOX_CENTER[0], BOX_CENTER[1], 0.0))
  devices["box_contact"] = config.Component(
      sim_device.ContactSensorDevice, fields=("found",)
  ).configure_spec(
      objtype="body", objname=_arm_bodies(), reftype="body", refname=f"{BOX}/{BOX}"
  )
  cfg.sim.nconmax = 96
  cfg.sim.njmax = 384

  # Reset: same arm/cube randomization, plus the box; the goal sits above the box.
  old = cfg.resetters["pick_cube"].runtime_params
  cfg.resetters["pick_cube"] = config.Component(
      PlaceCubeResetter,
      object_pos_range=old["object_pos_range"],
      target_pos_range=old["target_pos_range"],
      resampling_time_range=(1e9, 1e9),
      joint_perturbation=old.get("joint_perturbation", 0.1),
  )
  dr_terms = cfg.resetters["domain_randomization"].runtime_params["terms"]
  dr_terms["geom_rgba_box"] = config.Term(
      so101_task.randomize_uniform_geom_rgba,
      pattern=rf"{BOX}/.*",
      low=(0.10, 0.10, 0.10),
      high=(0.95, 0.95, 0.95),
      trigger="reset",
  )

  # Privileged critic inputs (the actor's observations are unchanged).
  critic = cfg.observation_adapter["critic"].terms
  critic["cube_to_box"] = config.Obs(obs_cube_to_box).configure(output_shape=(3,))
  critic["cube_in_box"] = config.Obs(obs_cube_in_box).configure(output_shape=(1,))
  critic["gripper_frame"] = config.Obs("ee/xmat").configure(output_shape=(9,))

  # ---- Rewards ----
  old_r = cfg.task_layer.rewards
  table_z = old_r["lift"].params["table_z"]
  penalties = {
      k: old_r[k]
      for k in ("action_rate", "joint_vel_hinge", "floor_contact", "arm_nominal_pose")
  }
  limits = old_r["joint_pos_limits"]
  limits.params["margin"] = 0.85
  limits.params["penalty_scale"] = -15.0
  penalties["joint_pos_limits"] = limits

  rewards = {
      # 1. reach the cube, 2. close the gripper on it, 3. lift it,
      # 4. carry it over the box (coarse + fine) -- all off once it is in the box.
      "reaching": config.Reward(
          reward_reaching, ee_xpos_key="ee/xpos", cube_xpos_key=CUBE_POS,
          reaching_std=0.05, weight=1.0,
      ),
      "grasp": config.Reward(
          reward_grasp, ee_xpos_key="ee/xpos", cube_xpos_key=CUBE_POS,
          qpos_key=GRIPPER_QPOS, gripper_joint_idx=-1, open_pos=GRIPPER_OPEN,
          closed_pos=GRIPPER_CLOSED, reaching_std=0.025, weight=1.5,
      ),
      "lift": config.Reward(
          reward_lift, cube_xpos_key=CUBE_POS,
          target_xpos_key="target/mocap_pos", table_z=table_z, weight=3.0,
      ),
      "bringing": config.Reward(
          reward_bringing, cube_xpos_key=CUBE_POS,
          target_xpos_key="target/mocap_pos", table_z=table_z, std=0.08, weight=3.0,
      ),
      "bringing_fine": config.Reward(
          reward_bringing, cube_xpos_key=CUBE_POS,
          target_xpos_key="target/mocap_pos", table_z=table_z, std=0.02, weight=2.0,
      ),
      # 5. open over the box, 6. cube in the box, 7. let go and lift clear.
      "release": config.Reward(reward_release_over_box, weight=2.0),
      "in_box": config.Reward(reward_in_box, weight=12.0),
      "released": config.Reward(reward_released_in_box, weight=2.0),
      "retreat": config.Reward(reward_retreat, weight=1.0),
      # Posture: gripper pointing down, jaws closing left-right.
      "gripper_down": config.Reward(reward_gripper_down, weight=0.5),
      "jaws_level": config.Reward(reward_jaws_level, weight=0.5),
      # Safety.
      "box_contact": config.Reward(reward_box_contact, weight=1.0),
      **penalties,
  }
  cfg.task_layer.rewards = rewards

  cfg.task_layer.metrics = {
      "success": config.Metric(metric_success, reduce="last"),
      "in_box": config.Metric(metric_in_box, reduce="last"),
      "cube_lifted": cfg.task_layer.metrics["cube_lifted"],
      "reaching_dist": cfg.task_layer.metrics["reaching_dist"],
      "box_xy_err": config.Metric(metric_box_xy_err, reduce="last"),
      "gripper_down_deg": config.Metric(metric_gripper_down_deg, reduce="mean"),
  }
  return cfg
