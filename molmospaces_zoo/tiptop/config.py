from molmo_spaces.configs.policy_configs import BasePolicyConfig, make_lenient
from molmo_spaces.configs.robot_configs import ActionNoiseConfig, FrankaRobotConfig
from molmo_spaces.evaluation.configs.evaluation_configs import JsonBenchmarkEvalConfig
from molmo_spaces.policy.base_policy import PolicyFactory

from molmospaces_zoo.tiptop.policy import TiptopPolicy


class TiptopPolicyConfig(BasePolicyConfig):
    policy_type: str = "tamp"
    policy_cls: type = TiptopPolicy
    policy_factory: PolicyFactory = make_lenient(TiptopPolicy)
    remote_config: dict = dict(host="localhost", port=8765, max_retries=5)

    # TiPToP requires depth from the wrist camera.
    force_enable_depth: bool = True

    # Arm moves here before the image capture that is sent to the TiPToP server.
    # Set to a list of 7 joint angles (radians) to enable; None disables the feature.
    cam_obs_qpos: list[float] | None = None
    # Number of interpolation steps to reach cam_obs_qpos (each step = one policy dt).
    cam_obs_n_steps: int = 200


class TiptopEvalConfig(JsonBenchmarkEvalConfig):
    robot_config: FrankaRobotConfig = FrankaRobotConfig(
        action_noise_config=ActionNoiseConfig(enabled=False)
    )
    policy_config: TiptopPolicyConfig = TiptopPolicyConfig(
        # This is the pose that the arm will move to at the beginning of the trajectory,
        # so that the wrist camera has a clear view of the scene.
        cam_obs_qpos=[0.0, -1.0, 0.0, -1.0, 0.0, 1.0, -3.0],
        cam_obs_n_steps=200,
    )
    policy_dt_ms: float = 20.0
