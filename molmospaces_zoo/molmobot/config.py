from molmo_spaces.policy.base_policy import PolicyFactory
import torch
from huggingface_hub import snapshot_download

from molmo_spaces.configs.robot_configs import FrankaRobotConfig, ActionNoiseConfig
from molmo_spaces.configs.policy_configs import BasePolicyConfig, make_lenient
from molmo_spaces.evaluation.configs.evaluation_configs import JsonBenchmarkEvalConfig

from molmospaces_zoo.molmobot.policy import MolmoBotPolicy


class MolmoBotDroidPolicyConfig(BasePolicyConfig):
    policy_type: str = "learned"
    action_type: str = "joint_pos"
    policy_cls: type = MolmoBotPolicy
    policy_factory: PolicyFactory = make_lenient(MolmoBotPolicy)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint_path: str = "hf://allenai/MolmoBot-DROID"
    camera_names: list[str] = ["exo_camera_1", "wrist_camera"]
    action_move_group_names: list[str] = ["arm", "gripper"]
    action_spec: dict[str, int] = {
        "arm": 7,  # 7-DOF arm joint positions
        "gripper": 1,  # gripper position
    }
    action_horizon: int = 16  # Number of action steps predicted per chunk
    execute_horizon: int = 8  # Number of actions to execute before re-querying

    clamp_gripper: bool = True
    gripper_representation_count: int = 1  # Number of gripper state values to input

    states_mode: str = "cross_attn"
    relative_max_joint_delta: list[float] | None = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        if self.checkpoint_path.startswith("hf://"):
            hf_repo = self.checkpoint_path[len("hf://") :]
            self.checkpoint_path = snapshot_download(hf_repo)


class MolmoBotDroidEvalConfig(JsonBenchmarkEvalConfig):
    policy_config: MolmoBotDroidPolicyConfig = MolmoBotDroidPolicyConfig()
    robot_config: FrankaRobotConfig = FrankaRobotConfig(
        action_noise_config=ActionNoiseConfig(enabled=False)
    )
    policy_dt_ms: float = 66.0
