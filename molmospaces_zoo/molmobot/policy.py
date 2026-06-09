from dataclasses import dataclass

import numpy as np
from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig
from molmo_spaces.policy.base_policy import InferencePolicy, StatefulPolicy

from olmo.models.molmobot.inference_wrapper import SynthManipMolmoInferenceWrapper


@dataclass
class MolmoBotPolicyState:
    action_buffer: list[dict[str, np.ndarray]] | None = None
    buffer_index: int = 0
    step_count: int = 0
    obs_history: list[dict] | None = None


class MolmoBotPolicy(InferencePolicy, StatefulPolicy):
    def __init__(self, config: MlSpacesExpConfig):
        super().__init__(config)
        self.camera_names = config.policy_config.camera_names
        self.action_move_group_names = config.policy_config.action_move_group_names
        self.action_spec = config.policy_config.action_spec
        self.action_horizon = config.policy_config.action_horizon
        self.execute_horizon = config.policy_config.execute_horizon
        self.action_type = config.policy_config.action_type
        self.relative_max_joint_delta = config.policy_config.relative_max_joint_delta
        if self.relative_max_joint_delta is not None:
            self.relative_max_joint_delta = np.array(self.relative_max_joint_delta)

        self.action_buffer: list[dict[str, np.ndarray]] = []
        self.buffer_index = 0
        self.step_count = 0
        self.obs_history: list[dict] = []
        self._prepared = False

        self.prepare_model()

        self.input_window_size = self.agent.model_config.n_obs_steps
        self.obs_step_delta = self.agent.model_config.obs_step_delta

    def get_state(self):
        return MolmoBotPolicyState(
            action_buffer=self.action_buffer,
            buffer_index=self.buffer_index,
            step_count=self.step_count,
            obs_history=self.obs_history,
        )

    def set_state(self, state: MolmoBotPolicyState):
        self.action_buffer = state.action_buffer
        self.buffer_index = state.buffer_index
        self.step_count = state.step_count
        self.obs_history = state.obs_history if state.obs_history is not None else []

    def prepare_model(self):
        if self._prepared:
            return
        self._prepared = True

        checkpoint_path = self.config.policy_config.checkpoint_path
        self.agent = SynthManipMolmoInferenceWrapper(
            checkpoint_path=checkpoint_path, states_mode=self.config.policy_config.states_mode
        )

    def reset(self):
        self.action_buffer = []
        self.buffer_index = 0
        self.step_count = 0
        self.obs_history = []

    def _populate_action_buffer(self, observation) -> None:
        """Call agent to get new action chunk and populate the buffer."""
        obs = observation[0] if isinstance(observation, list) else observation

        # Extract images from observations
        images = []
        for cam_name in self.camera_names:
            if cam_name not in obs:
                raise KeyError(
                    f"Camera '{cam_name}' not in observation. Available: {list(obs.keys())}"
                )

            cam_images = []
            # Simple case: single frame
            if self.input_window_size == 1:
                cam_images.append(obs[cam_name])
            else:
                # Multiple frames: calculate history indices using reference logic
                current_history_len = len(self.obs_history)

                # Only proceed if we have history images
                if current_history_len > 0:
                    # Calculate frame indices relative to current step (like reference implementation)
                    current_step = (
                        current_history_len - 1
                    )  # Current step is the last index in history

                    for i in range(self.input_window_size):
                        # Use the same logic as _get_camera_frames in synthmanip_dataset
                        frame_idx = (
                            current_step - (self.input_window_size - 1 - i) * self.obs_step_delta
                        )

                        # Only add valid indices (no padding)
                        if 0 <= frame_idx < current_history_len:
                            cam_images.append(self.obs_history[frame_idx][cam_name])

                assert cam_images, "No frames found when generating observations"

            # Always send a list of images
            images.extend(cam_images)

        # Extract qpos state
        qpos_parts = []
        for group_name in self.action_move_group_names:
            if "gripper" not in group_name:
                qpos_parts.append(obs["qpos"][group_name])
            else:
                qpos_parts.append(
                    obs["qpos"][group_name][
                        : self.config.policy_config.gripper_representation_count
                    ]
                )

        state = np.concatenate(qpos_parts).astype(np.float32)

        if "task" in obs:
            goal = obs["task"]
        else:
            goal = self.task.get_task_description()

        # Call agent
        pred_actions = self.agent.get_action_chunk(
            images=images,
            task_description=goal,
            state=state,
        )

        # Convert to list of action dicts and store in buffer
        self.action_buffer = []
        for t in range(pred_actions.shape[0]):
            action = {}
            start_idx = 0
            for group_name in self.action_move_group_names:
                dim = self.action_spec[group_name]
                selected_action = pred_actions[t, start_idx : start_idx + dim]
                if "gripper" in group_name and self.config.policy_config.clamp_gripper:
                    action[group_name] = np.where(selected_action > 128, 255, 0).astype(
                        selected_action.dtype
                    )
                else:
                    action[group_name] = pred_actions[t, start_idx : start_idx + dim]
                start_idx += dim
            self.action_buffer.append(action)

        self.buffer_index = 0

    def obs_to_model_input(self, obs) -> dict[str, np.ndarray]:
        return obs

    def model_output_to_action(self, model_output) -> dict[str, np.ndarray]:
        return model_output

    def inference_model(self, model_input) -> dict[str, np.ndarray]:
        """Return single action from buffer, refreshing when needed."""
        # Add current observation to history
        obs = model_input[0] if isinstance(model_input, list) else model_input
        self.obs_history.append(obs)

        # Refresh buffer if empty or executed enough actions
        if self.buffer_index >= self.execute_horizon or not self.action_buffer:
            self._populate_action_buffer(model_input)

        action = self.action_buffer[self.buffer_index]

        self.buffer_index += 1
        self.step_count += 1

        if self.action_type == "joint_pos_rel":
            predicted_deltas = action["arm"][:7]

            relative_scale = np.abs(predicted_deltas) / self.relative_max_joint_delta
            if np.max(relative_scale) > 1:
                scaled_predicted_deltas = predicted_deltas / np.max(relative_scale)
                action["arm"][:7] = scaled_predicted_deltas

        else:
            # calculate joint deltas
            obs = model_input[0] if isinstance(model_input, list) else model_input
            predicted_deltas = action["arm"][:7] - obs["qpos"]["arm"]

            # Find the largest value
            relative_scale = np.abs(predicted_deltas) / self.relative_max_joint_delta

            if np.max(relative_scale) > 1:
                scaled_predicted_deltas = predicted_deltas / np.max(relative_scale)
                action["arm"][:7] = obs["qpos"]["arm"] + scaled_predicted_deltas

        return action
