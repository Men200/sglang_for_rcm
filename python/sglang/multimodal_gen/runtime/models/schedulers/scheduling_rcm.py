import math
from dataclasses import dataclass
from typing import Any

import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import SchedulerMixin

from diffusers.utils import BaseOutput

from sglang.multimodal_gen.runtime.models.schedulers.base import BaseScheduler

@dataclass
class RcmFlowMatchSchedulerOutput(BaseOutput):
    """
    Output class for the scheduler's `step` function output.

    Args:
        prev_sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample `(x_{t-1})` of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
    """

    prev_sample: torch.FloatTensor


class RcmFlowMatchScheduler(BaseScheduler, ConfigMixin, SchedulerMixin):
    _compatibles: list[Any] = []
    config_name = "scheduler_config.json"
    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        sigma_max: float = 80.0,
        mid_t: tuple[float, ...] = (1.5, 1.4, 1.0), # [0.9877, 0.9338, 0.8529, 0.6090, 0.0000]
    ):
        self._shift = 1.0

        self.num_train_timesteps = num_train_timesteps
        self.num_inference_steps: int | None = None

        t_steps = torch.tensor(
            [math.atan(float(sigma_max)), *mid_t, 0.0],
            dtype=torch.float64,
        )
        t_steps = torch.sin(t_steps) / (torch.cos(t_steps) + torch.sin(t_steps))
        self._sigmas = t_steps.to(dtype=torch.float32)
        self.timesteps = self._sigmas * self.num_train_timesteps
        self.init_noise_sigma = self._sigmas[0]

        BaseScheduler.__init__(self)

    def set_begin_index(self, begin_index: int = 0) -> None:
        # T2V-only rCM path is stateless, but denoising stage still calls this hook.
        # Keep it as a no-op for interface compatibility.
        return None

    def set_shift(self, shift: float) -> None:
        self._shift = shift

    def set_timesteps(  # 在 timestep_preparation_stage.py 中被调用
        self,
        num_inference_steps: int = 4,
        device: str | torch.device | None = None,
    ) -> None:
        _SUPPORTED_STEPS = (1, 2, 3, 4)
        if num_inference_steps not in _SUPPORTED_STEPS:
            raise ValueError(
                f"rCM only supports 1, 2, 3, 4 inference steps, got {num_inference_steps}."
            )

        self.num_inference_steps = num_inference_steps
        self.sigmas = torch.cat(
            (self._sigmas[:num_inference_steps], self._sigmas[-1:]), dim=0
        ).to(device=device)
        self.timesteps = (self.sigmas * self.num_train_timesteps).to(device=device)

    def scale_model_input(
        self, sample: torch.Tensor, timestep: int | None = None
    ) -> torch.Tensor:
        return sample

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep_idx: float | torch.Tensor,   # 时间步索引
        sample: torch.FloatTensor,
        return_dict: bool = True,
        generator: torch.Generator | None = None,
    ) -> RcmFlowMatchSchedulerOutput | tuple[torch.FloatTensor, ...]:
        print("------------------------------------------------------------------------", self.timesteps[timestep_idx])

        if timestep_idx == len(self.timesteps) - 1:
            # 最后一步直接返回sample
            return RcmFlowMatchSchedulerOutput(prev_sample=sample) if return_dict else (sample,)
        
        t_cur = self.sigmas[timestep_idx]
        t_next = self.sigmas[timestep_idx + 1]
        
        prev_sample = (1 - t_next) * (sample - t_cur * model_output) + t_next * torch.randn(
            *sample.shape,
            dtype=torch.float32,
            device=sample.device,
            generator=generator[0],
        )

        if not return_dict:
            return (prev_sample,)

        return RcmFlowMatchSchedulerOutput(prev_sample=prev_sample)

EntryClass = RcmFlowMatchScheduler