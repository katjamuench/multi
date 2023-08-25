from typing import Callable, Dict, List, Optional, Tuple, Type, Union
import numpy as np
import torch
import torch.nn as nn
import gym
from gym import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from spr_rl.agent.params import Params
from stable_baselines3.common.policies import ActorCriticPolicy

class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict):
        super().__init__(observation_space, features_dim=1)

        extractors = {}
        num_sending_agents = 11
        total_concat_size = 0
        for key, subspace in observation_space.items():
            if key == "messages":
                extractors[key] = nn.Linear(subspace.shape[0], 11)
                total_concat_size += 11
            elif key == "observations":
                extractors[key] = nn.Linear(subspace.shape[0], 17)
                total_concat_size += 17
        #self.messages = nn.ModuleDict(extractors[messages])
        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

        def forward(self, observations) -> th.Tensor:
            encoded_tensor_list = []

            # self.extractors contain nn.Modules that do all the processing.
            for key, extractor in self.extractors.items():
                encoded_tensor_list.append(extractor(observations[key]))
            # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
            return th.cat(encoded_tensor_list, dim=1)

class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs,
    ):
        # Disable orthogonal initialization
        kwargs["ortho_init"] = False
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )


    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomCombinedExtractor(self.features_dim)



class MessageAggregator(nn.Module):
    def __init__(self, key_dim, value_dim, hidden_dim):
        super().__init__()

        self.key_dim = self.query_dim = key_dim
        self.value_dim = value_dim
        self.message_dim = self.key_dim + self.value_dim

        self.hidden_dim = hidden_dim

        self.query_predictor = nn.Linear(
            in_features=self.hidden_dim, out_features=self.query_dim, bias=False
        )
        self.scale = 1.0 / np.sqrt(self.query_dim)

    def forward(self, messages, hidden_states):
        # fmt: off
        assert messages.ndim == 3       # (B, T, Na * Dm)
        assert hidden_states.ndim == 3  # (B, T, Dh)
        B, T, joint_message_dim = messages.shape

        messages = messages.view(B, T, -1, self.message_dim)                   # (B, T, Na, Dm)
        keys, values = messages.split([self.key_dim, self.value_dim], dim=-1)  # (B, T, Na, *)

        queries = self.query_predictor(hidden_states)  # (B, T, Dq)
        queries = queries.unsqueeze(dim=-2)            # (B, T, 1, Dq)

        attns = self.scale * torch.matmul(queries, keys.transpose(-1, -2))  # (B, T, 1, Na)
        attns = attns.softmax(dim=-1)                                       # (B, T, 1, Na)
        outputs = torch.matmul(attns, values)                               # (B, T, 1, Dv)
        outputs = outputs.squeeze(dim=-2)                                   # (B, T, Dv)
        # fmt: on

        return outputs


