import numpy as np
import torch as th
from torch import nn
import collections
import copy
import warnings
from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from spr_rl.agent.params import Params
from stable_baselines3.common.policies import ActorCriticPolicy
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.preprocessing import get_flattened_obs_dim, is_image_space
from stable_baselines3.common.type_aliases import TensorDict
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    Distribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
    make_proba_distribution,
)
from stable_baselines3.common.preprocessing import get_action_dim, is_image_space, maybe_transpose, preprocess_obs
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    MlpExtractor,
    NatureCNN,
    create_mlp,
)
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.utils import get_device, is_vectorized_observation, obs_as_tensor
SelfBaseModel = TypeVar("SelfBaseModel", bound="BaseModel")


class CombinedExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: spaces.Dict,
        cnn_output_dim: int = 256,
        normalized_image: bool = False,
    ) -> None:
        super().__init__(observation_space, features_dim=1)

        extractors: Dict[str, nn.Module] = {}
        messages = []
        total_concat_size = 0
        for key, subspace in observation_space.spaces.items():
            extractors[key] = nn.Flatten()
            total_concat_size += get_flattened_obs_dim(subspace)
            if key == "messages":
                messages.append(subspace)
        self.messages = nn.ModuleList(messages)
        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations: TensorDict) -> th.Tensor:
        encoded_tensor_list = []

        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        return th.cat(encoded_tensor_list, dim=1)


class MultiInputActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = CombinedExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            use_sde,
            log_std_init,
            full_std,
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            share_features_extractor,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )

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


class TarMACModel(TorchRNN, nn.Module):
    def __init__(
        self,
        observation_space,
        action_space,
        num_outputs,
        model_config,
        name,
        # Extra MAPPOModel arguments
        actor_hiddens=None,
        actor_hidden_activation='tanh',
        critic_hiddens=None,
        critic_hidden_activation='tanh',
        lstm_cell_size=256,
        # Extra TarMACModel arguments
        message_key_dim=32,
        message_value_dim=32,
        critic_use_global_state=True,
        **kwargs,
    ):
        if actor_hiddens is None:
            actor_hiddens = [256, 256]

        if critic_hiddens is None:
            critic_hiddens = [256, 256]

        nn.Module.__init__(self)
        super().__init__(observation_space, action_space, num_outputs, model_config, name)
        self.observation_space = observation_space['nn_state']
        self.message_space = observation_space['m_state']
        self.flat_obs_dim = get_space_flat_size(self.observation_space)

