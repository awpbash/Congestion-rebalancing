from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
import torch as th
import torch.nn as nn
from torch.distributions import Categorical


class LargeFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=512):
        super(LargeFeatureExtractor, self).__init__(observation_space, features_dim)
        n_input = observation_space.shape[0]
        self.net = nn.Sequential(
            nn.Linear(n_input, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
        )

    def forward(self, observations):
        return self.net(observations)


class LargeBranchingPPOPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, net_arch=None, **kwargs):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=[dict(pi=[512, 512, 256, 128], vf=[512, 512, 256, 128])],
            features_extractor_class=LargeFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=512),
            **kwargs
        )

        self.branch_output_sizes = action_space.nvec.tolist()
        latent_pi_dim = 128  # last hidden layer size of the pi network
        self.policy_branches = nn.ModuleList([
            nn.Linear(latent_pi_dim, branch_size) for branch_size in self.branch_output_sizes
        ])

        self._build(lr_schedule)

    def forward(self, obs, deterministic=False):
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        values = self.value_net(latent_vf)

        logits_per_branch = [branch(latent_pi) for branch in self.policy_branches]
        distributions = [th.distributions.Categorical(logits=logits) for logits in logits_per_branch]

        if deterministic:
            actions = [dist.probs.argmax(dim=-1) for dist in distributions]
        else:
            actions = [dist.sample() for dist in distributions]

        actions_tensor = th.stack(actions, dim=1)

        # Sum the log probs of each branch
        log_probs = th.stack([dist.log_prob(action) for dist, action in zip(distributions, actions)], dim=1).sum(dim=1)

        return actions_tensor, values, log_probs


    def _get_action_dist_from_latent(self, latent_pi):
        logits_per_branch = [branch(latent_pi) for branch in self.policy_branches]
        return [Categorical(logits=logits) for logits in logits_per_branch]

    def evaluate_actions(self, obs, actions):
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        values = self.value_net(latent_vf)

        distributions = self._get_action_dist_from_latent(latent_pi)

        log_probs_per_branch = []
        entropy_per_branch = []

        for dist, action_branch in zip(distributions, actions.T):
            log_probs_per_branch.append(dist.log_prob(action_branch))
            entropy_per_branch.append(dist.entropy())

        log_probs = th.stack(log_probs_per_branch, dim=1).sum(dim=1)
        entropy = th.stack(entropy_per_branch, dim=1).mean(dim=1)

        return values, log_probs, entropy
