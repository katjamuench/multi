from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from tqdm.auto import tqdm
from .params import Params
from spr_rl.envs.spr_env import SprEnv
#import tensorflow as tf
#from tensorflow.nn import relu, tanh
import csv
import sys
from spr_rl.agent.TarMACPolicy import RecurrentPPO
#from sb3_contrib import RecurrentPPO
from stable_baselines3.common.evaluation import evaluate_policy


# Progress bar code from
# https://colab.research.google.com/github/araffin/rl-tutorial-jnrr19/blob/master/4_callbacks_hyperparameter_tuning.ipynb
class ProgressBarCallback(BaseCallback):
    """
    :param pbar: (tqdm.pbar) Progress bar object
    """
    def __init__(self, pbar):
        super(ProgressBarCallback, self).__init__()
        self._pbar = pbar

    def _on_step(self):
        # Update the progress bar:
        self._pbar.n = self.num_timesteps
        self._pbar.update(0)


# this callback uses the 'with' block, allowing for correct initialisation and destruction
class ProgressBarManager(object):
    def __init__(self, total_timesteps):  # init object with total timesteps
        self.pbar = None
        self.total_timesteps = total_timesteps

    def __enter__(self):  # create the progress bar and callback, return the callback
        self.pbar = tqdm(total=self.total_timesteps)

        return ProgressBarCallback(self.pbar)

    def __exit__(self, exc_type, exc_val, exc_tb):  # close the callback
        self.pbar.n = self.total_timesteps
        self.pbar.update(0)
        self.pbar.close()



class PPO_Agent:

    def __init__(self, params: Params):
        self.params: Params = params

    def create_model(self, n_envs=1):
        """ Create env and agent model """
        env_cls = SprEnv
        self.env = make_vec_env(env_cls, n_envs=n_envs, env_kwargs={"params": self.params}, seed=self.params.seed)
        self.model = RecurrentPPO(
            "MultiInputLstmPolicy",
            self.env,
            seed=self.params.seed
            #,policy_kwargs={"params": self.params}
        )

    def train(self):
        with ProgressBarManager(self.params.training_duration) as callback:
            self.model.learn(
                total_timesteps=self.params.training_duration,
                tb_log_name=self.params.tb_log_name,
                callback=callback)

    def test(self):
        self.params.test_mode = True
        obs = self.env.reset()
        self.setup_writer()
        episode = 1
        step = 0
        episode_reward = [0.0]
        done = False
        # Test for 1 episode
        while not done:
            action, _states = self.model.predict(obs)
            obs, reward, dones, info = self.env.step(action)
            episode_reward[episode - 1] += reward[0]
            if info[0]['sim_time'] >= self.params.testing_duration:
                done = True
                self.write_reward(episode, episode_reward[episode - 1])
                episode += 1
            sys.stdout.write(
                "\rTesting:" +
                f"Current Simulator Time: {info[0]['sim_time']}. Testing duration: {self.params.testing_duration}")
            sys.stdout.flush()
            step += 1
        print("")

    def save_model(self):
        """ Save the model to a zip archive """
        self.model.save(self.params.model_path)

    def load_model(self, path=None):
        """ Load the model from a zip archive """
        if path is not None:
            self.model = RecurrentPPO.load(path)
        else:
            self.model = RecurrentPPO.load(self.params.model_path)
            # Copy the model to the new directory
            self.model.save(self.params.model_path)

    def setup_writer(self):
        episode_reward_filename = f"{self.params.result_dir}/episode_reward.csv"
        episode_reward_header = ['episode', 'reward']
        self.episode_reward_stream = open(episode_reward_filename, 'a+', newline='')
        self.episode_reward_writer = csv.writer(self.episode_reward_stream)
        self.episode_reward_writer.writerow(episode_reward_header)

    def write_reward(self, episode, reward):
        self.episode_reward_writer.writerow([episode, reward])
