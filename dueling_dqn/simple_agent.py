import copy
import os
import numpy as np
import torch
import datetime

from torch.utils.tensorboard import SummaryWriter 
from dueling_dqn.utils import linear_schedule, replay_buffer, reward_recoder, select_action, set_init
from .model import simplenet

log_dir = './dueling_dqn/logs'
index = len(os.listdir(log_dir))
log_path = './dueling_dqn/logs/log'+str(index)
print(f"writing log in {log_path}")
writer = SummaryWriter(log_path)
torch.set_default_device('cuda')

class dqn_agent:
    def __init__(self, env, actions_per_robot, num_robots, num_observation ,args):
        self.env = env
        self.net = simplenet(actions_per_robot*num_robots, num_observation)
        self.actions_per_robot = actions_per_robot
        self.num_robots = num_robots
        self.args = args
        self.target_net = copy.deepcopy(self.net)
        self.target_net.load_state_dict(self.net.state_dict())
        
        if self.args.cuda:
            print(f"cuda available: {torch.cuda.is_available()}")
            self.net.cuda()
            self.target_net.cuda()

        
    def learn(self):
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.args.lr)
        self.buffer = replay_buffer(self.args.buffer_size)
        self.exploration_schedule = linear_schedule(int(self.args.total_timesteps * self.args.exploration_fraction), self.args.final_ratio, self.args.init_ratio)
        
        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)
        # set the environment folder
        self.model_path = os.path.join(self.args.save_dir, self.args.env_name)
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
        episode_reward = reward_recoder(self.env.robot_num)
        obs = self.env.reset()
        td_loss = 0
        for timestep in range(int(self.args.total_timesteps)):
            explore_eps = self.exploration_schedule.get_value(timestep)
            with torch.no_grad():
                obs_tensor = self._get_tensors(obs)
                action_value = self.net(obs_tensor)
            action = select_action(action_value, explore_eps)
            if self.env.robot_num == 1:
                action = [action]
            reward, obs_, done, _ = self.env.step(action)

            for i in range(len(obs)):
                self.buffer.add(obs[i], action[i], reward[i], obs_[i], float(done[i]))
                episode_reward.add_reward(reward[i],i)
            obs = obs_
            
            done = np.array(done).any()
            if done:
                obs = np.array(self.env.reset())
                writer.add_scalar("latest reward",episode_reward.latest[0], global_step=episode_reward.num_episodes)
                writer.add_scalar("random exploration",explore_eps,global_step=episode_reward.num_episodes)
                writer.add_scalar("mean reward", episode_reward.mean, global_step=episode_reward.num_episodes, walltime=None)
                episode_reward.start_new_episode() 
            
            if timestep > self.args.learning_starts and timestep % self.args.train_freq == 0:
                batch_sample = self.buffer.sample(self.args.bath_size)
                td_loss = self._update_network(batch_sample)
                writer.add_scalar("loss", td_loss, global_step=timestep, walltime=None)
                
            if timestep > self.args.learning_starts and timestep % self.args.target_network_update_freq == 0:
                for param, target_param in zip(self.net.parameters(), self.target_net.parameters()):
                    target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)
                # self.target_net.load_state_dict(self.net.state_dict())
            
            if done and episode_reward.num_episodes % self.args.display_interval == 0:
                print('[{}] Frames: {}, Episode: {}, Mean: {:.3f}, Loss: {:.3f}'.format(datetime.datetime.now(), timestep, episode_reward.num_episodes, \
                        episode_reward.mean, td_loss))
                torch.save(self.net.state_dict(), self.model_path + '/model'+str(td_loss)+'.pt')
    
    def learn_one(self):
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.args.lr)
        self.buffer = replay_buffer(self.args.buffer_size)
        self.exploration_schedule = linear_schedule(int(self.args.total_timesteps * self.args.exploration_fraction), self.args.final_ratio, self.args.init_ratio)
        
        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)
        # set the environment folder
        from datetime import datetime
        current_time = datetime.now()
        current_time_string = current_time.strftime("%Y-%m-%d-%H-%M-%S")
        self.model_path = os.path.join(self.args.save_dir, self.args.env_name+current_time_string)
        print(f"model saved in {self.model_path}")
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
        episode_reward = reward_recoder(self.env.robot_num)  # 
        obs = self.env.reset()
        td_loss = 0
        for timestep in range(int(self.args.total_timesteps)):
            explore_eps = self.exploration_schedule.get_value(timestep)
            with torch.no_grad():
                obs_tensor = self._get_tensors(obs)
                action_value = self.net(obs_tensor)
            action = select_action(action_value, explore_eps, self.actions_per_robot)
            reward, obs_, done_arr, _ = self.env.step(action)

            # Enumerate robot and compute data point for fitting
            # Q(s,a):  s: a large observation; a: action (0-actions_per_robot*num_robots-1) Q: reward for robot corresponding to action
            for i in range(self.num_robots):
                self.buffer.add(obs[0], i*self.actions_per_robot+action[i], reward[i], obs_[0], done_arr[i])
                # TODO change action it makes no sense now
                episode_reward.add_reward(reward[i],i)
            obs = obs_
            done = done_arr.any()
            if done:
                obs = np.array(self.env.reset())
                writer.add_scalar("mean reward", episode_reward.mean, global_step=episode_reward.num_episodes, walltime=None)
                writer.add_scalar("latest reward",episode_reward.latest[0], global_step=episode_reward.num_episodes)
                writer.add_scalar("random exploration",explore_eps,global_step=episode_reward.num_episodes)
                episode_reward.start_new_episode() 
            
            if timestep > self.args.learning_starts and timestep % self.args.train_freq == 0:
                batch_sample = self.buffer.sample(self.args.bath_size)
                td_loss = self._update_network(batch_sample)
                writer.add_scalar("loss", td_loss, global_step=timestep, walltime=None)
                
            if timestep > self.args.learning_starts and timestep % self.args.target_network_update_freq == 0:
                for param, target_param in zip(self.net.parameters(), self.target_net.parameters()):
                    target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)
                # self.target_net.load_state_dict(self.net.state_dict())
            
            if done and episode_reward.num_episodes % self.args.display_interval == 0:
                print('[{}] Frames: {}, Episode: {}, Mean: {:.3f}, Loss: {:.3f}, eps: {}'.format(datetime.now(), timestep, episode_reward.num_episodes, \
                        episode_reward.mean, td_loss, explore_eps))
                model_name = f"/model_{episode_reward.num_episodes}.pt"
                torch.save(self.net.state_dict(), self.model_path + model_name)                
        model_name = f"/model_{episode_reward.num_episodes}.pt"
        torch.save(self.net.state_dict(), self.model_path + model_name)

    def learn_fixed_path(self):
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.args.lr)
        self.buffer = replay_buffer(self.args.buffer_size)
        self.exploration_schedule = linear_schedule(int(self.args.total_timesteps * self.args.exploration_fraction), self.args.final_ratio, self.args.init_ratio)
        
        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)
        # set the environment folder
        from datetime import datetime
        current_time = datetime.now()
        current_time_string = current_time.strftime("%Y-%m-%d-%H-%M-%S")
        self.model_path = os.path.join(self.args.save_dir, self.args.env_name+current_time_string)
        print(f"model saved in {self.model_path}")
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
        episode_reward = reward_recoder(1)  # self.env.robot_num
        obs = self.env.reset()
        td_loss = 0
        for timestep in range(int(self.args.total_timesteps)):
            explore_eps = self.exploration_schedule.get_value(timestep)
            with torch.no_grad():
                obs_tensor = self._get_tensors(obs)
                action_value = self.net(obs_tensor)
            action = select_action(action_value, explore_eps, self.actions_per_robot)
            reward, obs_, done, _ = self.env.step(action)

            # Enumerate robot and compute data point for fitting
            # Q(s,a):  s: a large observation; a: action (0-actions_per_robot*num_robots-1) Q: reward for robot corresponding to action
            action_id = action[0]+action[1]*self.actions_per_robot+action[1]*self.actions_per_robot**2  # TODO 改进制
            self.buffer.add(obs[0], action_id, reward[0], obs_[0], done)# TODO done? done_arr  
            # TODO change action to an array of actions_per_robot^num_robots
            # TODO change reward to a single scalar
            # TODO change done to a single scalar
            episode_reward.add_reward(reward,0)  # TODO reward scalar
            obs = obs_
            # TODO avg reward problem  
            if done:
                obs = np.array(self.env.reset())
                writer.add_scalar("latest reward",episode_reward.latest[0], global_step=episode_reward.num_episodes)
                writer.add_scalar("random exploration",explore_eps,global_step=episode_reward.num_episodes)
                writer.add_scalar("mean reward", episode_reward.mean, global_step=episode_reward.num_episodes, walltime=None)
                episode_reward.start_new_episode() 
            
            if timestep > self.args.learning_starts and timestep % self.args.train_freq == 0:
                batch_sample = self.buffer.sample(self.args.bath_size)
                td_loss = self._update_network(batch_sample)
                writer.add_scalar("loss", td_loss, global_step=timestep, walltime=None)
                
            if timestep > self.args.learning_starts and timestep % self.args.target_network_update_freq == 0:
                for param, target_param in zip(self.net.parameters(), self.target_net.parameters()):
                    target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)
                # self.target_net.load_state_dict(self.net.state_dict())
            
            if done and episode_reward.num_episodes % self.args.display_interval == 0:
                print('[{}] Frames: {}, Episode: {}, Mean: {:.3f}, Loss: {:.3f}, eps: {}'.format(datetime.now(), timestep, episode_reward.num_episodes, \
                        episode_reward.mean, td_loss, explore_eps))
                model_name = f"/model_{episode_reward.num_episodes}.pt"
                torch.save(self.net.state_dict(), self.model_path + model_name)

    def load_dict(self, path):
        state_dict = torch.load(path)
        self.net.load_state_dict(state_dict)
        print("load successful")
    
    def _update_network(self, samples):
        obses, actions, rewards, obses_next, dones = samples
        # convert the data to tensor
        obses = self._get_tensors(obses)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(-1)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(-1)
        obses_next = self._get_tensors(obses_next)
        dones = torch.tensor(1 - dones, dtype=torch.float32).unsqueeze(-1)
        # convert into gpu
        if self.args.cuda:
            actions = actions.cuda()
            rewards = rewards.cuda()
            dones = dones.cuda()
        # calculate the target value
        with torch.no_grad():
            # if use the double network architecture
            if self.args.use_double_net:
                q_value_ = self.net(obses_next)
                action_max_idx = torch.argmax(q_value_, dim=1, keepdim=True)
                target_action_value = self.target_net(obses_next)
                target_action_max_value = target_action_value.gather(1, action_max_idx)
            else:
                target_action_value = self.target_net(obses_next)
                target_action_max_value, _ = torch.max(target_action_value, dim=1, keepdim=True)
        # target
        expected_value = rewards + self.args.gamma * target_action_max_value * dones
        # get the real q value
        action_value = self.net(obses)
        real_value = action_value.gather(1, actions)
        loss = (expected_value - real_value).pow(2).mean()
        # start to update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def _get_tensors(self, obs):
        # if obs.ndim == 3:
        #     obs = np.transpose(obs, (2, 0, 1))
        #     obs = np.expand_dims(obs, 0)
        # elif obs.ndim == 4:
        #     obs = np.transpose(obs, (0, 3, 1, 2))
        obs = torch.tensor(np.array(obs), dtype=torch.float32)
        if self.args.cuda:
            obs = obs.cuda()
        return obs
