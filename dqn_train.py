import argparse
import os
# from dueling_dqn.agent import dqn_agent
from dueling_dqn.simple_agent import dqn_agent
from simulator import Simulator

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # env name
    parser.add_argument("--env_name", default="MAPF", type=str)
    # use cuda
    parser.add_argument("--cuda", default=True, type=bool)
    # replay buffer size
    parser.add_argument("--buffer_size", default=20000, type=int)
    # learning rate
    parser.add_argument("--lr", default=1e-3, type=float)
    # bath size
    parser.add_argument("--bath_size", default=32, type=int)
    # gamma
    parser.add_argument("--gamma", default=0.95, type= float)
    # start learning time
    parser.add_argument("--learning_starts", default=500, type=int)
    # train frequency
    parser.add_argument("--train_freq", default=1, type=int)
    # target_network_update_freq
    parser.add_argument("--target_network_update_freq", default=2, type=int)
    # target network update tau
    parser.add_argument("--tau", default=0.005, type=float)
    # use double dqn
    parser.add_argument("--use_double_net", default=True, type=bool)
    # exploration fraction
    parser.add_argument("--exploration_fraction", default=0.9, type=float)  # 0.7
    # random exploration init ratio
    parser.add_argument("--init_ratio", default=0.8, type=float)  # 0.7
    # random exploration final ratio
    parser.add_argument("--final_ratio", default=0.1, type=float)
    # max time steps
    parser.add_argument("--total_timesteps", default=5e7, type=int)  #5e5
    # save dir
    parser.add_argument("--save_dir", default='./models', type=str)
    # save model frequency
    parser.add_argument("--display_interval", default=5_000, type=int)
    # load model to continue
    parser.add_argument("--load_model", default=False, type=bool)
    # number of robots
    parser.add_argument("--num_robots", default=3, type=int)
    # number of obstacles
    parser.add_argument("--num_obstacles", default=2, type=int)
    # map size
    parser.add_argument("--map_size", default=5, type=int)
    args = parser.parse_args()

    num_robots = args.num_robots
    obstacle_num=args.num_obstacles
    actions_per_robot = 5
    env = Simulator((35*args.map_size+1,35*args.map_size+1,3),num_robots, obstacle_num,visual=False)  # 601
    observation_per_robot = env.observation_per_robot
    model = dqn_agent(env, actions_per_robot, num_robots, observation_per_robot*num_robots,args)
    if args.load_model:
        model_path = os.path.join(args.save_dir, args.env_name)
        model.load_dict(model_path+"/model_152000.pt")
    model.learn_one()
