import argparse
import os
# from dueling_dqn.agent import dqn_agent
from dueling_dqn.simple_agent import dqn_agent
from simulator import Simulator
import datetime
"""


"""

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
    parser.add_argument("--exploration_fraction", default=0.7, type=int)
    # random exploration init ratio
    parser.add_argument("--init_ratio", default=0.7, type=float)
    # random exploration final ratio
    parser.add_argument("--final_ratio", default=0.1, type=float)
    # max time steps
    parser.add_argument("--total_timesteps", default=5e6, type=int)
    # parser.add_argument("--total_timesteps", default=5e5, type=int)

    # save dir
    parser.add_argument("--save_dir", default='./models', type=str)
    # save model frequency
    parser.add_argument("--display_interval", default=1000, type=int)
    # load model to continue
    parser.add_argument("--load_model", default=False, type=bool)

    # Close visualize for headless ssh connection
    parser.add_argument("--ssh", default=False, type=bool)
    parser.add_argument("--penalty_only", default=False, type=bool)

    parser.add_argument("--map_size", default=10, type=int)


    args = parser.parse_args()


    timestamp = datetime.datetime.now().strftime("%m%d_%H%M%S")
        

    # args.env_name += f"_{args.map_size}x{args.map_size}_{str(timestamp)}"

    # os.mkdir(f"models/{args.env_name}")

    print(f"args.env_name: {args.env_name}")
    print(f"args.penalty_only: {args.penalty_only}")
    print(f"args.load_model: {args.load_model}")


    # exit(0)


    # env = Simulator((args.map_size*35+1,args.map_size*35+1,3), 5, args=args)
    env = Simulator((args.map_size*35+1,args.map_size*35+1,3), 1, args=args)

    model = dqn_agent(env, args)
    if args.load_model:
        model_path = os.path.join(args.save_dir, "MAPF_Prior/model_304000_231_30.pt")
        model.load_dict(model_path)
        # model.load_dict(model_path+"/../MAPF_Prior/model_249000_231_0.pt")

        # model.load_dict(model_path+"/../MAPF_Prior/model0.980559229850769.pt")

        # 


        # model.load_dict(model_path+"/../MAPF_Prior/model40.983375549316406.pt")
        # model.load_dict(model_path+"/../MAPF_Prior/model1804.689453125.pt")

    model.learn_one()
