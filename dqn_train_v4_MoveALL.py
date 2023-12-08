import argparse
import os
# from dueling_dqn.agent import dqn_agent
from dueling_dqn.simple_agent import dqn_agent
from simulator import Simulator
import datetime


def get_args():
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
    parser.add_argument("--exploration_fraction", default=0.2, type=float)
    # random exploration init ratio
    parser.add_argument("--init_ratio", default=0.2, type=float)
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
    parser.add_argument("--load_model", action='store_true', default=False)


    # Close visualize for headless ssh connection
    parser.add_argument("--ssh", action='store_true', default=False)
    parser.add_argument("--penalty_only", action='store_true', default=False)


    parser.add_argument("--map_size", default=10, type=int)
    parser.add_argument("--num_robot", default=4, type=int)
    parser.add_argument("--reset_seed_inprocess", action='store_true', default=False)

    parser.add_argument("--requireDoneAll", action='store_true', default=False)

    parser.add_argument("--oldReward", action='store_true', default=False)

    parser.add_argument("--expand_obs", action='store_true', default=False)

    parser.add_argument("--save_gif", default=None, type=str)

    parser.add_argument("--vis_seed", default=5457, type=int)


    args = parser.parse_args()
    return args

"""


"""

if __name__ == "__main__":
    args = get_args()

    if args.expand_obs:
        args.load_model = False
        args.exploration_fraction = 0.7
        args.init_ratio = 0.7
    else:
        args.load_model = True


    print(args.exploration_fraction, args.init_ratio, args.final_ratio)
    # exit(0)

    timestamp = datetime.datetime.now().strftime("%m%d_%H%M%S")
        

    args.env_name += f"_{args.num_robot}"

    if args.reset_seed_inprocess:
        args.env_name += f"_ResetSeed"

    if args.requireDoneAll:
        args.env_name += f"_RequireDoneAll"


    print(f"args.env_name: {args.env_name}")
    print(f"args.penalty_only: {args.penalty_only}")
    print(f"args.load_model: {args.load_model}")


    # exit(0)

    
    assert args.num_robot > 1, "should assign num_robot > 1"


    # env = Simulator((args.map_size*35+1,args.map_size*35+1,3), 5, args=args)
    env = Simulator((args.map_size*35+1,args.map_size*35+1,3), args.num_robot, args=args)

    model = dqn_agent(env, args)


    

    if args.load_model:
        model_path = "/home/ziang/Public/AA228_Project/MAPF-hw/models_prior/MAPF_Dynamic_4/M_10x10/AllMove_4/model_E_962000_R_12_L_0.pt"


        # model_path = "/home/ziang/Public/AA228_Project/MAPF-hw/models/MAPF_Static_4/M_10x10_1205_143329/model_E_653000_R_27_L_6.pt"
        # model_path = os.path.join(args.save_dir, "MAPF_Prior/model_304000_231_30.pt")
        model.load_dict(model_path)
    # model.learn_one()

    model.learn()



"""
python dqn_train_v4_MoveALL.py --ssh 1 --env_name MAPF_Static --map_size 10 --num_robot 4 > txt_log/v3_Multi_10.txt

python dqn_train_v4_MoveALL.py --ssh --env_name MAPF_Move --map_size 10 --num_robot 4 --requireDoneAll > txt_log/requireDoneAll.txt
python dqn_train_v4_MoveALL.py --env_name MAPF_Move --map_size 10 --num_robot 4 --requireDoneAll > txt_log/requireDoneAll.txt

python dqn_train_v4_MoveALL.py --env_name MAPF_Move --map_size 10 --num_robot 4 --requireDoneAll --oldReward > txt_log/requireDoneAll_oldReward.txt
python dqn_train_v4_MoveALL.py --env_name MAPF_Move --map_size 10 --num_robot 4 --requireDoneAll --ssh --oldReward > txt_log/requireDoneAll_oldReward.txt


python dqn_train_v4_MoveALL.py --ssh 1 --env_name MAPF_Static --map_size 10 --num_robot 4 --reset_seed_inprocess > txt_log/v3_Multi_10_RESET_tmux4.txt


"""