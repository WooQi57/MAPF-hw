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
    parser.add_argument("--load_model", default=True, type=bool)

    # Close visualize for headless ssh connection
    parser.add_argument("--ssh", default=False, type=bool)
    parser.add_argument("--penalty_only", default=False, type=bool)

    parser.add_argument("--map_size", default=10, type=int)

    parser.add_argument("--num_robot", default=1, type=int)

    parser.add_argument("--reset_seed_inprocess", action='store_true', default=False)


    parser.add_argument("--debug", action='store_true', default=False)

    parser.add_argument("--moveAll", action='store_true', default=False)




    args = parser.parse_args()

    timestamp = datetime.datetime.now().strftime("%m%d_%H%M%S")

    print(f"args.env_name: {args.env_name}")
    print(f"args.penalty_only: {args.penalty_only}")
    print(f"args.load_model: {args.load_model}")



    model_path = ""


    args.num_robot=1

    # model_path = "/home/ziang/Public/AA228_Project/MAPF-hw/models/MAPF_ResetSeed/M_10x10_1205_185116/model_E_869000_R_90_L_23.pt"

    # model_path = "/home/ziang/Public/AA228_Project/MAPF-hw/models/MAPF/M_10x10_1205_182729/model_E_903000_R_96_L_0.pt"



    args.num_robot=4

    # RESET SEED TO 5457 at every time ...
    # model_path = "/home/ziang/Public/AA228_Project/MAPF-hw/models/MAPF_Static_4/M_10x10_1205_182846/model_E_582000_R_74_L_2.pt"
    # model_path = "/home/ziang/Public/AA228_Project/MAPF-hw/models/MAPF_Static_4/M_10x10_1205_182846/model_E_574000_R_74_L_5.pt"
    # model_path=  "/home/ziang/Public/AA228_Project/MAPF-hw/models/MAPF_Static_4/M_10x10_1205_182846/model_E_159000_R_33_L_4.pet"

    # model_path = "/home/ziang/Public/AA228_Project/MAPF-hw/models/MAPF_Static_4/M_10x10_1206_000841/model_E_62000_R_8_L_58.pt"

    # model_path = "/home/ziang/Public/AA228_Project/MAPF-hw/models/MAPF_Static_4_ResetSeed/M_10x10_1205_184623/model_E_574000_R_79_L_7.pt"
    # model_path = "/home/ziang/Public/AA228_Project/MAPF-hw/models/OLD_MAPF_Static_4_32_20000/M_10x10_1205_165856/model_E_123000_R_40_L_1.pt"




    ## LATEST SEED
    # model_path = "/home/ziang/Public/AA228_Project/MAPF-hw/models/MAPF_Static_4/M_10x10_1206_000841/model_E_554000_R_76_L_3.pt"

    
    if model_path != "":
        assert (not args.moveAll), "args.moveAll:"


    ### ALL MOVE TOGETHER:
    if args.moveAll:

        # model_path = "/home/ziang/Public/AA228_Project/MAPF-hw/models_prior/MAPF_Dynamic_4/M_10x10/AllMove_4/model_E_962000_R_12_L_0.pt"
        # model_path = "/home/ziang/Public/AA228_Project/MAPF-hw/models_prior/MAPF_Dynamic_4/M_10x10/AllMove_4/model_E_966000_R_11_L_9.pt"

        model_path = "/home/ziang/Public/AA228_Project/MAPF-hw/models_prior/MAPF_Dynamic_4/M_10x10/AllMove_4_SameSeed/model_E_928000_R_9_L_0.pt"
        # model_path = "/home/ziang/Public/AA228_Project/MAPF-hw/models_prior/MAPF_Dynamic_4/M_10x10/AllMove_4_SameSeed/model_E_929000_R_8_L_1.pt"

    assert model_path != "", "model_path != 0"



    env = Simulator((args.map_size*35+1,args.map_size*35+1,3), args.num_robot, args=args)
    model = dqn_agent(env, args)
    if args.load_model:
        model.load_dict(model_path)
    else:
        print("You shold Load model!!!")
        exit(0)
    # model.learn_one()

    if args.moveAll:
        model.render(random_level=0.1)
    else:
        model.render_one(random_level=0.1)


