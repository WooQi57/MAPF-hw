import argparse
import os
# from dueling_dqn.agent import dqn_agent
from dueling_dqn.simple_agent import dqn_agent
from simulator import Simulator
import datetime

from dqn_train_v4_MoveALL import get_args
"""


"""

if __name__ == "__main__":
    args = get_args()

    args.moveAll = True

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




    ## 





    
    if model_path != "":
        assert (not args.moveAll), "args.moveAll:"


    ### ALL MOVE TOGETHER:
    class_folder = "./models/MAPF_Move_4_RequireDoneAll/"
    if args.expand_obs:
        folder_path = "./M_10x10_1207_235305_expandObs"
        model_path = class_folder + folder_path + "/model_E_220000_Suc_3_Dist_9_LR_-6_MR_-10_L_12.pt"


    else:
        # args.save_gif = "MAPF_Move_4_RequireDoneAll"
        
        folder_path = "./M_10x10_1207_220013_oldReward"
        model_path = class_folder + folder_path + "/model_E_315000_Suc_4_Dist_0_LR_45_MR_8_L_39.pt"
        # model_path = class_folder + folder_path + "/model_E_316000_Suc_4_Dist_0_LR_44_MR_17_L_78.pt"


    args.save_gif = class_folder + folder_path + ".gif"

    args.requireDoneAll = True

    args.load_model = True

    assert model_path != "", "model_path != 0"


    env = Simulator((args.map_size*35+1,args.map_size*35+1,3), args.num_robot, args=args)
    model = dqn_agent(env, args)
    if args.load_model:
        model.load_dict(model_path)
    else:
        print("You shold Load model!!!")
        exit(0)
    # model.learn_one()

    if args.requireDoneAll:
        if args.save_gif != None:
            model.render(random_level=0)
        else:
            model.render(random_level=0.1)
    else:
        model.render_one(random_level=0.1)