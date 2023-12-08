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

    if args.wq_test:
        args.env_name = "Wq_Test_3_RequireDoneAll"
        args.map_size = 5
        args.num_robot = 3
        args.requireDoneAll = True
        args.oldReward = True
        class_folder = "./models/Wq_Test_3_RequireDoneAll/"
    else:
        class_folder = "./models/MAPF_Move_4_RequireDoneAll/"


    args.moveAll = True

    timestamp = datetime.datetime.now().strftime("%m%d_%H%M%S")

    model_path = ""

    # args.num_robot=1

    # model_path = "/home/ziang/Public/AA228_Project/MAPF-hw/models/MAPF_ResetSeed/M_10x10_1205_185116/model_E_869000_R_90_L_23.pt"
    # model_path = "/home/ziang/Public/AA228_Project/MAPF-hw/models/MAPF/M_10x10_1205_182729/model_E_903000_R_96_L_0.pt"



    # args.num_robot=4

    # RESET SEED TO 5457 at every time ...
    # model_path = "/home/ziang/Public/AA228_Project/MAPF-hw/models/MAPF_Static_4/M_10x10_1205_182846/model_E_582000_R_74_L_2.pt"
    # model_path = "/home/ziang/Public/AA228_Project/MAPF-hw/models/MAPF_Static_4/M_10x10_1205_182846/model_E_574000_R_74_L_5.pt"
    # model_path=  "/home/ziang/Public/AA228_Project/MAPF-hw/models/MAPF_Static_4/M_10x10_1205_182846/model_E_159000_R_33_L_4.pet"

    # model_path = "/home/ziang/Public/AA228_Project/MAPF-hw/models/MAPF_Static_4/M_10x10_1206_000841/model_E_62000_R_8_L_58.pt"

    # model_path = "/home/ziang/Public/AA228_Project/MAPF-hw/models/MAPF_Static_4_ResetSeed/M_10x10_1205_184623/model_E_574000_R_79_L_7.pt"
    # model_path = "/home/ziang/Public/AA228_Project/MAPF-hw/models/OLD_MAPF_Static_4_32_20000/M_10x10_1205_165856/model_E_123000_R_40_L_1.pt"


    ### ALL MOVE TOGETHER:
    assert args.moveAll, "move all together"
    # class_folder = "./models/MAPF_Move_4_RequireDoneAll/"
    if args.expand_obs:
        if args.wq_test:
            # folder_path = "./OBSFREE_M_5x5_1208_123537_oldReward_expandObs"
            # model_path = class_folder + folder_path + "/model_E_656000_Suc_3_Dist_0_LR_49_MR_44_L_2.pt"


            # /home/ziang/Public/AA228_Project/MAPF-hw/models/Wq_Test_3_RequireDoneAll/M_5x5_1208_152926_oldReward_expandObs/model_E_56000_Suc_3_Dist_0_LR_48_MR_4_L_91.pt

            folder_path = "./M_5x5_1208_152926_oldReward_expandObs"
            model_path = class_folder + folder_path + "/model_E_39000_Suc_2_Dist_4_LR_25_MR_3_L_172.pt" # KEEP IT!!
        else:
            folder_path = "./M_10x10_1207_235305_expandObs"
            model_path = class_folder + folder_path + "/model_E_278000_Suc_4_Dist_0_LR_4_MR_-3_L_4.pt"
    else:
        if args.wq_test:
            folder_path = "./OBSFREE_M_5x5_1208_122648_oldReward"
            model_path = class_folder + folder_path + "/model_E_661000_Suc_3_Dist_0_LR_48_MR_44_L_4.pt"

            # model_path = class_folder + folder_path + "/model_E_662000_Suc_2_Dist_4_LR_26_MR_44_L_91.pt"
        else:
            # folder_path = "./M_10x10_1207_220013_oldReward_CT"
            # model_path = class_folder + folder_path + "/model_E_315000_Suc_4_Dist_0_LR_45_MR_8_L_39.pt"

            folder_path = "./M_10x10_1208_022224_oldReward"
            model_path = class_folder + folder_path + "/model_E_300000_Suc_4_Dist_0_LR_43_MR_-24_L_193.pt"


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