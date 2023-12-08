from os import name, stat
import cv2
import numpy as np
from copy import deepcopy
import imageio
from matplotlib import pyplot as plt
import time

scale = 35

class Simulator:

    def __init__(self, size, robot_num, static=None, name ='', args=None):
        """
        Initialize simulator multi agent path finding
        robot: {index:(x,y,carry_index)}
        target: {index}:(box_x,box_y,target_x,target_y)
        """
        self.args = args
        self.no_vis = args.ssh

        self.canvas = np.ones(size, np.uint8)*255
        self.robot = dict()
        self.robot_last_pos = dict()
        self.target = dict()
        self.robot_carry = dict()
        self.size = size
        self.robot_num = robot_num
        self.frames = []
        self.name = name
        self.steps = 0
        if static != None:
            self.robot, self.target = static
        self.colours = self.assign_colour(robot_num*3)
        self.crash = []

        # Handel the map_reset at th most begining to reproduce the later training procedure
        rnd = np.random
        rnd.seed(self.args.seed)    # 5457
        if len(self.robot) == 0:
            pos = rnd.randint(1,size[0]//scale, size=(3*robot_num,2))
            pos = set([tuple(i) for i in pos])
            while len(pos) < 3*robot_num:
                temp = rnd.randint(1,size[0]//scale, size=(3*robot_num - len(pos),2))
                b = set([tuple(i) for i in temp])
                for i in b:
                    if i not in pos:
                        pos.add(i)
            pos = list(pos)


        self.fix_pos = pos


        self.generate_map(robot_num, size)    
        # cv2.namedWindow("Factory")
        # cv2.resizeWindow('Factory', tuple(np.array(list(size)[:2])+np.array([500,200])))


    
    def update_pairs(self, pairs):
        for pair in pairs:
            self.robot[pair[0]] = (self.robot[pair[0]][0], self.robot[pair[0]][1], pair[1])

    def generate_map(self, robot_num, size):
        """
        generate random map to increase the complexity
        """
        assert size[0]*size[1]>robot_num *scale*3
        for i in range(1,size[0]//scale):
            cv2.line(self.canvas, (scale*i,scale), (scale*i,(size[1]//scale-1)*scale), (0,0,0))
        for i in range(1,size[1]//scale):
            cv2.line(self.canvas, (scale,i*scale), ((size[0]//scale-1)*scale,i*scale), (0,0,0))
        if len(self.robot) == 0:
            pos = self.fix_pos
            for i in range(robot_num):
                self.robot[i] = (pos[i][0],pos[i][1],i)
                # self.target[i] = (pos[i+robot_num][0], pos[i+robot_num][1], pos[i+2*robot_num][0], pos[i+2*robot_num][1])
                self.target[i] = (pos[i+robot_num][0], pos[i+robot_num][1], pos[i+robot_num][0], pos[i+robot_num][1])



        for i in range(robot_num):
            self.draw_target(self.canvas, np.array(self.target[i][2:])*scale, self.colours[i+len(self.robot)], 5)
            self.robot_carry[i] = False    


        if self.args.reset_seed_inprocess:
            # reset rnd seed for the later training portion
            np.random.seed(int(time.time()))

    @staticmethod
    def assign_colour(num):
        def colour(x):
            x = hash(str(x+42))
            return ((x & 0xFF, (x >> 8) & 0xFF, (x >> 16) & 0xFF))
        colours = dict()
        for i in range(num):
            colours[i] = colour(i)
        return colours
    
    @staticmethod
    def draw_target(frame, point, color, thick):
        point1 = np.array(point)-np.array([scale//3,scale//3])
        point2 = np.array(point)+np.array([scale//3,scale//3])
        point3 = np.array(point)+np.array([scale//3,-scale//3])
        point4 = np.array(point)-np.array([scale//3,-scale//3])
        cv2.line(frame, tuple(point1), tuple(point2), color, thick)
        cv2.line(frame, tuple(point3), tuple(point4), color, thick)


    def show(self, wait=True, on_server=False):
        frame = deepcopy(self.canvas)
        font_scale = 1
        font_size = 0.4
        color = (255,255,255)
        for id_, pos in self.target.items():
            size, _ = cv2.getTextSize('{0}'.format(id_),cv2.FONT_HERSHEY_COMPLEX,font_size,font_scale)
            cv2.rectangle(frame, tuple(np.array(self.target[id_][:2])*scale-np.array([scale//3,scale//3])), tuple(np.array(self.target[id_][:2])*scale+np.array([scale//3,scale//3])), self.colours[id_+len(self.robot)],-1) 
            cv2.putText(frame,'{0}'.format(id_),tuple([self.target[id_][0]*scale-size[0]//2, self.target[id_][1]*scale+size[1]//2]),cv2.FONT_HERSHEY_COMPLEX,font_size,color,font_scale)
        for id_, pos in self.robot.items():
            size, _ = cv2.getTextSize('{0}'.format(pos[-1]),cv2.FONT_HERSHEY_COMPLEX,font_size,font_scale)
            cv2.circle(frame, tuple(np.array(pos)[:-1]*scale), scale//3, self.colours[id_], -1)
            cv2.putText(frame,'{0}'.format(pos[-1]),tuple([pos[0]*scale-size[0]//2, pos[1]*scale+size[1]//2]),cv2.FONT_HERSHEY_COMPLEX,font_size,color,font_scale)
        
        if on_server:
            self.frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            return
            
        cv2.imshow("Factory"+self.name,frame)
        if wait:
            cv2.waitKey(0)
        else:
            cv2.waitKey(100)
        self.frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    

    @staticmethod
    def out_of_map(pos, size):
        if pos[0] <= 0 or pos[0] >= size[0]//scale or pos[1] <= 0 or pos[1] >= size[1]//scale:
            return True
        return False

    def step(self, action, simple=False):
        path = {}
        reward = np.array([-0.3 for i in action])
        import copy
        if self.steps == 0:
            done = [False for i in action]
            prev_dist_list = [100 for i in action]
            self.reach_goal = [0 for i in action]
        else:
            done = copy.deepcopy(self.pre_done)
            prev_dist_list = self.prev_dist_list
        # done = [False for i in action]
        states = []
        end = {}

        if self.args.oldReward:
            gain = 0.4
            loss = 0.2
        else:
            gain = 1
            loss = 5


        for id_, pos in self.robot.items():
            # Terminated too long episode
            if self.steps > self.expected_length:
                done[id_] = True

            # Update the path for each robot with the action = [1, 2, 3, 4]
            if not done[id_]: 
                pos2 = self.target[pos[2]]
                end[id_] = (pos2[0], pos2[1])

                if (pos[0]-pos2[0])**2 + (pos[1]-pos2[1])**2 < 1:
                    reward[id_] += 50
                    # path[id_] = [(pos[0], pos[1])]
                    done[id_] = True
                    continue

                # action==0: stay still
                if action[id_] != 0:
                    if action[id_] == 1:
                        path[id_] = [(pos[0], pos[1]+1)]
                    elif action[id_] == 2:
                        path[id_] = [(pos[0]-1, pos[1])]
                    elif action[id_] == 3:
                        path[id_] = [(pos[0]+1, pos[1])]
                    elif action[id_] == 4:
                        path[id_] = [(pos[0], pos[1]-1)]
                    
                    if self.out_of_map(path[id_][0], self.size):
                        reward[id_] -= 20
                        done[id_] = True
                    
                        # Remove the path[id_] if it is out of map
                        del path[id_]

                    dist = abs(self.robot[id_][0]-end[id_][0])+abs(self.robot[id_][1]-end[id_][1])

                    prev_dist_list[id_] = dist
            else:
                continue


        self.steps += 1

        if path != {}:
            if self.args.save_gif != None:
                self.start(path, self.args.save_gif, False)
            else:
                self.start(path, None, False)


        # IMPORTANT
        self.crash_check()

        # Post Step
        if len(self.crash) > 0:
            for i in self.crash:
                # print("crash!!!")

                reward[i[0]] -= 20
                reward[i[1]] -= 20
                done[i[0]] = True
                done[i[1]] = True
        
        for id_ in self.robot.keys():
            # NOTE: USE THIS BRANCH
            # state = self.simple_state(id_)

            if self.args.expand_obs:
                state = self.expand_state(id_)
                # state = state[:7]
            else:
                state = self.simple_state(id_)


            states.append(state)

            if not done[id_]: 
                dist = abs(self.robot[id_][0]-end[id_][0])+abs(self.robot[id_][1]-end[id_][1])
                prev_dist_list[id_] = dist

                dist_change = prev_dist_list[id_] - dist
                if dist_change > 0:
                    reward[id_] += gain
                else:
                    reward[id_] -= loss

                if np.math.hypot(self.robot[id_][0]-end[id_][0], self.robot[id_][1]-end[id_][1])<1:
                    # reward[id_] += 30
                    reward[id_] += 50

                    self.reach_goal[id_] = 1

                    done[id_] = True
                    if not self.no_vis:
                        print("robot", id_, "reach the target")

        self.pre_done = done
        self.prev_dist_list = prev_dist_list

        info = {"num_reach_goal": np.sum(self.reach_goal),
                "crash": self.crash,
                "avg_dist_to_goal": np.sum(self.prev_dist_list),
                }

        return reward, np.array(states), done, info
    

    def expand_state(self, index, test=False):
        me = self.robot[index]
        state = np.zeros(7+8) # 2 + 4 + 1 + [4 (2xstep) + 4(corner) only consider robots!]

        state[0] = (self.target[self.robot[index][2]][0] - self.robot[index][0])/(self.size[0]//scale+1)
        state[1] = (self.target[self.robot[index][2]][1] - self.robot[index][1])/(self.size[1]//scale+1)

        if me[0] == 1:
            # left
            state[2] = 1
        elif me[0] == self.size[0]//scale-1:
            # right
            state[3] = 1
        if me[1] == 1:
            # up
            state[4] = 1
        elif me[1] == self.size[1]//scale-1:
            # down
            state[5] = 1

        # Has terminated
        if self.out_of_map(me, self.size):
            state[6] = 1
        
        for id_, pos in self.robot.items():
            # 2 + 4 + 1 + [4 (2xstep) + 4(corner) only consider robots!]
            if id_ == index:
                continue

            manhaton_dist = abs(self.robot[id_][0]-me[0])+abs(self.robot[id_][1]-me[1])

            # import pdb; pdb.set_trace()

            if manhaton_dist <= 2:
                left = (self.robot[id_][0]-me[0] == -1)
                right = (self.robot[id_][0]-me[0] == 1)
                up = (self.robot[id_][1]-me[1] == -1)
                down = (self.robot[id_][1]-me[1] == 1)

                left_2  = (self.robot[id_][0]-me[0] == -2)
                right_2 = (self.robot[id_][0]-me[0] == 2)
                up_2    = (self.robot[id_][1]-me[1] == -2)
                down_2  = (self.robot[id_][1]-me[1] == 2)

            if manhaton_dist == 0:
                state[6] = 1

            if manhaton_dist == 1:
                start_from = 2
                if left:
                    state[start_from+0] = 1    # left
                elif right:
                    state[start_from+1] = 1    # right
                elif up:
                    state[start_from+2] = 1    # up
                elif down:
                    state[start_from+3] = 1     # down                

            elif manhaton_dist == 2:
                start_from = 2+4+1

                if left_2:
                    state[start_from+0] = 1    # left_2
                elif right_2:
                    state[start_from+1] = 1    # right
                elif up_2:
                    state[start_from+2] = 1    # up
                elif down_2:
                    state[start_from+3] = 1     # down    

                start_from = 2+4+1+4

                if left and up:
                    state[start_from+0] = 1    # left and up
                elif left and down:
                    state[start_from+1] = 1    # left and down
                elif right and up:
                    state[start_from+2] = 1    # right and up
                elif right and down:
                    state[start_from+3] = 1    # right and down

        return state


    def simple_state(self, index, test=False):
        me = self.robot[index]
        state = np.zeros(7)

        state[0] = (self.target[self.robot[index][2]][0] - self.robot[index][0])/(self.size[0]//scale+1)
        state[1] = (self.target[self.robot[index][2]][1] - self.robot[index][1])/(self.size[1]//scale+1)
        # if test and self.robot_carry[index]==True:

        #     input("Not Use this branch")
        #     state[0] = (self.target[self.robot[index][2]][2] - self.robot[index][0])/(self.size[0]//scale+1)
        #     state[1] = (self.target[self.robot[index][2]][3] - self.robot[index][1])/(self.size[1]//scale+1)
        if me[0] == 1:
            # left
            state[2] = 1
        elif me[0] == self.size[0]//scale-1:
            # right
            state[3] = 1
        if me[1] == 1:
            # up
            state[4] = 1
        elif me[1] == self.size[1]//scale-1:
            # down
            state[5] = 1
        if self.out_of_map(me, self.size):
            state[6] = 1
        
        for id_, pos in self.robot.items():
            if id_ == index:
                continue
            if np.math.hypot(self.robot[id_][0]-me[0], self.robot[id_][1]-me[1])<1.2:
                if self.robot[id_][0]-me[0] < 0:
                    state[2] = 1
                elif self.robot[id_][0]-me[0] > 0:
                    state[3] = 1
                if self.robot[id_][1]-me[1] > 0:
                    state[5] = 1
                elif self.robot[id_][1]-me[1] < 0:
                    state[4] = 1
                if self.robot[id_][0]-me[0] == 0 and self.robot[id_][1]-me[1] == 0:
                    state[6] = 1
        return state

    def reset(self, simple=False):
        self.expected_length = 2 * self.args.map_size

        self.crash = []
        self.canvas = np.ones(self.size, np.uint8)*255
        self.robot = {}
        self.robot_carry = {}
        self.target = {}
        self.steps = 0
        states = []
        self.frames = []
        self.generate_map(self.robot_num, self.size)
        for id_ in self.robot.keys():

            if self.args.expand_obs:
                state = self.expand_state(id_)
                # state = state[:7]
            else:
                state = self.simple_state(id_)
                
            self.robot_carry[id_] = False
            states.append(state)
        return np.array(states)

    def crash_check(self):
        """
        check if there are any collision
        """
        for id_, pos in self.robot.items():
            lastmiddle1 = ((self.robot_last_pos[id_][0]+pos[0])/2, (self.robot_last_pos[id_][1]+pos[1])/2)
            for id2_, pos2 in self.robot.items():
                if id_ >= id2_:
                    continue
                lastmiddle = ((self.robot_last_pos[id2_][0]+pos2[0])/2, (self.robot_last_pos[id2_][1]+pos2[1])/2)
                # print(lastmiddle, lastmiddle1, np.math.hypot(lastmiddle1[0]-lastmiddle[0],lastmiddle1[1]-lastmiddle[1]))
                if np.math.hypot(pos[0]-pos2[0], pos[1]-pos2[1]) < 1 or np.math.hypot(lastmiddle1[0]-lastmiddle[0],lastmiddle1[1]-lastmiddle[1])<=0.5:
                    self.crash.append((id_,id2_))
                    # if np.math.hypot(pos[0]-pos2[0], pos[1]-pos2[1]) < 1:
                    #     # print(f"fucking robot crash{id_}:{pos},{id2_}:{pos2}")
                    # else:
                    #     # print("fucking lastmiddle crash")
                    return True
        return False
    
    def carry_check(self):
        """
        check if the robot carry the box
        """
        for id_, pos in self.robot.items():
            if pos[2] != -1:
                continue
            for id2_, pos2 in self.target.items():
                if (pos[0]-pos2[0])**2 + (pos[1]-pos2[1])**2 < 1:
                    self.robot[id_] = tuple(np.append(np.array(self.robot[id_])[:2], id2_))
                    break
    
    def information(self):
        return self.robot, self.target

    def start(self, path, save_gif=None, wait=False):
        try:
            i = 0
            while True:
                self.robot_last_pos = self.robot.copy()
                for id_ in path:
                    if i >= len(path[id_]) or np.math.hypot(path[id_][i][0]-self.robot[id_][0], path[id_][i][1]-self.robot[id_][1]) > 1.4:
                        continue
                    cv2.line(self.canvas, tuple(np.array(self.robot[id_][:2])*scale), tuple(np.array(path[id_][i])*scale), self.colours[id_],5)
                    if self.robot[id_][2] >= 0:
                        if self.target[self.robot[id_][2]][:2]==self.robot[id_][:2]:
                            self.robot[id_] = tuple(np.append(np.array(path[id_][i]),self.robot[id_][2]))
                            self.target[self.robot[id_][2]] = tuple([self.robot[id_][0], self.robot[id_][1], self.target[self.robot[id_][2]][2], self.target[self.robot[id_][2]][3]])
                        else:
                            self.robot[id_] = tuple(np.append(np.array(path[id_][i]),self.robot[id_][2]))
                    else:
                        self.robot[id_] = tuple(np.append(np.array(path[id_][i]),self.robot[id_][2]))
                        self.carry_check()

                if not self.no_vis:
                    if self.crash_check():
                        frame = np.ones(self.size, np.uint8)*255
                        cv2.putText(frame, "Crash", (self.size[0]//2-int(2.5*scale), self.size[1]//2), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 255), 2)
                        cv2.imshow("Factory"+self.name,frame)
                        cv2.waitKey(1000)
                        break    
                    self.show(wait)
                else:
                    self.show(wait, on_server=True)

                i += 1
                if i >= max([len(i) for i in path.values()]):
                    # input("here!!!!")
                    # import pdb; pdb.set_trace()
                    # print("over")
                    break
        except Exception as err:
            print(err)
            # print()
        if save_gif!=None:
            with imageio.get_writer(save_gif, fps=1, mode="I") as writer:

            # with imageio.get_writer("./image/"+save_gif, fps=1, mode="I") as writer:
                for idx, frame in enumerate(self.frames):
                    writer.append_data(frame)
    
        if not self.no_vis:
            cv2.waitKey(1)
        # cv2.destroyAllWindows()


if __name__ == "__main__":
    # random initialize
    # env1 = Simulator((601,601,3),8)
    # env1.get_state_map(0, True)

    # given state
    # static_origin = [{0:(1,1,1),1:(2,2,-1),2:(3,3,-1)}, {0:(8,5,7,3),1:(10,8,9,9),2:(5,10,11,2)}]
    # env2 = Simulator((601,601,3),3,static_origin)
    # env2.show()
    # state = env2.get_state_map(0, True)
    # # display

    # # get start and target
    # print(env2.information())

    # # given a path and show
    # static_origin = [{0:(1,1,0)},{0:(1,4,2,6)}]
    # path = {0:[(1,2),(1,3),(1,4),(2,4),(2,5),(2,6)]}
    # env = Simulator((601,601,3),1,static_origin)
    # env.start(path)

    # check collision
    # static_origin = [{0:(1,1,0),1:(1,3,1)},{0:(1,4,2,6),1:(10,8,9,7)}]
    # path = {0:[(1,2),(1,3),(1,4),(2,4),(2,5),(2,6)],1:[(1,3),(1,2)]}
    # env3 = Simulator((601,601,3),2,static_origin)
    # env3.start(path, None, True)

    # # check state map
    # static_origin = [{0:(1,1,0)},{0:(1,4,2,6)}]
    # action = [1,1,1,2,3,3]
    # env = Simulator((601,601,3),1,static_origin)
    # for i in action:
    #     reward, states, done, _ = env.step([i])
    #     print("reward:",reward)
    #     if done:
    #         print("done")
    #         break
    
    # check state map2
    static_origin = [{0:(1,1,0),1:(16,16,1)},{0:(1,4,2,6),1:(10,8,9,7)}]
    path = {0:[(1,2),(1,3),(1,4),(2,4),(2,5),(2,6)],1:[(1,4),(2,4),(2,5),(2,6)]}
    action = [[0,1],[1,3],[1,1],[3,1],[1,0],[1,0]]
    action = [[0,0],[3,0],[3,1],[3,1],[1,0],[4,0]]
    env = Simulator((601,601,3),2,static_origin)
    for i in action:
        reward, states, done, _ = env.step(i, True)
        print("reward:",states)
        if np.array(done).any():
            print("done")
            break

