from os import name, stat
import cv2
import numpy as np
import math
from copy import deepcopy
import imageio
from matplotlib import pyplot as plt
import time

scale = 35

class Simulator:

    def __init__(self, size, robot_num, static=None, visual=False, debug=False, name =''):
        """
        Initialize simulator multi agent path finding
        robot: {index:(x,y,carry_index)}
        target: {index}:(box_x,box_y,target_x,target_y)
        """
        self.canvas = np.ones(size, np.uint8)*255
        self.robot = dict()
        self.robot_last_pos = dict()
        self.target = dict()
        self.robot_carry = dict()
        self.size = size
        self.robot_num = robot_num
        self.observation_per_robot = 7
        self.frames = []
        self.name = name
        self.steps = 0
        if static != None:
            self.robot, self.target = static
        self.colours = self.assign_colour(robot_num*3)
        self.crash = []
        self.generate_map(robot_num, size)    
        self.visual = visual
        self.debug = debug
        # cv2.namedWindow("Factory")
        # cv2.resizeWindow('Factory', tuple(np.array(list(size)[:2])+np.array([500,200])))
    
    def update_pairs(self, pairs):
        for pair in pairs:
            self.robot[pair[0]] = (self.robot[pair[0]][0], self.robot[pair[0]][1], pair[1])

    def generate_map(self, robot_num, size):
        """
        generate random map to increase the complexity
        """
        rnd = np.random
        rnd.seed(5258)
        assert size[0]*size[1]>robot_num *scale*3
        self.canvas = np.ones(self.size, np.uint8)*255
        for i in range(1,size[0]//scale):
            cv2.line(self.canvas, (scale*i,scale), (scale*i,(size[1]//scale-1)*scale), (0,0,0))
        for i in range(1,size[1]//scale):
            cv2.line(self.canvas, (scale,i*scale), ((size[0]//scale-1)*scale,i*scale), (0,0,0))
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
            for i in range(robot_num):
                self.robot[i] = (pos[i][0],pos[i][1],i)
                self.target[i] = (pos[i+robot_num][0], pos[i+robot_num][1])
        self.init_robot = self.robot.copy()
        self.init_target = self.target.copy()
            
        # for i in range(robot_num):
        #     self.draw_target(self.canvas, np.array(self.target[i][2:])*scale, self.colours[i+len(self.robot)], 5)
        #     self.robot_carry[i] = False   

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


    def show(self, wait=True):
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
        
        cv2.imshow("Factory"+self.name,frame)
        if wait:
            cv2.waitKey(0)
        else:
            cv2.waitKey(100)
        self.frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    def get_state_map(self, index, show=False):
        state = np.zeros((self.size[0]//scale+1, self.size[1]//scale+1))
        for id_, pos in self.robot.items():
            if id_ == index:
                state[pos[0]][pos[1]] = -1
            else:
                state[pos[0]][pos[1]] -= -3
        for id2_, pos2 in self.target.items():
            if id2_ == self.robot[index][2]:
                if state[pos2[0]][pos2[1]] == -1:
                    self.robot_carry[id_] = True
                    state[pos2[0]][pos2[1]] = 2
                else:
                    state[pos2[0]][pos2[1]] += 4
                
        state = np.rot90(state, 1)
        if show:
            plt.figure()
            plt.imshow(state)
            for i in range(len(state)):
                for j in range(len(state[0])):
                    c = str(state[i][j])
                    plt.text(j, i, c, va='center', ha='center')
            plt.xlim((-0.5,len(state[0])-0.5))
            plt.ylim((-0.5,len(state)-0.5))
            plt.show()
        return np.array([state])
    
    @staticmethod
    def out_of_map(pos, size):
        if pos[0] <= 0 or pos[0] >= size[0]//scale or pos[1] <= 0 or pos[1] >= size[1]//scale:
            return True
        return False

    def step(self, action):
        # print(f"action:{action}")
        next_pos = {}
        reward = np.array([-0.3 for i in action])
        reached_goal = [False for i in range(self.robot_num)]
        collision = [False for i in range(self.robot_num)]
        out_bound = [False for i in range(self.robot_num)]
        done = False
        target_pos = {}
        self.robot_last_pos = self.robot.copy()

        # step simulation
        for id_, pos in self.robot.items():
            # take no action if reached the goal
            if action[id_] == 0:
                next_pos[id_] = [(pos[0], pos[1])]
            elif action[id_] == 1:
                next_pos[id_] = [(pos[0], pos[1]+1)]
            elif action[id_] == 2:
                next_pos[id_] = [(pos[0]-1, pos[1])]
            elif action[id_] == 3:
                next_pos[id_] = [(pos[0]+1, pos[1])]
            elif action[id_] == 4:
                next_pos[id_] = [(pos[0], pos[1]-1)]

            if self.robot_last_pos[id_][:2]==self.target[id_][:2]:
                next_pos[id_] = [(pos[0], pos[1])]

            self.robot[id_] = tuple(np.append(np.array(next_pos[id_][0]),self.robot[id_][2]))  

            # check if reached goal
            _pos = self.target[id_]
            target_pos[id_] = (_pos[0], _pos[1])
            if math.hypot(self.robot[id_][0]-target_pos[id_][0], self.robot[id_][1]-target_pos[id_][1])<1:
                reached_goal[id_] = True
            # check if out of map
            if self.out_of_map(next_pos[id_][0],self.size):
                out_bound[id_] = True

        assert math.hypot(next_pos[id_][0][0]-self.robot[id_][0], next_pos[id_][0][1]-self.robot[id_][1]) < 1.4, "Next step is too far away, which is so weird"
        assert self.robot[id_][2] >= 0, "robot defination id error"  # these asserts are from previous repo and I don't understand them

        # check collision between robots
        collision = self.collision_check(next_pos)
        done = np.array(collision).any()# or np.array(out_bound).any()  # wqwqwq

        obs = self.compute_obs(collision,out_bound,reached_goal)
        reward = self.compute_reward(action,collision,out_bound,reached_goal)
        self.steps += 1
        if self.steps > 80/2:
                done = True

        if self.visual:
            self.show_plot(next_pos, done, None, wait=self.debug)
        return reward, np.array(obs), done, {}
    
    def compute_obs(self,collision,out_bound,reached_goal):
        obs=[]
        for id_, pos in self.robot.items():
            state = np.zeros(self.observation_per_robot)
            state[:2] = self.target[id_]
            state[2:4] = pos[:2]
            state[4] = int(collision[id_])
            state[5] = np.min([pos[0],-pos[0]+self.size[0]//scale,pos[1],-pos[1]+self.size[1]//scale])
            state[6] = abs(self.target[id_][0] - pos[0])+abs(self.target[id_][1] - pos[1])
            obs.append(state)
        if self.debug:
            print(f"state: {obs}")
        return [np.array(obs).ravel()]
    
    def compute_reward(self,action,collision,out_bound,reached_goal):
        if self.debug:
            print(f"action:{action}")
        reward = np.zeros(self.robot_num)
        for id_, pos in self.robot.items():
            # reward for correct action
            target_pos = self.target[id_]
            if self.robot_last_pos[id_][:2]==target_pos[:2]:
                if action[id_] == 0:
                    reward[id_] += 2
            if action[id_] == 0:  # stay
                reward[id_] -= 2
            elif action[id_] == 1:  # down
                reward[id_] += -1 + (target_pos[1] - pos[1] > 0) * 2
            elif action[id_] == 2:  # left
                reward[id_] += -1 + (target_pos[0] - pos[0] < 0) * 2
            elif action[id_] == 3:  # right
                reward[id_] += -1 + (target_pos[0] - pos[0] > 0) * 2
            elif action[id_] == 4:  # up
                reward[id_] += -1 + (target_pos[1] - pos[1] < 0) * 2
            
            if self.debug:
                print(f"reward after action check:{reward}")
            # reward for goal
            if reached_goal[id_]:
                # print(f"robot {id_} reached the goal and gets rewards")
                reward[id_]+=10
            else:
                reward[id_]+=-(abs(target_pos[0] - pos[0])+abs(target_pos[1] - pos[1]))/(self.size[0]//scale) + 8

            # reward for collision and out of map
            if collision[id_] or out_bound[id_]:
                reward[id_] -= 20

            if self.debug:
                print(f"after reward for goal:{reward} {(abs(target_pos[0] - pos[0])+abs(target_pos[1] - pos[1]))}")

        return reward

    def collision_check(self,path):
        collision = [False for _ in range(self.robot_num)]
        for id_, pos in self.robot.items():
            lastmiddle1 = ((self.robot_last_pos[id_][0]+pos[0])/2, (self.robot_last_pos[id_][1]+pos[1])/2)
            for id2_, pos2 in self.robot.items():
                if id_ >= id2_:
                    continue
                lastmiddle = ((self.robot_last_pos[id2_][0]+pos2[0])/2, (self.robot_last_pos[id2_][1]+pos2[1])/2)  # last middle is for edge collision check
                if np.math.hypot(pos[0]-pos2[0], pos[1]-pos2[1]) < 1 or np.math.hypot(lastmiddle1[0]-lastmiddle[0],lastmiddle1[1]-lastmiddle[1])<=0.5:
                    collision[id_] = True
                    collision[id2_] = True
        return collision
    
    def show_plot(self, path, done=False, save_gif=None, wait=False):
        try:
            for id_ in path:
                cv2.line(self.canvas, tuple(np.array(self.robot_last_pos[id_][:2])*scale), tuple(np.array(path[id_][0])*scale), self.colours[id_],5)
            if done:
                frame = np.ones(self.size, np.uint8)*255
                cv2.putText(frame, "Done", (self.size[0]//2-int(2.5*scale), self.size[1]//2), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 255, 128), 2)
                cv2.imshow("Factory"+self.name,frame)
                cv2.waitKey(1000)
            self.show(wait)
        except Exception as err:
            print(err)
        if save_gif!=None:
            with imageio.get_writer("./image/"+save_gif, mode="I") as writer:
                for idx, frame in enumerate(self.frames):
                    writer.append_data(frame)
        cv2.waitKey(1)

    def simple_state(self, index, test=False):
        me = self.robot[index]
        state = np.zeros(7)
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

    def reset(self):
        self.crash = []
        self.robot = {}
        self.robot_carry = {}
        self.target = {}
        self.steps = 0
        states = []
        self.frames = []
        # self.generate_map(self.robot_num, self.size)

        # reset canvas
        self.canvas = np.ones(self.size, np.uint8)*255
        for i in range(1,self.size[0]//scale):
            cv2.line(self.canvas, (scale*i,scale), (scale*i,(self.size[1]//scale-1)*scale), (0,0,0))
        for i in range(1,self.size[1]//scale):
            cv2.line(self.canvas, (scale,i*scale), ((self.size[0]//scale-1)*scale,i*scale), (0,0,0))

        self.robot = self.init_robot.copy()
        self.target = self.init_target.copy()
        for id_ in self.robot.keys():
            state = np.zeros(self.observation_per_robot)
            states.append(state)
        return [np.array(states).ravel()]

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
                if np.math.hypot(pos[0]-pos2[0], pos[1]-pos2[1]) < 1 or np.math.hypot(lastmiddle1[0]-lastmiddle[0],lastmiddle1[1]-lastmiddle[1])<=0.5:
                    self.crash.append((id_,id2_))
                    if np.math.hypot(pos[0]-pos2[0], pos[1]-pos2[1]) < 1:
                        print(f"fucking robot crash{id_}:{pos},{id2_}:{pos2}")
                    else:
                        print("fucking lastmiddle crash")
                    return True
        return False
    
    def show_information(self):
        return self.robot, self.target

    def start_plot(self, path, save_gif=None, wait=False):
        print(path)
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
                        else:
                            self.robot[id_] = tuple(np.append(np.array(path[id_][i]),self.robot[id_][2]))
                    else:
                        self.robot[id_] = tuple(np.append(np.array(path[id_][i]),self.robot[id_][2]))
                        self.carry_check()
                if self.crash_check():
                    frame = np.ones(self.size, np.uint8)*255
                    cv2.putText(frame, "Crash", (self.size[0]//2-int(2.5*scale), self.size[1]//2), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 255), 2)
                    cv2.imshow("Factory"+self.name,frame)
                    cv2.waitKey(1000)
                    break    
                self.show(wait)
                i += 1
                if i >= max([len(i) for i in path.values()]):
                    print("over")
                    break
        except Exception as err:
            print(err)
        if save_gif!=None:
            with imageio.get_writer("./image/"+save_gif, mode="I") as writer:
                for idx, frame in enumerate(self.frames):
                    writer.append_data(frame)
        cv2.waitKey(1)

    def stepold(self, action, simple=False):
        path = {}
        reward = np.array([-0.3 for i in action])
        done = [False for i in action]
        states = []
        end = {}
        for id_, pos in self.robot.items():
            pos2 = self.target[pos[2]]
            end[id_] = (pos2[0], pos2[1])

            if action[id_] == 0:
                path[id_] = [(pos[0], pos[1])]
                reward[id_] -= 0.5
            elif action[id_] == 1:
                path[id_] = [(pos[0], pos[1]+1)]
                if end[id_][1] - pos[1] > 0:
                    reward[id_] += 1
                else:
                     reward[id_] -= 1
            elif action[id_] == 2:
                path[id_] = [(pos[0]-1, pos[1])]
                if end[id_][0] - pos[0] < 0:
                    reward[id_] += 1
                else:
                     reward[id_] -= 1
            elif action[id_] == 3:
                path[id_] = [(pos[0]+1, pos[1])]
                if end[id_][0] - pos[0] > 0:
                    reward[id_] += 1
                else:
                     reward[id_] -= 1
            elif action[id_] == 4:
                path[id_] = [(pos[0], pos[1]-1)]
                if end[id_][1] - pos[1] < 0:
                    reward[id_] += 1
                else:
                     reward[id_] -= 1
            if self.out_of_map(path[id_][0], self.size):
                reward[id_] -= 20
                done[id_] = True
            if self.steps > 80:
                done[id_] = True

        self.steps += 1
        self.start_plot(path, None, False)
        if len(self.crash) > 0:
            for i in self.crash:
                reward[i[0]] -= 20
                reward[i[1]] -= 20
                done[i[0]] = True
                done[i[1]] = True
        for id_ in self.robot.keys():
            if simple == False:
                state = self.get_state_map(id_, False)
            else:
                state = self.simple_state(id_, False)
            states.append(state)
            # reward -= 0.025*(abs(self.robot[id_][0]-end[id_][0])+abs(self.robot[id_][1]-end[id_][1]))
            if np.math.hypot(self.robot[id_][0]-end[id_][0], self.robot[id_][1]-end[id_][1])<1:
                reward[id_] += 30
                done[id_] = True

        return reward, np.array(states), done, {}
 
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

