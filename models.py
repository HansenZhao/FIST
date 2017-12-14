import fist_element
import random
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpim
import win32ui
import types
from enum import Enum

class HomoBMModel(fist_element.BasicModel):
    def __init__(self,field=fist_element.Field2D(20,20,1,False),agentNum=1000,D=0.01,dt=0.1):
        super().__init__(agentNum, dt)
        self.field = field
        self.D = D
        self.pointer = 0

    def initField(self):
        self.field.set_patch(lambda p,x:0,'D',[self.D])

    def initAgents(self):
        for n in range(self.agentNum):
            init_x = random.uniform(0,self.field.width)
            init_y = random.uniform(0, self.field.height)
            self.agents.append(fist_element.Agent(n,init_x,init_y,active = True))

    def next_step(self):
        for n in range(self.agentNum):
            if self.agents[n].get('active'):
                self.agents[n].move(self.delta_x_pool[self.pointer],self.delta_y_pool[self.pointer])
                self.pointer += 1
                x = self.agents[n].get('x')
                y = self.agents[n].get('y')
                if  x< 0 or y<0 or x>self.field.width or y>self.field.height:
                    if self.field.is_warp:
                        self.agents[n].set_cur_pos(math.fmod(x,self.field.width),math.fmod(y,self.field.width))
                    else:
                        self.agents[n].set('active',False)

    def simulate(self,step_num=1000):
        super().simulate(step_num)
        bmg = fist_element.BMGenerator2D(self.D,self.dt)
        dx, dy, x, y = bmg.get(step_num * self.agentNum)
        self.delta_x_pool, self.delta_y_pool = dx, dy
        for n in range(step_num):
            self.next_step()
        self.is_simulated = True

class SpatialHeteroBMModel(fist_element.BasicModel):

    def __init__(self, is_warp = False,agentNum=20000,dt=0.1, D_tuple=(0.1,0.3), angle_tuple=(math.pi/2,math.pi), sat_rate = 0.8):
        super().__init__(agentNum, dt)
        self.is_warp = is_warp
        self.p_width = 1
        self.D_tuple = D_tuple
        self.angle_tuple = angle_tuple
        self.D_to_delta_pool = dict()
        self.D_to_delta_pool[D_tuple[0]] = []
        self.D_to_delta_pool[D_tuple[1]] = []
        self.sat_rate = sat_rate

    def initField(self):
        dlg = win32ui.CreateFileDialog(1,'*.tif',None)
        dlg.SetOFNTitle('Please select TIFF figure for D')
        dlg.DoModal()

        file_extent = dlg.GetFileExt()
        file_name = dlg.GetPathName()

        im = mpim.imread(file_name,file_extent)
        height,width = im.shape
        self.field = fist_element.Field2D(width,height,self.p_width)
        self.field.set_patch(SpatialHeteroBMModel.ppos_2_index,'D',self.D_tuple,im>125)

        dlg = win32ui.CreateFileDialog(1, '*.tif', None)
        dlg.SetOFNTitle('Please select TIFF figure for angle')
        dlg.DoModal()

        file_extent = dlg.GetFileExt()
        file_name = dlg.GetPathName()

        im = mpim.imread(file_name, file_extent)
        self.field.set_patch(SpatialHeteroBMModel.ppos_2_index, 'angle', self.angle_tuple, im > 125)

    def initAgents(self):
        for n in range(self.agentNum):
            init_x = random.uniform(0,self.field.width)
            init_y = random.uniform(0, self.field.height)
            self.agents.append(fist_element.Agent(n,init_x,init_y,active = True,dir = random.uniform(-math.pi,math.pi)))

    def next_step(self):
        for agent in self.agents:
            if agent.get('active'):
                # get D based on location
                patch = self.field.get_nearest_patch(agent.get('x'),agent.get('y'))
                D = patch.get('D')
                # get step_length based on D
                step_len = self.D_to_delta_pool[D]
                step = step_len[self.D_to_pointer[D]]
                # refresh pointer
                self.D_to_pointer[D] += 1
                # define angle
                if random.uniform(0,1) > self.sat_rate:
                    angle = random.uniform(-math.pi,math.pi)
                else:
                    max_angle = patch.get('angle')
                    hist_dir = agent.get('dir')
                    angle = random.uniform(hist_dir-max_angle,hist_dir+max_angle)
                agent.move(step*math.cos(angle),step*math.sin(angle))
                # check if out of line
                x,y = (agent.get('x'),agent.get('y'))
                if  x< 0 or y<0 or x>self.field.width or y>self.field.height:
                    if self.field.is_warp:
                        agent.set_cur_pos(math.fmod(x,self.field.width),math.fmod(y,self.field.width))
                    else:
                        agent.set('active',False)

    def simulate(self,step_num=1000):
        super().simulate(step_num)
        self.pool_capacity = step_num*self.agentNum
        for item in self.D_tuple:
            sl = fist_element.BMGenerator2D.get_BM_step_len(item,self.dt,self.pool_capacity)
            self.D_to_delta_pool[item] = sl
        self.D_to_pointer = {self.D_tuple[0]:0, self.D_tuple[1]:0}

        for n in range(step_num):
            self.next_step()
        self.is_simulated = True

    @staticmethod
    def ppos_2_index(p,x):
        return 1 if x[0][p.pos_y,p.pos_x] else 0

    @staticmethod
    def vec_dot(vec_1,vec_2):
        mag_1 = np.array(vec_1)**2
        mag_1 = mag_1.sum()
        mag_2 = np.array(vec_2)**2
        mag_2 = mag_2.sum()
        return np.dot(vec_1,vec_2)/(mag_1*mag_2)

    @staticmethod
    def vec2angle(vec):
        alpha = math.acos(vec[0]/(vec[0]**2+vec[1]**2))
        return alpha if vec[1]>0 else (2*math.pi-alpha)

class DirPreferBMModel(fist_element.BasicModel):
    def __init__(self, field=fist_element.Field2D(20,20,1,False), agentNum = 10000, dt=0.1, D=0.01, init_phase=0.0, palstance=1.0, angle_tor=math.pi/3, sat_rate = 0.7):
        super().__init__(agentNum, dt)
        self.D = D
        self.field = field
        self.init_phase = init_phase
        self.palstance = palstance
        self.angle_tor = angle_tor
        self.cur_time = 0.0
        self.pointer = 0
        self.sat_rate = sat_rate

    def initField(self):
        self.field.set_global_prop('angle',self.init_phase)
        self.field.set_patch(lambda p,x:0,'D',[self.D])

    def initAgents(self):
        for n in range(self.agentNum):
            init_x = random.uniform(0,self.field.width)
            init_y = random.uniform(0, self.field.height)
            self.agents.append(fist_element.Agent(n,init_x,init_y,active = True,dir = random.uniform(-math.pi,math.pi)))

    def next_step(self):
        for agent in self.agents:
            if agent.get('active'):
                step_len = self.step_len_pool[self.pointer]
                self.pointer += 1
                if random.uniform(0,1) > self.sat_rate:
                    angle = random.uniform(-math.pi,math.pi)
                else:
                    angle = self.field.get_global_prop('angle') + random.uniform(-self.angle_tor,self.angle_tor)
                agent.move(step_len * math.cos(angle), step_len * math.sin(angle))
                # check if out of line
                x, y = (agent.get('x'), agent.get('y'))
                if x < 0 or y < 0 or x > self.field.width or y > self.field.height:
                    if self.field.is_warp:
                        agent.set_cur_pos(math.fmod(x, self.field.width), math.fmod(y, self.field.width))
                    else:
                        agent.set('active', False)
        # refresh global prop
        self.cur_time += self.dt
        new_angle = math.fmod(self.field.get_global_prop('angle')+self.palstance*self.dt,2*math.pi)
        self.field.set_global_prop('angle',new_angle)

    def simulate(self,step_num=1000):
        super().simulate(step_num)
        self.step_len_pool = fist_element.BMGenerator2D.get_BM_step_len(self.D,self.dt,step_num*self.agentNum)
        for n in range(step_num):
            self.next_step()
        self.is_simulated = True

class Ant(fist_element.Agent):
    def __init__(self,a_id,x,y,food=0,dir=np.random.randint(0,8,1),max_turn = 3,var_angle=math.pi/10):
        super().__init__(a_id,x,y,food=food)
        self.dir = dir
        self.max_turn = max_turn
        self.var_angle = var_angle
    def turn_back(self):
        if self.dir < 4:
            self.dir += 4
        else:
            self.dir -= 4
    def try_rand_walk(self):
        dir_var = np.random.randint(-self.max_turn,self.max_turn+1,1)
        new_dir = self.dir + dir_var[0]
        if new_dir > 7:
            new_dir -= 8
        elif new_dir < 0:
            new_dir += 8
        self.dir = new_dir
        return self.__try_goto(new_dir)
    def try_move_along(self,dx,dy):
        dir = Ant.vec_to_num(AntField.normalize_vector(np.array([dx,dy])))
        distance = Ant.dir_distance(dir,self.dir)
        if distance > self.max_turn:
            if Ant.is_left(self.dir,dir):
                dir = Ant.turn_num(self.dir,self.max_turn)
            else:
                dir = Ant.turn_num(self.dir,-self.max_turn)
        self.dir = dir
        return self.__try_goto(dir)
    def try_look_for_food(self,sense):
        valid_num = sum(np.array(sense)>0)
        if valid_num <= 1:
            return self.try_rand_walk()
        else:
            pass
    def __try_goto(self,dir_num):
        angle = Ant.num_to_angle(dir_num) + np.random.uniform(-self.var_angle, self.var_angle)
        return np.cos(angle), np.sin(angle)
    @staticmethod
    def is_left(origin,target):
        value_tag = (target - origin) >= 0
        overlap_tag = abs(target-origin) != Ant.dir_distance(target,origin)
        return value_tag ^ overlap_tag
    @staticmethod
    def turn_num(dir,num):
        target = dir + num
        if target > 7:
            return target - 8
        elif target < 0:
            return target + 8
    @staticmethod
    def num_to_vec(dir_num):
        if dir_num in [1,0,7]:
            x = 1
        elif dir_num in [3,4,5]:
            x = -1
        else:
            x = 0
        if dir_num in [1,2,3]:
            y = 1
        elif dir_num in [5,6,7]:
            y = -1
        else:
            y = 0
        return AntField.normalize_vector(np.array([x,y]))
    @staticmethod
    def vec_to_num(vec,thres=0.1):
        x = vec[0]
        if abs(x) < thres:
            x = 0
        elif x > 0:
            x = 1
        else:
            x = -1
        y = vec[1]
        if abs(y) < thres:
            y=0
        elif y > 0:
            y=1
        else:
            y = -1
        if [x,y] == [1,0]:
            return 0
        elif [x,y] == [1,1]:
            return 1
        elif [x,y] == [0,1]:
            return 2
        elif [x,y] == [-1,1]:
            return 3
        elif [x,y] == [-1,0]:
            return 4
        elif [x,y] == [-1,-1]:
            return 5
        elif [x,y] == [0,-1]:
            return 6
        elif [x,y] == [1,-1]:
            return 7
        else:
            return None
    @staticmethod
    def dir_distance(dir_a,dir_b):
        dis = abs(dir_a-dir_b)
        if dis > 4:
            dis = 8 - dis
        return dis
    @staticmethod
    def num_to_angle(dir_a):
        if dir_a <= 4:
            return dir_a * math.pi/4
        else:
            return (dir_a-8) * math.pi/4


class AntField(fist_element.Field2D):
    def __init__(self,width, height, p_width, is_warp=False, evapor_rate=0.1, food_amount = 50):
        super().__init__(width,height,p_width,is_warp,evapor_rate=evapor_rate,food_amount=food_amount)
        self.max_food = food_amount
        self.set_patch(lambda p,arg:0,'signal',[0,0],0)
        self.set_patch(lambda p,arg:0,'food',[0,0],0)
    def set_food_location(self,x,y,r):
        self.set_patch(AntField.isInside2D,'food',[None,self.get_global_prop('food_amount')],[x,y],r)
    def set_nest_location(self,x,y,r):
        self.nest_x = x
        self.nest_y = y
        self.nest_r = r
        self.set_patch(AntField.isInside2D,'home',[0,1],[x,y],r)
    def evapor(self):
        for p in self.patches:
            signal_amount = p.get('signal')
            if signal_amount > 0:
                p.set('signal',max([0,signal_amount-self.get_global_prop('evapor_rate')]))
    def check_for_food(self,x,y):
        p = self.get_nearest_patch(x,y)
        food_amount = p.get('food')
        if food_amount >= 1:
            p.set('food',food_amount-1)
            return 1
        return 0
    def check_for_home(self,x,y):
        p = self.get_nearest_patch(x, y)
        return p.get('home') > 0
    def check_for_signal(self,x,y,r,dir = None):
        check_list = []
        for p in self.patches:
            if AntField.isInside2D(p,([x,y],r)) and (dir is None or AntField.isAhead(p,([x,y],dir))):
                check_list.append([p.pos_x,p.pos_y,p.get('signal')])
        check_list = np.array(check_list)
        #print(check_list.shape[0])
        if check_list.shape[0] == 0:
            return 0,0
        if np.max(check_list[:,2]) == 0:
            return 0,0
        else:
            signal_value = check_list[:,2]
            valid_signal_number = sum(signal_value>0)
            max_pos = check_list[signal_value==np.max(signal_value),:2]
            max_pos = max_pos[0]
            if valid_signal_number == 1:
                return 1,AntField.normalize_vector(max_pos-np.array([x,y]))
                #return 0,0
            else:
                valid_signal = check_list[signal_value>0,:]
                value = valid_signal[:,2]
                min_pos = valid_signal[value==np.min(value),:2]
                min_pos = min_pos[0]
            return 1,AntField.normalize_vector(min_pos-max_pos+min_pos-np.array([x,y]))

    def add_signal(self,x,y):
        p = self.get_nearest_patch(x, y)
        if p.get('signal') == 0:
            p.set('signal',1)
    def to_home(self,x,y):
        vec = np.array([self.nest_x,self.nest_y]) - np.array([x,y])
        return vec/np.sqrt(np.sum(np.power(vec,2)))
    @staticmethod
    def isAhead(p,args):
        xy0 = args[0]
        dir = args[1]
        vec = np.array([p.pos_x,p.pos_y])-np.array(xy0)
        if np.sum(vec)==0:
            return 0
        to_patch = AntField.normalize_vector(vec)
        cos_value = np.dot(dir,to_patch)
        #print(cos_value)
        return 1 if cos_value > 0.3 else 0
    @staticmethod
    def isInside2D(p,args):
        xy0 = args[0]
        r = args[1]
        dist = np.sqrt(np.sum(np.power(np.array([p.pos_x,p.pos_y])-np.array(xy0),2)))
        return 1 if dist<=r else 0
    @staticmethod
    def normalize_vector(x):
        return x/np.sqrt(np.sum(x**2))
    @property
    def food_left(self):
        food = 0
        for p in self.patches:
            food += p.get('food')
        return food
    @property
    def signal_left(self):
        signal = 0
        for p in self.patches:
            signal += p.get('signal')
        return signal

class AntModel(fist_element.BasicModel):
    def __init__(self, field, agentNum = 100, dt=0.1, speed = 1, vision=0.5):
        super().__init__(agentNum,dt)
        self.field = field
        self.speed = speed
        self.vision = vision
    def initAgents(self):
        for n in np.arange(self.agentNum):
            self.agents.append(fist_element.Agent(n,self.field.nest_x,self.field.nest_y,food = 0,dir_x = 1, dir_y = 0))
    def initField(self):
        pass
    def next_step(self,frame):
        print('Frmae: {:d}'.format(frame))
        # refresh evapor
        self.field.evapor()

        # refresh agent position
        for agent in self.agents:
            x,y = (agent.get('x'),agent.get('y'))
            has_food = agent.get('food') > 0
            agent_dir = [agent.get('dir_x'),agent.get('dir_y')]
            if has_food:
                self.field.add_signal(x,y)
            if self.field.check_for_food(x,y) and not has_food:
                agent.set('food',1)
                self.field.add_signal(x, y)
                print('Frame {:d}: pick food by Ant {:d}'.format(frame,agent.a_id))
                has_food = True
            if self.field.check_for_home(x,y) and has_food:
                agent.set('food',0)
                print('Frame {:d}: put food by Ant {:d}'.format(frame, agent.a_id))
                has_food = False

            if has_food:
                dir = self.field.to_home(x, y)
            else:
                bIndex, dir = self.field.check_for_signal(x,y,self.vision,dir=agent_dir)
                if (not bIndex) or (np.dot(dir,np.array(agent_dir))<0):
                    # random walk
                    angle = random.uniform(-math.pi,math.pi)
                    dir = [np.cos(angle),np.sin(angle)]
                else:
                    print('Ant:{:d} sense food at {:f},{:f}'.format(agent.a_id,dir[0],dir[1]))
            dx, dy = (dir[0] * self.speed * self.dt, dir[1] * self.speed * self.dt)
            if type(dx) is np.ndarray:
                print('s')

            dx,dy = self.handle_move(x,y,dx,dy)
            agent.set('dir_x',dx)
            agent.set('dir_y',dy)
            agent.move(dx,dy)

    def handle_move(self,x,y,dx,dy):
        target_x = x + dx
        target_y = y + dy
        is_left_out = target_x < 0
        is_right_out = target_x > self.field.width
        is_top_out = target_y < 0
        is_bottom_out = target_y > self.field.height
        if is_left_out | is_right_out:
            dx = -dx
        if is_top_out | is_bottom_out:
            dy = -dy
        return dx,dy


    def simulate(self,step_num=1000,display = False,saveFig = False):
        super().simulate(step_num)
        self.food_dynamic = []
        self.signal_dynamic = []
        if display:
            fig = plt.figure(figsize=(10,5),dpi=100)
            ax1 = plt.subplot(121,)
            ax2 = plt.subplot(122)
            plt.ion()
            plt.show()
        for n in np.arange(step_num):
            self.next_step(n)
            self.food_dynamic.append(self.field.food_left)
            self.signal_dynamic.append(self.field.signal_left)
            if display:
                ax1.cla()
                ax2.cla()
                food_mat = self.field.patch_prop_list('food')
                chemical_mat = self.field.patch_prop_list('signal')
                has_food = np.array(self.agent_prop('food')) > 0
                x = np.array(self.agent_pos_x)
                y = np.array(self.agent_pos_y)
                ax1.matshow(food_mat,cmap='Reds',vmin=0,vmax=self.field.max_food)
                ax1.scatter(x[has_food==False]/self.field.p_width,y[has_food==False]/self.field.p_width,s=15)
                ax1.scatter(x[has_food ] / self.field.p_width, y[has_food] / self.field.p_width,s=15)
                ax1.set_xlim([0,self.field.n_width])
                ax1.set_ylim([0,self.field.n_height])
                ax2.matshow(chemical_mat,cmap='Blues')
                ax2.set_xlim([0, self.field.n_width])
                ax2.set_ylim([0, self.field.n_height])
                plt.draw()
                if saveFig:
                    plt.savefig('fig_{:04d}.png'.format(n))
                plt.pause(0.01)

        self.is_simulated = True
        if display:
            plt.figure()
            plt.plot(np.arange(step_num)*self.dt,self.food_dynamic,label="Food")
            plt.plot(np.arange(step_num)*self.dt,self.signal_dynamic,label="Signal")
            plt.show()

    def save_csv(self,path):
        super().save_csv('{:s}.csv'.format(path))
        m = np.vstack((np.array(self.food_dynamic),np.array(self.signal_dynamic)))
        m = m.T
        row_fmt = '{0:f},{1:f}\n'
        with open('{:s}_dynamic.csv'.format(path),'w') as f:
            for row in m:
                f.write(row_fmt.format(row[0],row[1]))




