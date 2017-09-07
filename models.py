import fist_element
import random
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpim
import win32ui

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
        self.cur_time += self.dt
        new_angle = math.fmod(self.field.get_global_prop('angle')+self.palstance*self.dt,2*math.pi)
        self.field.set_global_prop('angle',new_angle)

    def simulate(self,step_num=1000):
        super().simulate(step_num)
        self.step_len_pool = fist_element.BMGenerator2D.get_BM_step_len(self.D,self.dt,step_num*self.agentNum)
        for n in range(step_num):
            self.next_step()
        self.is_simulated = True



