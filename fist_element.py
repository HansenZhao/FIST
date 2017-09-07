import math
import numpy as np
import random
from abc import ABC,abstractmethod
import matplotlib.pyplot as plt

class Patch (object):
    def __init__(self, p_id=None, pos_x=None, pos_y=None, **kwargs):
        self.p_id = p_id
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.prop = dict()
        self.prop['p_id'] = self.p_id
        self.prop['pos_x'] = self.pos_x
        self.prop['pos_y'] = self.pos_y
        for key, value in kwargs.items():
            self.prop[key] = value

    def __str__(self):
        return "Patch {0} at [{1},{2}] with {3} props".format(self.p_id, self.pos_x, self.pos_y, len(self.prop))

    def set(self, name, value):
        self.prop[name] = value

    def get(self, name, default_value=None):
        if name in self.prop.keys():
            return self.prop[name]
        elif default_value:
            self.prop[name] = default_value
            return default_value
        else:
            return None


class Agent (object):
    def __init__(self, a_id=None, x=0, y=0, **kwargs):
        self.a_id = a_id
        self.prop = dict()
        self.__traj = []
        self.__traj.append((x, y))
        self.prop['x'] = self.__traj[-1][0]
        self.prop['y'] = self.__traj[-1][1]
        for key, value in kwargs.items():
            self.prop[key] = value

    def __str__(self):
        return "Agent {0} at [{1},{2}] with {3} props".format(self.a_id, self.get('x'), self.get('y'), len(self.prop))

    def __len__(self):
        return len(self.__traj)

    def move(self, dx, dy):
        self.__traj.append((self.get('x')+dx, self.get('y')+dy))
        self.prop['x'], self.prop['y'] = self.__traj[-1]

    def get_dimension(self,dim):
        x = np.mat(self.__traj)
        return x[:,dim]

    def get_particle_mat(self):
        m = np.mat(self.__traj)
        m_id = np.ones(shape=(self.__len__(),1))*self.a_id
        m_frame = np.arange(self.__len__()).transpose()
        res = np.column_stack((m_id, m_frame, m, np.zeros(shape=(self.__len__(),1))))
        return res

    def set_cur_pos(self,x,y):
        self.__traj[-1] = (x,y)

    def set(self, name, value):
        self.prop[name] = value

    def get(self, name, default_value=None):
        if name in self.prop.keys():
            return self.prop[name]
        elif default_value:
            self.prop[name] = default_value
            return default_value
        else:
            return None


class Field2D (object):
    def __init__(self,width, height, p_width, is_warp=False, **kwargs):
        self.width = math.ceil(width/p_width)*p_width
        self.height = math.ceil(height/p_width)*p_width
        self.p_width = p_width
        self.n_width = int(self.width/self.p_width)
        self.n_height = int(self.height/self.p_width)
        self.patches = []
        self.is_warp = is_warp

        for y in range(self.n_height):
            for x in range(self.n_width):
                self.patches.append(Patch(x+y*self.n_width, x, y))

        self.__global_prop = dict()
        for key,value in kwargs.items():
            self.__global_prop[key] = value

    def __len__(self):
        return len(self.patches)

    def set_patch(self,func,name,value_list,*args):
        for p in self.patches:
            I = func(p,args)
            p.set(name,value_list[I])

    def __xy2index(self,x, y):
        return x+y*self.n_width

    def disp(self,name):
        for nline in range(self.n_height):
            v = []
            for ncol in range(self.n_width):
                v.append(self.patches[self.__xy2index(ncol,nline)].get(name))
            print(v)

    def get_nearest_patch(self,x,y):
        if x < 0 or y < 0 or x > self.width or y > self.height:
            return None
        return self.patches[self.__xy2index(math.floor(x/self.p_width),math.floor(y/self.p_width))]

    def get_global_prop(self,name,default_value=None):
        if name in self.__global_prop.keys():
            return self.__global_prop[name]
        elif default_value:
            self.__global_prop[name] = default_value
            return default_value
        else:
            return None

    def set_global_prop(self,name,value):
        self.__global_prop[name] = value



class BasicModel (ABC):
    def __init__(self, agentNumber, dt):
        self.agentNum = agentNumber
        self.dt = dt
        self.agents = []
        self.is_simulated = False
    @abstractmethod
    def initField(self):pass
    @abstractmethod
    def initAgents(self):pass
    @abstractmethod
    def simulate(self,step_num=1000):
        self.initField()
        self.initAgents()
    def overview(self):
        if self.is_simulated:
            for n in range(self.agentNum):
                plt.plot(self.agents[n].get_dimension(0),self.agents[n].get_dimension(1))
            plt.show()
    def save_csv(self,path):
        m = self.agents[0].get_particle_mat()
        for agent in self.agents[1:]:
            m = np.row_stack((m,agent.get_particle_mat()))
        row_fmt = '{0:f},{1:f},{2:f},{3:f},{4:f}\n'
        with open(path,'w') as f:
            for row in m:
                f.write(row_fmt.format(row[0,0],row[0,1],row[0,2],row[0,3],row[0,4]))


class BMGenerator2D(object):
    def __init__(self,D=0.01,delta_t=0.1,eps = 0.0001):
        self.D = D
        self.delta_t = delta_t
        self.u0 = math.sqrt(-4*D*delta_t*math.log(eps))
        self.eps = eps

    def get(self,stepNumber):
        u,prop = BMGenerator2D.get_BM_distribution(self.D,self.delta_t,self.u0/300,self.eps)
        step_len = np.random.choice(u,size = stepNumber-1,p=prop)
        delta_x = [0]
        delta_y = [0]
        for n in range(stepNumber-1):
            angle = random.uniform(0,2*math.pi)
            delta_x.append(step_len[n]*math.cos(angle))
            delta_y.append(step_len[n]*math.sin(angle))
        x = np.array(delta_x).cumsum()
        y = np.array(delta_y).cumsum()
        return delta_x,delta_y,x,y

    @staticmethod
    def step_len_intens(u, D, delta_t):
        return u/(2*D*delta_t)*math.exp(-math.pow(u, 2)/(4*D*delta_t))
    @staticmethod
    def get_BM_distribution(D, delta_t, interval, eps):
        u = [0.0]
        p = [BMGenerator2D.step_len_intens(u[0],D,delta_t)*interval]
        while((1-sum(p))> eps):
            new_u = u[-1]+interval
            u.append(new_u)
            p.append(BMGenerator2D.step_len_intens(new_u,D,delta_t)*interval)
        u.append(u[-1]+interval)
        p.append(1-sum(p))
        return  u,p
    @staticmethod
    def get_BM_step_len(D,delta_t,step_num = 1000, eps=0.0001):
        u0 = math.sqrt(-4*D*delta_t*math.log(eps))
        u,prop = BMGenerator2D.get_BM_distribution(D,delta_t,u0/300,eps)
        step_len = np.random.choice(u, size=step_num, p=prop)
        return step_len
