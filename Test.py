import fist_element
import models
import math
import matplotlib.pyplot as plt

'''
a = fist_element.Patch(1,1,1,D=0.01,asym = 100)
print(a)


b = fist_element.Agent(2)
print(b)
b.move(0,1)
b.move(1,0)
b.move(1,1)
print(b)
print(b.get_particle_mat())

def func(p,x):
    return 0 if p.pos_x < x[0] else 1

c = fist_element.Field2D(10,10,0.5)
c.set_patch(func,'D',[0.1,0.01],5)
c.disp('D')
c.disp('pos_x')
c.disp('pos_y')

d = fist_element.BMGenerator2D()
x,y = d.get(10000)
plt.plot(x,y)
plt.show()

for m in range(10):
    hbmm = models.HomoBMModel(fist_element.Field2D(38,38,1,False),agentNum=200,D=0.05)
    hbmm.simulate(step_num=3000)
    hbmm.save_csv('homo_200_{:d}.csv'.format(m))

sh = models.SpatialHeteroBMModel(False,500,0.1,D_tuple=(0.02,0.06),sat_rate=0.7)
sh.simulate(step_num=3000)
#sh.overview()
sh.save_csv('hetero_D_THU.csv')

sh = models.SpatialHeteroBMModel(False,500,0.1,D_tuple=(0.02,0.06),sat_rate=0.7)
sh.simulate(step_num=3000)
#sh.overview()
sh.save_csv('hetero_D_angle_THU_s.csv')


dp = models.DirPreferBMModel(agentNum=500,palstance=0.1,sat_rate=0.0,angle_tor=math.pi*1.0)
dp.simulate(step_num=3000)
dp.save_csv('dirprefer_high_00_10.csv')

dp = models.DirPreferBMModel(agentNum=500,palstance=0.1,sat_rate=0.8,angle_tor=math.pi*0.8)
dp.simulate(step_num=3000)
dp.save_csv('dirprefer_high_08_08.csv')

dp = models.DirPreferBMModel(agentNum=500,palstance=0.1,sat_rate=0.8,angle_tor=math.pi*0.2)
dp.simulate(step_num=3000)
dp.save_csv('dirprefer_high_08_02.csv')

dp = models.DirPreferBMModel(agentNum=500,palstance=0.1,sat_rate=0.2,angle_tor=math.pi*0.8)
dp.simulate(step_num=3000)
dp.save_csv('dirprefer_high_02_08.csv')


dp = models.DirPreferBMModel(agentNum=500,palstance=0.1,sat_rate=0.2,angle_tor=math.pi*0.2)
dp.simulate(step_num=3000)
dp.save_csv('dirprefer_high_02_02.csv')
'''
f = models.AntField(10,10,0.5,evapor_rate=0.1,food_amount=5)
f.set_nest_location(5,5,1)
f.set_food_location(1,1,1)
f.set_food_location(1,9,1)
f.set_food_location(9,5,1)


am = models.AntModel(f,agentNum=200,dt=0.1,speed=5,vision=1)
am.simulate(step_num=1000,display=True)
am.save_csv('ant')

'''
f = models.AntField(7,7,0.5,evapor_rate=0.05,food_amount=1000)
f.set_nest_location(1,1,1)
f.set_food_location(4,4,1)

am = models.AntModel(f,agentNum=20,dt=1,speed=0.5,vision=1.5)
am.simulate(step_num=1000,display=True)
'''

