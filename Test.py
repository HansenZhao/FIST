import fist_element
import models
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


hbmm = models.HomoBMModel()
hbmm.simulate()
hbmm.save_csv('test.csv')
'''

sh = models.SpatialHeteroBMModel(False,1000,0.1)
sh.simulate(3000)
#sh.overview()
sh.save_csv('simu_hetero.csv')




