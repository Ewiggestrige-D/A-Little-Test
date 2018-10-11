
# coding: utf-8

# ##2.6 Use the Euler method to calculate cannon shell trajectories ignoring both air drag and the effect of air density 
# ##(actually, ignoring the former automatically rules out the latter). 
# ##Compare your results with those in Figure 2.4, and with the exact solution.

# In[14]:


import numpy as np
import math
import matplotlib.pyplot as plt
class trajectory:
    """
    Simulation of cannon shell trajectories ignoring air
    Program by Rongyu Dong
    """
    def __init__(self, initial_position_x = 0, initial_position_y = 0, initial_velocity = 0.7,
                 initial_angle = math.pi/ 4,acceleration_of_gravity = 0.0098, time_step = 0.05):
        self.x = [initial_position_x]
        self.y = [initial_position_y]
        self.vx = [initial_velocity*math.cos(initial_angle)]
        self.vy = [initial_velocity*math.sin(initial_angle)]
        self.g = acceleration_of_gravity
        self.dt = time_step
        self.t = [0]
        print("initial postion x ->",initial_position_x)
        print("initial position y ->",initial_position_y)
        print("initial velocity  ->",initial_velocity)
        print("initial angle ->",initial_angle)
        print("time step ->",time_step)
        print("acceleration of gravity ->",acceleration_of_gravity)
    def calculate(self): 
        i = 0
        while(self.y[i] > -0.000000001):
            self.x.append(self.x[i] + self.vx[i] * self.dt)
            self.y.append(self.y[i] + self.vy[i] * self.dt)
            self.vx.append(self.vx[i])
            self.vy.append(self.vy[i] - self.g * self.dt)
            self.t.append(self.t[i] + self.dt)
            i = i + 1
    def show_results(self,degree = '45°', x_y = [25,12.5],x_yt = [55,12.5]):
        font = {'family': 'serif',
              'color': 'darkred',
              'weight': 'normal',
              'size': 16,
               }
        plt.plot(self.x,self.y)
        plt.title('Cannon shell trajectory ignoring air',fontdict = font)
        plt.xlabel('x ($km$)')
        plt.ylabel('y ($km$)')
        plt.xlim(0,60)
        plt.ylim(0,20)
        plt.annotate(degree, xy = tuple(x_y), xytext=tuple(x_yt),arrowprops=dict(facecolor='black', shrink=0.01,width=1),
            
            )

a = trajectory()
a.calculate()
a.show_results()
b = trajectory(initial_angle=math.pi/6)
b.calculate()
b.show_results('35°',[25,5],[55,5])
c = trajectory(initial_angle=math.pi*35/180)
c.calculate()
c.show_results('35°',[25,7.5],[55,7.5])
d = trajectory(initial_angle=math.pi*40/180)
d.calculate()
d.show_results('40°',[25,10],[55,10])
e = trajectory(initial_angle=math.pi*50/180)
e.calculate()
e.show_results('50°',[25,14],[55,14])
f = trajectory(initial_angle=math.pi*55/180)
f.calculate()
f.show_results('55°',[25,17],[55,17])
plt.show()

class exact_solution(trajectory):
    def show_results(self):
        self.ex = []
        self.ey = []
        for n in range(len(self.t)):
            self.ex.append(self.vx[0] * self.t[n])
            self.ey.append(self.vy[0] * self.t[n]-1/ 2 * self.g * (self.t[n]**2))
        plt.plot(self.ex,self.ey,'*')
        font = {'family': 'serif',
              'color': 'darkred',
              'weight': 'normal',
              'size': 16,
               }
        plt.plot(self.x,self.y)
        plt.title('Cannon shell trajectory ignoring air at 45°',fontdict = font)
        plt.xlabel('x ($km$)')
        plt.ylabel('y ($km$)')
        plt.xlim(0,60)
        plt.ylim(0,20)
        plt.legend(["exact_solution","numerical_method"],loc='upper right')
        plt.show()         


# In[15]:


m = exact_solution(time_step=2)
m.calculate()
m.show_results()


# # 2.10 Generalize the program developed for the previous problem so that it can deal with situations in which the target is at a different altitude than the cannon. 
# ##Consider cases in which the target is higher and lower than the cannon. 
# ##Also investigate how the minimum firing velocity required to hit the target varies as the altitude of the target is varied. 
# ##（需要如同书上图2.4一样考虑风阻的影响）

# In[16]:


class trajectory2:
    """
    Simulation of cannon shell trajectories 
    """
    def __init__(self, initial_position_x = 0, initial_position_y = 0, initial_velocity = 0.7, initial_angle = math.pi/4,
                 target_altitude = 0, b2_m = 0.04, temperature = 300, a = 6.5, exponent_constant = 2.5, r = 6371,
                 acceleration_of_gravity=0.0098,time_step=0.05):
        self.x = [initial_position_x]
        self.y = [initial_position_y]
        self.v = [initial_velocity]
        self.vx = [initial_velocity * math.cos(initial_angle)]
        self.vy = [initial_velocity * math.sin(initial_angle)]
        self.g = acceleration_of_gravity
        self.dt = time_step
        self.altitude = target_altitude
        self.b2_m = b2_m
        self.temp = temperature
        self.r = r
        self.a = a
        self.exp_c = exponent_constant
        self.t = [0]
        print("target altitude ->",target_altitude)
        print("B2/m ->",b2_m)
        print("temperature of the earth ground",temperature,"K")
        print("initial velocity  ->",initial_velocity)
        print("initial angle ->",initial_angle)
        print("time step ->",time_step)
    def calculate(self): 
        i=0
        self.x.append(self.x[i] + self.vx[i] * self.dt)
        self.y.append(self.y[i] + self.vy[i] * self.dt)
        self.vx.append(self.vx[i] - self.b2_m * self.v[i] * self.vx[i] * ((1 - self.a *s elf.y[i+1]/ self.temp)**self.exp_c) * self.dt)
        self.vy.append(self.vy[i] - self.g * (self.r**2)/ ((self.r + self.y[i+1])**2) * self.dt-
                        self.b2_m  * self.v[i] * self.vy[i] * ((1-self.a * self.y[i+1]/self.temp)**self.exp_c) * self.dt)
        self.v.append((self.vx[-1]**2 + self.vy[-1]**2)**(1/2))
        self.t.append(self.t[i] + self.dt)
        i = i + 1
        while((self.y[i]  >  self.altitude) or (self.y[i] > self.y[i-1])):
            self.x.append(self.x[i] + self.vx[i] * self.dt)
            self.y.append(self.y[i] + self.vy[i] * self.dt)
            self.vx.append(self.vx[i] - self.b2_m * self.v[i] * self.vx[i] * ((1 - self.a * self.y[i+1]/ self.temp)**self.exp_c) * self.dt)
            self.vy.append(self.vy[i] - self.g * (self.r**2)/ ((self.r + self.y[i+1])**2) * self.dt -
                          self.b2_m * self.v[i] * self.vy[i] * ((1 - self.a * self.y[i+1]/self.temp)**self.exp_c) * self.dt)
            self.v.append((self.vx[-1]**2 + self.vy[-1]**2)**(1/2))
            self.t.append(self.t[i] + self.dt)
            i = i + 1
    def show_results(self,degree='45°'):
        font={'family': 'serif',
              'color': 'darkred',
              'weight': 'normal',
              'size': 16,
             }
        plt.plot(self.x,self.y)
        plt.title('Cannon shell trajectory with density correction',fontdict = font)
        plt.xlabel('x ($km$)')
        plt.ylabel('y ($km$)')
        plt.xlim()
        plt.ylim()

a=trajectory2(initial_velocity=0.59)
a.calculate()
a.show_results()
b=trajectory2(initial_velocity=0.59,initial_angle=math.pi/6)
b.calculate()
b.show_results('30°')
c=trajectory2(initial_velocity=0.59,initial_angle=math.pi*35/180)
c.calculate()
c.show_results('35°')
d=trajectory2(initial_velocity=0.59,initial_angle=math.pi*40/180)
d.calculate()
d.show_results('40°')
e=trajectory2(initial_velocity=0.59,initial_angle=math.pi*50/180)
e.calculate()
e.show_results('50°')
f=trajectory2(initial_velocity=0.59,initial_angle=math.pi*55/180)
f.calculate()
f.show_results('55°')
plt.legend(['45°','30°','35°','40°','50°','55°'],loc='upper right')
plt.show()


# # 以下为选做题
# ##2.12 Add the effect of the Earth's revolution about its own axis, that is, consider the Coriolis force. 
# ##（在2.10的基础上加入科里奥利力，要求达到的效果是命中你设定的目标）

# In[11]:


import pylab as pl
import math
class projectile_motion:
    def __init__(self, velocity = 700, time_step=0.1, acceleration_y = 9.8, theta = math.pi/4, 
                 coefficient_air_drag = 4*(10**-5), relative_height=0,z0 = 1*10**4, phi = math.pi/4, omega = 7.292*10**(-5)):
        '''initial_vx = total_v*math.sin(theta), initial_vy = total_v*math.cos(theta), '''
        self.v = velocity
        self.vx = [velocity * math.cos(theta)]
        self.vz = [velocity * math.sin(theta)]
        self.vy = [0]
        self.x = [0]
        self.z = [0]
        self.y = [0]
        self.dt = time_step
        self.g = acceleration_y
        self.phi = phi
        self.omega = omega#自转角速度
        self.b = coefficient_air_drag
        self.h = relative_height
        self.z0 = z0
        self.xl = 0
        self.r = 0
        self.nsteps = int(2 * self.vz[0] // acceleration_y + 1)*10
        print('initial vx->', self.vx[0])
        print('initial vz->', self.vz[0])
        print('time step->', time_step)
        print('acceleration_y->', acceleration_y)
    def calculate(self):
        for i in range(self.nsteps):
            tmp_x = self.x[i] + self.vx[i] * self.dt
            tmp_vx = self.vx[i] - self.b * self.vx[i] * self.dt * self.v * math.exp(- self.z[-1]/self.z0) + 2 * self.omega * self.vy[i] * math.sin(self.phi)
            tmp_z = self.z[i] + self.vz[i] * self.dt
            tmp_vz = self.vz[i] - self.g * self.dt - self.b * self.vz[i] * self.dt * self.v * math.exp(- self.z[-1]/self.z0) + 2 * self.omega * self.vy[i] * math.cos(self.phi)
            tmp_y = self.y[i] + self.vy[i] *self.dt
            tmp_vy = -2 * self.omega * (self.vz[i] * math.cos(self.phi) + self.vx[i] * math.sin(self.phi))
            self.x.append(tmp_x)
            self.z.append(tmp_z)
            self.y.append(tmp_y)
            self.vx.append(tmp_vx)
            self.vz.append(tmp_vz)
            self.vy.append(tmp_vy)
            if (tmp_z - self.h) < 0 and tmp_x>10000 :
                break
        self.r = - (self.z[-2] - self.h) / (self.z[-1] - self.h) 
        self.xl = (self.x[i-1] + self.r * self.x[i]) / (self.r+1)
            
    def show_results(self):
        font = {'family': 'serif',
                'color':  'darkred',
                'weight': 'normal',
                'size': 12,
        }
        pl.plot(self.x, self.z,label='trajectory')
        pl.plot(self.x[-1],self.z[-1],'^',label='landing point')
        pl.xlabel('x(m)')
        pl.ylabel('z(m)')
        pl.legend(loc='upper left')
        pl.xlim(0,self.xl*1.1)
        pl.title('cannon shell trajectories with $Coriolis Force$', fontdict = font)
        print('xl=',self.xl,'向东偏移为：',self.y[-1])
        pl.show()


# In[12]:


b = projectile_motion()
b.calculate()
b.show_results()


# <a>在系统中引入科里奥利力之后，将系统修正为地球的赤道-极轴系（xoz coordinate），考虑xoz平面的轨迹，不考虑东西方向的偏移，x、z方向各多了一项加速度。  
# ax = 2 * omega * vy * sin（phi）  
# az = 2 * omega * vy * cos（phi）  
# ay = -2 * omega * （vz * cos（phi）+ vx * sin（phi））  
# 其中Omega是地球的自转角速度， phi为 纬度  
# Coriolis Force简称为科氏力，是对旋转体系中进行直线运动的质点由于惯性相对于旋转体系产生的直线运动的偏移的一种描述。科里奥利力来自于物体运动所具有的惯性。  
#  在本题中地球系中的炮弹飞行受到地球本身自转的影响产生一个横向的位移。  
#     F= -2mv'×ω  
# 式中F为科里奥利力；m为质点的质量；v'为相对于转动参考系质点的运动速度（矢量）；ω为旋转体系的角速度（矢量）；×表示两个向量的外积符号（方向满足右手螺旋定则）则在北半球，炮弹向东飞行时，会受到一个向右的力，这个力即为Coriolis Force。</a>

# In[9]:


import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import math
class trajectory_3D:
    """
    Simulation of cannon shell trajectories 
    """
    def __init__(self, initial_position_x = 0, initial_position_y = 0, initial_position_z = 0, initial_velocity = 0.7,
                 initial_phi = math.pi/4, initial_theta = math.pi/4, latitude = math.pi/4, omega = 7.292 * (10**(-5)),
                 target_altitude = 0, coefficient_air_drag = 4*(10**-5), temperature = 300, a = 6.5, exponent_constant = 2.5,r = 6371,
                 acceleration_of_gravity = 0.0098,time_step = 0.05):
        #上述单位由m转化为Km
        self.x = [initial_position_x]
        self.y = [initial_position_y]
        self.z = [initial_position_z]
        self.v = [initial_velocity]
        self.vx = [initial_velocity * math.sin(initial_phi) * math.cos(initial_theta)]
        self.vy = [initial_velocity * math.sin(initial_phi) * math.sin(initial_theta)]
        self.vz = [initial_velocity * math.cos(initial_phi)]
        self.g = acceleration_of_gravity
        self.dt = time_step
        self.altitude = target_altitude
        self.latitude = latitude
        self.omega = omega#地球自转角速度
        self.b = coefficient_air_drag
        self.temp = temperature#地球表面温度，单位为K（开尔文）
        self.r = r#地球半径的平均值
        self.a = a
        self.exp_c = exponent_constant
        self.t = [0]
        print("Mean value of the earth's radius  ->",self.r)
        print("spin velocity of the earth  ->",self.omega)
        print("target altitude ->",target_altitude)
        print("initial velocity of the cannon  ->",initial_velocity)
        print("time step ->",time_step)
    def calculate(self): 
        i=0
        self.x.append(self.x[i] + self.vx[i] * self.dt)
        self.y.append(self.y[i] + self.vy[i] * self.dt)
        self.z.append(self.z[i] + self.vz[i] * self.dt)  
        self.vx.append(self.vx[i] - self.b * self.v[i] * self.vx[i] * ((1-self.a * self.z[i + 1]/ self.temp)**self.exp_c) * self.dt
                       + 2 * self.omega * self.vy[i] * math.sin(self.latitude) * self.dt)
        self.vy.append(self.vy[i]- self.b * self.v[i] * self.vy[i] * ((1-self.a * self.z[i+1]/self.temp)**self.exp_c) * self.dt
                       - 2 * self.omega * (self.vz[i] * math.cos(self.latitude) + self.vx[i] * math.sin(self.latitude)) * self.dt)
        self.vz.append(self.vz[i]-self.g * (self.r**2)/((self.r+self.z[i+1])**2) * self.dt-
        self.b * self.v[i] * self.vy[i] * ((1 - self.a * self.z[i+1]/self.temp)**self.exp_c) * self.dt
                        + 2 * self.omega * self.vy[i] * math.cos(self.latitude) * self.dt)
        self.v.append((self.vx[-1]**2 + self.vy[-1]**2 + self.vz[-1]**2)**(1/2))
        self.t.append(self.t[i] + self.dt)
        i=i+1
        while((self.z[i]  >  self.altitude) or (self.z[i] > self.z[i-1])):
            self.x.append(self.x[i] + self.vx[i] * self.dt)
            self.y.append(self.y[i] + self.vy[i] * self.dt)
            self.z.append(self.z[i] + self.vz[i] * self.dt)  
            self.vx.append(self.vx[i] - self.b * self.v[i] * self.vx[i] * ((1 - self.a * self.z[i+1]/ self.temp)**self.exp_c) * self.dt
                          + 2 * self.omega * self.vy[i] * math.sin(self.latitude) * self.dt)
            self.vy.append(self.vy[i] - self.b * self.v[i] * self.vy[i] * ((1 - self.a * self.z[i+1]/ self.temp)**self.exp_c) * self.dt
                          - 2 * self.omega * (self.vz[i] * math.cos(self.latitude) + self.vx[i] * math.sin(self.latitude)) * self.dt)
            self.vz.append(self.vz[i] - self.g * (self.r**2)/ ((self.r + self.z[i+1])**2) * self.dt -
                           self.b * self.v[i] * self.vy[i] * ((1 - self.a * self.z[i+1]/ self.temp)**self.exp_c) * self.dt
                          + 2 * self.omega * self.vy[i] * math.cos(self.latitude) * self.dt)
            self.v.append((self.vx[-1]**2 + self.vy[-1]**2 + self.vz[-1]**2)**(1/2))
            self.t.append(self.t[i] + self.dt)
            i=i+1
    def show_results(self,degree='45°'):
        font={'family': 'serif',
              'color': 'darkred',
              'weight': 'normal',
              'size': 12,
             }
       
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot(self.x, self.y, self.z, label='trajectory') 
        plt.title('Cannon shell trajectory with $Coriolis force$ in 3D',fontdict = font)
        ax.set_xlabel('x ($km$)')
        ax.set_ylabel('y ($km$)')
        ax.set_zlabel('z ($km$)')
        plt.show()


# In[10]:


b = trajectory_3D()
b.calculate()
b.show_results()

