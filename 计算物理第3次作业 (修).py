
# coding: utf-8

# In[1]:


##2.6 Use the Euler method to calculate cannon shell trajectories ignoring both air drag and the effect of air density 
##(actually, ignoring the former automatically rules out the latter). 
##Compare your results with those in Figure 2.4, and with the exact solution.


# In[2]:


##2.10 Generalize the program developed for the previous problem so that it can deal with situations in which the target is at a different altitude than the cannon. 
##Consider cases in which the target is higher and lower than the cannon. 
##Also investigate how the minimum firing velocity required to hit the target varies as the altitude of the target is varied. 
##（需要如同书上图2.4一样考虑风阻的影响）


# In[3]:


##以下为选做题
##2.12 Add the effect of the Earth's revolution about its own axis, that is, consider the Coriolis force. 
##（在2.10的基础上加入科里奥利力，要求达到的效果是命中你设定的目标）


# In[4]:


class bicycle:
    def __init__(self, power, mass,                 time_step, total_time, initial_velocity):
        pass
    def run(self):
        pass
    def show_results(self):
        pass


# In[5]:


def __init__(self, power=10, mass=1, time_step=0.1,                total_time=20, initial_velocity=1):
       self.v = [initial_velocity]
       self.t = [0]
       self.m = mass
       self.p = power
       self.dt = time_step
       self.time = total_time


# In[6]:


def run(self):
        _time = 0
        while(_timer < self.time):
            self.v.append(self.v[-1] +                          self.dt * self.p / (self.m * self.v[-1]))
            self.t.append(_time)
            _time += self.dt


# In[7]:


def show_results(self):
        pl.plot(self.t, self.v)
        pl.xlabel('time ($s$)')
        pl.ylabel('velocity')
        pl.show()


# In[8]:


import pylab as pl
class bicycle:
    def __init__(self, power=10, mass=1, time_step=0.1,                 total_time=20, initial_velocity=1):
        self.v = [initial_velocity]
        self.t = [0]
        self.m = mass
        self.p = power
        self.dt = time_step
        self.time = total_time   
    def run(self):
        _time = 0
        while(_time < self.time):
            self.v.append(self.v[-1] +                          self.dt * self.p / (self.m * self.v[-1]))
            self.t.append(_time)
            _time += self.dt    
    def show_results(self):
        font = {'family': 'serif',
                'color':  'darkred',
                'weight': 'normal',
                'size': 16,
        }
        pl.plot(self.t, self.v)
        pl.title('Bicycling without air resistance', fontdict = font)
        pl.xlabel('time ($s$)')
        pl.ylabel('velocity')
        pl.text(0.2 * self.time, 0.9 * self.v[-1],                'velocity with time', fontdict = font)
        pl.show()


# In[9]:


a = bicycle()
a.run()
a.show_results()


# In[10]:


#第一题
import pylab as pl
import math
class cannon:
    def __init__(self, distance_x = 0, distance_y = 0, velocity = 700,theta=math. pi/6, acceleration_x = 0,acceleration_y = 9.8, 
                 time_step = 0.1, ):
        '''选择炮弹的出射角与地面夹角为30度，重力加速度为9.8m/s^2'''
        self.x =  [distance_x]
        self.y =  [distance_y]
        self.v =  velocity
        self.vx = [velocity * math. cos(theta)]
        self.vy = [velocity * math. sin(theta)]
        self.ax = acceleration_x
        self.ay = acceleration_y
        self.theta = theta
        self.t = [0]
        self.dt = time_step 
        self.nsteps = int(2 *  self.vy[0]// acceleration_y + 1)*10
        print("Initial Velocity in x-dierection ->",self.vx[0])
        print("Initial Velocity in y-dierection ->",self.vy[0])
        print("time step -> ", time_step)
        print("acceleration in y-direction -> ", acceleration_y  )
    def run(self):
        for i in range(self.nsteps):
            temp1 = self.x[i] + self.vx[0] * self.dt
            temp2 = self.y[i] + self.vy[i] * self.dt
            temp3 = self.vy[i] - self.ay * self.dt
            self.x. append(temp1)
            self.y. append(temp2)
            self.vy. append(temp3)
            if temp2 < 0 :
                    break
                    
    def show_results(self):
        font = {'family': 'serif',
                'color':  'darkred',
                'weight': 'normal',
                'size': 12,
        }
        pl.plot(self.x, self.y)
        pl.title('cannon shell trajectories without air resistance', fontdict = font)
        pl.xlabel('X-direction ($m$)')
        pl.ylabel('Y-direction ($m$)')
        pl.text( 0.2 * self.x[0] , 0.9 * self.y[0], 'trajectory', fontdict = font)
        pl.show()


# In[11]:


b = cannon()
b.run()
b.show_results()


# <a>第一题中控制初速度为700*m/s*，角度theta = pi/4。在没有空气阻力的情况下，炮弹的轨迹为标准的抛物线。</a>

# In[12]:


import pylab as pl
import math
class exact_result_check(cannon):
    def show_result(self):
        self.ety = [] #创建新列表，用于记录精确解的纵坐标
        for i in range(len(self.x)-2):
            self.ety.append(-1/2 * self.ay/self.v**2/math.cos(self.theta)**2 * self.x[i] * 
                            (self.x[i] - 2 * self.v**2 * math.sin(self.theta) * math.cos(self.theta)/self.ay))
        self.etx = self.x[:len(self.x)-2]
        self.ety.append(0)
        self.etx.append(2 * self.v**2 * math. sin(self.theta) * math. cos(self.theta)/self.ay)

        pl.plot(self.etx, self.ety,label='exact_result')
        pl.plot(self.x, self.y,label='approximate_result')
        pl.legend(loc='upper right')
        pl.xlabel('X-direction ($m$)')
        pl.ylabel('Y-direction ($m$)')
        pl.title('exact_result_comparision')
        pl.show()


# In[13]:


c = exact_result_check()
c.run()
c.show_result()


# <a>精确的轨迹方程为：y = -g/(2v^2 * cos^2(theta)) * x * (x - 2v^2sin(theta)cos(thets)),将精确方程和用欧拉法得到的方程结合。  
# 由于差距主要来源于高阶小项，且途中x、y比例较大，故差别不是很明显</a>

# In[14]:


import pylab as pl
import math
class cannon_with_air :
    def __init__(self, distance_x = 0, distance_y = 0, theta=math. pi/4, acceleration_x = 0,acceleration_y = 9.8, 
                 time_step = 0.1, coefficient_air_drag = 4*10**(-5),target_x = 20000,
                target_y=500):
        '''选择炮弹的出射角与地面夹角为30度，重力加速度为9.8m/s^2'''
        self.x0 = target_x
        self.y0 = target_y
        self.x =  [distance_x]
        self.y =  [distance_y]
        self.x1 =  [distance_x]
        self.y1 =  [distance_y]
        self.ay = acceleration_y
        self.theta = theta
        self.t = [0]
        self.b = coefficient_air_drag
        self.dt = time_step 
        print('炮弹的出射角 -> ',theta)
        print("time step -> ", time_step)
        print("acceleration in y-direction -> ", acceleration_y  )
        print("coefficient of the air drag in x,y-direction -> ", coefficient_air_drag  )
    def run(self):
        if math.tan(self.theta) <=self.y0/self.x0 :
            print('在该角度下炮弹无法命中目标点')
        else:
            self.v = [math.sqrt(self.x0**2 * self.ay/ 2 / math.cos(self.theta)**2 / (math.
                    tan(self.theta) * self.x0 - self.y0))]
            self.vx = [self.v[0] * math. cos(self.theta)]
            self.vy = [self.v[0] * math. sin(self.theta)]
            self.vx1 = [self.v[0] * math. cos(self.theta)]
            self.vy1 = [self.v[0] * math. sin(self.theta)]
            self.time = self.x0 /self.v[0] /math.cos(self.theta)
        for i in range(int(self.time/self.dt)):
            temp1 = self.x[i] + self.vx[i] * self.dt
            temp2 = self.y[i] + self.vy[i] * self.dt
            temp3 = self.x1[i] + self.vx1[i] * self.dt
            temp4 = self.y1[i] + self.vy1[i] * self.dt
            temp5 = self.vy[i] - self.ay * self.dt
            temp6 = self.vx[i]
            temp7 = self.vy1[i] - self.ay * self.dt - (self.b * self.v[i] * self.vy1[i] * self.dt)
            temp8 = self.vx1[i]  - (self.b * self.v[i] * self.vx1[i] * self.dt)
            temp9 = math.sqrt(self.vx[i]**2 + self.vy[i]**2)
            self.x. append(temp1)
            self.y. append(temp2)
            self.x1. append(temp3)
            self.y1. append(temp4)
            self.vy. append(temp5)
            self.vx. append(temp6)
            self.vy1. append(temp7)
            self.vx1. append(temp8)
            self.v.append(temp9)
           
    def show_results(self):
        font = {'family': 'serif',
                'color':  'darkred',
                'weight': 'normal',
                'size': 12,
        }
        pl.plot(self.x, self.y,'b-',label='without air resistance')
        pl.plot(self.x1, self.y1,'r-',label='with air resistance')
        pl.plot(self.x0, self.y0,'^',label='target point')
        pl.legend(loc='upper right')
        pl.title('cannon shell trajectories with air resistance', fontdict = font)
        pl.xlabel('X-direction ($m$)')
        pl.ylabel('Y-direction ($m$)')
        pl.text( 1.2 * self.x[0] , 0.9 * self.y[0], 'trajectory with air drag ', fontdict = font)
        pl.show()


# In[15]:


b = cannon_with_air()
b.run()
b.show_results()


# <a>`给定目标点的位置，发射角也自由选取，但如果发射角不满足tan（\theta）>y0/x0，则炮弹无论如何也不能落到目标点。给定目标点之后，可以算出击中目标点得炮弹轨迹方程为y = （y0-tan（\theta）*x0）/ x0^2,根据理论上的位置和平抛运动的规律可解出炮弹的轨迹方程。  
# 由计算可得出v^2(min) = x0^2 * g/ (sqrt(x0^2+y0^2)-y0)</a>

# In[16]:


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


# In[17]:


b = projectile_motion()
b.calculate()
b.show_results()


# <a>在系统中引入科里奥利力之后，将系统修正为地球的赤道-极轴系（xoz coordinate），考虑xoz平面的轨迹，不考虑东西方向的偏移，x、z方向各多了一项加速度。  
# ax = 2 * omega * vy * sin（phi）  
# az = 2 * omega * vy * cos（phi）  
# ay = -2 * omega * （vz * cos（phi）+ vx * sin（phi））  
# 其中Omega是地球的自转角速度， phi为 纬度</a>

# In[48]:


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


# In[49]:


b = trajectory_3D()
b.calculate()
b.show_results()


# <a>将二维的平面移到三维，可以看出轨迹在x方向的偏移</a>

# <a>致谢
# 感谢胡维宇同学在我写Coriolis Force in 3D过程中提供的帮助，他在对待题目的认证程度上远超与我。本来没打算把3D部分写出来，用xoz平面也能基本展示科里奥利力的影响，但是在他的鼓励和帮助下，我重新从网上找到了python 3D作图的教程，并且把这部分完成。</a>
