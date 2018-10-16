
# coding: utf-8

# **2.17** Investigate the trajectories of knuckleballs as a function of the angular velocity $\omega$, the initial angular orientation, and the (center of mass) velocity.

# **2.20** Calculate the effect of the knuckball force on a batted ball. Assume that the ball is a line drive hit at an initial speed of 90 mph and an angle of $20^\circ$ and that the ball does not spin at all (clearly a gross assumption). Let the rough side of the ball always face one side, which is perpendicular to the direction the ball is hit, and use Figure 2.11 to estimate the magnitude of the lateral force. With all of these assumptions, calculate the lateral deflection of the ball. Such balls hit with very little spin are observed by outfielders to effectively "flutter" from side to side as they move through the air.

# In[1]:


import numpy as np
import math
import matplotlib.pyplot as plt
class trajectory:
    """
    Program about the trajectory of the knuckleball with rough surface which causes a special aerodynamics pattern
    coding by Rongyu Dong
    
    """
    def __init__(self,velocity = 29.069444, angular_velocity = 0.2 * 2 * math.pi, initial_theta = 0, 
                 initial_angle = math.pi/4, time_step = 0.01, acceleration_of_gravity = 9.8,
                 v_d = 35, delta = 5, constant_1 = 0.0039, constant_2 = 0.0058,):#单位采用m/s
        self.x = [0]
        self.y = [1.8]#假设投球手身高1.8m
        self.z = [0]
        self.theta = [initial_theta]
        self.v = [velocity]
        self.vx = [velocity*math.cos(initial_angle)]
        self.vz = [velocity*math.sin(initial_angle)]
        self.vy = [0]
        self.g = acceleration_of_gravity
        self.angular_velocity = angular_velocity 
        self.v_d = v_d
        self.delta = delta
        self.constant_1 = constant_1
        self.constant_2 = constant_2
        self.f_lateral = [0.5 * (math.sin(4 * self.theta[0]) - 0.25 * math.sin(8 * self.theta[0]) + 0.08 * math.sin(12 * self.theta[0])
                               - 0.025 * math.sin(16 * self.theta[0]))]
        self.b_m = [0.0039 + 0.0058/ (1 + math.e**(self.v[0]/ 5))]
        self.dt = time_step
        self.t = [0]
        print("initial velocity  ->", velocity)
        print("angular velocity ->", angular_velocity)
        print("initial_theta ->", initial_theta)
        print("initial_angle ->", initial_angle)
        print("time step ->", time_step)
    def calculate(self):
        i=0
        while(self.z[i] > -0.000000001):
            self.x.append(self.x[i] + self.vx[i] * self.dt)
            self.y.append(self.y[i] + self.vy[i] * self.dt)
            self.z.append(self.z[i] + self.vz[i] * self.dt)
            self.vx.append(self.vx[i] - self.b_m[-1] * self.v[-1] * self.vx[-1] * self.dt)
            self.vy.append(self.vy[-1]+ self.f_lateral[-1] * 9.8 * self.dt)
            self.vz.append(self.vz[i]-9.8 * self.dt-self.b_m[-1] * self.v[-1] * self.vz[-1] * self.dt)
            self.v.append((self.vx[-1]**2 + self.vy[-1]**2 + self.vz[-1]**2)**(1/2))
            self.theta.append(self.theta[-1] + self.angular_velocity * self.dt)
            self.b_m.append(self.constant_1 - (self.constant_2/ (1+ math.exp(self.v[-1] - self.v_d)/self.delta)))
            self.f_lateral.append(0.5 * (math.sin(4 * self.theta[-1]) - 0.25* math.sin(8 * self.theta[-1]) 
                                    + 0.08 * math.sin(12 * self.theta[-1]) - 0.025 * math.sin(16 * self.theta[-1])))
            self.t.append(self.t[i]+self.dt)
            i=i+1
    def show_results(self):
        font={'family': 'serif',
              'color': 'darkred',
              'weight': 'normal',
              'size': 12,
             }
        plt.plot(self.x,self.y)
        plt.plot(self.x,self.z)
        plt.title('trajectory of knuckleball with spin in orientation',fontdict = font)
        plt.xlabel('x ($m$)')
        plt.ylabel('y or z ($m$)')
        plt.legend(["y-x","z-x"],loc='upper right')
        plt.show()


# In[2]:


a =trajectory()
a.calculate()
a.show_results()


# <a>由上图可以看出y方向的偏转具有一定的周期性，这个周期和球的自转角速度$\omega$有一定的关系。</a>

# In[3]:


a1=trajectory(angular_velocity=0)
a1.calculate()
a2=trajectory(angular_velocity=0.5*math.pi)
a2.calculate()
a3=trajectory(angular_velocity=1*math.pi)
a3.calculate()
a4=trajectory(angular_velocity=1.5*math.pi)
a4.calculate()
a5=trajectory(angular_velocity=2*math.pi)
a5.calculate()
a6=trajectory(angular_velocity=2.5*math.pi)
a6.calculate()
 
fig = plt.figure()  

ax1 = fig.add_subplot(231)  
ax1.plot(a1.x, a1.y)  
  
ax2 = fig.add_subplot(232)  
ax2.plot(a2.x,a2.y)  
  
ax3 = fig.add_subplot(233)  
ax3.plot(a3.x,a3.y)  
  
ax4 = fig.add_subplot(234)  
ax4.plot(a4.x, a4.y)  

ax5 = fig.add_subplot(235)  
ax5.plot(a5.x, a5.y)  

ax6 = fig.add_subplot(236)  
ax6.plot(a6.x, a6.y)  
  
plt.show()  


# <a>上图的对比可以看出，当自旋$\omega$从零开始增加时，在Y方向上的偏转会逐渐增大；  
#     同时自旋$\omega$增大，偏转的周期性将会增大，逐渐趋于线性</a>

# In[4]:


#第一题
import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class knuckleballs:
    def __init__(self, time_step=0.1, initial_x=0, initial_y=1.8, initial_z=0,launch_angle=math.pi/4,
                 initial_angular_orientation=0, initial_velocity=15, gravity_constant=9.8, 
                 home_plate=20,angular_velocity=0.2 ):
        self.dt=time_step
        self.x=[initial_x]
        self.y=[initial_y]
        self.z=[initial_z]
        self.V=initial_velocity
        self.Vx=[initial_velocity*math.cos(launch_angle)]
        self.Vy=[initial_velocity*math.sin(launch_angle)]
        self.Vz=[0]
        self.g=gravity_constant
        self.a=launch_angle
        self.b=[initial_angular_orientation]
        self.w=angular_velocity
        self.home_plate=home_plate
    
    def run(self):
        self.B_m=[0.0039+0.0058/(1+math.exp((self.V-35)/5))]#根据书上2.26公式得到
        while self.x[-1]<self.home_plate :#当棒球的横坐标小于本垒时进行如下循环
            self.x.append(self.x[-1]+self.Vx[-1]*self.dt)
            self.y.append(self.y[-1]+self.Vy[-1]*self.dt)
            self.z.append(self.z[-1]+self.Vz[-1]*self.dt)
            self.Vx.append(self.Vx[-1]-self.B_m[-1]*self.V*self.Vx[-1]*self.dt)
            self.Vy.append(self.Vy[-1]-self.g*self.dt-self.B_m[-1]*self.V*self.Vy[-1]*self.dt)
            self.Vz.append(self.Vz[-1]+0.5*self.g*(math.sin(4*self.b[-1])-0.25*math.sin(8*self.b[-1])+0.08*math.sin(12*self.b[-1])-0.025*math.sin(16*self.b[-1]))*self.dt)
            self.b.append(self.b[-1]+self.w*self.dt)
            self.B_m.append(0.0039+0.0058/(1+math.exp(((math.sqrt(self.Vx[-1]**2+self.Vy[-1]**2)-35)/5))))
    
    def show(self):
        mpl.rcParams['legend.fontsize']=20
        font={'color':'b','style' : 'oblique','size' : 20,'weight' : 'bold'}
        fig=plt.figure(figsize=(16,12))
        ax=fig.gca(projection='3d')
        ax.set_xlabel("X", fontdict=font)
        ax.set_ylabel("Z", fontdict=font)
        ax.set_zlabel("Y", fontdict=font)
        ax.plot(self.x,self.z,self.y,label='trajectories of knuckleballs')
        ax.legend(loc='upper right')
        plt.show()


# In[5]:


a =knuckleballs()
a.run()
a.show()


# <a>由题目的要求可知，本题中自旋$\omega$=0，同时取向角保持20度，则将上题中数据进行简单修改即可</a>

# In[6]:


a1 =trajectory(velocity=40,angular_velocity=0,initial_theta=20/180*math.pi)
a1.calculate()
a1.show_results()
font={'family': 'serif',
      'color': 'darkred',
      'weight': 'normal',
      'size': 12,
     }
plt.plot(a1.x,a1.y)
plt.title('trajectory of knuckleball in Y-Direction without spin in orientation of $\pi/9$',fontdict = font)
plt.show()


# <a>由上图可以看出当自转角速度$\omega$消失后，y的偏移将大大增加，函数趋向于二阶多项式  
# 在实际的球场上，则可以预见Knuckle Ball的偏移将会很大，给击球手造成很大的困难。  
# 那么在三维情况下的图如下所示</a>

# In[7]:


#第2题
import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class knuckleballs_2:
    def __init__(self, time_step=0.01, initial_x=0, initial_y=1.8, initial_z=0,launch_angle=math.pi/4,
                 initial_angular_orientation=math.pi/9, initial_velocity=40.25, gravity_constant=9.8, 
                 home_plate=20 ,angular_velocity=0,):#假设投球手身高1.8m
        self.dt=time_step
        self.x=[initial_x]
        self.y=[initial_y]
        self.z=[initial_z]
        self.V=initial_velocity
        self.Vx=[initial_velocity*math.cos(launch_angle)]
        self.Vy=[initial_velocity*math.sin(launch_angle)]
        self.Vz=[0]
        self.g=gravity_constant
        self.a=launch_angle
        self.b=[initial_angular_orientation]
        self.w=angular_velocity
        self.home_plate=home_plate
    
    def run(self):
        self.B_m=[0.0039+0.0058/(1+math.exp((self.V-35)/5))]#根据书上2.26公式得到
        while self.y[-1]>0 :
            self.x.append(self.x[-1]+self.Vx[-1]*self.dt)
            self.y.append(self.y[-1]+self.Vy[-1]*self.dt)
            self.z.append(self.z[-1]+self.Vz[-1]*self.dt)
            self.Vx.append(self.Vx[-1]-self.B_m[-1]*self.V*self.Vx[-1]*self.dt)
            self.Vy.append(self.Vy[-1]-self.g*self.dt-self.B_m[-1]*self.V*self.Vy[-1]*self.dt)
            self.Vz.append(self.Vz[-1]+0.5*self.g*(math.sin(4*self.b[-1])-0.25*math.sin(8*self.b[-1])+0.08*math.sin(12*self.b[-1])-0.025*math.sin(16*self.b[-1]))*self.dt)
            self.b.append(self.b[-1]+self.w*self.dt)
            self.B_m.append(0.0039+0.0058/(1+math.exp(((math.sqrt(self.Vx[-1]**2+self.Vy[-1]**2)-35)/5))))
    
    def show(self):
        mpl.rcParams['legend.fontsize']=16
        font={'color':'r','style' : 'oblique','size' : 16,'weight' : 'bold'}
        fig=plt.figure(figsize=(16,12))
        ax=fig.gca(projection='3d')
        ax.set_xlabel("X", fontdict=font)
        ax.set_ylabel("Z", fontdict=font)
        ax.set_zlabel("Y", fontdict=font)
        ax.plot(self.x,self.z,self.y,label='trajectories of knuckleballs without spin in 3D')
        ax.legend(loc='upper right')
        plt.show()


# In[8]:


a =knuckleballs_2()
a.run()
a.show()


# <a>显然，相对于有自旋$\omega$时候的情况，没有自旋$\omega$时候的$y$方向偏转较大（击球点到本垒约20$m$）。</a>
