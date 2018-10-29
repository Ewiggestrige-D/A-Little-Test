
# coding: utf-8

# **3.11** For the three values of $F_D$ shown in Figure 3.6, compute and plot the total energy of the system as a function of the time and discuss the results.

# **3.13** Write a program to calculate and compare the behavior of two, nearly identical pendulums. Use it to calculate the divergence of two nearby trajectories in the chaotic regime, as in Figure 3.7, and make a qualitative estimate of the corresponding Lyapunov exponent from the slope of a plot of $\log(\Delta\theta)$ as a function of $t$.

# In[173]:


#第一题
import numpy as np
import math
import matplotlib.pyplot as plt
class pendulum:
    """
    Simulation of the physical pendulum
    Program by Rongyu Dong
    """
    def __init__(self,initial_angularvelocity=0,initial_theta=0.2,g=9.8,l=9.8,q=1/2,omega=2/3,fd=0,mass=1,time_step=0.004):
        self.theta=[initial_theta]
        self.angular_velocity=[initial_angularvelocity]
        self.g=g
        self.l=l
        self.gl=g/l
        self.omega=omega
        self.fd=fd
        self.q=q
        self.mass=mass
        self.energy=[(self.mass*(self.l**2)*(self.angular_velocity[0]**2))/2+self.mass*self.g*self.l*(1-math.cos(self.theta[0]))]
        self.dt=time_step
        self.t=[0]
        print("initial angular velocity ->",initial_angularvelocity)
        print("initial angle->",initial_theta)
        print("omega->",omega)
        print("q->",q)
        print("fd->",fd)
        print("time step->",time_step)
    def calculate(self):
        i=0
        while(self.t[i]<100):
            self.angular_velocity.append(self.angular_velocity[i]+self.dt*(-self.gl*math.sin(self.theta[i])
                                        -self.q*self.angular_velocity[i]+self.fd*math.sin(self.omega*self.t[i])))
            self.theta.append(self.theta[i]+self.angular_velocity[i+1]*self.dt)
            self.energy.append((self.mass*(self.l**2)*(self.angular_velocity[i+1]**2))/2+self.mass*self.g*self.l*(1-math.cos(self.theta[i+1])))
            self.t.append(self.t[i]+self.dt)
            i=i+1
    def show_results(self,fd=0):
        font={'family': 'serif',
              'color': 'darkred',
              'weight': 'normal',
              'size': 16,
             }
        plt.plot(self.t,self.angular_velocity)
        plt.title('angular_velocity versus time',fontdict = font)
        plt.xlabel('t ($s$)')
        plt.ylabel('angular_velocity ($ω/s$)')
        
a =pendulum()
a.calculate()
a.show_results('fd=0')
b =pendulum(fd=0.5)
b.calculate()
b.show_results('fd=0.5')
c =pendulum(fd=1.2)
c.calculate()
c.show_results('fd=1.2')
plt.legend(['fd=0','fd=0.7','fd=1.5'],loc='upper left')
plt.show()  


# <a>由上图可知当外力$F$=0时， 系统的角速度$\omega$随时间保持一个稳恒态；  
# 当外力$F$=0.5时，系统的角速度$\omega$随时间做简谐周期运动，系统的轨迹是存在解析解；  
# 当外力$F$=1.2时，系统的角速度$\omega$随时间做无规则运动，形成一个混沌；   
# 经多次反复测试，当外力$F$=1.1时，系统的角速度$\omega$开始有无规则运动的倾向。</a>

# In[174]:


#第一题
import numpy as np
import math
import matplotlib.pyplot as plt
class pendulum:
    """
    Simulation of the physical pendulum
    Program by Rongyu Dong
    """
    def __init__(self,initial_angularvelocity=0,initial_theta=0.2,g=9.8,l=9.8,q=1/2,omega=2/3,fd=0,mass=1,time_step=0.004):
        self.theta=[initial_theta]
        self.angular_velocity=[initial_angularvelocity]
        self.g=g
        self.l=l
        self.gl=g/l
        self.omega=omega
        self.fd=fd
        self.q=q
        self.mass=mass
        self.energy=[(self.mass*(self.l**2)*(self.angular_velocity[0]**2))/2+self.mass*self.g*self.l*(1-math.cos(self.theta[0]))]
        self.dt=time_step
        self.t=[0]
        print("initial angular velocity ->",initial_angularvelocity)
        print("initial angle->",initial_theta)
        print("omega->",omega)
        print("q->",q)
        print("fd->",fd)
        print("time step->",time_step)
    def calculate(self):
        i=0
        while(self.t[i]<100):
            self.angular_velocity.append(self.angular_velocity[i]+self.dt*(-self.gl*math.sin(self.theta[i])
                                        -self.q*self.angular_velocity[i]+self.fd*math.sin(self.omega*self.t[i])))
            self.theta.append(self.theta[i]+self.angular_velocity[i+1]*self.dt)
            self.energy.append((self.mass*(self.l**2)*(self.angular_velocity[i+1]**2))/2+self.mass*self.g*self.l*(1-math.cos(self.theta[i+1])))
            self.t.append(self.t[i]+self.dt)
            i=i+1
    def show_results(self,fd=0):
        font={'family': 'serif',
              'color': 'darkred',
              'weight': 'normal',
              'size': 16,
             }
        plt.plot(self.t,self.theta)
        plt.title('angle of rotation versus time',fontdict = font)
        plt.xlabel('t ($s$)')
        plt.ylabel('angle')
    
a =pendulum()
a.calculate()
a.show_results('fd=0')
b =pendulum(fd=0.5)
b.calculate()
b.show_results('fd=0.5')
c =pendulum(fd=1.2)
c.calculate()
c.show_results('fd=1.2')
plt.legend(['fd=0','fd=0.5','fd=1.2'],loc='lower right')
plt.show()


# <a>由上图可知当外力$F$=0时， 系统的初始角度$\theta$随时间保持一个稳恒态；  
# 当外力$F$=0.5时，系统的初始角度$\theta$随时间做简谐周期运动，系统的轨迹是存在解析解；  
# 当外力$F$=1.2时，系统的初始角度$\theta$随时间做无规则运动，形成一个混沌；   
# 经多次反复测试，当外力$F$=1.095时，系统的初始角度$\theta$开始有无规则运动的倾向。</a>

# In[175]:


#第一题
import numpy as np
import math
import matplotlib.pyplot as plt
class pendulum:
    """
    Simulation of the physical pendulum
    Program by Rongyu Dong
    """
    def __init__(self,initial_angularvelocity=0,initial_theta=0.2,g=9.8,l=9.8,q=1/2,omega=2/3,fd=0,mass=1,time_step=0.004):
        self.theta=[initial_theta]
        self.angular_velocity=[initial_angularvelocity]
        self.g=g
        self.l=l
        self.gl=g/l
        self.omega=omega
        self.fd=fd
        self.q=q
        self.mass=mass
        self.energy=[(self.mass*(self.l**2)*(self.angular_velocity[0]**2))/2+self.mass*self.g*self.l*(1-math.cos(self.theta[0]))]
        self.dt=time_step
        self.t=[0]
        print("initial angular velocity ->",initial_angularvelocity)
        print("initial angle->",initial_theta)
        print("omega->",omega)
        print("q->",q)
        print("fd->",fd)
        print("time step->",time_step)
    def calculate(self):
        i=0
        while(self.t[i]<100):
            self.angular_velocity.append(self.angular_velocity[i]+self.dt*(-self.gl*math.sin(self.theta[i])
                                        -self.q*self.angular_velocity[i]+self.fd*math.sin(self.omega*self.t[i])))
            self.theta.append(self.theta[i]+self.angular_velocity[i+1]*self.dt)
            self.energy.append((self.mass*(self.l**2)*(self.angular_velocity[i+1]**2))/2+self.mass*self.g*self.l*(1-math.cos(self.theta[i+1])))
            self.t.append(self.t[i]+self.dt)
            i=i+1
    def show_results(self,fd=0):
        font={'family': 'serif',
              'color': 'darkred',
              'weight': 'normal',
              'size': 16,
             }
        plt.plot(self.t,self.energy)
        plt.title('energy of the pendulum system versus time',fontdict = font)
        plt.xlabel('t ($s$)')
        plt.ylabel('energy $J$')
    
a =pendulum()
a.calculate()
a.show_results('fd=0')
b =pendulum(fd=0.5)
b.calculate()
b.show_results('fd=0.5')
c =pendulum(fd=1.2)
c.calculate()
c.show_results('fd=1.2')
plt.legend(['fd=0','fd=0.5','fd=1.2'],loc='lower right')
plt.show()


# <a>由上图可知当外力$F$=0时， 系统的总能量$E$保持不变；  
# 当外力$F$=0.5时，系统的总能量$E$随时间做简谐周期运动，系统的轨迹是存在解析解；  
# 当外力$F$=1.2时，系统的总能量$E$随时间做无规则运动，形成一个混沌；   
# 经多次反复测试，当外力$F$<1.09时，       
# 系统的总能量$E$在一个长的时间段上仍保持能量的守恒，波形为双峰。   
# 这与上两幅图的结论保持一致</a>

# In[176]:


#第二题
import numpy as np
import math
import matplotlib.pyplot as plt
class pendulum_comparison:
    """
    Simulation of the physical pendulum
    Program by Rongyu Dong
    """
    def __init__(self,initial_angular_velocity_1 = 0,initial_angular_velocity_2 = 0,initial_theta_1 = 0.2,initial_theta_2 = 0.201,
                 g = 9.8,l = 9.8,q = 1/2,omega = 2/3,fd = 0.5,mass = 1,time_step = 0.004):
        self.theta_1 = [initial_theta_1]
        self.theta_2 = [initial_theta_2]
        self.angular_velocity_1 = [initial_angular_velocity_1]
        self.angular_velocity_2 = [initial_angular_velocity_2]
        self.g = g
        self.l = l
        self.gl = g/l
        self.omega = omega
        self.fd = fd
        self.q = q
        self.mass = mass
        self.Lyapunov_exponent = [self.angular_velocity_2[0]-self.angular_velocity_1[0]]
        self.dt = time_step
        self.t = [0]
        print("initial angular velocity 1->",initial_angular_velocity_1)
        print("initial angular velocity 2->",initial_angular_velocity_2)
        print("initial angle 1->",initial_theta_1)
        print("initial angle 2->",initial_theta_2)
        print("omega->",omega)
        print("q->",q)
        print("fd->",fd)
        print("time step->",time_step)
    def calculate(self):
        i = 0
        while(self.t[i]<300):
            self.angular_velocity_1.append(self.angular_velocity_1[i] + self.dt * (-self.gl * math.sin(self.theta_1[i])
                                        -self.q * self.angular_velocity_1[i] + self.fd * math.sin(self.omega * self.t[i])))
            self.theta_1.append(self.theta_1[i]+self.angular_velocity_1[i+1]*self.dt)
            self.angular_velocity_2.append(self.angular_velocity_2[i]+self.dt*(-self.gl*math.sin(self.theta_2[i])
                                        -self.q*self.angular_velocity_2[i]+self.fd*math.sin(self.omega*self.t[i])))
            self.theta_2.append(self.theta_2[i]+self.angular_velocity_2[i+1]*self.dt)
            self.Lyapunov_exponent.append(math.log(abs(self.theta_2[i]-self.theta_1[i])))
            self.t.append(self.t[i]+self.dt)
            i=i+1
    def show_results(self,fd=0):
        font={'family': 'serif',
              'color': 'darkred',
              'weight': 'normal',
              'size': 16,
             }
        plt.plot(self.t,self.Lyapunov_exponent)
        plt.title('$Δθ$ comparison between two nearly identical pendulum',fontdict = font)
        plt.xlabel('t ($s$)')
        plt.ylabel('$log(Δθ)$ ')
        


# In[177]:


b =pendulum_comparison(fd=0.7)
b.calculate()
b.show_results('fd=0.7')
c =pendulum_comparison(fd=1.5)
c.calculate()
c.show_results('fd=1.5')
plt.legend(['fd=0.7','fd=1.5'],loc='upper left')
plt.show()


# <a>由上图可知当外力$F$=0.7时，$Δθ$的Lyapunov exponent取对数后呈现出明显的线性关系；  
# 是一条斜率为负的直线
# 当外力$F$=1.5时，$Δθ$的Lyapunov exponent取对数后和线性函数的相关性很弱；  
# 当t>85时，$Δθ$的Lyapunov exponent取对数后基本稳定在2.5-3之间，并没有发散或收敛至0。
# </a>
