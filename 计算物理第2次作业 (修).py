
# coding: utf-8

# In[15]:


#1. Consider a radioactive decay problem involving two types of nuclei,  A  and  B , with populations  NA(t)  and  NB(t) . 
#Suppose that type  A  nuclei decay to form type  B  nuclei, which then also decay, according to the differential equations
dNA/dt=−NA/τA
 
dNB/dt=NA/τA−NB/τB
 
#where  τA  and  τB  are the decay time constant for each type of nucleus. 
#Use the Euler method to solve these coupled equations for  NA  and  NB  as functions of time.
#This problem can also be solved exactly, as was the case with our original nuclear decay problem. 
#Obtain the analytic solutions for  NA(t)  and  NB(t) , and compare them with your numerical results. 
#It is also interesting to explore the behavior found for different values of the ratio  τA/τB . 
#In particular, try to interpret the short and long time behaviors for different values of this ratio.


# In[73]:


#2. 尝试改进第一章正文中的例子，在泰勒展开式中保留二阶项进行计算，讨论与只保留一阶项的 Euler 法在误差上的差别。


# In[2]:


import pylab as pl
class particles_AandB_decay:
    
    def __init__(self, number_of_nuclei_A = 100, number_of_nuclei_B = 0, 
                 time_constant_A = 1,time_constant_B = 2, time_of_duration = 5, time_step = 0.05):
        self.n_a = [number_of_nuclei_A]
        self.n_b = [number_of_nuclei_B]
        self.t = [0]
        self.tauA = time_constant_A
        self.tauB = time_constant_B
        self.dt = time_step
        self.time = time_of_duration
        self.nsteps = int(time_of_duration // time_step + 1)
        print("Initial number of nuclei A ->", number_of_nuclei_A)
        print("Initial number of nuclei B->", number_of_nuclei_B)
        print("Time constant A ->", time_constant_A)
        print("Time constant B ->", time_constant_B)
        print("time step -> ", time_step)
        print("total time -> ", time_of_duration)
    def calculate(self):
        for i in range(self.nsteps):
            tmp1 = self.n_a[i] - self.n_a[i] / self.tauA * self.dt
            tmp2 = self.n_b[i] - self.n_b[i] / self.tauB * self.dt + self.n_a[i] / self.tauA* self.dt
            self.n_a.append(tmp1)
            self.n_b.append(tmp2)
            self.t.append(self.t[i] + self.dt)
    def show_results(self):
        pl.plot(self.t, self.n_a, color="blue", label="NA Numerical ",)
        pl.plot(self.t, self.n_b, color="red", label="NB Numerical",)
        pl.legend(loc='upper right')
        pl.xlabel('time ($s$)')
        pl.ylabel('Number of Nuclei')
        pl.show()
         
    def store_results(self):
        myfile = open('particles_AandB_decay.txt', 'w')
        for i in range(len(self.t)):
            print(self.t[i], self.n_a[i], self.n_a[i], file = myfile)
        myfile.close()


# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')
a = particles_AandB_decay()
a.calculate()
a.show_results()
a.store_results()


# In[4]:


from math import *
class exact_results_check(particles_AandB_decay):
    def show_results(self):
        self.et = []
        for i in range(len(self.t)):
            temp = self.n_a[0] * exp(- self.t[i] / self.tauA)
            self.et.append(temp)
        pl.plot(self.t, self.et, label="NA Analytical")
        pl.plot(self.t, self.n_a, 'o', label = "NA Numerical")
        pl.legend(loc='upper right')
        pl.xlabel('time ($s$)')
        pl.ylabel('Number of Nuclei')
        pl.xlim(0, self.time)
        pl.show()


# In[5]:


b = exact_results_check(number_of_nuclei_A=100, time_constant_A=1, time_step=0.2)
b.calculate()
b.show_results()


# In[8]:


from math import *
class exact_results_check_b(particles_AandB_decay):
    def show_results(self):
        self.et_a = []
        self.et_b = []
        for i in range(len(self.t)):
            temp1 = self.n_a[0] * exp(- self.t[i] / self.tauA)
            self.et_a.append(temp1)
            temp2 = self.n_a[0] / self.tauA * self.tauA * self.tauB  / (self.tauA - self.tauB ) * (exp(- self.t[i]/self.tauA ) - exp(- self.t[i]/self.tauB))
            self.et_b.append(temp2)
        pl.plot(self.t, self.et_a, label="NA Analytical")
        pl.plot(self.t, self.n_a, '*', label = "NA Numerical")
        pl.plot(self.t, self.et_b, label="NB Analytical")
        pl.plot(self.t, self.n_b, 'o', label = "NB Numerical")
        pl.legend(loc='upper right')
        pl.xlabel('time ($s$)')
        pl.ylabel('Number of Nuclei')
        pl.xlim(0, self.time)
        pl.show()


# In[9]:


c = exact_results_check_b(time_constant_B=2, time_step=0.2)
c.calculate()
c.show_results()


# In[12]:


import pylab as pl
class uranium_decay:
    def __init__(self, number_of_nuclei = 100, 
                 time_constant = 1, time_of_duration = 5, time_step = 0.05):
        # unit of time is second
        self.n_uranium_first = [number_of_nuclei]
        self.n_uranium_second = [number_of_nuclei]
        self.t = [0]
        self.tau = time_constant
        self.dt = time_step
        self.time = time_of_duration
        self.nsteps = int(time_of_duration // time_step + 1)
        print("Initial number of nuclei ->", number_of_nuclei)
        print("Time constant ->", time_constant)
        print("time step -> ", time_step)
        print("total time -> ", time_of_duration)
    def calculate(self):
        for i in range(self.nsteps):
            tmp_first = self.n_uranium_first[i] - self.n_uranium_first[i] / self.tau * self.dt
            tmp_second = self.n_uranium_second[i] - self.n_uranium_second[i] / self.tau * self.dt + self.n_uranium_second[i]/ self.tau** 2 * 1/2 * self.dt ** 2
            self.n_uranium_first.append(tmp_first)
            self.n_uranium_second.append(tmp_second)
            self.t.append(self.t[i] + self.dt)
    def show_results(self):
        pl.plot(self.t, self.n_uranium_first , '*', color = "red", label ="First-Ordered")
        pl.plot(self.t, self.n_uranium_second, color="green",label ="Second-Ordered")
        pl.legend(loc='upper right')
        pl.xlabel('time ($s$)')
        pl.ylabel('Number of Nuclei')
        pl.show()
    def store_results(self):
        myfile = open('nuclei_decay_data.txt', 'w')
        for i in range(len(self.t)):
            print(self.t[i], self.n_uranium_first[i], file = myfile)
        myfile.close()


# In[13]:


get_ipython().run_line_magic('matplotlib', 'inline')
a = uranium_decay()
a.calculate()
a.show_results()
a.store_results()

