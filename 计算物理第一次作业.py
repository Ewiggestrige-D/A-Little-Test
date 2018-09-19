
# coding: utf-8

# In[ ]:


#第一题 创建一个函数addNumbers(n)，它接受一个正数作为参数，并返回0和该数字（包括）之间所有数字的总和。


# In[1]:


def addNumber(n):
    if n >=0:
        return sum(range(0,n+1))
    else:
        pass

addNumber(100)
# In[2]:


addNumber(100)


# In[ ]:


#第二题 根据指定的迭代次数创建一个计算  π  近似值的函数piApprox(n)。


pi can be computed by 4*(1-1/3+1/5-1/7+1/9- ...).


# In[3]:


def pi(n): 
    s=0 
    for i in range(n): 
        if i%2==1: 
            s=s-1.0/(1+2*i) 
        else: 
            s=s+1.0/(1+2*i) 
    else: 
        print (s*4)


# In[ ]:


#第三题 根据印度数学家斯里尼瓦瑟·拉马努金(Srinivasa Ramanujan)发现的公式，编写一个函数estimatePi()来估计并返回  π  的值。 它应该使用while循环来计算求和项，直到最后一项小于  10−15 。 计算公式如下：
1
#π=22⎯⎯√9801∑k=0∞(4k)!(1103+26390k)(k!)43964k
 
这里  k!  是  k  的阶乘。

提示： 阶乘可以使用递归进行计算


# In[2]:


def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n-1)


# In[4]:


import math

def estimatePi(n):
    k=n
    pi=0
    while p>10**(-15):
        p=(factorial(4*k)*(1103+26390*k))/(pow(factorial(k),4)*pow(396,4*k))
        k=n+1
        pi=pi+s
    else:
        a=(math.sqrt(8)*pi)/9801
        print('pi=',1/a)
        #失败代码 
        #注意循环结构 和语句位置


# In[13]:


import math
def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n-1)
    
def estimatePi():
    k = 0
    p = 0
    while True:
        s = (factorial(4*k)*(1103+26390*k))/(pow(factorial(k),4)*pow(396,4*k))
        p = p+s
        k = k+1
        if s < 1e-15 :
            break
    return p
pi = 9801/(math.sqrt(8)*estimatePi())
print('pi = ',pi)

