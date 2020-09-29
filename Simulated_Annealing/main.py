from functions import func
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

f=open('input.txt','r')
x=f.readline().split(",")
y=f.readline().split(",")
def Brute_force():
    max=10000
    for a in range(int(x[0]),int(x[1])):
        for b in range(int(y[0]),int(y[1])):
            if (func(a,b)<max):
                max=func(a,b)
                ans_a=a
                ans_b=b   
    return ans_a,ans_b,round(func(ans_a,ans_b),4)
def Simulated_Annealing(a_search_space,b_search_space, func, t):
    scale=np.sqrt(t)
    a_init=np.random.choice(a_search_space)
    b_init=np.random.choice(b_search_space)
    a=a_init
    b=b_init
    cur=func(a,b)
    a_history=[a]
    b_history=[b]
    iterations=2000
    for i in range(iterations):
        prop_a=int(a+np.random.normal()*scale)
        prop_b=int(b+np.random.normal()*scale)
        if (prop_a >=-60 and prop_a<=60 and prop_b>=-30 and prop_b<=70):
            new_out=func(prop_a,prop_b)
            diff=new_out-cur
            if diff<0 or np.exp(-diff/(k*t))>np.random.rand():
                #time.sleep(1)
                a=prop_a
                b=prop_b
                cur=func(a,b)
                a_history.append(a)
                b_history.append(b)
        t=0.9*t


    return a,b,round(func(a,b),4),len(a_history)
    


#print(Brute_force())
a_area=np.linspace(int(x[0]),int(x[1]),int(x[1])-int(x[0])+1)
b_area=np.linspace(int(y[0]),int(y[1]),int(y[1])-int(y[0])+1)
k=1
print(Simulated_Annealing(a_area,b_area,func,100))



