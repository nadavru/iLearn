import numpy as np
#from abc import ABCMeta, abstractmethod
import math
from math import exp, pi
from numpy.linalg import inv
from optimizers import *


class Trainer:
    def __init__(self, grid_x, grid_y, lr=0.5, epochs=50, opt=SGD, x=None, y=None): 
        self.lr=lr
        self.x_cur = 2*grid_x*np.random.random_sample()-grid_x if x is None else x
        self.y_cur = 2*grid_y*np.random.random_sample()-grid_y if y is None else y
        self.epochs=epochs

        x = np.linspace(-grid_x, grid_x, 1000)
        y = np.linspace(-grid_y, grid_y, 1000)
        self.X, self.Y = np.meshgrid(x, y)
        self.opt = opt(self.x_cur, self.y_cur, self.lr)

    def __call__(self): 
        return self.X, self.Y, np.where(True, self.calc(self.X, self.Y), 0)
    
    def train(self): 
        self.Xs = [self.x_cur]
        self.Ys = [self.y_cur]
        self.Zs = [self.calc(self.x_cur, self.y_cur)]

        for _ in range(self.epochs):
            d = self.derive(self.x_cur, self.y_cur)
            d2 = self.derive2(self.x_cur, self.y_cur)
            #print(d)
            self.x_cur, self.y_cur = self.opt.step(d, d2)

            self.Xs.append(self.x_cur)
            self.Ys.append(self.y_cur)
            self.Zs.append(self.calc(self.x_cur, self.y_cur))
        return self.Xs, self.Ys, self.Zs

 #   @abstractmethod
    def display(self): 
      pass
    
    def derive2(self, x, y): 
      return None
    
  #  @abstractmethod
    def calc(self, x, y): 
      pass

   # @abstractmethod
    def derive(self, x, y): 
      pass







class SVM(Trainer):

  def __init__(self, group_size, p0, p1, grid_x=50, grid_y=50, lr=0.5, epochs=10, opt=SGD): 
        super().__init__(grid_x, grid_y, lr, epochs, opt)
        (x0, y0, var0) = p0
        (x1, y1, var1) = p1
        self.group_size = group_size
        group0 = np.array(list(zip(list(np.random.normal(x0, var0, group_size)), list(np.random.normal(y0, var0, group_size)),[-1]*group_size)))
        group1 = np.array(list(zip(list(np.random.normal(x1, var1, group_size)), list(np.random.normal(y1, var0, group_size)),[1]*group_size)))

        self.data = np.concatenate((group0, group1))

        url = "https://i-learn-ml.oa.r.appspot.com/viewer/viewerSVM.html#"
        #grid = 30
        self.disp_url = url+f"xmax={grid_x}&xmin=-{grid_x}&ymax={grid_y}&ymin=-{grid_y}&"
        
        url2 = "https://i-learn-ml.oa.r.appspot.com/viewer/viewerGD.html#"
        self.disp_url2 = url2+f"xmax={grid_x}&xmin=-{grid_x}&ymax={grid_y}&ymin=-{grid_y}&"

        #self.X, self.Y = np.concatenate((self.group0[:,0],self.group1[:,0])), np.concatenate((self.group0[:,1],self.group1[:,1]))

  def calc(self, x, y): 
    '''act0 = np.tanh(x*self.group0[:,0]+y*self.group0[:,1])+1
    act1 = 1-np.tanh(x*self.group1[:,0]+y*self.group1[:,1])

    loss = (np.sum(act0)+np.sum(act1))/(2*self.group_size)'''
    pred = np.sign(np.tanh(x*self.data[:,0]+y*self.data[:,1]))
    #print(pred, self.data[:,2])
    acc = np.sum(np.equal(pred, self.data[:,2]))/(2*self.group_size)
    #print(acc)
    #loss = np.mean(1-np.tanh(self.data[:,2]*(x*self.data[:,0]+y*self.data[:,1])))
    '''self.scores = (x*self.data[:,0]+y*self.data[:,1])
    self.keep = np.exp(-self.data[:,2]*self.scores)/(1+np.exp(-self.data[:,2]*self.scores))
    loss = np.mean(self.keep)'''
    loss = np.mean(-self.data[:,2]*(x*self.data[:,0]+y*self.data[:,1]))
    return loss
  
  def pred(self):
    pred_y = np.sign(np.tanh(self.x_cur*self.data[:,0]+self.y_cur*self.data[:,1]))
    acc = np.sum(np.equal(pred_y, self.data[:,2]))/(2*self.group_size)
    return acc
  
  def derive(self, x, y): 
    '''der0 = 1/(2*self.group_size*np.cosh(x*self.group0[:,0]+y*self.group0[:,1])**2)
    der1 = -1/(2*self.group_size*np.cosh(x*self.group1[:,0]+y*self.group1[:,1])**2)
    dx = np.sum(self.group0[:,0]*der0) + np.sum(self.group1[:,0]*der1)
    dy = np.sum(self.group0[:,1]*der0) + np.sum(self.group1[:,1]*der1)'''
    
    '''der = -1/np.cosh(self.data[:,2]*(x*self.data[:,0]+y*self.data[:,1]))**2
    dx = np.mean(self.data[:,2]*self.data[:,0]*der)
    dy = np.mean(self.data[:,2]*self.data[:,1]*der)'''
    
    '''der = -self.keep*(1-self.keep)
    dx = np.mean(self.data[:,2]*self.data[:,0]*der)
    dy = np.mean(self.data[:,2]*self.data[:,1]*der)'''

    dx = np.mean(-self.data[:,2]*self.data[:,0])
    dy = np.mean(-self.data[:,2]*self.data[:,1])

    return [dx, dy]

  def display(self): 
    points_str = ""
    x = 30
    for i in range(len(self.Xs)):
      points_str+=(str)(-x)+","+(str)(x*self.Xs[i]/self.Ys[i])+","+(str)(0)+"|"+(str)(x)+","+(str)(-x*self.Xs[i]/self.Ys[i])+","+(str)(0)+"|"
    points_str=points_str[:-1]

    points_red = ""
    for (x,y) in self.data[:self.group_size,:2]:
      points_red += f"{x},{y},0"+"|"
    points_red=points_red[:-1]
    
    points_blue = ""
    for (x,y) in self.data[self.group_size:,:2]:
      points_blue += f"{x},{y},0"+"|"
    points_blue=points_blue[:-1]

    src=self.disp_url+f"points_red={points_red}&points_blue={points_blue}&vec={points_str}"
    return src
  
  def display2(self): 
    points_str = ""
    for i in range(len(self.Xs)):
      points_str+=(str)(self.Xs[i])+","+(str)(self.Ys[i])+","+(str)(self.Zs[i])+"|"
    points_str=points_str[:-1]

    x_sum = np.mean(-self.data[:,2]*self.data[:,0])
    y_sum = np.mean(-self.data[:,2]*self.data[:,1])
    
    f_str = f"{x_sum}*x+{y_sum}*y".replace("+-","-")

    src=self.disp_url2+f"func={f_str}&points={points_str}"
    return src



    