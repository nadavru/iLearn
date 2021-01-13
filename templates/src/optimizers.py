import numpy as np
from abc import ABCMeta, abstractmethod
import math
from math import exp, pi
from numpy.linalg import inv

class opt:
    def __init__(self, x, y, lr, momentum): 
      self.x=x
      self.y=y
      self.lr=lr
      self.momentum=momentum

    @abstractmethod
    def step(self, d:list, d2):
      pass

class SGD(opt):
  def __init__(self, x, y, lr, momentum=0): 
      super().__init__(x, y, lr, momentum)
  def step(self, d, d2=None):
    self.x = self.x-self.lr*d[0]
    self.y = self.y-self.lr*d[1]
    return (self.x, self.y)

class MomentumSGD(opt):
  def __init__(self, x, y, lr, momentum=0.9): 
      super().__init__(x, y, lr, momentum)
      self.dx, self.dy = 0, 0
  def step(self, d, d2=None):
    self.dx = self.momentum*self.dx - self.lr*d[0]
    self.dy = self.momentum*self.dy - self.lr*d[1]
    self.x = self.x + self.dx
    self.y = self.y + self.dy
    return (self.x, self.y)

class Newton(opt):
  def __init__(self, x, y, lr, momentum=0): 
      super().__init__(x, y, lr, momentum)
  def step(self, d1, d2=None):
    d1 = np.array(d1)
    d2 = np.array(d2)
    d2 = inv(d2)
    d = -np.matmul(d2, d1)
    self.x = self.x + self.lr*d[0]
    self.y = self.y + self.lr*d[1]
    return (self.x, self.y)
