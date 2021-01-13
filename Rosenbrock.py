from learning import Trainer
from math import exp
from optimizers import *


class Rosenbrock(Trainer):
    def __init__(self, a=1, b=100, grid_x=3, grid_y=3, lr=0.0001, epochs=50, opt=SGD, x=None, y=None): 
        super().__init__(grid_x, grid_y, lr, epochs, opt, x, y)

        url = "https://i-learn-ml.oa.r.appspot.com/viewer/viewerGD.html#"
        self.disp_url = url+f"xmax={grid_x}&xmin=-{grid_x}&ymax={grid_y}&ymin=-{grid_y}&"

        self.a, self.b = a, b
        self.f_str = f"({a}-x)^2+{b}*(y-x^2)^2"
        
        
    def calc(self, x, y): 
        return (self.a-x)**2+self.b*(y-x**2)**2

    def derive(self, x, y): 
        dx = 2*x*(-2*self.b*(y-x**2)-1)
        dy = 2*self.b*(y-x**2)
        return [dx, dy]
    
    def derive2(self, x, y): 
        dxx = 12*self.b*x**2-4*self.b*y+2
        dxy = -4*self.b*x
        dyx = -4*self.b*x
        dyy = 2*self.b
        return [[dxx, dxy], [dyx, dyy]]
    
    def display(self): 
        points_str = ""
        for i in range(len(self.Xs)):
          points_str+=(str)(self.Xs[i])+","+(str)(self.Ys[i])+","+(str)(self.Zs[i])+"|"
        points_str=points_str[:-1]

        src= f"func={self.f_str}&points={points_str}"
        return src

def runRosenbrock():
    Ro = Rosenbrock(lr=0.1**5, opt=SGD, grid_x=10, grid_y=10, epochs=100, x=5, y=-5)
    Xs, Ys, Zs = Ro.train()
    src = Ro.display()
    return src
    
