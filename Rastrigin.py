import math
from learning import Trainer
from math import exp
from optimizers import *
pi = math.pi

class Rastrigin(Trainer):

    def __init__(self, A=10, grid_x=6, grid_y=6, lr=0.0001, epochs=50, opt=SGD, x=None, y=None): 
        super().__init__(grid_x, grid_y, lr, epochs, opt, x, y)

        url = "https://i-learn-ml.oa.r.appspot.com/viewer/viewerGD.html#"
        self.disp_url = f"xmax={grid_x}$xmin=-{grid_x}$ymax={grid_y}$ymin=-{grid_y}$"

        self.A = A
        self.f_str = f"{2*A}+x^2-{A}*cos({2*pi}*x)+y^2-{A}*cos({2*pi}*y)"
        
        
    def calc(self, x, y): 
        return 2*self.A+x**2-self.A*math.cos(2*pi*x)+y**2-self.A*math.cos(2*pi*y)

    def derive(self, x, y): 
        dx = 2*(pi*self.A*math.sin(2*pi*x)+x)
        dy = 2*(pi*self.A*math.sin(2*pi*y)+y)
        return [dx, dy]
    
    def display(self): 
        points_str = ""
        for i in range(len(self.Xs)):
          points_str+=(str)(self.Xs[i])+","+(str)(self.Ys[i])+","+(str)(self.Zs[i])+"|"
        points_str=points_str[:-1]

        src=self.disp_url+f"func={self.f_str}$points={points_str}"
        return src

def runRastrigin(lr=0.0001,epochs=50, opt=SGD, x=None, y=None):
    Ra = Rastrigin(lr=lr,epochs=epochs,opt=opt,x=x,y=y)
    Xs, Ys, Zs = Ra.train()
    src = Ra.display()
    return src