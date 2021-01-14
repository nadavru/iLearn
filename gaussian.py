from learning import Trainer
from math import exp
from optimizers import *

e = exp(1)

class Gaussian_helper: 
    def __init__(self, a, px, py, c=1): 
        self.a=a
        self.px=px
        self.py=py
        self.c=c

    def calc(self, x, y): 
        return -self.a*e**(-((x-self.px)**2+(y-self.py)**2)/(2*self.c**2))

    def derive(self, x, y): 
        tmp = (self.a/self.c**2)*e**(-((x-self.px)**2+(y-self.py)**2)/(2*self.c**2))
        return (tmp*(x-self.px), tmp*(y-self.py))

class Gaussian(Trainer):

    def __init__(self, apc:list, grid_x, grid_y, lr=0.5, epochs=50, opt=SGD, x=None, y=None): 
        super().__init__(grid_x, grid_y, lr, epochs, opt, x, y)
        self.g_list = [Gaussian_helper(*i) for i in apc]

        self.disp_url = f"xmax={grid_x}$xmin=-{grid_x}$ymax={grid_y}$ymin=-{grid_y}$"

        self.f_str = ""
        for a, px, py, c in apc:
            s_x = "-"+(str)(px) if px>0 else "+"+(str)(-px)
            s_y = "-"+(str)(py) if py>0 else "+"+(str)(-py)
            self.f_str += f"-{a}*e^(-((x{s_x})^2+(y{s_y})^2)/(2*{c}^2))"
        
    def calc(self, x, y): 
        return sum([G.calc(x, y) for G in self.g_list])

    def derive(self, x, y): 
        d_list = [G.derive(x, y) for G in self.g_list]
        return [sum(i) for i in zip(*d_list)]
    
    def display(self): 
        points_str = ""
        for i in range(len(self.Xs)):
          points_str+=(str)(self.Xs[i])+","+(str)(self.Ys[i])+","+(str)(self.Zs[i])+"|"
        points_str=points_str[:-1]

        src=self.disp_url+f"func={self.f_str}$points={points_str}"
        return src

def runGaussian(x=None,y=None,lr=0.5,opt=SGD):
    sizes = []
    for ix in [-15,0,15]:
        for iy in [-15,0,15]:
            sizes.append((50,ix,iy,5))
    grid_x = 30
    grid_y = 30
    G = Gaussian(sizes, grid_x, grid_y,x=x,y=y,lr=lr,opt=opt)
    Xs, Ys, Zs = G.train()
    src = G.display()
    return src

