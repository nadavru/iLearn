from learning import Trainer
from math import exp
from optimizers import *

class Himmelblau(Trainer):

    def __init__(self, grid_x=6, grid_y=6, lr=0.001, epochs=50, opt=SGD, x=None, y=None): 
        super().__init__(grid_x, grid_y, lr, epochs, opt, x, y)

        url = "https://i-learn-ml.oa.r.appspot.com/viewer/viewerGD.html#"
        self.disp_url = url+f"xmax={grid_x}&xmin=-{grid_x}&ymax={grid_y}&ymin=-{grid_y}&"

        self.f_str = f"(x^2+y-11)^2+(x+y^2-7)^2"
        
        self.f_str_arr = []
        
        
    def calc(self, x, y): 
        return (x**2+y-11)**2+(x+y**2-7)**2

    def derive(self, x, y): 
        dx = 2*(2*x*(x**2+y-11)+x+y**2-7)
        dy = 2*(2*y*(x+y**2-7)+x**2+y-11)

        self.f_str_arr.append(self.display2())
        return [dx, dy]
    
    def derive2(self, x, y): 
        dxx = 12*x**2+4*y-42
        dxy = 4*(x+y)
        dyx = 4*(x+y)
        dyy = 12*y**2+4*x-26
        return [[dxx, dxy], [dyx, dyy]]
    
    def display(self): 
        points_str = ""
        for i in range(len(self.Xs)):
          points_str+=(str)(self.Xs[i])+","+(str)(self.Ys[i])+","+(str)(self.Zs[i])+"|"
        points_str=points_str[:-1]

        src=self.disp_url+f"func={self.f_str}&points={points_str}"
        return src
    
    def display2(self): 
        dx = "2*(2*x*(x^2+y-11)+x+y^2-7)"
        dy = "2*(2*y*(x+y^2-7)+x^2+y-11)"
        dxx = "(12*x^2+4*y-42)"
        dxy = "4*(x+y)"
        dyx = "4*(x+y)"
        dyy = "(12*y^2+4*x-26)"
        f_str = f"(x^2+y-11)^2+(x+y^2-7)^2"
        Hx = f"((x-{self.x_cur})*{dxx}+(y-{self.y_cur})*{dyx})"
        Hy = f"((x-{self.x_cur})*{dxy}+(y-{self.y_cur})*{dyy})"
        A = f"0.5*(x-{self.x_cur})*{Hx}+(y-{self.y_cur})*{Hy}"
        B = f"((x-{self.x_cur})*{dx}+(y-{self.y_cur})*{dy})"
        f_str2 = f"{f_str}+{A}+{B}"
        src=f"func={f_str2}"
        return src

def runHimmelblau():
    Hi = Himmelblau(opt=Newton, lr=0.1, x=4, y=4)
    Xs, Ys, Zs = Hi.train()
    src = Hi.display()
    return src