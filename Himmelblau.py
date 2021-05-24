from learning import Trainer
from math import exp
from optimizers import *

class Himmelblau(Trainer):

    def __init__(self, grid_x=6, grid_y=6, lr=0.001, epochs=50, opt=SGD, x=None, y=None): 
        super().__init__(grid_x, grid_y, lr, epochs, opt, x, y)

        url = "https://i-learn-ml.oa.r.appspot.com/viewer/viewerGD.html#"
        self.disp_url = f"xmax={grid_x}$xmin=-{grid_x}$ymax={grid_y}$ymin=-{grid_y}$"

        self.f_str = f"(x^2+y-11)^2+(x+y^2-7)^2"
        
        self.f_str_arr = []
        
        
    def calc(self, x, y): 
        return (x**2+y-11)**2+(x+y**2-7)**2

    def derive(self, x, y): 
        dx = 2*(2*x*(x**2+y-11)+x+y**2-7)
        dy = 2*(2*y*(x+y**2-7)+x**2+y-11)

        self.f_str_arr.append(self.calc_f())
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

        src=self.disp_url+f"func={self.f_str}$points={points_str}"
        return src
    
    def calc_f(self): 

        '''dx = f"2*(2*{self.x_cur}*({self.x_cur}^2+{self.y_cur}-11)+{self.x_cur}+{self.y_cur}^2-7)"
        dy = f"2*(2*{self.y_cur}*({self.x_cur}+{self.y_cur}^2-7)+{self.x_cur}^2+{self.y_cur}-11)"
        dxx = f"(12*{self.x_cur}^2+4*{self.y_cur}-42)"
        dxy = f"4*({self.x_cur}+{self.y_cur})"
        dyx = f"4*({self.x_cur}+{self.y_cur})"
        dyy = f"(12*{self.y_cur}^2+4*{self.x_cur}-26)"
        f_str = f"({self.x_cur}^2+{self.y_cur}-11)^2+({self.x_cur}+{self.y_cur}^2-7)^2"'''

        x, y = self.x_cur, self.y_cur
        d = self.derive2(self.x_cur, self.y_cur)

        dx = 2*(2*x*(x**2+y-11)+x+y**2-7)
        dy = 2*(2*y*(x+y**2-7)+x**2+y-11)
        dxx, dxy = d[0]
        dyx, dyy = d[1]
        f_val = self.calc(self.x_cur, self.y_cur)
        
        Hx = f"((x-{self.x_cur})*{dxx}+(y-{self.y_cur})*{dyx})"
        Hy = f"((x-{self.x_cur})*{dxy}+(y-{self.y_cur})*{dyy})"

        A = f"0.5*(x-{self.x_cur})*{Hx}+(y-{self.y_cur})*{Hy}"

        B = f"((x-{self.x_cur})*{dx}+(y-{self.y_cur})*{dy})"

        f_str2 = f"{f_val}+{A}+{B}"

        src=f_str2
        return src

    def display2(self): 
        points_str = ""
        for i in range(len(self.Xs)):
          points_str+=(str)(self.Xs[i])+","+(str)(self.Ys[i])+","+(str)(self.Zs[i])+"|"
        points_str=points_str[:-1]

        src=self.disp_url+f"func={self.f_str}$points={points_str}$curves="
        for func in self.f_str_arr:
          src += func + "|"
        
        src = src[:-1]
        
        return src
def runHimmelblau(opt=SGD,epochs=50,lr=0.001, x=4, y=4):
    Hi = Himmelblau(opt=opt,epochs=epochs, lr=lr, x=x, y=y)
    Xs, Ys, Zs = Hi.train()
    if((opt == SGD) or (opt == MomentumSGD)):
        src = Hi.display()
    if(opt==Newton):
        src = Hi.display2()
    return src, Hi.error
