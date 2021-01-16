from learning import Trainer
from math import exp
from optimizers import *

class Absfunc(Trainer):
    def __init__(self, grid_x=10, grid_y=10, lr=0.0001, epochs=50, opt=SGD, x=None, y=None): 
        super().__init__(grid_x, grid_y, lr, epochs, opt, x, y)

        url = "https://i-learn-ml.oa.r.appspot.com/viewer/viewerGD.html#"
        self.disp_url = f"xmax={grid_x}$xmin=-{grid_x}$ymax={grid_y}$ymin=-{grid_y}$"

        #self.f_str = "x^4+y^4+50*sin(30*(x^2+y^2)*atan(y/x))"
        '''t = "3*atan(y/x)"
        self.f_str = f"x^2+y^2+20*abs({t}-floor({t})-0.5)"'''
        self.f_str = "abs(x)+abs(y)"

    def calc(self, x, y): 
        #return x**4+y**4+50*math.sin(30*(x**2+y**2)*math.atan(y/x))
        #return x**2+y**2+20*abs(3*math.atan(y/x)-math.floor(3*math.atan(y/x))-0.5)
        return abs(x)+abs(y)

    def derive(self, x, y): 
        #dx = (4 * (x**5 + x**3 * y**2 - 625 * y * math.cos(50 * math.atan(y/x))))/(x**2 + y**2)
        #dx = 4 * (x**3 + 750 * x * math.atan(y/x) * math.cos(30 * (x**2 + y**2) * math.atan(y/x)) - 375 * y * math.cos(30 * (x**2 + y**2) * math.atan(y/x)))
        #dy = 4 * ((625 * x * math.cos(50 * math.atan(y/x)))/(x**2 + y**2) + y**3)
        #dy = 4 * (750 * y * math.atan(y/x) * math.cos(30 * (x**2 + y**2) * math.atan(y/x)) + 375 * x * math.cos(30 * (x**2 + y**2) * math.atan(y/x)) + y**3)
        
        '''if math.sin(3 * pi * math.atan(y/x))!=0 and 3 * math.atan(y/x) - math.floor(3 * math.atan(y/x))>0.5:
          dx = 2 * x - (60 * y)/(x**2 + y**2)
          dy = 2 * y + (60 * x)/(x**2 + y**2)
        else:
          dx = 2 * x +  (60 * y)/(x**2 + y**2)
          dy = 2 * y - (60 * x)/(x**2 + y**2)'''
        
        dx = x/abs(x) if x!=0 else 0
        dy = y/abs(y) if y!=0 else 0

        return [dx, dy]
    
    def display(self): 
        points_str = ""
        for i in range(len(self.Xs)):
          points_str+=(str)(self.Xs[i])+","+(str)(self.Ys[i])+","+(str)(self.Zs[i])+"|"
        points_str=points_str[:-1]

        src=self.disp_url+f"func={self.f_str}$points={points_str}"
        return src

def runABS(opt=SGD,epochs=50, lr=1, x=10, y=0.5):
    Abs = Absfunc(opt=SGD,epochs =epochs,lr=1, x=10, y=0.5)
    Xs, Ys, Zs = Abs.train()
    src = Abs.display()
    return src