from learning import Trainer
from math import exp
from optimizers import *


class Beale(Trainer):

    def __init__(self, grid_x=4, grid_y=4, lr=0.0001, epochs=50, opt=SGD, x=None, y=None): 
        super().__init__(grid_x, grid_y, lr, epochs, opt, x, y)

        url = "https://i-learn-ml.oa.r.appspot.com/viewer/viewerGD.html#"
        self.disp_url = f"xmax={grid_x}$xmin=-{grid_x}$ymax={grid_y}$ymin=-{grid_y}$"

        self.f_str = f"(1.5-x+x*y)^2+(2.25-x+x*y^2)^2+(2.625-x+x*y^3)^2"
        
        
    def calc(self, x, y): 
        return (1.5-x+x*y)**2+(2.25-x+x*y**2)**2+(2.625-x+x*y**3)**2

    def derive(self, x, y): 
        dx = 2 * x *(y**6 + y**4 - 2 * y**3 - y**2 - 2 * y + 3) + 5.25 * y**3 + 4.5 * y**2 + 3 * y - 12.75
        dy = 6 * x * (x * (y**5 + 0.666667 * y**3 - y**2 - 0.333333 * y - 0.333333) + 2.625 * y**2 + 1.5 * y + 0.5)
        return [dx, dy]
    
    def display(self): 
        points_str = ""
        for i in range(len(self.Xs)):
          points_str+=(str)(self.Xs[i])+","+(str)(self.Ys[i])+","+(str)(self.Zs[i])+"|"
        points_str=points_str[:-1]

        src=self.disp_url+f"func={self.f_str}$points={points_str}"
        return src

def runBeale(x=4,y=4,epochs=50,opt=MomentumSGD,lr=0.000001):
    Be = Beale(opt=opt,epochs=epochs, lr=lr, x=x, y=y)
    Xs, Ys, Zs = Be.train()
    src = Be.display()
    return src, Be.error
