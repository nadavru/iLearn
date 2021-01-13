from learning import Trainer
from math import exp
from optimizers import *
class SVM(Trainer):

  def __init__(self, group_size, p0, p1, grid_x=50, grid_y=50, lr=0.5, epochs=10): 
        super().__init__(grid_x, grid_y, lr, epochs)
        (x0, y0, var0) = p0
        (x1, y1, var1) = p1
        self.group_size = group_size
        group0 = np.array(list(zip(list(np.random.normal(x0, var0, group_size)), list(np.random.normal(y0, var0, group_size)),[-1]*group_size)))
        group1 = np.array(list(zip(list(np.random.normal(x1, var1, group_size)), list(np.random.normal(y1, var0, group_size)),[1]*group_size)))

        self.data = np.concatenate((group0, group1))

        url = "https://htmlpreview.github.io/?https://github.com/nadavru/iLearn/blob/iLearnML/scatter.html#"
        #grid = 30
        self.disp_url = url+f"xmax={grid_x}&xmin=-{grid_x}&ymax={grid_y}&ymin=-{grid_y}&"
        
        url2 = "https://htmlpreview.github.io/?https://github.com/nadavru/iLearn/blob/iLearnML/ilearnml_graph_viewer.html#"
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

    src=f"func={f_str}&points={points_str}"
    return src
def runSVM():
    T = SVM(50, (-5,-5,5), (5,5,5), lr=0.5, epochs=30)
    Xs, Ys, Zs = T.train()
    src = T.display()
    return src
