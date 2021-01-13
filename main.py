from flask import Flask,render_template
import numpy as np
from math import exp
#from abc import ABCMeta, abstractmethod


class Trainer:
    def __init__(self, grid_x, grid_y, lr=0.5, epochs=50): 
        self.lr=lr
        self.x_cur = 2*grid_x*np.random.random_sample()-grid_x
        self.y_cur = 2*grid_y*np.random.random_sample()-grid_y
        self.epochs=epochs

        x = np.linspace(-grid_x, grid_x, 1000)
        y = np.linspace(-grid_y, grid_y, 1000)
        self.X, self.Y = np.meshgrid(x, y)

    def __call__(self): 
        return self.X, self.Y, np.where(True, self.calc(self.X, self.Y), 0)
    
    def train(self): 
        self.Xs = [self.x_cur]
        self.Ys = [self.y_cur]
        self.Zs = [self.calc(self.x_cur, self.y_cur)]

        for _ in range(self.epochs):
            d = self.derive(self.x_cur, self.y_cur)
            #print(d)
            self.x_cur -= self.lr*d[0]
            self.y_cur -= self.lr*d[1]

            self.Xs.append(self.x_cur)
            self.Ys.append(self.y_cur)
            self.Zs.append(self.calc(self.x_cur, self.y_cur))
        return self.Xs, self.Ys, self.Zs

 #   @abstractmethod
    def display(self): 
      pass
    
  #  @abstractmethod
    def calc(self, x, y): 
      pass

  #  @abstractmethod
    def derive(self, x, y): 
      pass

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
    def __init__(self, apc, grid_x, grid_y, lr=0.5, epochs=50): 
        super().__init__(grid_x, grid_y, lr, epochs)
        self.g_list = [Gaussian_helper(*i) for i in apc]

        url = "https://htmlpreview.github.io/?https://github.com/nadavru/iLearn/blob/iLearnML/ilearnml_graph_viewer.html#"
        self.disp_url = url+f"xmax={grid_x}&xmin=-{grid_x}&ymax={grid_y}&ymin=-{grid_y}&"

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

        src=self.disp_url+f"func={self.f_str}&points={points_str}"
        return src

# If `entrypoint` is not defined in app.yaml, App Engine will look for an app
# called `app` in `main.py`.


app = Flask(__name__)

@app.route('/')
def main():
    return render_template('index.html')


if __name__ == '__main__':
    # This is used when running locally only. When deploying to Google App
    # Engine, a webserver process such as Gunicorn will serve the app. This
    # can be configured by adding an `entrypoint` to app.yaml.
    app.run(host='127.0.0.1', port=8080, debug=True)
# [END gae_python3_app]
# [END gae_python38_app]
