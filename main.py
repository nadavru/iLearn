from flask import Flask,render_template,request
from optimizers import *
from gaussian import runGaussian
from Rosenbrock import runRosenbrock
from Himmelblau import runHimmelblau
from svm import runSVM
from Rastrigin import runRastrigin
from absFunc import runABS
from Beale import runBeale
x=-1
y=-1
learningRate=-1
def castToOpt(optString):
    if(optString == "Newton"):
        return Newton
    if(optString == "SGD"):
        return SGD
    if(optString == "MomentumSGD"):
        return MomentumSGD
    return SGD
app = Flask(__name__)
@app.route('/Gaussian/',methods=['POST','GET'])
def showGaussian():    
    learningRate = float(request.args.get("lr", 0.5))
    x = float(request.args.get("x", 1))
    y = float(request.args.get("y", 1))
    optString = request.args.get("opt", "Newton") 
    opt = castToOpt(optString)
    return render_template("viewerGDpyver.html",pyString = runGaussian(lr=learningRate,x=x,y=y,opt=opt) ,opt = optString,lr = learningRate, func = "Gaussian")

@app.route('/Beale/',methods=['POST','GET'])
def showBeale():    
    learningRate = float(request.args.get("lr", 0.000001))
    x = float(request.args.get("x", 4))
    y = float(request.args.get("y", 4))
    optString = request.args.get("opt", "Newton") 
    opt = castToOpt(optString)
    return render_template("viewerGDpyver.html",pyString = runBeale(lr=learningRate,x=x,y=y,opt=opt) ,opt = optString,lr = learningRate, func = "Beale")

@app.route('/Himmelblau/',methods=['POST','GET'])
def showHimmelblau():    
    learningRate = float(request.args.get("lr", 0.1))
    x = float(request.args.get("x", 4))
    y = float(request.args.get("y", 4))
    optString = request.args.get("opt", "Newton") 
    opt = castToOpt(optString)
    return render_template("viewerGDpyver.html",pyString = runHimmelblau(lr=learningRate,x=x,y=y,opt=opt),opt = optString,lr = learningRate,func = "Himmelblau")

@app.route('/Rosenbrock/',methods=['POST','GET'])
def showRosenbrock():    
    learningRate = float(request.args.get("lr", 0.1**5))
    x = float(request.args.get("x", 5))
    y = float(request.args.get("y", -5))
    optString = request.args.get("opt", "Newton") 
    opt = castToOpt(optString)
    return render_template("viewerGDpyver.html",pyString = runRosenbrock(lr=learningRate,x=x,y=y,opt=opt),opt = optString,lr = learningRate,func = "Rosenbrock")

@app.route('/ABS/',methods=['POST','GET'])
def showABS():    
    learningRate = float(request.args.get("lr", 1))
    x = float(request.args.get("x", 10))
    y = float(request.args.get("y", 0.5))
    optString = request.args.get("opt", "Newton") 
    opt = castToOpt(optString)
    return render_template("viewerGDpyver.html",pyString = runABS(lr=learningRate,x=x,y=y,opt=opt),opt = optString,lr = learningRate,func = "ABS")


@app.route('/')
def main():
    algo = request.args.get('algo')
    if(algo == "Gaussian"):
        return render_template("viewerGDpyver.html",pyString = runGaussian(),func = "Gaussian")
    elif(algo == "Beale"):
        return render_template("viewerGDpyver.html",pyString = runBeale(),func = "Beale")
    elif(algo == "Himmelblau"):
        return render_template("viewerGDpyver.html",pyString = runHimmelblau(),func = "Himmelblau")
    elif(algo == "Rastrigin"):
        return render_template("viewerGDpyver.html",pyString = runRastrigin(),func = "Rastrigin")
    elif(algo == "SVM"):
        return render_template("viewerGDpyver.html",pyString = runSVM(),func = "ABS")
    elif(algo == "Rosenbrock"):
        return render_template("viewerGDpyver.html",pyString = runRosenbrock(),func = "Rosenbrock")
    elif(algo == "ABS"):
        return render_template("viewerGDpyver.html",pyString = runABS(),func = "ABS")
 




if __name__ == '__main__':
    # This is used when running locally only. When deploying to Google App
    # Engine, a webserver process such as Gunicorn will serve the app. This
    # can be configured by adding an `entrypoint` to app.yaml.
    app.run(host='127.0.0.1', port=8080, debug=True)
# [END gae_python3_app]
# [END gae_python38_app]
