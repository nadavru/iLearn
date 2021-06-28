from flask import Flask,render_template,request
from optimizers import *
from gaussian import runGaussian
from Rosenbrock import runRosenbrock
from Himmelblau import runHimmelblau
from svm import runSVM
from Rastrigin import runRastrigin
from absFunc import runABS
from Beale import runBeale


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
    learningRate = float(request.args.get("lr", 2))
    x = float(request.args.get("x", 1))
    y = float(request.args.get("y", 1))
    epochs = int(request.args.get("epochs", 50))
    optString = request.args.get("opt", "SGD") 
    opt = castToOpt(optString)
    pyString,error_val = runGaussian(lr=learningRate,x=x,y=y,opt=opt,epochs=epochs)
    return render_template("viewerGDpyver.html",epochs=epochs,maxLR=2,x=x,y=y,pyString = pyString ,opt = optString,lrVal = learningRate, func = "Gaussian")

@app.route('/Beale/',methods=['POST','GET'])
def showBeale():    
    learningRate = float(request.args.get("lr", 0.1**5))
    x = float(request.args.get("x", 4))
    y = float(request.args.get("y", 4))
    epochs = int(request.args.get("epochs", 50))
    optString = request.args.get("opt", "SGD") 
    opt = castToOpt(optString)
    pyString,error_val = runBeale(lr=learningRate,x=x,y=y,opt=opt,epochs=epochs)
    return render_template("viewerGDpyver.html",pyString = pyString,x=x,y=y,maxLR=1*0.1**5,error_val = error_val ,epochs=epochs,opt = optString,lrVal = learningRate, func = "Beale")

@app.route('/Himmelblau/',methods=['POST','GET'])
def showHimmelblau():    
    learningRate = float(request.args.get("lr", 0.01))
    x = float(request.args.get("x", 4))
    y = float(request.args.get("y", 4))
    epochs = int(request.args.get("epochs", 50))
    optString = request.args.get("opt", "SGD") 
    opt = castToOpt(optString)
    pyString,error_val= runHimmelblau(lr=learningRate,x=x,y=y,opt=opt,epochs=epochs)
    return render_template("viewerGDpyver.html",pyString = pyString,x=x,y=y,maxLR=0.02,error_val = error_val ,epochs=epochs,opt = optString,lrVal = learningRate,func = "Himmelblau")

@app.route('/Rosenbrock/',methods=['POST','GET'])
def showRosenbrock():    
    learningRate = float(request.args.get("lr", 0.4*0.1**4))
    x = float(request.args.get("x", 3))
    y = float(request.args.get("y", -3))
    epochs = int(request.args.get("epochs", 100))
    optString = request.args.get("opt", "SGD") 
    opt = castToOpt(optString)
    pyString,error_val = runRosenbrock(lr=learningRate,x=x,y=y,opt=opt,epochs=epochs)
    return render_template("viewerGDpyver.html",pyString = pyString,x=x,y=y,maxLR=0.4*0.1**4,epochs=epochs,opt = optString,lrVal = learningRate,func = "Rosenbrock")

@app.route('/ABS/',methods=['POST','GET'])
def showABS():    
    learningRate = float(request.args.get("lr", 2))
    x = float(request.args.get("x", 10))
    y = float(request.args.get("y", 0.5))
    epochs = int(request.args.get("epochs", 50))
    optString = request.args.get("opt", "SGD") 
    opt = castToOpt(optString)
    pyString,error_val = runABS(lr=learningRate,x=x,y=y,opt=opt,epochs=epochs)
    return render_template("viewerGDpyver.html",pyString = pyString,x=x,y=y,maxLR=2,error_val = error_val ,epochs=epochs,opt = optString,lrVal = learningRate,func = "ABS")

@app.route('/Rastrigin/',methods=['POST','GET'])
def showRastrigin():    
    learningRate = float(request.args.get("lr", 0.0001))
    x = float(request.args.get("x", 10))
    y = float(request.args.get("y", 10))
    epochs = int(request.args.get("epochs", 50))
    optString = request.args.get("opt", "SGD") 
    opt = castToOpt(optString)
    pyString,error_val = runRastrigin(lr=learningRate,x=x,y=y,opt=opt,epochs=epochs)
    return render_template("viewerGDpyver.html",pyString = pyString,x=x,y=y,maxLR= 0.0001, error_val = error_val ,epochs=epochs,opt = optString,lrVal = learningRate,func = "Rastrigin")

@app.route('/SVM/',methods=['POST','GET'])
def showSVM():   
    try:
        learningRate = float(request.args.get("lr", 0.0001))
    except:
        learningRate = 0.0001
    try:
        epochs = int(request.args.get("epochs", 30))
    except:
        epochs = 30
    return render_template("viewerSVMpyver.html",pyString = runSVM(lr=learningRate,epochs = epochs),epochs = epochs,lr = learningRate,func = "SVM")

@app.route('/Newton/',methods=['POST','GET'])

 




if __name__ == '__main__':
    # This is used when running locally only. When deploying to Google App
    # Engine, a webserver process such as Gunicorn will serve the app. This
    # can be configured by adding an `entrypoint` to app.yaml.
    app.run(host='127.0.0.1', port=8080, debug=True)
# [END gae_python3_app]
# [END gae_python38_app]
