from flask import Flask,render_template,request
from gaussian import runGaussian
from Rosenbrock import runRosenbrock
from Himmelblau import runHimmelblau
from svm import runSVM
from Rastrigin import runRastrigin
from absFunc import runABS
from Beale import runBeale


app = Flask(__name__)

@app.route('/')
def main():
    algo = request.args.get('algo')
    if(algo == "Gaussian"):
        s = runGaussian()
    elif(algo == "Beale"):
        s = runBeale()
    elif(algo == "Himmelblau"):
        s = runHimmelblau()
    elif(algo == "Rastrigin"):
        s = runRastrigin()
    elif(algo == "SVM"):
        s = runSVM()
    elif(algo == "Rosenbrock"):
        s = runRosenbrock()
    elif(algo == "ABS"):
        s = runABS()
    return render_template("viewerGDpyver.html",pyString = s)
 




if __name__ == '__main__':
    # This is used when running locally only. When deploying to Google App
    # Engine, a webserver process such as Gunicorn will serve the app. This
    # can be configured by adding an `entrypoint` to app.yaml.
    app.run(host='127.0.0.1', port=8080, debug=True)
# [END gae_python3_app]
# [END gae_python38_app]
