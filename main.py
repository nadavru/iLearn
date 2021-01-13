from flask import Flask,render_template
from gaussian import runGaussian
from Rosenbrock import runRosenbrock
from Himmelblau import runHimmelblau
from svm import runSVM
from Rastrigin import runRastrigin
from absFunc import runABS
from Beale import runBeale

import requests

app = Flask(__name__)

@app.route('/')
def main():
    try:
        page = request.GET['page']
        return page
    except Exception as e:
        print(e) # handle your errors
        return e
    page = 1
    runSVM()
    runRosenbrock()
    runHimmelblau()
    runGaussian()
    return a


if __name__ == '__main__':
    # This is used when running locally only. When deploying to Google App
    # Engine, a webserver process such as Gunicorn will serve the app. This
    # can be configured by adding an `entrypoint` to app.yaml.
    app.run(host='127.0.0.1', port=8080, debug=True)
# [END gae_python3_app]
# [END gae_python38_app]
