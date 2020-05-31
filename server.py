from flask import Flask, request
from visuarl_main import execute_solver
import json

app = Flask(__name__);

@app.route('/trainer', methods=['POST'])
def train():
    return execute_solver(request.get_json())