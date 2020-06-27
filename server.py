from flask import Flask, request
from visuarl_main import execute_dp, execute_ql
import json

app = Flask(__name__);

@app.route('/dp', methods=['POST'])
def dp_train():
    return execute_dp(request.get_json())

@app.route('/ql', methods=['POST'])
def ql_train():
    return execute_ql(request.get_json())
