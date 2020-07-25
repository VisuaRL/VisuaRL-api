from flask import Flask, request
from visuarl_main import execute_dp, execute_td_learning
import json

app = Flask(__name__)


@app.route('/dp', methods=['POST'])
def dp_train():
    return execute_dp(request.get_json())


@app.route('/ql', methods=['POST'])
def ql_train():
    return execute_td_learning(request.get_json(), policy='ql')


@app.route('/sarsa', methods=['POST'])
def sarsa_train():
    return execute_td_learning(request.get_json(), policy='sarsa')
