import server
import pytest
import json

@pytest.fixture
def client():
    return server.app.test_client()

def test_trainer(client):
    with open('data/sample.json', 'r') as f:
        response = client.post('/dp', data=f, content_type='application/json')
        print(response.data)
        assert False
