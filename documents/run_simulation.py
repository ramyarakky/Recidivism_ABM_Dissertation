import json
import pandas as pd
from model.RecidivismModel import RecidivismModel

def load_risk_weights(path="risk_weights.json"):
    with open(path) as f:
        return json.load(f)

def load_peer_influence_config(path="peer_influence_config.json"):
    with open(path) as f:
        return json.load(f)


# Then use them in your model setup
risk_weights = load_risk_weights()
peer_config = load_peer_influence_config()
model = RecidivismModel(risk_weights=risk_weights, peer_config=peer_config)
