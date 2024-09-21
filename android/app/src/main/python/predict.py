import xgboost as xgb
import numpy as np
import os

model_path = os.path.join(os.path.dirname(__file__), 'model.json')
m = xgb.XGBRegressor()
m.load_model(model_path)

def predict(inputs):
    inputs = np.array(inputs).reshape(-1, 33 * 3)
    preds = m.predict(inputs)
    return preds

if 'main' in __name__:
    print(predict(np.random.uniform(0, 1, (4 * 33 * 3))))