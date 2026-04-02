import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_treatment import prepare_historic_data
from xgboost import XGBRegressor

def xgb_model():

    X_y = prepare_historic_data()
    X, y = X_y[0], X_y[1]

    xgb_model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        random_state=42
    )
    xgb_model.fit(X, y)

    return xgb_model

model = xgb_model()