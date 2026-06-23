import numpy as np


def predict_with_threshold(model, X, classification_threshold: float ): 

    """ 
    Devuelve: 
    - probabilidad de riesgo 
    - predicción final (0/1) usando un umbral 

    Esto es lo que alimenta Power BI: 
    - risk_probability 
    - risk_prediction 

    """ 

    prob = model.predict_proba(X)[:, 1] 
    pred = (prob >= classification_threshold).astype(int) 

    return prob, pred 