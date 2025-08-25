# model-server/app.py
from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

@app.route("/")
def home():
    return "Supply Chain Risk Prediction API"

# Load model once
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.joblib")
model = joblib.load(MODEL_PATH)

MODEL_PATH2 = os.path.join(os.path.dirname(__file__), "body_chain_model.joblib")
model2 = joblib.load(MODEL_PATH2)

# # Simple heuristic risk score (demo)
# def compute_risk_percent(delay_days, geo, transport_status):
#     # raw score: delay contributes most, low geo (1-geo) increases risk, transport_status adds small risk
#     raw = delay_days * 0.5 + (1 - geo) * 0.4 + transport_status * 0.1
#     max_raw = 5 * 0.5 + (1 - 0.2) * 0.4 + 1 * 0.1  # delay max=5, geo min=0.2 -> (1-0.2)=0.8
#     pct = (raw / max_raw) * 100
#     return round(min(100, max(0, pct)), 2)

def compute_risk_percent(delay_days, geo, transport_status, money_loss, defective_rate=0.0):
    """
    delay_days: int (0–5 days typically)
    geo: float (0.2–1.0, higher = safer)
    transport_status: 
    money_loss: float (negative = loss, positive = gain)
    defective_rate: float (0.0–0.5 range, higher = more defective pieces)
    """

    # Base raw score from delay, geo, transport
    raw = delay_days * 0.5 + (geo) * 0.4 + transport_status * 0.1

    # Add defective_rate contribution
    defective_factor = min(defective_rate / 0.5, 1.0) * 0.3  # max 0.3 extra risk
    raw += defective_factor

    # Normalize money loss into risk contribution
    if money_loss < 0:  # losses increase risk
        money_factor = min(1.0, abs(money_loss) / 100000) * 0.7
    else:  # gains reduce risk slightly
        money_factor = -(min(1.0, money_loss / 100000) * 0.3)

    raw += money_factor

    # Max possible raw (for scaling to %)
    max_raw = 5 * 0.5 + (1 - 0.2) * 0.4 + 1 * 0.1 + 0.7 + 0.3  # include defective max
    pct = (raw / max_raw) * 100

    return round(min(100, max(0, pct)), 2)


@app.route("/predict", methods=["POST"])
def predict():
    """
    expected JSON body:
    {
      "delay_days": 3,
      "geopolitical_points_bounds": 0.6,
      "transport_status": 1,
      "required_material": 1000   # optional
    }
    """
    data = request.get_json(force=True)
    delay = float(data.get("delay_days", 0))
    geo = float(data.get("geopolitical_points_bounds", 0.2))
    transport = int(data.get("transport_status", 0))
    required = data.get("required_material", 1000)

    # model input must match training columns order
    X = np.array([[delay, geo, transport]])
    pred = model.predict(X)[0]
    pred_int = int(round(pred))

    risk_pct = compute_risk_percent(delay, geo, transport,pred_int - float(required))
    loss = pred_int - float(required)
  
    # recommendation (very simple demo logic)
    recommendation = "OK"
    status = "Normal"
    if transport == 1:
        status = "Disrupted"
    # if risk_pct >= 70:
    #     recommendation = "High risk — consider backup supplier or increase safety stock"
    # elif risk_pct >= 40:
    #     recommendation = "Moderate risk — monitor & consider partial pre-order"
    # else:
    #     recommendation = "Low risk — proceed as planned"
    if risk_pct >= 60:
        recommendation = (
            f"The supply risk is high at {risk_pct:.1f}%. The predicted shortage is {abs(loss)} units. "
            f"This high risk is due to a delivery delay of {delay} days, a geopolitical risk score of {geo}, "
            f"transport issues with status {status}, "
            f"It is recommended to consider a backup supplier, increase safety stock, or split orders to mitigate this risk."
        )
    elif risk_pct >= 40:
        recommendation = (
            f"The supply risk is moderate at {risk_pct:.1f}%. The predicted shortage is {abs(loss)} units. "
            f"This moderate risk arises from a delivery delay of {delay} days, a geopolitical risk score of {geo}, "
            f"transport issues with status {status} "
            f"It is advised to monitor the supplier closely and consider partial pre-orders to reduce potential loss."
        )
    else:
        recommendation = (
            f"The supply risk is low at {risk_pct:.1f}%. The predicted shortage or surplus is {loss} units. "
            f"All key factors, including delivery delay, geopolitical risk, transport status, and supplier reliability, "
            f"are within acceptable limits. You can proceed with the planned orders as scheduled."
        )


    result = {
        "predicted_material": pred_int,
        "risk_pct": risk_pct,
        "recommendation": recommendation
    }
    result["required_material"] = required
    result["loss"] = float(loss)

    return jsonify(result)











@app.route("/predictBody",methods=["POST"])
def predictBody():
    """
    expected JSON body:
    { 
      "defective_rate": 0.05,
      "delay_days": 3,
      "geopolitical_points_bounds": 0.6,
      "transport_status": 2,
      "supplier_reliability": 0.8,
      "required_material": 1000 , 
    }
    """
    data = request.get_json(force=True)
    defective = float(data.get("defective_rate", 0.0))
    delay = float(data.get("delay_days", 0))
    geo = float(data.get("geopolitical_points_bounds", 0.2))
    transport = int(data.get("transport_status", 0))
    supplier = float(data.get("supplier_reliability", 1.0))
    required = data.get("required_material", 1000)

    # model input must match training columns order
    X = np.array([[defective, delay, geo, transport, supplier]])
    pred = model2.predict(X)[0]
    pred_int = int(round(pred))

    risk_pct = compute_risk_percent(delay, geo, transport,pred_int - float(required),defective)
    loss = pred_int - float(required)

    # recommendation (very simple demo logic)
    recommendation = "OK"
    # if risk_pct >= 60:
    #     recommendation = "High risk — consider backup supplier or increase safety stock"
    # elif risk_pct >= 40:
    #     recommendation = "Moderate risk — monitor & consider partial pre-order"
    # else:
    #     recommendation = "Low risk — proceed as planned"
    status = "Smooth"
    if transport == 1:
        status = "Disrupted"
    elif transport == 2:
        status = "Severely Disrupted"  

    if risk_pct >= 60:
        recommendation = (
            f"The supply risk is high at {risk_pct:.1f}%. The predicted shortage is {abs(loss)} units. "
            f"This high risk is due to a delivery delay of {delay} days, a geopolitical risk score of {geo}, "
            f"transport issues with status {status}, "
            f"It is recommended to consider a backup supplier, increase safety stock, or split orders to mitigate this risk."
        )
    elif risk_pct >= 40:
        recommendation = (
            f"The supply risk is moderate at {risk_pct:.1f}%. The predicted shortage is {abs(loss)} units. "
            f"This moderate risk arises from a delivery delay of {delay} days, a geopolitical risk score of {geo}, "
            f"transport issues with status {status} "
            f"It is advised to monitor the supplier closely and consider partial pre-orders to reduce potential loss."
        )
    else:
        recommendation = (
            f"The supply risk is low at {risk_pct:.1f}%. The predicted shortage or surplus is {loss} units. "
            f"All key factors, including delivery delay, geopolitical risk, transport status, and supplier reliability, "
            f"are within acceptable limits. You can proceed with the planned orders as scheduled."
        )


    result = {
        "predicted_material": pred_int,
        "risk_pct": risk_pct,
        "recommendation": recommendation
    }

    if required is not None:
        result["required_material"] = required
        result["loss"] = float(loss)

    return jsonify(result)

        


if __name__ == "__main__":
    app.run(port=5000, debug=True)