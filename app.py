from flask import Flask, request, jsonify
import joblib
import requests
import os

app = Flask(__name__)

# Load model AI
model_path = os.path.join(os.path.dirname(__file__), 'aqi_model.pkl')
model = joblib.load(model_path)

# Link tới cục Realtime Database của bro (Thay bằng link Firebase của bro)
FIREBASE_URL = "https://iotbytienpham-default-rtdb.firebaseio.com/"

@app.route('/predict', methods=['POST'])
def predict():
    # 1. Nhận data từ ESP32 gửi lên
    data = request.get_json()
    pm25 = data.get('pm25', 0)
    temp = data.get('temp', 0)
    hum = data.get('hum', 0)
    
    # 2. Đưa vào Model dự đoán
    prediction = model.predict([[pm25, temp, hum]])
    predicted_aqi = prediction[0]
    
    # 3. Ghi kết quả lên Firebase bằng REST API (Cực dễ, không cần cấu hình lằng nhằng)
    payload = {
        "pm25": pm25, "temp": temp, "hum": hum,
        "predicted_aqi": predicted_aqi
    }
    requests.put(f"{FIREBASE_URL}/ai_forecast/latest.json", json=payload)
    
    # Báo cáo lại cho ESP32 biết là đã xong
    return jsonify({"status": "Thành công", "aqi": predicted_aqi}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)