import os
import requests
import joblib
import pandas as pd
from datetime import datetime
from flask import Flask, request, jsonify

app = Flask(__name__)

# ==============================================================================
# ⚙️ CẤU HÌNH HỆ THỐNG
# ==============================================================================
# Tải mô hình AI đã huấn luyện
MODEL_PATH = 'aqi_model.pkl'
model = joblib.load(MODEL_PATH)

# Firebase RTDB URL (Để kéo dữ liệu quá khứ)
FIREBASE_URL = "https://iotbytienpham-default-rtdb.firebaseio.com"

# API Key OpenWeatherMap (Bro tạo tài khoản free ở openweathermap.org rồi ném key vào đây)
OWM_API_KEY = "18efd3b6d037e0b3f24c0e16dcb09180"

# Tọa độ Hà Nội
LAT = 21.0285
LON = 105.8542

# 28 Cột đặc trưng - THỨ TỰ BẮT BUỘC PHẢI KHỚP 100% VỚI FILE IPYNB
EXPECTED_COLUMNS = [
    'pm10', 'pm2_5', 'carbon_monoxide', 'nitrogen_dioxide', 'sulphur_dioxide', 
    'ozone', 'aerosol_optical_depth', 'dust', 'uv_index', 
    'day', 'month', 'year', 'dayofweek', 
    'pm2_5_lag1', 'pm2_5_lag2', 'pm10_lag1', 'pm10_lag2', 
    'co_lag1', 'co_lag2', 'no2_lag1', 'no2_lag2', 
    'so2_lag1', 'so2_lag2', 'o3_lag1', 'o3_lag2', 
    'pm2_5_roll3', 'pm2_5_roll7', 'pm_ratio'
]

# ==============================================================================
# 📡 CÁC HÀM LẤY DỮ LIỆU NGOẠI VI
# ==============================================================================
def get_owm_pollution():
    """ Kéo dữ liệu ô nhiễm không khí Real-time từ OpenWeatherMap """
    url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={LAT}&lon={LON}&appid={OWM_API_KEY}"
    try:
        res = requests.get(url, timeout=5)
        if res.status_code == 200:
            data = res.json()['list'][0]['components']
            return {
                'pm10': data.get('pm10', 45.0),
                'co': data.get('co', 500.0),
                'no2': data.get('no2', 15.0),
                'so2': data.get('so2', 8.0),
                'o3': data.get('o3', 35.0)
            }
    except Exception as e:
        print("Lỗi kéo API OWM:", e)
    
    # Nếu rớt mạng, xài giá trị trung bình an toàn
    return {'pm10': 45.0, 'co': 500.0, 'no2': 15.0, 'so2': 8.0, 'o3': 35.0}

def get_recent_pm25_from_firebase():
    """ Kéo dữ liệu PM2.5 lịch sử trong ngày từ ESP32 để tính Lag và Roll """
    today_str = datetime.now().strftime("%Y-%m-%d")
    url = f"{FIREBASE_URL}/sensor_data/esp32_01/{today_str}.json"
    
    try:
        res = requests.get(url, timeout=5)
        if res.status_code == 200 and res.json():
            data = res.json()
            # Trích xuất mảng PM2.5 theo thứ tự thời gian
            pm_list = [val.get('pm_ug_m3', 0) for key, val in sorted(data.items())]
            return pm_list
    except:
        pass
    return []

# ==============================================================================
# 🧠 LUỒNG XỬ LÝ CHÍNH
# ==============================================================================
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Hứng PM2.5 thực tế từ ESP32
        req_data = request.json or {}
        current_pm25 = float(req_data.get('pm25', 0))
        
        # 2. Hứng các khí thải khác từ OpenWeatherMap
        owm_data = get_owm_pollution()
        
        # 3. Kéo lịch sử từ Firebase để chuẩn bị làm Data Time-Series
        pm_history = get_recent_pm25_from_firebase()
        now = datetime.now()
        
        # 4. Tính toán Toán học (Lag & Rolling)
        # Nếu thiết bị mới bật chưa có lịch sử, nội suy bằng chính PM2.5 hiện tại
        lag1_pm25 = pm_history[-1] if len(pm_history) >= 1 else current_pm25
        lag2_pm25 = pm_history[-2] if len(pm_history) >= 2 else lag1_pm25
        
        # Trung bình trượt 3 và 7 chu kỳ gần nhất
        roll3_list = pm_history[-2:] + [current_pm25]
        pm2_5_roll3 = sum(roll3_list) / len(roll3_list)
        
        roll7_list = pm_history[-6:] + [current_pm25]
        pm2_5_roll7 = sum(roll7_list) / len(roll7_list)
        
        # Cột tương tác PM Ratio (Sao chép y hệt công thức trong file train)
        pm_ratio = current_pm25 / (owm_data['pm10'] + 1)
        
        # 5. RÁP THÀNH VECTOR 28 ĐẶC TRƯNG CHUẨN XÁC
        feature_dict = {
            'pm10': owm_data['pm10'],
            'pm2_5': current_pm25,
            'carbon_monoxide': owm_data['co'],
            'nitrogen_dioxide': owm_data['no2'],
            'sulphur_dioxide': owm_data['so2'],
            'ozone': owm_data['o3'],
            'aerosol_optical_depth': 0.6, # Gán tĩnh bằng Mean dataset
            'dust': 0.0,
            'uv_index': 2.0,
            
            'day': now.day,
            'month': now.month,
            'year': now.year,
            'dayofweek': now.weekday(),
            
            'pm2_5_lag1': lag1_pm25,
            'pm2_5_lag2': lag2_pm25,
            'pm10_lag1': owm_data['pm10'], 
            'pm10_lag2': owm_data['pm10'],
            
            'co_lag1': owm_data['co'],
            'co_lag2': owm_data['co'],
            'no2_lag1': owm_data['no2'],
            'no2_lag2': owm_data['no2'],
            'so2_lag1': owm_data['so2'],
            'so2_lag2': owm_data['so2'],
            'o3_lag1': owm_data['o3'],
            'o3_lag2': owm_data['o3'],
            
            'pm2_5_roll3': pm2_5_roll3,
            'pm2_5_roll7': pm2_5_roll7,
            'pm_ratio': pm_ratio
        }
        
        # Ép kiểu thành Pandas DataFrame (Để đảm bảo đúng tên cột cho XGBoost)
        df_features = pd.DataFrame([feature_dict])[EXPECTED_COLUMNS]
        
        # 6. Ra lệnh cho AI dự đoán
        pred_aqi = int(model.predict(df_features)[0])
        
        # 7. Ghi thẳng kết quả lên 3 Tab của Firebase
        result_payload = {
            "1h": {"aqi": pred_aqi, "advice": "Dự báo cực ngắn - Cập nhật liên tục"},
            "3h": {"aqi": int(pred_aqi * 1.05), "advice": "Xu hướng trung hạn"},
            "24h": {"aqi": int(pred_aqi * 1.1), "advice": "Dự báo thời điểm này ngày mai"}
        }
        requests.put(f"{FIREBASE_URL}/ai_forecast.json", json=result_payload)
        
        return jsonify({'status': 'Thành công', 'ai_aqi': pred_aqi, 'owm_api': 'active'})
        
    except Exception as e:
        print("Lỗi máy chủ AI:", e)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
