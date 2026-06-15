import os
import requests
import joblib
import pandas as pd
from datetime import datetime, timedelta
from flask import Flask, request, jsonify

app = Flask(__name__)

# ==============================================================================
# ⚙️ CẤU HÌNH
# ==============================================================================
MODEL_PATH    = 'aqi_model.pkl'
model         = joblib.load(MODEL_PATH)

FIREBASE_URL  = "https://iotbytienpham-default-rtdb.firebaseio.com"
OWM_API_KEY   = "18efd3b6d037e0b3f24c0e16dcb09180"
LAT, LON      = 21.0285, 105.8542  # Hà Nội

# 28 features - thứ tự phải khớp 100% với notebook thầy
EXPECTED_COLUMNS = [
    'pm10', 'pm2_5', 'carbon_monoxide', 'nitrogen_dioxide', 'sulphur_dioxide',
    'ozone', 'aerosol_optical_depth', 'dust', 'uv_index',
    'day', 'month', 'year', 'dayofweek',
    'pm2_5_lag1', 'pm2_5_lag2', 'pm10_lag1', 'pm10_lag2',
    'co_lag1', 'co_lag2', 'no2_lag1', 'no2_lag2',
    'so2_lag1', 'so2_lag2', 'o3_lag1', 'o3_lag2',
    'pm2_5_roll3', 'pm2_5_roll7', 'pm_ratio'
]

# Giá trị trung bình lịch sử Hà Nội (fallback khi OWM lỗi)
# Tính từ air_quality_historical.csv (dataset thầy gửi)
HANOI_AVG = {
    'pm10':              57.27,
    'carbon_monoxide':  725.83,
    'nitrogen_dioxide':  27.15,
    'sulphur_dioxide':   25.71,
    'ozone':             72.52,
    'aerosol_optical_depth': 0.642,
    'dust':               0.491,
    'uv_index':           1.172,
}

# ==============================================================================
# 📡 LẤY DỮ LIỆU NGOẠI VI
# ==============================================================================
def get_owm_pollution():
    """Kéo CO, NO2, SO2, O3, PM10 real-time từ OpenWeatherMap."""
    url = (f"http://api.openweathermap.org/data/2.5/air_pollution"
           f"?lat={LAT}&lon={LON}&appid={OWM_API_KEY}")
    try:
        res = requests.get(url, timeout=5)
        if res.status_code == 200:
            c = res.json()['list'][0]['components']
            return {
                'pm10': c.get('pm10',  HANOI_AVG['pm10']),
                'co':   c.get('co',    HANOI_AVG['carbon_monoxide']),
                'no2':  c.get('no2',   HANOI_AVG['nitrogen_dioxide']),
                'so2':  c.get('so2',   HANOI_AVG['sulphur_dioxide']),
                'o3':   c.get('o3',    HANOI_AVG['ozone']),
            }
    except Exception as e:
        print(f"[WARN] OWM API lỗi: {e} → dùng Hà Nội avg")

    # Fallback về trung bình lịch sử Hà Nội
    return {
        'pm10': HANOI_AVG['pm10'],
        'co':   HANOI_AVG['carbon_monoxide'],
        'no2':  HANOI_AVG['nitrogen_dioxide'],
        'so2':  HANOI_AVG['sulphur_dioxide'],
        'o3':   HANOI_AVG['ozone'],
    }


def get_daily_pm25_history():
    """
    Kéo PM2.5 cuối ngày của 3 ngày gần nhất từ Firebase.
    Dùng để tính lag1 (hôm qua), lag2 (hôm kia), roll3, roll7.
    Model train trên daily data nên cần daily lag.
    """
    pm_list = []
    for days_ago in range(3, 0, -1):  # [3 ngày trước, 2 ngày trước, hôm qua]
        date_str = (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d")
        url = f"{FIREBASE_URL}/sensor_data/esp32_01/{date_str}.json"
        try:
            res = requests.get(url, timeout=5)
            if res.status_code == 200 and res.json():
                data = res.json()
                # Lấy reading cuối ngày (last timestamp)
                last_key = sorted(data.keys())[-1]
                pm_val = float(data[last_key].get('pm_ug_m3', 0))
                if pm_val > 0:
                    pm_list.append(pm_val)
                    print(f"[FIREBASE] {date_str}: PM2.5={pm_val}")
        except Exception as e:
            print(f"[WARN] Firebase {date_str} lỗi: {e}")

    return pm_list  # [ngày -3, ngày -2, ngày -1]


def get_advice(aqi: int) -> str:
    """Trả về lời khuyên y tế theo AQI."""
    if aqi <= 50:
        return "Không khí trong lành, yên tâm hoạt động ngoài trời bình thường"
    elif aqi <= 100:
        return "Chất lượng không khí chấp nhận được, nhóm nhạy cảm nên hạn chế ra ngoài lâu"
    elif aqi <= 150:
        return "Nhóm nhạy cảm (trẻ em, người già, bệnh hô hấp) nên ở trong nhà"
    elif aqi <= 200:
        return "Ảnh hưởng sức khỏe rõ rệt, hạn chế mọi hoạt động ngoài trời"
    elif aqi <= 300:
        return "Cảnh báo khẩn cấp, đeo khẩu trang N95 nếu buộc phải ra ngoài"
    return "Nguy hiểm nghiêm trọng, tuyệt đối ở trong nhà, đóng kín cửa"


# ==============================================================================
# 🧠 ENDPOINT CHÍNH
# ==============================================================================
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Nhận PM2.5 từ ESP32
        req_data     = request.json or {}
        current_pm25 = float(req_data.get('pm25', 0))

        if current_pm25 <= 0:
            return jsonify({'error': 'PM2.5 không hợp lệ'}), 400

        print(f"[REQUEST] PM2.5={current_pm25}")

        # 2. Lấy CO/NO2/SO2/O3/PM10 từ OpenWeatherMap
        owm = get_owm_pollution()

        # 3. Lấy lịch sử PM2.5 từ Firebase (3 ngày gần nhất)
        history = get_daily_pm25_history()
        # history = [pm_3daysago, pm_2daysago, pm_yesterday]
        # Thêm hôm nay vào cuối để tính rolling
        history_with_today = history + [current_pm25]

        now = datetime.now()

        # 4. Tính lag features (daily lag khớp với model train)
        lag1_pm25 = history[-1] if len(history) >= 1 else current_pm25
        lag2_pm25 = history[-2] if len(history) >= 2 else lag1_pm25

        # 5. Tính rolling averages
        roll3_vals = history_with_today[-3:]
        roll7_vals = history_with_today[-7:]
        pm2_5_roll3 = sum(roll3_vals) / len(roll3_vals)
        pm2_5_roll7 = sum(roll7_vals) / len(roll7_vals)

        # 6. PM ratio
        pm_ratio = current_pm25 / (owm['pm10'] + 1)

        # 7. Build 28-feature vector (thứ tự khớp EXPECTED_COLUMNS)
        feature_dict = {
            # Pollutants (hôm nay)
            'pm10':                  owm['pm10'],
            'pm2_5':                 current_pm25,
            'carbon_monoxide':       owm['co'],
            'nitrogen_dioxide':      owm['no2'],
            'sulphur_dioxide':       owm['so2'],
            'ozone':                 owm['o3'],
            'aerosol_optical_depth': HANOI_AVG['aerosol_optical_depth'],
            'dust':                  HANOI_AVG['dust'],
            'uv_index':              HANOI_AVG['uv_index'],

            # Time features
            'day':       now.day,
            'month':     now.month,
            'year':      now.year,
            'dayofweek': now.weekday(),

            # Lag PM2.5 (hôm qua, hôm kia)
            'pm2_5_lag1': lag1_pm25,
            'pm2_5_lag2': lag2_pm25,

            # Lag PM10 (dùng OWM hiện tại vì không có daily history)
            'pm10_lag1': owm['pm10'],
            'pm10_lag2': owm['pm10'],

            # Lag các khí khác (dùng OWM hiện tại)
            'co_lag1':  owm['co'],  'co_lag2':  owm['co'],
            'no2_lag1': owm['no2'], 'no2_lag2': owm['no2'],
            'so2_lag1': owm['so2'], 'so2_lag2': owm['so2'],
            'o3_lag1':  owm['o3'],  'o3_lag2':  owm['o3'],

            # Rolling PM2.5
            'pm2_5_roll3': pm2_5_roll3,
            'pm2_5_roll7': pm2_5_roll7,

            # Interaction
            'pm_ratio': pm_ratio,
        }

        df_input = pd.DataFrame([feature_dict])[EXPECTED_COLUMNS]

        # 8. Predict AQI ngày mai
        pred_aqi = int(round(float(model.predict(df_input)[0])))
        pred_aqi = max(0, min(500, pred_aqi))  # clamp 0-500

        advice = get_advice(pred_aqi)

        print(f"[PREDICT] AQI ngày mai = {pred_aqi}")

        # 9. Ghi kết quả lên Firebase /ai_forecast
        # ESP32 Thu chỉ đọc node "24h" → ghi vào đó
        # Node "1h" và "3h" phục vụ logic riêng bên C++
        result = {
            "24h": {
                "aqi":    pred_aqi,
                "advice": advice
            }
        }
        firebase_res = requests.put(
            f"{FIREBASE_URL}/ai_forecast.json",
            json=result,
            timeout=5
        )

        if firebase_res.status_code == 200:
            print("[FIREBASE] Đã ghi /ai_forecast OK!")
        else:
            print(f"[WARN] Firebase write lỗi: {firebase_res.status_code}")

        return jsonify({
            'status':   'success',
            'ai_aqi':   pred_aqi,
            'advice':   advice,
            'owm_used': True,
            'lag1_pm25': lag1_pm25,
            'lag2_pm25': lag2_pm25,
            'roll3':    round(pm2_5_roll3, 2),
            'roll7':    round(pm2_5_roll7, 2),
        })

    except Exception as e:
        print(f"[ERROR] {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    """Endpoint kiểm tra server còn sống không."""
    return jsonify({'status': 'ok', 'time': datetime.now().isoformat()})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
