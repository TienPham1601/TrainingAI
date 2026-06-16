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
model        = joblib.load('aqi_model.pkl')
FIREBASE_URL = "https://iotbytienpham-default-rtdb.firebaseio.com"
OWM_API_KEY  = "18efd3b6d037e0b3f24c0e16dcb09180"
LAT, LON     = 21.0285, 105.8542  # Hà Nội

# 28 features - thứ tự khớp 100% với model
EXPECTED_COLUMNS = [
    'pm10', 'pm2_5', 'carbon_monoxide', 'nitrogen_dioxide', 'sulphur_dioxide',
    'ozone', 'aerosol_optical_depth', 'dust', 'uv_index',
    'day', 'month', 'year', 'dayofweek',
    'pm2_5_lag1', 'pm2_5_lag2', 'pm10_lag1', 'pm10_lag2',
    'co_lag1', 'co_lag2', 'no2_lag1', 'no2_lag2',
    'so2_lag1', 'so2_lag2', 'o3_lag1', 'o3_lag2',
    'pm2_5_roll3', 'pm2_5_roll7', 'pm_ratio'
]

# Trung bình lịch sử Hà Nội (từ air_quality_historical.csv)
HANOI_AVG = {
    'pm10':                   57.27,
    'carbon_monoxide':       725.83,
    'nitrogen_dioxide':       27.15,
    'sulphur_dioxide':        25.71,
    'ozone':                  72.52,
    'aerosol_optical_depth':  0.642,
    'dust':                   0.491,
    'uv_index':               1.172,
}

# ==============================================================================
# 📡 HÀM LẤY DỮ LIỆU NGOẠI VI
# ==============================================================================
def get_owm_current():
    """Lấy ô nhiễm HIỆN TẠI tại Hà Nội từ OWM (cho local prediction)."""
    url = (f"http://api.openweathermap.org/data/2.5/air_pollution"
           f"?lat={LAT}&lon={LON}&appid={OWM_API_KEY}")
    try:
        res = requests.get(url, timeout=5)
        if res.status_code == 200:
            c = res.json()['list'][0]['components']
            return {
                'pm10': c.get('pm10', HANOI_AVG['pm10']),
                'co':   c.get('co',   HANOI_AVG['carbon_monoxide']),
                'no2':  c.get('no2',  HANOI_AVG['nitrogen_dioxide']),
                'so2':  c.get('so2',  HANOI_AVG['sulphur_dioxide']),
                'o3':   c.get('o3',   HANOI_AVG['ozone']),
            }
    except Exception as e:
        print(f"[WARN] OWM current lỗi: {e}")
    return {
        'pm10': HANOI_AVG['pm10'],
        'co':   HANOI_AVG['carbon_monoxide'],
        'no2':  HANOI_AVG['nitrogen_dioxide'],
        'so2':  HANOI_AVG['sulphur_dioxide'],
        'o3':   HANOI_AVG['ozone'],
    }


def get_owm_forecast_tomorrow():
    """
    Lấy DỰ BÁO ô nhiễm ngày mai tại Hà Nội từ OWM forecast API.
    Trả về trung bình các giờ trong ngày mai.
    """
    url = (f"http://api.openweathermap.org/data/2.5/air_pollution/forecast"
           f"?lat={LAT}&lon={LON}&appid={OWM_API_KEY}")
    try:
        res = requests.get(url, timeout=5)
        if res.status_code == 200:
            all_data   = res.json()['list']
            tomorrow   = (datetime.now() + timedelta(days=1)).date()

            tomorrow_entries = [
                e for e in all_data
                if datetime.fromtimestamp(e['dt']).date() == tomorrow
            ]

            if tomorrow_entries:
                keys = ['pm2_5', 'pm10', 'co', 'no2', 'so2', 'o3']
                avg  = {}
                for k in keys:
                    vals   = [e['components'].get(k, 0) for e in tomorrow_entries]
                    avg[k] = sum(vals) / len(vals) if vals else 0

                print(f"[OWM FORECAST] {len(tomorrow_entries)} entries cho {tomorrow}")
                print(f"  PM2.5={avg['pm2_5']:.1f} PM10={avg['pm10']:.1f} "
                      f"NO2={avg['no2']:.1f} O3={avg['o3']:.1f}")
                return avg
    except Exception as e:
        print(f"[WARN] OWM forecast lỗi: {e}")

    # Fallback về Hà Nội avg
    return {
        'pm2_5': HANOI_AVG['pm10'] / 1.305,
        'pm10':  HANOI_AVG['pm10'],
        'co':    HANOI_AVG['carbon_monoxide'],
        'no2':   HANOI_AVG['nitrogen_dioxide'],
        'so2':   HANOI_AVG['sulphur_dioxide'],
        'o3':    HANOI_AVG['ozone'],
    }


def get_daily_pm25_history():
    """Lấy PM2.5 cuối ngày của 3 ngày gần nhất để tính lag/rolling."""
    pm_list = []
    for days_ago in range(3, 0, -1):
        date_str = (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d")
        url = f"{FIREBASE_URL}/sensor_data/esp32_01/{date_str}.json"
        try:
            res = requests.get(url, timeout=5)
            if res.status_code == 200 and res.json():
                data    = res.json()
                last_k  = sorted(data.keys())[-1]
                pm_val  = float(data[last_k].get('pm_ug_m3', 0))
                if pm_val > 0:
                    pm_list.append(pm_val)
        except Exception as e:
            print(f"[WARN] Firebase {date_str}: {e}")
    return pm_list


def get_advice(aqi: int) -> str:
    if aqi <= 50:
        return "Không khí trong lành, yên tâm hoạt động ngoài trời"
    elif aqi <= 100:
        return "Chấp nhận được, nhóm nhạy cảm nên hạn chế ra ngoài lâu"
    elif aqi <= 150:
        return "Nhóm nhạy cảm (trẻ em, người già) nên ở trong nhà"
    elif aqi <= 200:
        return "Ảnh hưởng sức khỏe rõ rệt, hạn chế hoạt động ngoài trời"
    elif aqi <= 300:
        return "Cảnh báo khẩn cấp, đeo khẩu trang N95 nếu ra ngoài"
    return "Nguy hiểm! Tuyệt đối ở trong nhà, đóng kín cửa"


def build_features(pm2_5, owm, history, target_date, history_pm_for_lag=None):
    """
    Xây dựng 28-feature vector cho XGBoost.
    target_date: ngày dự đoán (datetime object)
    """
    history_all = (history_pm_for_lag or history) + [pm2_5]

    lag1 = history[-1] if len(history) >= 1 else pm2_5
    lag2 = history[-2] if len(history) >= 2 else lag1

    roll3_vals = history_all[-3:]
    roll7_vals = history_all[-7:]
    roll3 = sum(roll3_vals) / len(roll3_vals)
    roll7 = sum(roll7_vals) / len(roll7_vals)

    pm10     = owm.get('pm10', HANOI_AVG['pm10'])
    pm_ratio = pm2_5 / (pm10 + 1)

    return {
        'pm10':                  pm10,
        'pm2_5':                 pm2_5,
        'carbon_monoxide':       owm.get('co',  HANOI_AVG['carbon_monoxide']),
        'nitrogen_dioxide':      owm.get('no2', HANOI_AVG['nitrogen_dioxide']),
        'sulphur_dioxide':       owm.get('so2', HANOI_AVG['sulphur_dioxide']),
        'ozone':                 owm.get('o3',  HANOI_AVG['ozone']),
        'aerosol_optical_depth': HANOI_AVG['aerosol_optical_depth'],
        'dust':                  HANOI_AVG['dust'],
        'uv_index':              HANOI_AVG['uv_index'],
        'day':       target_date.day,
        'month':     target_date.month,
        'year':      target_date.year,
        'dayofweek': target_date.weekday(),
        'pm2_5_lag1': lag1, 'pm2_5_lag2': lag2,
        'pm10_lag1':  pm10,  'pm10_lag2':  pm10,
        'co_lag1':  owm.get('co',  HANOI_AVG['carbon_monoxide']),
        'co_lag2':  owm.get('co',  HANOI_AVG['carbon_monoxide']),
        'no2_lag1': owm.get('no2', HANOI_AVG['nitrogen_dioxide']),
        'no2_lag2': owm.get('no2', HANOI_AVG['nitrogen_dioxide']),
        'so2_lag1': owm.get('so2', HANOI_AVG['sulphur_dioxide']),
        'so2_lag2': owm.get('so2', HANOI_AVG['sulphur_dioxide']),
        'o3_lag1':  owm.get('o3',  HANOI_AVG['ozone']),
        'o3_lag2':  owm.get('o3',  HANOI_AVG['ozone']),
        'pm2_5_roll3': roll3,
        'pm2_5_roll7': roll7,
        'pm_ratio':    pm_ratio,
    }


# ==============================================================================
# 🧠 ENDPOINT CHÍNH - trả về CẢ 2 dự đoán trong 1 lần gọi
# ==============================================================================
@app.route('/predict', methods=['POST'])
def predict():
    try:
        req_data     = request.json or {}
        current_pm25 = float(req_data.get('pm25', 0))
        if current_pm25 <= 0:
            return jsonify({'error': 'PM2.5 không hợp lệ'}), 400

        print(f"\n[REQUEST] PM2.5={current_pm25}")

        # Lấy dữ liệu từ OWM và Firebase
        owm_current  = get_owm_current()
        owm_forecast = get_owm_forecast_tomorrow()
        history      = get_daily_pm25_history()

        today    = datetime.now()
        tomorrow = today + timedelta(days=1)

        # ── DỰ BÁO 1: Vị trí bạn (dùng sensor Sharp + OWM hiện tại) ──────────
        feats_local = build_features(current_pm25, owm_current, history, today)
        df_local    = pd.DataFrame([feats_local])[EXPECTED_COLUMNS]
        aqi_local   = int(max(0, min(500, round(float(model.predict(df_local)[0])))))
        adv_local   = get_advice(aqi_local)
        print(f"[LOCAL]  AQI ngày mai tại vị trí bạn = {aqi_local}")

        # ── DỰ BÁO 2: Hà Nội ngày mai (dùng OWM forecast + XGBoost) ──────────
        # Dùng OWM forecast pm2_5 thay vì sensor của Tiến
        hanoi_pm25 = owm_forecast.get('pm2_5', HANOI_AVG['pm10'] / 1.305)

        # lag cho Hanoi: dùng lịch sử sensor làm proxy (cùng địa bàn Hà Nội)
        feats_hanoi = build_features(hanoi_pm25, owm_forecast, history, tomorrow)
        df_hanoi    = pd.DataFrame([feats_hanoi])[EXPECTED_COLUMNS]
        aqi_hanoi   = int(max(0, min(500, round(float(model.predict(df_hanoi)[0])))))
        adv_hanoi   = get_advice(aqi_hanoi)
        print(f"[HANOI]  AQI Hà Nội ngày mai = {aqi_hanoi} (PM2.5 forecast={hanoi_pm25:.1f})")

        # ── Ghi cả 2 kết quả lên Firebase ────────────────────────────────────
        firebase_payload = {
            "local": {
                "aqi":    aqi_local,
                "advice": adv_local,
            },
            "hanoi": {
                "aqi":    aqi_hanoi,
                "pm25":   round(hanoi_pm25, 1),
                "advice": adv_hanoi,
            }
        }
        fb_res = requests.put(
            f"{FIREBASE_URL}/ai_forecast.json",
            json=firebase_payload,
            timeout=5
        )
        print(f"[FIREBASE] Write {'OK' if fb_res.status_code == 200 else 'FAIL'}")

        return jsonify({
            'status': 'success',
            'local':  {'aqi': aqi_local,  'advice': adv_local},
            'hanoi':  {'aqi': aqi_hanoi,  'pm25': round(hanoi_pm25, 1), 'advice': adv_hanoi},
        })

    except Exception as e:
        print(f"[ERROR] {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'time': datetime.now().isoformat()})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
