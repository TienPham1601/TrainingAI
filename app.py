import os
import requests
import joblib
import pandas as pd
from datetime import datetime, timedelta
import pytz
from flask import Flask, request, jsonify

app = Flask(__name__)

# ==============================================================================
# ⚙️ CẤU HÌNH HỆ THỐNG - KHÓA CHẶT TẠI HÀ NỘI
# ==============================================================================
model        = joblib.load('aqi_model.pkl')
FIREBASE_URL = "https://iotbytienpham-default-rtdb.firebaseio.com"
OWM_API_KEY  = "18efd3b6d037e0b3f24c0e16dcb09180" 
AQICN_TOKEN  = "a63cb2de5e93820f8a97c997d0e650dcad7d6aea"

LAT, LON     = 21.0285, 105.8542 # TỌA ĐỘ CHUẨN CỦA HÀ NỘI
# LƯU Ý CHO BRO: Trong chuẩn quốc tế IANA, toàn bộ Việt Nam (GMT+7) dùng chung mã "Asia/Ho_Chi_Minh". Không có mã Hanoi.
VN_TZ        = pytz.timezone('Asia/Ho_Chi_Minh') 

EXPECTED_COLUMNS = ['pm10', 'pm2_5', 'carbon_monoxide', 'nitrogen_dioxide', 'sulphur_dioxide', 'ozone', 'aerosol_optical_depth', 'dust', 'uv_index', 'day', 'month', 'year', 'dayofweek', 'pm2_5_lag1', 'pm2_5_lag2', 'pm10_lag1', 'pm10_lag2', 'co_lag1', 'co_lag2', 'no2_lag1', 'no2_lag2', 'so2_lag1', 'so2_lag2', 'o3_lag1', 'o3_lag2', 'pm2_5_roll3', 'pm2_5_roll7', 'pm_ratio']
HANOI_AVG = {'pm10': 57.27, 'carbon_monoxide': 725.83, 'nitrogen_dioxide': 27.15, 'sulphur_dioxide': 25.71, 'ozone': 72.52, 'aerosol_optical_depth': 0.642, 'dust': 0.491, 'uv_index': 1.172}

def get_advice(aqi: int) -> str:
    if aqi <= 50: return "✅ Tuyệt vời! Không khí trong lành, lý tưởng để hoạt động ngoài trời."
    elif aqi <= 100: return "⚠️ Chấp nhận được. Nhóm người nhạy cảm nên cân nhắc nếu ở ngoài quá lâu."
    elif aqi <= 150: return "😷 Kém. Trẻ em, người già và người có bệnh hô hấp cần hạn chế ra ngoài."
    elif aqi <= 200: return "🚨 XẤU! Ảnh hưởng sức khỏe toàn dân. Bắt buộc đeo khẩu trang N95."
    return "🔴 NGUY HIỂM! Mức báo động. Hãy ở trong nhà và bật máy lọc không khí."

def get_aqi_color(aqi: int) -> str:
    if aqi <= 50: return "#15803d" 
    elif aqi <= 100: return "#b45309" 
    elif aqi <= 150: return "#c2410c" 
    elif aqi <= 200: return "#dc2626" 
    return "#7c3aed" 

def get_owm_current():
    try:
        res = requests.get(f"http://api.openweathermap.org/data/2.5/air_pollution?lat={LAT}&lon={LON}&appid={OWM_API_KEY}", timeout=5).json()
        c = res['list'][0]['components']
        return {'pm10': c.get('pm10', HANOI_AVG['pm10']), 'co': c.get('co', HANOI_AVG['carbon_monoxide']), 'no2': c.get('no2', HANOI_AVG['nitrogen_dioxide']), 'so2': c.get('so2', HANOI_AVG['sulphur_dioxide']), 'o3': c.get('o3', HANOI_AVG['ozone'])}
    except:
        return {'pm10': HANOI_AVG['pm10'], 'co': HANOI_AVG['carbon_monoxide'], 'no2': HANOI_AVG['nitrogen_dioxide'], 'so2': HANOI_AVG['sulphur_dioxide'], 'o3': HANOI_AVG['ozone']}

def get_daily_pm25_history():
    pm_list = []
    for days_ago in range(3, 0, -1):
        date_str = (datetime.now(VN_TZ) - timedelta(days=days_ago)).strftime("%Y-%m-%d")
        url = f"{FIREBASE_URL}/sensor_data/esp32_01/{date_str}.json"
        try:
            res = requests.get(url, timeout=5)
            if res.status_code == 200 and res.json():
                data = res.json()
                last_k = sorted(data.keys())[-1]
                pm_val = float(data[last_k].get('pm_ug_m3', 0))
                if pm_val > 0: pm_list.append(pm_val)
        except: pass
    return pm_list

def build_features(pm2_5, owm, history, target_date):
    history_all = history + [pm2_5]
    lag1 = history[-1] if len(history) >= 1 else pm2_5
    lag2 = history[-2] if len(history) >= 2 else lag1
    roll3_vals = history_all[-3:] if len(history_all) >=3 else [pm2_5]*3
    roll7_vals = history_all[-7:] if len(history_all) >=7 else [pm2_5]*7
    roll3 = sum(roll3_vals) / len(roll3_vals)
    roll7 = sum(roll7_vals) / len(roll7_vals)
    pm10 = owm.get('pm10', HANOI_AVG['pm10'])
    return {
        'pm10': pm10, 'pm2_5': pm2_5,
        'carbon_monoxide': owm['co'], 'nitrogen_dioxide': owm['no2'], 'sulphur_dioxide': owm['so2'], 'ozone': owm['o3'],
        'aerosol_optical_depth': HANOI_AVG['aerosol_optical_depth'], 'dust': HANOI_AVG['dust'], 'uv_index': HANOI_AVG['uv_index'],
        'day': target_date.day, 'month': target_date.month, 'year': target_date.year, 'dayofweek': target_date.weekday(),
        'pm2_5_lag1': lag1, 'pm2_5_lag2': lag2, 'pm10_lag1': pm10, 'pm10_lag2': pm10,
        'co_lag1': owm['co'], 'co_lag2': owm['co'], 'no2_lag1': owm['no2'], 'no2_lag2': owm['no2'],
        'so2_lag1': owm['so2'], 'so2_lag2': owm['so2'], 'o3_lag1': owm['o3'], 'o3_lag2': owm['o3'],
        'pm2_5_roll3': roll3, 'pm2_5_roll7': roll7, 'pm_ratio': pm2_5 / (pm10 + 1)
    }

@app.route('/predict', methods=['POST'])
def predict():
    try:
        req_data = request.json or {}
        current_pm25 = float(req_data.get('pm25', 0))
        if current_pm25 <= 0: return jsonify({'error': 'Invalid PM2.5'}), 400

        print(f"\n[AI] Nhận Sensor PM2.5 từ ESP32 = {current_pm25}")
        today_vn = datetime.now(VN_TZ)

        # 1. AI XGBOOST
        owm_current = get_owm_current()
        history     = get_daily_pm25_history()
        feats_local = build_features(current_pm25, owm_current, history, today_vn)
        df_local    = pd.DataFrame([feats_local])[EXPECTED_COLUMNS]
        aqi_local   = int(max(0, min(500, round(float(model.predict(df_local)[0])))))
        adv_local   = get_advice(aqi_local)

        # 2. AQICN HÀ NỘI
        resp_aqicn = requests.get(f"https://api.waqi.info/feed/hanoi/?token={AQICN_TOKEN}", timeout=10).json()
        days_list, hanoi_tomorrow = [], {}
        
        if resp_aqicn.get('status') == 'ok':
            daily_pm25 = resp_aqicn['data']['forecast']['daily']['pm25']
            daily_temp = resp_aqicn['data']['forecast']['daily'].get('t', []) 
            temp_dict = {t['day']: t for t in daily_temp}

            for item in daily_pm25:
                date_str = item['day']
                date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()
                if date_obj < today_vn.date(): continue 
                
                aqi_val = int(item['avg'])
                t_data = temp_dict.get(date_str, {})
                
                day_item = {
                    "date_label": date_obj.strftime("%d/%m/%Y"), 
                    "day_label": "Hôm nay" if date_obj == today_vn.date() else "Ngày mai" if date_obj == (today_vn + timedelta(days=1)).date() else date_obj.strftime("%A"),
                    "is_today": date_obj == today_vn.date(),
                    "is_tomorrow": date_obj == (today_vn + timedelta(days=1)).date(),
                    "aqi": aqi_val,
                    "aqi_color": get_aqi_color(aqi_val),
                    "temp_max": int(t_data.get('max', 0)) if t_data else "--",
                }
                
                vn_days = {"Monday":"Thứ 2", "Tuesday":"Thứ 3", "Wednesday":"Thứ 4", "Thursday":"Thứ 5", "Friday":"Thứ 6", "Saturday":"Thứ 7", "Sunday":"Chủ Nhật"}
                if day_item['day_label'] in vn_days: day_item['day_label'] = vn_days[day_item['day_label']]
                days_list.append(day_item)
                if day_item['is_tomorrow']: hanoi_tomorrow = day_item

        # 3. ĐẨY LÊN FIREBASE
        firebase_payload = {
            "ai_forecast": {
                "local": { "aqi": aqi_local, "color": get_aqi_color(aqi_local), "advice": adv_local },
                "hanoi_tomorrow": { "aqi": hanoi_tomorrow.get('aqi', '--'), "color": hanoi_tomorrow.get('aqi_color', '#64748b'), "date_label": hanoi_tomorrow.get('date_label', '--'), "advice": get_advice(hanoi_tomorrow.get('aqi', 0)) }
            },
            "weather_7day": { "updated": today_vn.strftime("%H:%M:%S - %d/%m/%Y"), "days": days_list[:7] }
        }
        
        fb_res = requests.patch(f"{FIREBASE_URL}/.json", json=firebase_payload, timeout=8)
        print(f"[FIREBASE] Đã cập nhật AI Local và Forecast 7 ngày Hà Nội: HTTP {fb_res.status_code}")
        return jsonify({'status': 'success'})

    except Exception as e:
        print(f"[ERROR] {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
