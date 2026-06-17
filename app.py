import os, time, requests, joblib, pandas as pd
from datetime import datetime, timedelta
from flask import Flask, request, jsonify

app = Flask(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
model        = joblib.load('aqi_model.pkl')
FIREBASE_URL = "https://iotbytienpham-default-rtdb.firebaseio.com"
FIREBASE_AUTH= "BaP7SSMKkNLJkVFFcXpqYPJOpKQBWFnNknESzsoH" # Đã thêm khóa bảo mật
OWM_KEY      = "18efd3b6d037e0b3f24c0e16dcb09180"
AQICN_TOKEN  = "a63cb2de5e93820f8a97c997d0e650dcad7d6aea"
LAT, LON     = 21.0285, 105.8542

EXPECTED_COLUMNS = [
    'pm10','pm2_5','carbon_monoxide','nitrogen_dioxide','sulphur_dioxide',
    'ozone','aerosol_optical_depth','dust','uv_index',
    'day','month','year','dayofweek',
    'pm2_5_lag1','pm2_5_lag2','pm10_lag1','pm10_lag2',
    'co_lag1','co_lag2','no2_lag1','no2_lag2',
    'so2_lag1','so2_lag2','o3_lag1','o3_lag2',
    'pm2_5_roll3','pm2_5_roll7','pm_ratio'
]

HANOI_AVG = {
    'pm10':57.27,'carbon_monoxide':725.83,'nitrogen_dioxide':27.15,
    'sulphur_dioxide':25.71,'ozone':72.52,
    'aerosol_optical_depth':0.642,'dust':0.491,'uv_index':1.172,
}

VN_DAYS = ['Thứ Hai','Thứ Ba','Thứ Tư','Thứ Năm','Thứ Sáu','Thứ Bảy','Chủ Nhật']

WEATHER_EMOJI = {
    '01':'☀️','02':'⛅','03':'🌥️','04':'☁️',
    '09':'🌧️','10':'🌦️','11':'⛈️','13':'❄️','50':'🌫️'
}

# Cache 7-day forecast (update mỗi giờ)
_forecast_cache = None
_forecast_time  = 0

# ── Helpers ───────────────────────────────────────────────────────────────────
def pm25_to_aqi(pm25):
    bps = [(0,12,0,50),(12.1,35.4,51,100),(35.5,55.4,101,150),
           (55.5,150.4,151,200),(150.5,250.4,201,300),
           (250.5,350.4,301,400),(350.5,500.4,401,500)]
    for lo,hi,alo,ahi in bps:
        if lo <= pm25 <= hi:
            return int(round((ahi-alo)/(hi-lo)*(pm25-lo)+alo))
    return 500

def aqi_level_info(aqi):
    if aqi <= 50:  return ('Tốt',       '#15803d')
    if aqi <= 100: return ('Trung bình', '#b45309')
    if aqi <= 150: return ('Kém',        '#c2410c')
    if aqi <= 200: return ('Xấu',        '#dc2626')
    if aqi <= 300: return ('Rất xấu',    '#7c3aed')
    return                ('Nguy hiểm',  '#1e1b4b')

def w_emoji(icon): return WEATHER_EMOJI.get(icon[:2], '🌤️')

def get_advice(aqi):
    if aqi <= 50:  return "Không khí trong lành, yên tâm hoạt động ngoài trời"
    if aqi <= 100: return "Chấp nhận được, nhóm nhạy cảm nên hạn chế ra ngoài"
    if aqi <= 150: return "Nhóm nhạy cảm (trẻ em, người già) nên ở trong nhà"
    if aqi <= 200: return "Ảnh hưởng sức khỏe rõ rệt, hạn chế hoạt động ngoài"
    if aqi <= 300: return "Cảnh báo khẩn, đeo khẩu trang N95 nếu buộc ra ngoài"
    return "Nguy hiểm! Tuyệt đối ở trong nhà, đóng kín cửa"

# ── Data sources ──────────────────────────────────────────────────────────────
def get_owm_current():
    try:
        r = requests.get(
            f"http://api.openweathermap.org/data/2.5/air_pollution?lat={LAT}&lon={LON}&appid={OWM_KEY}",
            timeout=5)
        if r.status_code == 200:
            c = r.json()['list'][0]['components']
            return {'pm10':c.get('pm10',HANOI_AVG['pm10']),
                    'co':  c.get('co',  HANOI_AVG['carbon_monoxide']),
                    'no2': c.get('no2', HANOI_AVG['nitrogen_dioxide']),
                    'so2': c.get('so2', HANOI_AVG['sulphur_dioxide']),
                    'o3':  c.get('o3',  HANOI_AVG['ozone'])}
    except Exception as e: print(f"[WARN] OWM current: {e}")
    return {'pm10':HANOI_AVG['pm10'],'co':HANOI_AVG['carbon_monoxide'],
            'no2':HANOI_AVG['nitrogen_dioxide'],'so2':HANOI_AVG['sulphur_dioxide'],
            'o3':HANOI_AVG['ozone']}

def get_daily_pm25_history():
    pm_list = []
    for days_ago in range(3, 0, -1):
        ds = (datetime.now()-timedelta(days=days_ago)).strftime("%Y-%m-%d")
        try:
            # Đã gắn thẻ Auth vào đường link GET
            r = requests.get(f"{FIREBASE_URL}/sensor_data/esp32_01/{ds}.json?auth={FIREBASE_AUTH}", timeout=5)
            if r.status_code == 200 and r.json():
                data = r.json()
                lk   = sorted(data.keys())[-1]
                pm   = float(data[lk].get('pm_ug_m3', 0))
                if pm > 0: pm_list.append(pm)
        except: pass
    return pm_list

def get_aqicn_data():
    """Lấy AQI hiện tại + 7-day PM2.5 forecast từ AQICN (trạm đo thực tế Hà Nội)."""
    try:
        r = requests.get(
            f"https://api.waqi.info/feed/hanoi/?token={AQICN_TOKEN}",
            timeout=8)
        if r.status_code == 200:
            d = r.json()
            if d['status'] == 'ok':
                current_aqi   = int(d['data']['aqi'])
                forecast_pm25 = []
                if 'forecast' in d['data'] and 'daily' in d['data']['forecast']:
                    forecast_pm25 = d['data']['forecast']['daily'].get('pm25', [])
                print(f"[AQICN] Current AQI={current_aqi}, forecast days={len(forecast_pm25)}")
                return current_aqi, forecast_pm25
    except Exception as e: print(f"[WARN] AQICN: {e}")
    return None, []

def get_owm_weather_forecast():
    """OWM 5-day forecast, aggregate theo ngày (UTC+7)."""
    try:
        r = requests.get(
            f"https://api.openweathermap.org/data/2.5/forecast?lat={LAT}&lon={LON}&appid={OWM_KEY}&units=metric",
            timeout=5)
        if r.status_code == 200:
            by_date = {}
            for e in r.json()['list']:
                dt_local = datetime.utcfromtimestamp(e['dt']) + timedelta(hours=7)
                dk = dt_local.strftime('%Y-%m-%d')
                by_date.setdefault(dk, []).append(e)
            result = {}
            for dk, entries in by_date.items():
                temps  = [e['main']['temp']     for e in entries]
                humids = [e['main']['humidity']  for e in entries]
                mid    = next(
                    (e for e in entries
                     if '11' <= (datetime.utcfromtimestamp(e['dt'])+timedelta(hours=7)).strftime('%H') < '14'),
                    entries[len(entries)//2])
                result[dk] = {
                    'temp_max': round(max(temps)),
                    'temp_min': round(min(temps)),
                    'humidity': round(sum(humids)/len(humids)),
                    'emoji':    w_emoji(mid['weather'][0]['icon']),
                    'desc':     mid['weather'][0]['description'].capitalize(),
                }
            return result
    except Exception as e: print(f"[WARN] OWM weather: {e}")
    return {}

def build_7day_forecast():
    """Kết hợp AQICN (AQI chính xác) + OWM (nhiệt độ, thời tiết)."""
    current_aqi, aqicn_pm25 = get_aqicn_data()
    owm_weather = get_owm_weather_forecast()

    today = datetime.now().date()

    # Map ngày → PM2.5 avg từ AQICN
    aqicn_map = {}
    for entry in aqicn_pm25:
        try: aqicn_map[entry['day']] = float(entry['avg'])
        except: pass

    days = []
    for i in range(7):
        d     = today + timedelta(days=i)
        d_str = d.strftime('%Y-%m-%d')

        # AQI: dùng AQICN hiện tại cho hôm nay, forecast cho các ngày sau
        if i == 0 and current_aqi:
            aqi = current_aqi
        elif d_str in aqicn_map:
            aqi = pm25_to_aqi(aqicn_map[d_str])
        else:
            aqi = 0

        level, color = aqi_level_info(aqi) if aqi > 0 else ('N/A', '#64748b')
        w = owm_weather.get(d_str, {})

        if i == 0:   day_label = 'Hôm nay'
        elif i == 1: day_label = 'Ngày mai'
        else:        day_label = VN_DAYS[d.weekday()]

        days.append({
            'date':       d_str,
            'day_label':  day_label,
            'date_label': d.strftime('%d/%m'),
            'weekday':    VN_DAYS[d.weekday()],
            'is_today':   i == 0,
            'is_tomorrow':i == 1,
            'aqi':        aqi,
            'aqi_label':  level,
            'aqi_color':  color,
            'pm25':       round(aqicn_map.get(d_str, 0), 1),
            'temp_max':   w.get('temp_max', '--'),
            'temp_min':   w.get('temp_min', '--'),
            'humidity':   w.get('humidity', '--'),
            'emoji':      w.get('emoji', '🌤️'),
            'desc':       w.get('desc', ''),
        })

    return {
        'updated':          datetime.now().strftime('%d/%m/%Y %H:%M'),
        'hanoi_current_aqi':current_aqi or 0,
        'days':             days,
    }

def get_or_refresh_forecast():
    """Cache 7-day forecast, chỉ rebuild mỗi 1 tiếng."""
    global _forecast_cache, _forecast_time
    now = time.time()
    if _forecast_cache is None or (now - _forecast_time) > 3600:
        print("[FORECAST] Refreshing 7-day forecast...")
        _forecast_cache = build_7day_forecast()
        _forecast_time  = now
        try:
            # Đã gắn thẻ Auth vào đường link PUT
            requests.put(f"{FIREBASE_URL}/weather_7day.json?auth={FIREBASE_AUTH}",
                        json=_forecast_cache, timeout=5)
            print("[FIREBASE] /weather_7day updated OK")
        except Exception as e:
            print(f"[WARN] Firebase weather write: {e}")
    return _forecast_cache

def build_features(pm2_5, owm, history, target_date):
    h    = history + [pm2_5]
    lag1 = history[-1] if history else pm2_5
    lag2 = history[-2] if len(history) >= 2 else lag1
    r3   = sum(h[-3:])/len(h[-3:]) if h else pm2_5
    r7   = sum(h[-7:])/len(h[-7:]) if h else pm2_5
    pm10 = owm.get('pm10', HANOI_AVG['pm10'])
    return {
        'pm10':pm10,'pm2_5':pm2_5,
        'carbon_monoxide':  owm.get('co',  HANOI_AVG['carbon_monoxide']),
        'nitrogen_dioxide': owm.get('no2', HANOI_AVG['nitrogen_dioxide']),
        'sulphur_dioxide':  owm.get('so2', HANOI_AVG['sulphur_dioxide']),
        'ozone':            owm.get('o3',  HANOI_AVG['ozone']),
        'aerosol_optical_depth':HANOI_AVG['aerosol_optical_depth'],
        'dust':HANOI_AVG['dust'],'uv_index':HANOI_AVG['uv_index'],
        'day':target_date.day,'month':target_date.month,
        'year':target_date.year,'dayofweek':target_date.weekday(),
        'pm2_5_lag1':lag1,'pm2_5_lag2':lag2,
        'pm10_lag1':pm10,'pm10_lag2':pm10,
        'co_lag1':owm.get('co',HANOI_AVG['carbon_monoxide']),
        'co_lag2':owm.get('co',HANOI_AVG['carbon_monoxide']),
        'no2_lag1':owm.get('no2',HANOI_AVG['nitrogen_dioxide']),
        'no2_lag2':owm.get('no2',HANOI_AVG['nitrogen_dioxide']),
        'so2_lag1':owm.get('so2',HANOI_AVG['sulphur_dioxide']),
        'so2_lag2':owm.get('so2',HANOI_AVG['sulphur_dioxide']),
        'o3_lag1':owm.get('o3',HANOI_AVG['ozone']),
        'o3_lag2':owm.get('o3',HANOI_AVG['ozone']),
        'pm2_5_roll3':r3,'pm2_5_roll7':r7,
        'pm_ratio':pm2_5/(pm10+1),
    }

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/predict', methods=['POST'])
def predict():
    try:
        current_pm25 = float((request.json or {}).get('pm25', 0))
        if current_pm25 <= 0:
            return jsonify({'error': 'PM2.5 không hợp lệ'}), 400

        print(f"\n[REQUEST] PM2.5={current_pm25}")

        owm     = get_owm_current()
        history = get_daily_pm25_history()
        today   = datetime.now()

        # XGBoost dự đoán AQI tại vị trí sensor
        feats     = build_features(current_pm25, owm, history, today)
        df        = pd.DataFrame([feats])[EXPECTED_COLUMNS]
        aqi_local = int(max(0, min(500, round(float(model.predict(df)[0])))))
        adv_local = get_advice(aqi_local)
        print(f"[LOCAL] AQI={aqi_local}")

        # Đã gắn thẻ Auth vào đường link PUT
        requests.put(f"{FIREBASE_URL}/ai_forecast.json?auth={FIREBASE_AUTH}",
                    json={"local": {"aqi": aqi_local, "advice": adv_local}},
                    timeout=5)

        # Refresh 7-day forecast nếu cache hết hạn
        get_or_refresh_forecast()

        return jsonify({'status': 'success', 'local_aqi': aqi_local})

    except Exception as e:
        print(f"[ERROR] {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'time': datetime.now().isoformat()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 10000)))
