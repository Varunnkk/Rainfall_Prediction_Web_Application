import os, json, pickle, warnings, math
import numpy as np
from flask import Flask, jsonify, request, send_from_directory
warnings.filterwarnings('ignore')

app = Flask(__name__, static_folder='static')

BASE = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE, 'best_model.pkl'), 'rb') as f:
    payload = pickle.load(f)

MODEL    = payload['model']
SCALER   = payload['scaler']
FEATURES = payload['features']

with open(os.path.join(BASE, 'project_data.json')) as f:
    DATA = json.load(f)

print(f"✅ Model loaded: {payload['model_name']}")

def predict_for_date(date_str):
    from datetime import datetime
    d     = datetime.strptime(date_str, '%Y-%m-%d')
    doy   = d.timetuple().tm_yday
    month = d.month
    week  = min(53, (doy - 1) // 7 + 1)
    clim  = DATA['climatology'][str(month)]
    feat_vec = [
        d.year, doy, month, week,
        math.sin(2 * math.pi * doy / 365),
        math.cos(2 * math.pi * doy / 365),
        clim['ALLSKY_SFC_SW_DWN'],
        clim['T2M'], clim['T2MDEW'], clim['T2MWET'],
        clim['T2M_MAX'], clim['T2M_MIN'],
        clim['RH2M'], clim['QV2M'], clim['WS2M'], clim['GWETTOP']
    ]
    X    = SCALER.transform([feat_vec])
    pred = float(np.clip(MODEL.predict(X)[0], 0, None))
    return round(pred, 3), clim

@app.route('/api/overview')
def api_overview():
    return jsonify({
        'best_model'    : DATA['best_model'],
        'r2'            : DATA['r2'],
        'rmse'          : DATA['rmse'],
        'mae'           : DATA['mae'],
        'model_results' : DATA['model_results'],
        'hist_monthly'  : DATA['hist_monthly'],
        'hist_yearly'   : DATA['hist_yearly'],
    })

@app.route('/api/future')
def api_future():
    return jsonify({'predictions': DATA['future_predictions']})

@app.route('/api/predict')
def api_predict():
    date_str = request.args.get('date', '')
    if not date_str:
        return jsonify({'error': 'Please provide ?date=YYYY-MM-DD'}), 400
    try:
        pred, clim = predict_for_date(date_str)
        if pred < 0.5:   intensity, level = 'No Rain / Trace', 0
        elif pred < 2.5: intensity, level = 'Light Rain', 1
        elif pred < 7.5: intensity, level = 'Moderate Rain', 2
        elif pred < 15:  intensity, level = 'Heavy Rain', 3
        else:            intensity, level = 'Very Heavy Rain', 4
        return jsonify({
            'date'                  : date_str,
            'predicted_rainfall_mm' : pred,
            'intensity'             : intensity,
            'level'                 : level,
            'weather_context': {
                'temperature_C' : round(clim['T2M'], 1),
                'max_temp_C'    : round(clim['T2M_MAX'], 1),
                'min_temp_C'    : round(clim['T2M_MIN'], 1),
                'humidity_pct'  : round(clim['RH2M'], 1),
                'dew_point_C'   : round(clim['T2MDEW'], 1),
                'wind_speed_ms' : round(clim['WS2M'], 2),
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/range')
def api_range():
    start = request.args.get('start', '')
    end   = request.args.get('end', '')
    if not start or not end:
        return jsonify({'error': 'Provide ?start=YYYY-MM-DD&end=YYYY-MM-DD'}), 400
    from datetime import datetime, timedelta
    try:
        s = datetime.strptime(start, '%Y-%m-%d')
        e = datetime.strptime(end,   '%Y-%m-%d')
        if (e - s).days > 365:
            return jsonify({'error': 'Max range is 365 days'}), 400
        results = []
        cur = s
        while cur <= e:
            ds   = cur.strftime('%Y-%m-%d')
            pred, clim = predict_for_date(ds)
            results.append({'date': ds, 'predicted_rainfall_mm': pred, 'temp_C': round(clim['T2M'], 1)})
            cur += timedelta(days=1)
        total = round(sum(r['predicted_rainfall_mm'] for r in results), 1)
        rainy = sum(1 for r in results if r['predicted_rainfall_mm'] > 0.5)
        return jsonify({'start': start, 'end': end, 'days': len(results),
                        'total_rainfall_mm': total, 'rainy_days': rainy, 'daily': results})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/test_records')
def api_test_records():
    return jsonify({'records': DATA['test_records']})

@app.route('/')
def index():
    return send_from_directory(os.path.join(BASE, 'static'), 'index.html')

if __name__ == '__main__':
    print("🌧️  Open browser → http://localhost:5000")
    app.run(host='0.0.0.0', port=5001, debug=False)