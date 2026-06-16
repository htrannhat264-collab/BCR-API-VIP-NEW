import requests
from urllib.parse import unquote
import time
import threading
from flask import Flask, jsonify
import re
import os
import json
from collections import deque, Counter
import random
import math
from statistics import mean, stdev

# ======================
# CẤU HÌNH
# ======================
BASE = "https://aibcr.me"
LOGIN_URL = f"{BASE}/login"
LOBBY_URL = f"{BASE}/ae/lobby"
GETNEWRESULT_URL = f"{BASE}/baccarat/getnewresult"

USERNAME = os.environ.get("BACCARAT_USER", "WangLin20199")
PASSWORD = os.environ.get("BACCARAT_PASS", "WangFlang1")

CONFIG = {
    'min_history': 5,
    'max_history': 100,
    'update_interval': 1,
    'version': '3.0.0',
    'confidence_threshold': 55
}

# ======================
# SESSION
# ======================
session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7"
})

# ======================
# BIẾN TOÀN CỤC
# ======================
history_data = {}  # Lưu lịch sử kết quả
data_cache = []
last_results = {}
last_data = []
running = True
prediction_sessions = {}

# ======================
# LỚP THUẬT TOÁN DỰ ĐOÁN
# ======================
class BaccaratPredictor:
    def __init__(self):
        self.history_size = CONFIG['max_history']
        self.algorithms = {
            'theo_do_chieu': self._method_trend,
            'dao_chieu': self._method_reversal,
            'tan_suat': self._method_frequency,
            'mau_hinh': self._method_pattern,
            'thong_minh': self._method_smart,
            'big_data': self._method_bigdata,
            'ml_basic': self._method_ml,
            'ema_signal': self._method_ema
        }
    
    def update_history(self, table_name, result):
        if table_name not in history_data:
            history_data[table_name] = deque(maxlen=self.history_size)
        history_data[table_name].append(result)
    
    def _get_streak(self, history):
        if not history:
            return []
        streak = []
        last = history[-1]
        for i in range(len(history)-1, -1, -1):
            if history[i] == last:
                streak.append(history[i])
            else:
                break
        return streak
    
    def _method_trend(self, history):
        if len(history) < 2:
            return None, 0
        last = history[-1]
        streak = self._get_streak(history)
        if len(streak) >= 4:
            return last, min(85, 50 + len(streak) * 8)
        elif len(streak) >= 2:
            return last, 55
        else:
            return 'P' if last == 'B' else 'B', 50
    
    def _method_reversal(self, history):
        if not history:
            return None, 0
        last = history[-1]
        prediction = 'P' if last == 'B' else 'B'
        if len(history) >= 3:
            recent = history[-3:]
            if all(x != recent[0] for x in recent[1:]):
                return last, 60
        return prediction, 55
    
    def _method_frequency(self, history):
        if len(history) < 5:
            return None, 0
        recent = history[-10:] if len(history) >= 10 else history
        counts = Counter(recent)
        total = len(recent)
        b_rate = counts.get('B', 0) / total
        p_rate = counts.get('P', 0) / total
        if abs(b_rate - p_rate) > 0.2:
            if b_rate > p_rate:
                return 'B', int(b_rate * 80 + 20)
            else:
                return 'P', int(p_rate * 80 + 20)
        return None, 0
    
    def _method_pattern(self, history):
        if len(history) < 10:
            return None, 0
        pattern_len = 4
        recent_pattern = ''.join(history[-pattern_len:])
        matches = []
        for i in range(0, len(history) - pattern_len):
            pattern = ''.join(history[i:i+pattern_len])
            if pattern == recent_pattern and i + pattern_len < len(history):
                matches.append(history[i+pattern_len])
        if matches:
            next_result = Counter(matches).most_common(1)[0]
            confidence = min(75, 50 + next_result[1] * 8)
            return next_result[0], confidence
        return None, 0
    
    def _method_smart(self, history):
        if len(history) < 3:
            return None, 0
        score = {'B': 0, 'P': 0}
        
        # 1. Xu hướng gần đây
        last_10 = history[-10:] if len(history) >= 10 else history
        b_count = sum(1 for x in last_10 if x == 'B')
        p_count = len(last_10) - b_count
        if b_count > p_count:
            score['B'] += (b_count / len(last_10)) * 30
        else:
            score['P'] += (p_count / len(last_10)) * 30
        
        # 2. Đảo chiều
        if len(history) >= 2:
            if history[-1] == history[-2]:
                score['P' if history[-1] == 'B' else 'B'] += 20
            else:
                score[history[-1]] += 20
        
        # 3. Tổng thể
        total_b = sum(1 for x in history if x == 'B')
        total_p = len(history) - total_b
        if total_b > total_p:
            score['B'] += (total_b / len(history)) * 25
        else:
            score['P'] += (total_p / len(history)) * 25
        
        # 4. Streak
        streak = self._get_streak(history)
        if len(streak) >= 3:
            score['P' if streak[0] == 'B' else 'B'] += 25
        else:
            score[streak[0] if streak else 'B'] += 25
        
        if score['B'] > score['P']:
            return 'B', min(85, score['B'])
        else:
            return 'P', min(85, score['P'])
    
    def _method_bigdata(self, history):
        if len(history) < 15:
            return None, 0
        numeric = [1 if x == 'B' else 0 for x in history]
        mean_val = mean(numeric)
        std_val = stdev(numeric) if len(numeric) > 1 else 0
        cv = std_val / mean_val if mean_val > 0 else 0
        if cv < 0.2:
            if mean_val > 0.6:
                return 'B', 80
            elif mean_val < 0.4:
                return 'P', 80
        return None, 0
    
    def _method_ml(self, history):
        if len(history) < 20:
            return None, 0
        features = []
        labels = []
        for i in range(5, len(history) - 1):
            window = history[i-5:i]
            features.append([1 if x == 'B' else 0 for x in window])
            labels.append(1 if history[i] == 'B' else 0)
        if len(features) < 5:
            return None, 0
        weights = [0] * 5
        for feature, label in zip(features, labels):
            for j in range(5):
                if feature[j] == label:
                    weights[j] += 1
        last_5 = [1 if x == 'B' else 0 for x in history[-5:]]
        score = sum(weights[j] * last_5[j] for j in range(5))
        prediction = 'B' if score > sum(weights) / 2 else 'P'
        confidence = min(80, 50 + abs(score - sum(weights) / 2) / sum(weights) * 30)
        return prediction, confidence
    
    def _method_ema(self, history):
        if len(history) < 10:
            return None, 0
        numeric = [1 if x == 'B' else 0 for x in history]
        def calculate_ema(data, period):
            if len(data) < period:
                return []
            ema = []
            sma = sum(data[:period]) / period
            ema.append(sma)
            multiplier = 2 / (period + 1)
            for i in range(period, len(data)):
                ema.append(data[i] * multiplier + ema[-1] * (1 - multiplier))
            return ema
        ema_5 = calculate_ema(numeric, 5)
        ema_10 = calculate_ema(numeric, 10)
        if len(ema_5) >= 2 and len(ema_10) >= 2:
            if ema_5[-1] > ema_10[-1] and ema_5[-2] <= ema_10[-2]:
                return 'B', 70
            elif ema_5[-1] < ema_10[-1] and ema_5[-2] >= ema_10[-2]:
                return 'P', 70
        return None, 0
    
    def predict(self, table_name, history):
        if len(history) < CONFIG['min_history']:
            return None, 0, {}
        
        all_predictions = {}
        valid_predictions = []
        
        for name, method in self.algorithms.items():
            try:
                pred, conf = method(history)
                if pred:
                    all_predictions[name] = {
                        'du_doan': pred,
                        'do_tin_cay': conf
                    }
                    valid_predictions.append((pred, conf))
            except Exception as e:
                print(f"Lỗi {name}: {e}")
        
        if not valid_predictions:
            return None, 0, all_predictions
        
        pred_counts = Counter([p[0] for p in valid_predictions])
        most_common = pred_counts.most_common(1)[0]
        final_pred = most_common[0]
        
        avg_conf = sum(p[1] for p in valid_predictions) / len(valid_predictions)
        consensus_factor = (most_common[1] / len(all_predictions)) * 30
        final_conf = min(95, avg_conf + consensus_factor)
        
        return final_pred, round(final_conf, 1), all_predictions

# ======================
# TẠO INSTANCE
# ======================
predictor = BaccaratPredictor()

# ======================
# HÀM GET CSRF TOKEN
# ======================
def get_csrf_token(html):
    match = re.search(r'name="_token" value="([^"]+)"', html)
    if match:
        return match.group(1)
    match = re.search(r'<meta name="csrf-token" content="([^"]+)"', html)
    if match:
        return match.group(1)
    return None

# ======================
# ĐĂNG NHẬP
# ======================
def login():
    try:
        print("Đang đăng nhập...")
        r = session.get(LOGIN_URL, timeout=15)
        token = get_csrf_token(r.text)
        payload = {"username": USERNAME, "password": PASSWORD, "action": "Login"}
        if token:
            payload["_token"] = token
        headers = {
            "Referer": LOGIN_URL,
            "Origin": BASE,
            "Content-Type": "application/x-www-form-urlencoded"
        }
        resp = session.post(LOGIN_URL, data=payload, headers=headers, timeout=15)
        print(f"Login: {resp.status_code}")
        return resp.status_code == 200
    except Exception as e:
        print(f"Login error: {e}")
        return False

def go_to_lobby():
    try:
        resp = session.get(LOBBY_URL, timeout=15)
        print(f"Lobby: {resp.status_code}")
        return resp.status_code == 200
    except Exception as e:
        print(f"Lobby error: {e}")
        return False

# ======================
# LẤY DỮ LIỆU
# ======================
def fetch_loop():
    global data_cache, last_results, last_data
    
    print("🔄 Bắt đầu vòng lặp...")
    
    while running:
        try:
            xsrf = unquote(session.cookies.get("XSRF-TOKEN", ""))
            headers = {
                "Referer": LOBBY_URL,
                "Origin": BASE,
                "X-Requested-With": "XMLHttpRequest",
                "X-XSRF-TOKEN": xsrf,
                "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8"
            }
            
            resp = session.post(GETNEWRESULT_URL, headers=headers, data={"gameCode": "ae"}, timeout=10)
            
            if resp.status_code == 200:
                try:
                    json_data = resp.json()
                    data = json_data.get("data", [])
                    
                    if data:
                        new_data = []
                        for t in data:
                            tb_name = t.get("table_name", "")
                            curr = t.get("result", "")
                            prev = last_results.get(tb_name, "")
                            
                            if curr and curr != prev:
                                last_results[tb_name] = curr
                                new_data.append({
                                    "table_name": tb_name,
                                    "result": curr,
                                    "round": t.get("round", ""),
                                    "shoeId": t.get("shoeId", ""),
                                    "goodRoad": t.get("goodRoad", ""),
                                    "time": time.strftime("%H:%M:%S")
                                })
                                # Cập nhật lịch sử cho thuật toán
                                predictor.update_history(tb_name, curr)
                        
                        if new_data:
                            data_cache = new_data
                            last_data = data
                            print(f"✅ Updated: {len(new_data)} tables")
                except Exception as e:
                    print(f"JSON error: {e}")
            else:
                print(f"API error: {resp.status_code}")
                
            time.sleep(CONFIG['update_interval'])
            
        except Exception as e:
            print(f"Loop error: {e}")
            time.sleep(3)

# ======================
# HÀM DỰ ĐOÁN
# ======================
def get_prediction_details(table_name):
    history = list(history_data.get(table_name, []))
    
    if len(history) < CONFIG['min_history']:
        return {
            'du_doan': 'Chưa đủ dữ liệu',
            'do_tin_cay': 0,
            'chi_tiet': f'Cần {CONFIG["min_history"]} lịch sử, hiện có {len(history)}',
            'khuyen_nghi': 'Chờ thêm kết quả'
        }
    
    final_pred, final_conf, algorithm_details = predictor.predict(table_name, history)
    
    if not final_pred:
        return {
            'du_doan': 'Không thể dự đoán',
            'do_tin_cay': 0,
            'chi_tiet': 'Không có thuật toán nào đưa ra dự đoán',
            'khuyen_nghi': 'Thử lại sau'
        }
    
    streak = predictor._get_streak(history)
    risk_level = 'THẤP' if final_conf >= 70 else 'TRUNG BÌNH' if final_conf >= 55 else 'CAO'
    
    if final_conf >= 70:
        strategy = 'NÊN ĐẶT CƯỢC - Tỷ lệ thắng cao'
        bet_amount = 'Vừa phải (1-2% vốn)'
    elif final_conf >= 55:
        strategy = 'CÓ THỂ THỬ - Rủi ro chấp nhận được'
        bet_amount = 'Nhỏ (0.5-1% vốn)'
    else:
        strategy = 'QUAN SÁT - Không nên đặt cược'
        bet_amount = '0 (Chờ cơ hội khác)'
    
    # Đếm số thuật toán đồng thuận
    consensus_count = sum(1 for d in algorithm_details.values() if d['du_doan'] == final_pred)
    
    return {
        'du_doan': final_pred,
        'do_tin_cay': final_conf,
        'so_thuat_toan_dong_thuan': f"{consensus_count}/{len(algorithm_details)}",
        'muc_do_rui_ro': risk_level,
        'chien_luoc': strategy,
        'so_tien_khuyen_nghi': bet_amount,
        'chuoi_hien_tai': len(streak),
        'tong_lich_su': len(history),
        'chi_tiet_thuat_toan': algorithm_details
    }

# ======================
# FLASK API
# ======================
app = Flask(__name__)

@app.route("/")
def home():
    return jsonify({
        "status": "running",
        "version": CONFIG['version'],
        "tables": len(data_cache),
        "algorithms": len(predictor.algorithms),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    })

@app.route("/data")
def get_data():
    return jsonify({
        "status": "success",
        "data": data_cache,
        "total": len(data_cache),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    })

@app.route("/predict/all")
def predict_all():
    """Dự đoán tất cả bàn"""
    results = []
    for table_name in history_data.keys():
        pred = get_prediction_details(table_name)
        results.append({
            'ban': table_name,
            'du_doan': pred['du_doan'],
            'do_tin_cay': pred['do_tin_cay'],
            'muc_do_rui_ro': pred['muc_do_rui_ro'],
            'chuoi': pred['chuoi_hien_tai'],
            'thoi_gian': time.strftime("%H:%M:%S")
        })
    results.sort(key=lambda x: x['do_tin_cay'], reverse=True)
    return jsonify({
        'tong_ban': len(results),
        'du_doan': results,
        'thoi_gian': time.strftime("%Y-%m-%d %H:%M:%S")
    })

@app.route("/predict/<string:table_name>")
def predict_one(table_name):
    """Dự đoán một bàn"""
    if table_name not in history_data:
        return jsonify({'loi': f'Không tìm thấy bàn "{table_name}"'}), 404
    
    pred = get_prediction_details(table_name)
    history = list(history_data.get(table_name, []))
    
    return jsonify({
        'ban': table_name,
        'du_doan_chi_tiet': pred,
        'lich_su_gan_day': history[-10:] if history else [],
        'thoi_gian': time.strftime("%Y-%m-%d %H:%M:%S")
    })

@app.route("/predict/best")
def predict_best():
    """Bàn có tỷ lệ thắng cao nhất"""
    results = []
    for table_name in history_data.keys():
        pred = get_prediction_details(table_name)
        if pred['do_tin_cay'] >= CONFIG['confidence_threshold']:
            results.append({
                'ban': table_name,
                'du_doan': pred['du_doan'],
                'do_tin_cay': pred['do_tin_cay'],
                'muc_do_rui_ro': pred['muc_do_rui_ro'],
                'chien_luoc': pred['chien_luoc']
            })
    results.sort(key=lambda x: x['do_tin_cay'], reverse=True)
    return jsonify({
        'khuyen_nghi_tot_nhat': results[:5] if results else [],
        'thoi_gian': time.strftime("%Y-%m-%d %H:%M:%S")
    })

@app.route("/health")
def health():
    return "OK", 200

# ======================
# KHỞI ĐỘNG
# ======================
if __name__ == "__main__":
    print("="*60)
    print("🎰 BACCARAT PREDICTOR - THUẬT TOÁN CAO CẤP")
    print("="*60)
    print(f"📊 Phiên bản: {CONFIG['version']}")
    print(f"🔢 Số thuật toán: {len(predictor.algorithms)}")
    print("="*60)
    
    login()
    go_to_lobby()
    
    threading.Thread(target=fetch_loop, daemon=True).start()
    time.sleep(2)
    
    print("="*60)
    print("✅ SERVER ĐANG CHẠY")
    print("📌 ENDPOINTS:")
    print("  /              - Thông tin")
    print("  /data          - Dữ liệu")
    print("  /predict/all   - Dự đoán tất cả")
    print("  /predict/<ban> - Dự đoán 1 bàn")
    print("  /predict/best  - Bàn tốt nhất")
    print("="*60)
    
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
