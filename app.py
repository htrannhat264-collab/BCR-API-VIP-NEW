import requests
from bs4 import BeautifulSoup
from urllib.parse import unquote
import time
import threading
from flask import Flask, jsonify, request
from collections import deque, Counter
import random
from datetime import datetime
import uuid
import os
import json
import math
from statistics import mean, stdev

# ======================
# CẤU HÌNH NÂNG CẤP
# ======================
BASE = "https://aibcr.me"
LOGIN_URL = f"{BASE}/login"
LOBBY_URL = f"{BASE}/ae/lobby"
GETNEWRESULT_URL = f"{BASE}/baccarat/getnewresult"

USERNAME = os.environ.get("BACCARAT_USER", "WangLin20199")
PASSWORD = os.environ.get("BACCARAT_PASS", "WangFlang1")

CONFIG = {
    'min_history': 5,
    'max_history': 200,  # Tăng lên 200 để phân tích sâu hơn
    'confidence_threshold': 55,  # Giảm ngưỡng để có nhiều cơ hội hơn
    'update_interval': 0.5,  # Giảm xuống 0.5 giây để cập nhật nhanh hơn
    'version': '3.0.0-PRO',
    'streak_weight': 1.5,  # Trọng số chuỗi
    'pattern_depth': 5,  # Độ sâu phân tích pattern
    'variance_threshold': 0.3  # Ngưỡng biến động
}

# ======================
# BIẾN TOÀN CỤC
# ======================
session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7"
})

last_results = {}
filtered_data = []
auto_running = True
history_data = {}
prediction_sessions = {}
predictor = None
perfomance_stats = {
    'total_predictions': 0,
    'correct_predictions': 0,
    'accuracy_history': deque(maxlen=100)
}

# ======================
# THUẬT TOÁN NÂNG CẤP
# ======================
class AdvancedBaccaratPredictor:
    def __init__(self):
        self.history_size = CONFIG['max_history']
        self.algorithms = {
            'theo_do_chieu_nang_cao': self._method_advanced_trend,
            'dao_chieu_thong_minh': self._method_smart_reversal,
            'tan_suat_dong': self._method_dynamic_frequency,
            'mau_hinh_sau': self._method_deep_pattern,
            'du_doan_thong_minh_plus': self._method_smart_plus,
            'phan_tich_du_lieu_lon': self._method_big_data,
            'machine_learning_co_ban': self._method_ml_basic,
            'trung_binh_di_dong': self._method_moving_average
        }
        self.weights = {name: 1.0 for name in self.algorithms.keys()}
        self.performance_history = {name: deque(maxlen=50) for name in self.algorithms.keys()}
    
    def update_history(self, table_name, result):
        if table_name not in history_data:
            history_data[table_name] = deque(maxlen=self.history_size)
        history_data[table_name].append(result)
    
    def _calculate_weights(self):
        """Cập nhật trọng số dựa trên hiệu suất thực tế"""
        for name in self.algorithms.keys():
            if self.performance_history[name]:
                avg_accuracy = sum(self.performance_history[name]) / len(self.performance_history[name])
                # Tăng trọng số cho thuật toán chính xác hơn
                self.weights[name] = 0.5 + (avg_accuracy / 100) * 0.5
    
    def _method_advanced_trend(self, history):
        """1. Theo dõi xu hướng nâng cao với phân tích chu kỳ"""
        if len(history) < 3:
            return None, 0
        
        # Phân tích chu kỳ 5 kết quả gần nhất
        last_5 = history[-5:] if len(history) >= 5 else history
        cycle_pattern = Counter(last_5)
        
        # Dự đoán theo chu kỳ
        if len(cycle_pattern) == 1:
            # Toàn B hoặc toàn P -> tiếp tục với độ tin cậy cao
            return last_5[0], min(90, 70 + len(cycle_pattern) * 10)
        elif len(cycle_pattern) == 2:
            # Có cả B và P -> phân tích tỷ lệ
            b_count = cycle_pattern.get('B', 0)
            p_count = cycle_pattern.get('P', 0)
            if b_count > p_count * 1.5:
                return 'B', 70
            elif p_count > b_count * 1.5:
                return 'P', 70
        return None, 0
    
    def _method_smart_reversal(self, history):
        """2. Đảo chiều thông minh với xác suất"""
        if len(history) < 4:
            return None, 0
        
        last = history[-1]
        # Phân tích 3 kết quả cuối
        recent_3 = history[-3:]
        
        # Nếu 3 kết quả cuối giống nhau, khả năng đảo chiều cao
        if all(x == recent_3[0] for x in recent_3):
            prediction = 'P' if last == 'B' else 'B'
            return prediction, 65
        
        # Nếu có pattern xen kẽ
        if recent_3[0] == recent_3[2] and recent_3[0] != recent_3[1]:
            # Pattern B-P-B hoặc P-B-P -> dự đoán ngược lại
            prediction = 'P' if last == 'B' else 'B'
            return prediction, 60
        
        return None, 0
    
    def _method_dynamic_frequency(self, history):
        """3. Tần suất động với trọng số thời gian"""
        if len(history) < 5:
            return None, 0
        
        # Lấy các đoạn lịch sử khác nhau
        recent_10 = history[-10:] if len(history) >= 10 else history
        recent_20 = history[-20:] if len(history) >= 20 else history
        recent_30 = history[-30:] if len(history) >= 30 else history
        
        # Tính tỷ lệ có trọng số
        weighted_b = 0
        weighted_p = 0
        total_weight = 0
        
        segments = [
            (recent_10, 3),  # Trọng số cao cho đoạn gần nhất
            (recent_20, 2),
            (recent_30, 1)
        ]
        
        for seg, weight in segments:
            if seg:
                b_count = sum(1 for x in seg if x == 'B')
                p_count = len(seg) - b_count
                weighted_b += (b_count / len(seg)) * weight
                weighted_p += (p_count / len(seg)) * weight
                total_weight += weight
        
        if total_weight > 0:
            b_rate = weighted_b / total_weight
            p_rate = weighted_p / total_weight
            
            if b_rate > p_rate + 0.15:
                return 'B', int(60 + (b_rate - p_rate) * 50)
            elif p_rate > b_rate + 0.15:
                return 'P', int(60 + (p_rate - b_rate) * 50)
        
        return None, 0
    
    def _method_deep_pattern(self, history):
        """4. Mẫu hình sâu với độ dài biến động"""
        if len(history) < 10:
            return None, 0
        
        max_pattern_length = min(CONFIG['pattern_depth'], len(history) - 2)
        
        for pattern_len in range(max_pattern_length, 2, -1):
            recent_pattern = ''.join(history[-pattern_len:])
            matches = []
            
            for i in range(0, len(history) - pattern_len - 1):
                pattern = ''.join(history[i:i+pattern_len])
                if pattern == recent_pattern and i + pattern_len < len(history):
                    matches.append(history[i+pattern_len])
            
            if matches:
                next_result = Counter(matches).most_common(1)[0]
                confidence = min(80, 50 + (len(matches) * 5))
                return next_result[0], confidence
        
        return None, 0
    
    def _method_smart_plus(self, history):
        """5. Dự đoán thông minh plus với nhiều chỉ báo"""
        if len(history) < 5:
            return None, 0
        
        scores = {'B': 0, 'P': 0}
        
        # Chỉ báo 1: Xu hướng gần đây (trọng số 0.25)
        recent_10 = history[-10:] if len(history) >= 10 else history
        b_10 = sum(1 for x in recent_10 if x == 'B')
        p_10 = len(recent_10) - b_10
        if b_10 > p_10:
            scores['B'] += (b_10 / len(recent_10)) * 25
        else:
            scores['P'] += (p_10 / len(recent_10)) * 25
        
        # Chỉ báo 2: Biến động (trọng số 0.15)
        if len(history) >= 10:
            recent_5 = history[-5:]
            b_5 = sum(1 for x in recent_5 if x == 'B')
            if b_5 >= 4:
                scores['P'] += 15  # Khả năng đảo chiều cao
            elif b_5 <= 1:
                scores['B'] += 15
        
        # Chỉ báo 3: Chu kỳ (trọng số 0.2)
        if len(history) >= 8:
            last_8 = history[-8:]
            # Tìm chu kỳ 4
            if last_8[:4] == last_8[4:]:
                scores[last_8[-1]] += 20
        
        # Chỉ báo 4: Tương quan (trọng số 0.2)
        if len(history) >= 20:
            first_half = history[:len(history)//2]
            second_half = history[len(history)//2:]
            b_first = sum(1 for x in first_half if x == 'B')
            b_second = sum(1 for x in second_half if x == 'B')
            if b_first > b_second:
                scores['B'] += (b_first - b_second) / len(first_half) * 20
            else:
                scores['P'] += (b_second - b_first) / len(second_half) * 20
        
        # Chỉ báo 5: Streak hiện tại (trọng số 0.2)
        streak = self._get_streak(history)
        if len(streak) >= 4:
            # Chuỗi dài -> khả năng đảo chiều
            scores['P' if streak[0] == 'B' else 'B'] += 20
        elif len(streak) >= 2:
            scores[streak[0]] += 10
        
        # Chọn kết quả có điểm cao nhất
        if scores['B'] > scores['P']:
            return 'B', min(90, scores['B'])
        elif scores['P'] > scores['B']:
            return 'P', min(90, scores['P'])
        return None, 0
    
    def _method_big_data(self, history):
        """6. Phân tích dữ liệu lớn - thống kê nâng cao"""
        if len(history) < 15:
            return None, 0
        
        # Chuyển đổi sang số để tính toán thống kê
        numeric = [1 if x == 'B' else 0 for x in history]
        
        # Tính trung bình và độ lệch chuẩn
        mean_val = mean(numeric)
        std_val = stdev(numeric) if len(numeric) > 1 else 0
        
        # Tính hệ số biến động
        cv = std_val / mean_val if mean_val > 0 else 0
        
        # Phân tích xu hướng
        if cv < 0.2:
            # Biến động thấp -> có xu hướng rõ ràng
            if mean_val > 0.6:
                return 'B', 80
            elif mean_val < 0.4:
                return 'P', 80
        else:
            # Biến động cao -> dự đoán ngược lại xu hướng gần đây
            last = history[-1]
            if len(history) >= 3 and all(x == last for x in history[-3:]):
                return 'P' if last == 'B' else 'B', 60
        
        return None, 0
    
    def _method_ml_basic(self, history):
        """7. Machine learning cơ bản - logistic regression đơn giản"""
        if len(history) < 20:
            return None, 0
        
        # Tạo feature đơn giản
        features = []
        labels = []
        
        for i in range(5, len(history) - 1):
            # Feature: 5 kết quả gần nhất
            window = history[i-5:i]
            features.append([1 if x == 'B' else 0 for x in window])
            labels.append(1 if history[i] == 'B' else 0)
        
        if len(features) < 5:
            return None, 0
        
        # Tính trọng số đơn giản (tần suất)
        weights = [0] * 5
        for feature, label in zip(features, labels):
            for j in range(5):
                if feature[j] == label:
                    weights[j] += 1
        
        # Dự đoán dựa trên 5 kết quả gần nhất
        last_5 = [1 if x == 'B' else 0 for x in history[-5:]]
        score = sum(weights[j] * last_5[j] for j in range(5))
        
        prediction = 'B' if score > sum(weights) / 2 else 'P'
        confidence = min(80, 50 + abs(score - sum(weights) / 2) / sum(weights) * 30)
        
        return prediction, confidence
    
    def _method_moving_average(self, history):
        """8. Trung bình động với phân tích EMA"""
        if len(history) < 10:
            return None, 0
        
        # Chuyển đổi sang số
        numeric = [1 if x == 'B' else 0 for x in history]
        
        # Tính EMA với các period khác nhau
        ema_5 = self._calculate_ema(numeric, 5)
        ema_10 = self._calculate_ema(numeric, 10)
        ema_20 = self._calculate_ema(numeric, 20)
        
        # Kiểm tra tín hiệu crossover
        if len(ema_5) >= 2 and len(ema_10) >= 2:
            if ema_5[-1] > ema_10[-1] and ema_5[-2] <= ema_10[-2]:
                return 'B', 70
            elif ema_5[-1] < ema_10[-1] and ema_5[-2] >= ema_10[-2]:
                return 'P', 70
        
        # Kiểm tra xu hướng
        if ema_5 and ema_10 and ema_20:
            if ema_5[-1] > ema_10[-1] > ema_20[-1]:
                return 'B', 65
            elif ema_5[-1] < ema_10[-1] < ema_20[-1]:
                return 'P', 65
        
        return None, 0
    
    def _calculate_ema(self, data, period):
        """Tính EMA (Exponential Moving Average)"""
        if len(data) < period:
            return []
        
        ema = []
        sma = sum(data[:period]) / period
        ema.append(sma)
        multiplier = 2 / (period + 1)
        
        for i in range(period, len(data)):
            ema.append(data[i] * multiplier + ema[-1] * (1 - multiplier))
        
        return ema
    
    def _get_streak(self, history):
        """Lấy chuỗi kết quả hiện tại"""
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
    
    def predict(self, table_name, history):
        """Dự đoán tổng hợp với trọng số động"""
        if len(history) < CONFIG['min_history']:
            return None, 0, {}
        
        # Cập nhật trọng số trước khi dự đoán
        self._calculate_weights()
        
        all_predictions = {}
        valid_predictions = []
        
        for name, method in self.algorithms.items():
            try:
                pred, conf = method(history)
                if pred:
                    # Điều chỉnh độ tin cậy theo trọng số của thuật toán
                    weight = self.weights.get(name, 1.0)
                    adjusted_conf = min(95, conf * (0.7 + 0.3 * weight))
                    
                    all_predictions[name] = {
                        'du_doan': pred,
                        'do_tin_cay': round(adjusted_conf, 1),
                        'trong_so': round(weight, 2)
                    }
                    valid_predictions.append((pred, adjusted_conf, weight))
            except Exception as e:
                print(f"Lỗi {name}: {e}")
        
        if not valid_predictions:
            return None, 0, {}
        
        # Tổng hợp có trọng số
        prediction_weights = {'B': 0, 'P': 0}
        total_weight = 0
        
        for pred, conf, weight in valid_predictions:
            prediction_weights[pred] += conf * weight
            total_weight += conf * weight
        
        if total_weight == 0:
            return None, 0, all_predictions
        
        # Chuẩn hóa và chọn kết quả
        final_pred = 'B' if prediction_weights['B'] > prediction_weights['P'] else 'P'
        final_conf = (max(prediction_weights.values()) / total_weight) * 100
        
        # Điều chỉnh độ tin cậy cuối cùng
        consensus_count = sum(1 for p in valid_predictions if p[0] == final_pred)
        consensus_boost = (consensus_count / len(valid_predictions)) * 20
        final_conf = min(95, final_conf + consensus_boost)
        
        return final_pred, round(final_conf, 1), all_predictions
    
    def update_performance(self, algorithm_name, correct):
        """Cập nhật hiệu suất của thuật toán"""
        if algorithm_name in self.performance_history:
            self.performance_history[algorithm_name].append(1 if correct else 0)
            # Cập nhật trọng số ngay lập tức
            self._calculate_weights()

# ======================
# HÀM XỬ LÝ CHÍNH
# ======================
predictor = AdvancedBaccaratPredictor()

def get_prediction_details(table_name, history):
    """Lấy chi tiết dự đoán nâng cao"""
    if len(history) < CONFIG['min_history']:
        return {
            'du_doan': 'Chưa đủ dữ liệu',
            'do_tin_cay': 0,
            'chi_tiet': f'Cần {CONFIG["min_history"]} lịch sử, hiện có {len(history)}',
            'khuyen_nghi': 'Chờ thêm kết quả'
        }
    
    # Dự đoán với thuật toán nâng cao
    final_pred, final_conf, algorithm_details = predictor.predict(table_name, history)
    
    if not final_pred:
        return {
            'du_doan': 'Không thể dự đoán',
            'do_tin_cay': 0,
            'chi_tiet': 'Không có thuật toán nào đưa ra dự đoán',
            'khuyen_nghi': 'Thử lại sau'
        }
    
    # Phân tích thêm
    streak = predictor._get_streak(history) if hasattr(predictor, '_get_streak') else []
    risk_level = 'RẤT THẤP' if final_conf >= 85 else 'THẤP' if final_conf >= 70 else 'TRUNG BÌNH' if final_conf >= 55 else 'CAO'
    
    # Chiến lược chi tiết
    if final_conf >= 85:
        strategy = 'TỐI ƯU - Đặt cược lớn (3-5% vốn)'
        bet_amount = 'Lớn'
        action = 'BẮT BUỘC'
    elif final_conf >= 70:
        strategy = 'TỐT - Đặt cược vừa (2-3% vốn)'
        bet_amount = 'Vừa'
        action = 'NÊN'
    elif final_conf >= 55:
        strategy = 'CHẤP NHẬN - Đặt cược nhỏ (1-2% vốn)'
        bet_amount = 'Nhỏ'
        action = 'CÓ THỂ'
    else:
        strategy = 'RỦI RO CAO - Quan sát hoặc cược rất nhỏ (0.5% vốn)'
        bet_amount = 'Rất nhỏ'
        action = 'HẠN CHẾ'
    
    # Dự đoán lợi nhuận kỳ vọng
    expected_return = final_conf - 50  # Baccarat có tỷ lệ thắng cơ bản ~50%
    expected_return = max(-20, min(45, expected_return))  # Giới hạn trong khoảng -20% đến 45%
    
    # Tạo chi tiết thuật toán có thông tin trọng số
    algo_details_with_weights = {}
    for name, details in algorithm_details.items():
        algo_details_with_weights[name] = {
            'du_doan': details['du_doan'],
            'do_tin_cay': details['do_tin_cay'],
            'trong_so_hien_tai': details['trong_so']
        }
    
    return {
        'du_doan': final_pred,
        'do_tin_cay': final_conf,
        'so_thuat_toan': len(algorithm_details),
        'so_thuat_toan_dong_thuan': sum(1 for d in algorithm_details.values() if d['du_doan'] == final_pred),
        'muc_do_rui_ro': risk_level,
        'hanh_dong': action,
        'chien_luoc': strategy,
        'so_tien_khuyen_nghi': bet_amount,
        'loi_nhuan_ky_vong': round(expected_return, 1),
        'chuoi_hien_tai': len(streak),
        'tong_lich_su': len(history),
        'chi_tiet_thuat_toan': algo_details_with_weights,
        'thong_ke_nhanh': {
            'B_gan_day': sum(1 for x in history[-10:] if x == 'B') if len(history) >= 10 else 0,
            'P_gan_day': sum(1 for x in history[-10:] if x == 'P') if len(history) >= 10 else 0,
            'ty_le_thang': f"{sum(1 for x in history if x == 'B') / len(history) * 100:.1f}%"
        }
    }

# ======================
# API MỚI NÂNG CẤP
# ======================
@app.route("/du-doan/cao-cap")
def predict_advanced():
    """Dự đoán nâng cao với phân tích chi tiết"""
    results = []
    for table_name in history_data.keys():
        history = list(history_data[table_name])
        if len(history) >= CONFIG['min_history']:
            prediction = get_prediction_details(table_name, history)
            results.append({
                'ban': table_name,
                'du_doan': prediction['du_doan'],
                'do_tin_cay': prediction['do_tin_cay'],
                'muc_do_rui_ro': prediction['muc_do_rui_ro'],
                'hanh_dong': prediction['hanh_dong'],
                'so_tien': prediction['so_tien_khuyen_nghi'],
                'loi_nhuan_ky_vong': prediction['loi_nhuan_ky_vong'],
                'chuoi': prediction['chuoi_hien_tai'],
                'thoi_gian': datetime.now().strftime('%H:%M:%S')
            })
    
    # Sắp xếp theo lợi nhuận kỳ vọng
    results.sort(key=lambda x: x['loi_nhuan_ky_vong'], reverse=True)
    
    return jsonify({
        'tong_ban': len(results),
        'ket_qua': results,
        'thoi_gian': datetime.now().strftime('%H:%M:%S %d/%m/%Y')
    })

@app.route("/du-doan/thong-minh/<string:table_name>")
def predict_smart(table_name):
    """Dự đoán thông minh cho một bàn với phân tích sâu"""
    history = list(history_data.get(table_name, []))
    if not history:
        return jsonify({'loi': f'Không tìm thấy bàn "{table_name}"'}), 404
    
    prediction = get_prediction_details(table_name, history)
    
    # Thêm phân tích nâng cao
    analysis = {
        'do_phan_tan': calculate_dispersion(history),
        'xu_huong': detect_trend(history),
        'diem_mua_ban': generate_trading_signals(history)
    }
    
    return jsonify({
        'ban': table_name,
        'du_doan': prediction,
        'phan_tich_nang_cao': analysis,
        'thoi_gian': datetime.now().strftime('%H:%M:%S %d/%m/%Y')
    })

@app.route("/he-thong/hieu-suat")
def system_performance():
    """Thống kê hiệu suất hệ thống"""
    total_sessions = len(prediction_sessions)
    total_preds = sum(sess['tong_du_doan'] for sess in prediction_sessions.values())
    correct_preds = sum(sess['du_doan_dung'] for sess in prediction_sessions.values())
    
    accuracy = (correct_preds / total_preds * 100) if total_preds > 0 else 0
    
    # Thống kê theo thuật toán
    algo_stats = {}
    for name in predictor.algorithms.keys():
        if name in predictor.performance_history and predictor.performance_history[name]:
            algo_stats[name] = {
                'do_chinh_xac': round(mean(predictor.performance_history[name]) * 100, 1),
                'so_lan_test': len(predictor.performance_history[name]),
                'trong_so': round(predictor.weights.get(name, 1.0), 2)
            }
    
    return jsonify({
        'hieu_suat_he_thong': {
            'tong_du_doan': total_preds,
            'du_doan_dung': correct_preds,
            'ty_le_chinh_xac': round(accuracy, 1),
            'so_phien': total_sessions,
            'thoi_gian_chay': f"{int((datetime.now() - datetime.strptime('2026-06-16', '%Y-%m-%d')).total_seconds() / 3600)} giờ"
        },
        'chi_tiet_thuat_toan': algo_stats,
        'thoi_gian': datetime.now().strftime('%H:%M:%S %d/%m/%Y')
    })

# ======================
# HÀM PHÂN TÍCH NÂNG CAO
# ======================
def calculate_dispersion(history):
    """Tính độ phân tán của dữ liệu"""
    if len(history) < 10:
        return {'do_phan_tan': 'Chưa đủ dữ liệu'}
    
    numeric = [1 if x == 'B' else 0 for x in history[-30:]]
    mean_val = mean(numeric)
    variance = sum((x - mean_val) ** 2 for x in numeric) / len(numeric)
    
    return {
        'do_phan_tan': round(variance, 3),
        'do_lech_chuan': round(math.sqrt(variance), 3),
        'muc_do': 'CAO' if variance > 0.25 else 'TRUNG BÌNH' if variance > 0.1 else 'THẤP'
    }

def detect_trend(history):
    """Phát hiện xu hướng trong dữ liệu"""
    if len(history) < 10:
        return {'xu_huong': 'Chưa đủ dữ liệu'}
    
    # Chia thành 3 phần
    part_size = len(history) // 3
    part1 = history[:part_size]
    part2 = history[part_size:2*part_size]
    part3 = history[2*part_size:]
    
    b1 = sum(1 for x in part1 if x == 'B') / len(part1)
    b2 = sum(1 for x in part2 if x == 'B') / len(part2)
    b3 = sum(1 for x in part3 if x == 'B') / len(part3)
    
    if b1 < b2 < b3:
        return {'xu_huong': 'TĂNG DẦN', 'huong': 'B', 'do_manh': round((b3 - b1) * 100, 1)}
    elif b1 > b2 > b3:
        return {'xu_huong': 'GIẢM DẦN', 'huong': 'P', 'do_manh': round((b1 - b3) * 100, 1)}
    else:
        return {'xu_huong': 'DAO ĐỘNG', 'huong': 'KHÔNG RÕ', 'do_manh': 0}

def generate_trading_signals(history):
    """Tạo tín hiệu giao dịch"""
    if len(history) < 20:
        return {'tin_hieu': 'Chưa đủ dữ liệu'}
    
    # Tính RSI đơn giản
    gains = []
    losses = []
    for i in range(1, len(history)):
        diff = 1 if history[i] == 'B' else -1 if history[i] == 'P' else 0
        if diff > 0:
            gains.append(diff)
            losses.append(0)
        elif diff < 0:
            gains.append(0)
            losses.append(abs(diff))
        else:
            gains.append(0)
            losses.append(0)
    
    avg_gain = sum(gains[-14:]) / 14 if len(gains) >= 14 else 0
    avg_loss = sum(losses[-14:]) / 14 if len(losses) >= 14 else 1
    
    rsi = 100 - (100 / (1 + (avg_gain / avg_loss))) if avg_loss > 0 else 100
    
    return {
        'rsi': round(rsi, 1),
        'tin_hieu': 'MUA' if rsi < 30 else 'BÁN' if rsi > 70 else 'TRUNG LẬP',
        'khoang_cach': 'Nhiều' if abs(rsi - 50) > 20 else 'Ít'
    }

# ======================
# KHỞI ĐỘNG
# ======================
if __name__ == "__main__":
    print("="*70)
    print("🎰 HỆ THỐNG DỰ ĐOÁN BACCARAT AI - NÂNG CẤP CAO CẤP 🎰")
    print("="*70)
    print(f"📊 Phiên bản: {CONFIG['version']}")
    print(f"🔢 Số thuật toán: {len(predictor.algorithms)}")
    print("="*70)
    
    # Đăng nhập
    if login():
        print("✅ Đăng nhập thành công")
        go_to_lobby()
        print("✅ Đã vào lobby")
    else:
        print("❌ Đăng nhập thất bại")
    
    # Khởi động thread
    threading.Thread(target=auto_loop, daemon=True).start()
    print("✅ Đã khởi động vòng lặp lấy dữ liệu")
    print("="*70)
    
    # Chạy server
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)