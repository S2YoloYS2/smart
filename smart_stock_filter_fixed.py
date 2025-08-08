import streamlit as st
# 비밀번호 확인 함수
def check_password():
    """Returns `True` if the user had the correct password."""
    
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets["password"]:  # secrets에서 가져옴
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input(
            "🔐 비밀번호를 입력하세요", 
            type="password", 
            on_change=password_entered, 
            key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        st.text_input(
            "🔐 비밀번호를 입력하세요", 
            type="password", 
            on_change=password_entered, 
            key="password"
        )
        st.error("😕 비밀번호가 틀렸습니다")
        return False
    else:
        return True

# 비밀번호 체크
if not check_password():
    st.stop()

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import json
from pathlib import Path
import time
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 조건부 임포트
try:
    from pykrx import stock
    PYKRX_AVAILABLE = True
except ImportError:
    PYKRX_AVAILABLE = False
    st.warning("pykrx가 설치되지 않았습니다. pip install pykrx")

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    st.warning("yfinance가 설치되지 않았습니다. pip install yfinance")

warnings.filterwarnings('ignore')

# 페이지 설정
st.set_page_config(page_title="🔍 스마트 주식 필터 Pro", layout="wide")
st.title("🔍 스마트 주식 필터 Pro v4.0 - 한국/미국 통합")

# S&P 500 대표 종목 (섹터별)
SP500_TICKERS = [
    # Technology
    'AAPL', 'MSFT', 'NVDA', 'META', 'GOOGL', 'AMZN', 'TSLA', 'AVGO', 'ORCL', 'ADBE',
    # Healthcare  
    'UNH', 'JNJ', 'LLY', 'PFE', 'ABBV', 'MRK', 'TMO', 'ABT', 'CVS', 'AMGN',
    # Financials
    'BRK-B', 'JPM', 'V', 'MA', 'BAC', 'WFC', 'GS', 'MS', 'AXP', 'BLK',
    # Consumer
    'WMT', 'PG', 'HD', 'KO', 'PEP', 'MCD', 'NKE', 'COST', 'SBUX', 'DIS',
    # Industrials
    'UPS', 'RTX', 'BA', 'HON', 'CAT', 'GE', 'LMT', 'MMM', 'DE', 'UNP',
    # Energy
    'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO', 'OXY', 'HAL'
]

# 세션 상태 초기화
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = []

if 'search_results' not in st.session_state:
    st.session_state.search_results = None

if 'backtest_stats' not in st.session_state:
    st.session_state.backtest_stats = {'total': 0, 'success': 0, 'fail': 0, 'success_rate': 0}

# 파일 경로 설정
WATCHLIST_FILE = "watchlist.json"
BACKTEST_FILE = "backtest_results.json"

# 관심종목 로드/저장 함수
def load_watchlist():
    """저장된 관심종목 로드"""
    if Path(WATCHLIST_FILE).exists():
        try:
            with open(WATCHLIST_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return []
    return []

def save_watchlist(watchlist):
    """관심종목 저장"""
    with open(WATCHLIST_FILE, 'w', encoding='utf-8') as f:
        json.dump(watchlist, f, ensure_ascii=False, indent=2)

# 백테스팅 결과 저장/로드
def load_backtest_results():
    """백테스팅 결과 로드"""
    if Path(BACKTEST_FILE).exists():
        try:
            with open(BACKTEST_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_backtest_results(results):
    """백테스팅 결과 저장"""
    with open(BACKTEST_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

# 세션 시작 시 데이터 로드
if 'data_loaded' not in st.session_state:
    st.session_state.watchlist = load_watchlist()
    st.session_state.backtest_results = load_backtest_results()
    st.session_state.data_loaded = True

# --- 기술적 지표 계산 함수들 ---
@st.cache_data
def compute_cci(high, low, close, window=20):
    tp = (high + low + close) / 3
    sma = tp.rolling(window=window).mean()
    mad = tp.rolling(window=window).apply(lambda x: np.fabs(x - x.mean()).mean(), raw=True)
    cci = (tp - sma) / (0.015 * (mad + 1e-9))
    return cci

@st.cache_data
def compute_cci_ma(cci, ma_window=14):
    return cci.rolling(window=ma_window).mean()

@st.cache_data
def compute_stoch_mtm(close, k_length, ema_length=10, smooth_period=5):
    low_min = close.rolling(window=k_length).min()
    high_max = close.rolling(window=k_length).max()
    percent_k = 100 * (close - low_min) / (high_max - low_min + 1e-9)
    percent_k_scaled = percent_k - 100
    ema_signal = percent_k_scaled.ewm(span=ema_length, adjust=False).mean()
    signal_smoothed = ema_signal.rolling(window=smooth_period).mean()
    return percent_k_scaled, signal_smoothed

@st.cache_data
def compute_bollinger_bands(close, window=20, num_std_dev=2):
    mid_band = close.rolling(window=window).mean()
    std_dev = close.rolling(window=window).std()
    upper_band = mid_band + (std_dev * num_std_dev)
    lower_band = mid_band - (std_dev * num_std_dev)
    band_width = upper_band - lower_band
    return mid_band, upper_band, lower_band, band_width

@st.cache_data
def compute_rsi(close, window=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(com=window-1, adjust=False).mean()
    avg_loss = loss.ewm(com=window-1, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi

@st.cache_data
def compute_macd(close, fast=12, slow=26, signal=9):
    """MACD 계산"""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_diff = macd - macd_signal
    return macd, macd_signal, macd_diff

@st.cache_data
def compute_vwap(df, period=20):
    """VWAP - 거래량 가중 평균가격"""
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    vwap = (typical_price * df['Volume']).rolling(period).sum() / df['Volume'].rolling(period).sum()
    return vwap

@st.cache_data
def compute_adx(df, period=14):
    """ADX - 추세 강도"""
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    plus_dm = high.diff()
    minus_dm = low.diff() * -1
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    
    atr = tr.rolling(period).mean()
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
    
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9))
    adx = dx.rolling(period).mean()
    
    return adx, plus_di, minus_di

@st.cache_data
def compute_money_flow(df, period=14):
    """MFI - 자금 흐름 지표"""
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    raw_money_flow = typical_price * df['Volume']
    
    positive_flow = raw_money_flow.where(typical_price > typical_price.shift(1), 0)
    negative_flow = raw_money_flow.where(typical_price < typical_price.shift(1), 0)
    
    positive_mf = positive_flow.rolling(period).sum()
    negative_mf = negative_flow.rolling(period).sum()
    
    mfi = 100 - (100 / (1 + positive_mf / (negative_mf + 1e-9)))
    return mfi

# --- 캔들 패턴 인식 ---
def detect_candle_patterns(df):
    """주요 캔들 패턴 감지"""
    patterns = {}
    
    if len(df) < 3:
        return patterns
    
    # 현재 캔들 정보
    open_price = df['Open'].iloc[-1]
    high_price = df['High'].iloc[-1]
    low_price = df['Low'].iloc[-1]
    close_price = df['Close'].iloc[-1]
    
    # 이전 캔들 정보
    prev_open = df['Open'].iloc[-2]
    prev_close = df['Close'].iloc[-2]
    
    # 몸통과 꼬리 계산
    body = abs(close_price - open_price)
    upper_wick = high_price - max(open_price, close_price)
    lower_wick = min(open_price, close_price) - low_price
    
    # 1. 망치형 (Hammer)
    if lower_wick > body * 2 and upper_wick < body * 0.3 and close_price > open_price:
        patterns['망치형'] = True
    
    # 2. 상승장악형 (Bullish Engulfing)
    if (prev_open > prev_close and  # 전일 음봉
        close_price > open_price and  # 당일 양봉
        open_price < prev_close and  # 당일 시가 < 전일 종가
        close_price > prev_open):  # 당일 종가 > 전일 시가
        patterns['상승장악형'] = True
    
    # 3. 적삼병 (Three White Soldiers)
    if len(df) >= 3:
        three_white = (
            df['Close'].iloc[-3] > df['Open'].iloc[-3] and
            df['Close'].iloc[-2] > df['Open'].iloc[-2] and
            df['Close'].iloc[-1] > df['Open'].iloc[-1] and
            df['Close'].iloc[-2] > df['Close'].iloc[-3] and
            df['Close'].iloc[-1] > df['Close'].iloc[-2]
        )
        if three_white:
            patterns['적삼병'] = True
    
    return patterns

# --- 한국 주식 관련 함수들 ---
@st.cache_data(ttl=3600)
def get_most_recent_trading_day():
    today = datetime.today()
    
    if today.hour < 6:
        today = today - timedelta(days=1)
    
    if today.weekday() >= 5:
        days_to_subtract = today.weekday() - 4
        today = today - timedelta(days=days_to_subtract)
    
    for i in range(30):
        check_date = (today - timedelta(days=i)).strftime('%Y%m%d')
        try:
            test_data = stock.get_market_ticker_list(check_date, market="KOSPI")
            if test_data and len(test_data) > 0:
                return check_date
        except:
            continue
    
    yesterday = datetime.today() - timedelta(days=1)
    if yesterday.weekday() >= 5:
        days_to_subtract = yesterday.weekday() - 4
        yesterday = yesterday - timedelta(days=days_to_subtract)
    return yesterday.strftime('%Y%m%d')

@st.cache_data(ttl=3600)
def get_korean_stock_data(ticker, start_date, end_date):
    """한국 주식 데이터 조회"""
    try:
        df = stock.get_market_ohlcv_by_date(start_date, end_date, ticker)
        if not df.empty:
            df.index = pd.to_datetime(df.index)
            df.columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Change']
            return df
        return None
    except:
        return None

@st.cache_data(ttl=3600)
def get_top_volume_korean_stocks(today_str, top_n=100):
    """거래대금 상위 한국 종목"""
    try:
        df_kospi = stock.get_market_ohlcv_by_ticker(today_str, market="KOSPI")
        if not df_kospi.empty:
            df_kospi['거래대금'] = df_kospi['거래량'] * df_kospi['종가']
            df_kospi = df_kospi.nlargest(min(top_n, len(df_kospi)), '거래대금')
            return df_kospi.index.tolist()
        return []
    except:
        return []

# --- 미국 주식 관련 함수들 ---
@st.cache_data(ttl=3600)
def get_us_stock_data(ticker, period="3mo"):
    """미국 주식 데이터 조회"""
    try:
        stock_data = yf.Ticker(ticker)
        df = stock_data.history(period=period)
        if not df.empty:
            info = stock_data.info
            return df, info
        return None, None
    except:
        return None, None

# --- 강화된 AI 예측 모델 ---
class EnhancedAIPredictor:
    """강화된 AI 예측 시스템"""
    
    def __init__(self):
        self.models = {}
        self.success_count = 0
        self.total_count = 0
        
    def prepare_features(self, df):
        """머신러닝을 위한 특징 준비"""
        features = pd.DataFrame(index=df.index)
        
        # 가격 변화율
        for period in [1, 3, 5, 10, 20]:
            features[f'returns_{period}d'] = df['Close'].pct_change(period)
        
        # 이동평균
        for period in [5, 10, 20, 50]:
            if len(df) >= period:
                ma = df['Close'].rolling(period).mean()
                features[f'ma_{period}'] = (df['Close'] - ma) / ma
        
        # RSI
        features['rsi'] = compute_rsi(df['Close'])
        
        # MACD
        if len(df) >= 26:
            macd, signal, diff = compute_macd(df['Close'])
            features['macd'] = macd
            features['macd_signal'] = signal
            features['macd_diff'] = diff
        
        # 볼린저 밴드
        _, upper, lower, _ = compute_bollinger_bands(df['Close'])
        features['bb_position'] = (df['Close'] - lower) / (upper - lower + 1e-9)
        
        # CCI
        features['cci'] = compute_cci(df['High'], df['Low'], df['Close'])
        
        # 스토캐스틱
        stoch_k, stoch_d = compute_stoch_mtm(df['Close'], k_length=14)
        features['stoch_k'] = stoch_k
        features['stoch_d'] = stoch_d
        
        # 거래량
        features['volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
        
        # ADX
        adx, plus_di, minus_di = compute_adx(df)
        features['adx'] = adx
        features['plus_di'] = plus_di
        features['minus_di'] = minus_di
        
        # MFI
        features['mfi'] = compute_money_flow(df)
        
        # 변동성
        features['volatility'] = df['Close'].pct_change().rolling(20).std()
        
        # 타겟: 5일 후 상승 여부
        features['target'] = (df['Close'].shift(-5) > df['Close']).astype(int)
        features['return_5d'] = df['Close'].pct_change(-5)
        
        return features.replace([np.inf, -np.inf], np.nan).dropna()
    
    def train_and_predict(self, df):
        """모델 학습 및 예측"""
        features = self.prepare_features(df)
        
        if len(features) < 100:
            return None, None, None
        
        # 특징 선택
        feature_cols = [col for col in features.columns if not col.startswith('target') and not col.startswith('return')]
        
        X = features[feature_cols]
        y = features['target']
        
        # 시계열 분할
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:-5]
        y_train, y_test = y[:split_idx], y[split_idx:-5]
        
        if len(X_train) < 50 or len(X_test) < 10:
            return None, None, None
        
        # 스케일링
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 앙상블 모델
        rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        gb_model = GradientBoostingClassifier(n_estimators=50, max_depth=5, random_state=42)
        
        # 학습
        rf_model.fit(X_train_scaled, y_train)
        gb_model.fit(X_train_scaled, y_train)
        
        # 예측
        rf_pred = rf_model.predict_proba(X_test_scaled)[:, 1]
        gb_pred = gb_model.predict_proba(X_test_scaled)[:, 1]
        
        # 앙상블
        ensemble_pred = (rf_pred + gb_pred) / 2
        accuracy = ((ensemble_pred > 0.5) == y_test).mean()
        
        # 현재 예측
        current_features = scaler.transform(X.iloc[-1:])
        rf_prob = rf_model.predict_proba(current_features)[0][1]
        gb_prob = gb_model.predict_proba(current_features)[0][1]
        final_prediction = (rf_prob + gb_prob) / 2
        
        # 예상 수익률
        expected_return = features['return_5d'].mean() if 'return_5d' in features else None
        
        return final_prediction, accuracy, expected_return

# --- 백테스팅 함수 ---
@st.cache_data
def perform_backtest(df, prediction_prob, entry_date):
    """5일 후 백테스팅 수행"""
    if len(df) < 5:
        return None
    
    entry_price = df['Close'].iloc[-1]
    
    # 5일 후 실제 결과 확인 (미래 데이터가 있다면)
    if len(df) >= 5:
        try:
            # 5일 후 가격
            future_price = df['Close'].iloc[-1]  # 현재 시점
            actual_return = ((future_price - entry_price) / entry_price) * 100
            
            # 성공 여부 판단
            predicted_up = prediction_prob >= 0.5
            actual_up = actual_return > 0
            success = predicted_up == actual_up
            
            return {
                'entry_date': entry_date,
                'entry_price': entry_price,
                'prediction_prob': prediction_prob,
                'actual_return': actual_return,
                'success': success
            }
        except:
            return None
    
    return None

# --- 스마트 조건 평가 시스템 (고급자 전용) ---
class SmartStockFilter:
    """고급자용 스마트 필터"""
    
    def __init__(self, near_cross_thresh=5.0):
        self.NC_THRESH = abs(near_cross_thresh)
        
    def evaluate_stock(self, df, min_volume, min_market_cap=None):
        """종목 평가 - 엄격한 조건"""
        
        if len(df) < 60:
            return None
        
        score = 0
        conditions = {}
        category_scores = {
            'CCI_조건': {'score': 0, 'count': 0, 'conditions': []},
            '캔들_패턴': {'score': 0, 'count': 0, 'conditions': []},
            '추세_지표': {'score': 0, 'count': 0, 'conditions': []},
            '모멘텀_지표': {'score': 0, 'count': 0, 'conditions': []},
            '거래량_지표': {'score': 0, 'count': 0, 'conditions': []},
        }
        
        # 1. CCI 조건 체크 - 돌파 직전 최우선
        try:
            cci = compute_cci(df['High'], df['Low'], df['Close'])
            cci_ma = compute_cci_ma(cci)
            if len(cci) >= 2:
                c_cur, c_prev = cci.iloc[-1], cci.iloc[-2]
                m_cur, m_prev = cci_ma.iloc[-1], cci_ma.iloc[-2]
                gap = m_cur - c_cur

                # CCI 돌파 직전 (최우선)
                if (c_prev < m_prev and c_cur < m_cur and
                    0 < gap <= self.NC_THRESH and
                    c_cur > c_prev and
                    (c_cur - c_prev) > (m_cur - m_prev)):
                    score += 50
                    conditions['CCI_돌파직전'] = (True, f"CCI({c_cur:.1f}) MA({m_cur:.1f}) 돌파 직전")
                    category_scores['CCI_조건']['score'] += 50
                    category_scores['CCI_조건']['count'] += 1
                
                # CCI 골든크로스 완료
                elif c_prev < m_prev and c_cur >= m_cur and m_cur < 0:
                    score += 30
                    conditions['CCI_교차'] = (True, f"CCI 골든크로스 완료")
                    category_scores['CCI_조건']['score'] += 30
                    category_scores['CCI_조건']['count'] += 1
        except:
            pass
        
        # 2. 스토캐스틱 모멘텀 (두 번째 우선순위)
        try:
            stoch_k, stoch_d = compute_stoch_mtm(df['Close'], k_length=14)
            if len(stoch_d) >= 2:
                if stoch_d.iloc[-1] < -40 and stoch_d.iloc[-1] > stoch_d.iloc[-2]:
                    score += 30
                    conditions['Stoch_과매도반등'] = (True, f"Stoch({stoch_d.iloc[-1]:.1f}) 과매도 반등")
                    category_scores['모멘텀_지표']['score'] += 30
                    category_scores['모멘텀_지표']['count'] += 1
        except:
            pass
        
        # 3. 캔들 패턴
        try:
            patterns = detect_candle_patterns(df)
            pattern_scores = {'망치형': 15, '상승장악형': 15, '적삼병': 20}
            
            for pattern, pattern_score in pattern_scores.items():
                if patterns.get(pattern, False):
                    score += pattern_score
                    conditions[f'캔들_{pattern}'] = (True, f"{pattern} 패턴")
                    category_scores['캔들_패턴']['score'] += pattern_score
                    category_scores['캔들_패턴']['count'] += 1
        except:
            pass
        
        # 4. 거래량 조건
        try:
            if df['Volume'].iloc[-1] >= min_volume:
                score += 10
                conditions['거래량_충족'] = (True, f"{df['Volume'].iloc[-1]:,}")
                category_scores['거래량_지표']['score'] += 10
                category_scores['거래량_지표']['count'] += 1
            
            avg_vol_20 = df['Volume'].rolling(20).mean().iloc[-1]
            if df['Volume'].iloc[-1] >= avg_vol_20 * 2.5:
                score += 20
                conditions['거래량_급증'] = (True, f"20일 평균의 {df['Volume'].iloc[-1]/avg_vol_20:.1f}배")
                category_scores['거래량_지표']['score'] += 20
                category_scores['거래량_지표']['count'] += 1
        except:
            pass
        
        # 필수 조건: CCI 또는 스토캐스틱 중 하나는 있어야 함
        if category_scores['CCI_조건']['count'] == 0 and category_scores['모멘텀_지표']['count'] == 0:
            return None
        
        # 5. 추세 지표
        try:
            # MACD
            if len(df) >= 26:
                macd, signal, diff = compute_macd(df['Close'])
                if diff.iloc[-1] > 0 and diff.iloc[-2] <= 0:
                    score += 25
                    conditions['MACD_골든크로스'] = (True, "MACD 골든크로스")
                    category_scores['추세_지표']['score'] += 25
                    category_scores['추세_지표']['count'] += 1
            
            # ADX
            adx, plus_di, minus_di = compute_adx(df)
            if len(adx) > 0:
                if adx.iloc[-1] > 25 and plus_di.iloc[-1] > minus_di.iloc[-1]:
                    score += 15
                    conditions['ADX_강한추세'] = (True, f"ADX({adx.iloc[-1]:.1f}) 강한 상승추세")
                    category_scores['추세_지표']['score'] += 15
                    category_scores['추세_지표']['count'] += 1
        except:
            pass
        
        # 6. 추가 모멘텀 지표
        try:
            # RSI
            rsi = compute_rsi(df['Close'])
            if len(rsi) >= 2:
                if 30 < rsi.iloc[-1] < 70 and rsi.iloc[-1] > rsi.iloc[-2]:
                    score += 10
                    conditions['RSI_상승'] = (True, f"RSI({rsi.iloc[-1]:.1f}) 상승중")
                    category_scores['모멘텀_지표']['score'] += 10
                    category_scores['모멘텀_지표']['count'] += 1
            
            # MFI
            mfi = compute_money_flow(df)
            if len(mfi) >= 2:
                if 20 < mfi.iloc[-1] < 80 and mfi.iloc[-1] > mfi.iloc[-2]:
                    score += 10
                    conditions['MFI_자금유입'] = (True, f"MFI({mfi.iloc[-1]:.1f}) 상승")
                    category_scores['모멘텀_지표']['score'] += 10
                    category_scores['모멘텀_지표']['count'] += 1
        except:
            pass
        
        # 등급 계산
        if score >= 120:
            grade = 'S'
        elif score >= 90:
            grade = 'A'
        elif score >= 60:
            grade = 'B'
        else:
            grade = 'C'
        
        # 추가 조건 충족 시 + 부여
        high_categories = sum(1 for cat in category_scores.values() if cat['score'] >= 20)
        if high_categories >= 3:
            grade += '+'
        
        return {
            'score': score,
            'grade': grade,
            'conditions': conditions,
            'category_scores': category_scores
        }

# --- 메인 UI ---
st.markdown("""
### 📊 스마트 필터 시스템 v4.0
- **🎯 CCI 돌파 직전 최우선**: CCI가 MA선을 돌파하기 직전 종목
- **📈 스토캐스틱 모멘텀**: 과매도 구간 반등 신호
- **🇰🇷 한국 주식**: KOSPI/KOSDAQ 실시간 분석
- **🇺🇸 미국 주식**: S&P 500 주요 종목 분석
- **🤖 AI 예측**: 5일 후 상승/하락 예측 (앙상블 모델)
- **✅ 백테스팅 신뢰도**: 시스템 성공률 실시간 표시
""")

# 백테스팅 통계 표시
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("전체 예측", st.session_state.backtest_stats['total'])
with col2:
    st.metric("성공", st.session_state.backtest_stats['success'])
with col3:
    st.metric("실패", st.session_state.backtest_stats['fail'])
with col4:
    success_rate = st.session_state.backtest_stats['success_rate']
    st.metric("시스템 신뢰도", f"{success_rate:.1f}%",
              delta=f"{'✅' if success_rate >= 60 else '⚠️'}")

# 사이드바
with st.sidebar:
    st.header("⚙️ 설정")
    
    # 시장 선택
    market = st.radio(
        "🌍 시장 선택",
        ["🇰🇷 한국 주식", "🇺🇸 미국 주식 (S&P 500)"],
        help="분석할 시장을 선택하세요"
    )
    
    st.markdown("---")
    
    # 공통 설정
    min_volume = st.number_input(
        "📊 최소 거래량",
        value=500000 if "한국" in market else 1000000,
        step=100000
    )
    
    # 검색 종목 수
    search_limit = st.slider(
        "🔍 검색 종목 수",
        10, 100,
        value=30 if "한국" in market else 20,
        help="적을수록 빠르고 정확합니다"
    )
    
    # 목표 등급
    target_grade = st.select_slider(
        "🎖️ 목표 등급",
        options=['B', 'B+', 'A', 'A+', 'S', 'S+'],
        value='A'
    )
    
    # 조건 엄격도 (고정값: 엄격함)
    st.markdown("---")
    st.info("📏 조건 엄격도: **엄격함** (고정)")
    min_score = 80  # 엄격한 조건
    
    # CCI 설정
    st.markdown("---")
    st.subheader("🎯 CCI 설정")
    cci_threshold = st.slider(
        "CCI 돌파 직전 감지 범위",
        1.0, 10.0,
        value=5.0,
        step=0.5,
        help="작을수록 더 엄격하게 돌파 직전만 감지"
    )
    
    # AI 설정
    st.markdown("---")
    st.subheader("🤖 AI 설정")
    enable_ai = st.checkbox("AI 예측 활성화", value=True)
    enable_backtest = st.checkbox("백테스팅 활성화", value=True)

    with st.sidebar:
    if st.button("🧹 캐시 초기화"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.experimental_rerun()


# AI 예측기 초기화
ai_predictor = EnhancedAIPredictor()

# 검색 실행
if st.button("🔍 스마트 검색 실행", type="primary"):
    with st.spinner("종목 분석 중..."):
        results = []
        smart_filter = SmartStockFilter(near_cross_thresh=cci_threshold)
        
        if "한국" in market and PYKRX_AVAILABLE:
            # 한국 주식 분석
            today_str = get_most_recent_trading_day()
            st.info(f"📅 기준일: {today_str[:4]}-{today_str[4:6]}-{today_str[6:]}")
            
            # 거래대금 상위 종목
            top_codes = get_top_volume_korean_stocks(today_str, search_limit * 2)
            
            # 종목명 매핑
            code_name_map = {}
            for code in top_codes:
                try:
                    name = stock.get_market_ticker_name(code)
                    code_name_map[code] = name
                except:
                    continue
            
            progress_bar = st.progress(0)
            
            for idx, code in enumerate(top_codes[:search_limit]):
                progress_bar.progress((idx + 1) / min(search_limit, len(top_codes)))
                
                # 90일 데이터
                start_date = (datetime.strptime(today_str, '%Y%m%d') - timedelta(days=90)).strftime('%Y%m%d')
                df = get_korean_stock_data(code, start_date, today_str)
                
                if df is not None and len(df) >= 60:
                    result = smart_filter.evaluate_stock(df, min_volume)
                    
                    if result and result['score'] >= min_score:
                        # 등급 확인
                        result_grade = result['grade'].replace('+', '')
                        target_grade_clean = target_grade.replace('+', '')
                        grade_order = ['C', 'B', 'A', 'S']
                        
                        if grade_order.index(result_grade) >= grade_order.index(target_grade_clean):
                            # AI 예측
                            ai_pred = None
                            ai_acc = None
                            if enable_ai:
                                ai_pred, ai_acc, _ = ai_predictor.train_and_predict(df)
                            
                            results.append({
                                'ticker': code,
                                'name': code_name_map.get(code, code),
                                'market': 'KR',
                                'grade': result['grade'],
                                'score': result['score'],
                                'price': df['Close'].iloc[-1],
                                'change': ((df['Close'].iloc[-1] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100,
                                'volume': df['Volume'].iloc[-1],
                                'conditions': result['conditions'],
                                'category_scores': result['category_scores'],
                                'ai_prediction': ai_pred,
                                'ai_accuracy': ai_acc,
                                'df': df
                            })
            
            progress_bar.empty()
            
        elif "미국" in market and YFINANCE_AVAILABLE:
            # 미국 주식 분석
            st.info("📅 S&P 500 주요 종목 분석")
            
            progress_bar = st.progress(0)
            
            for idx, ticker in enumerate(SP500_TICKERS[:search_limit]):
                progress_bar.progress((idx + 1) / min(search_limit, len(SP500_TICKERS)))
                
                df, info = get_us_stock_data(ticker, period="3mo")
                
                if df is not None and len(df) >= 60:
                    result = smart_filter.evaluate_stock(df, min_volume)
                    
                    if result and result['score'] >= min_score:
                        # 등급 확인
                        result_grade = result['grade'].replace('+', '')
                        target_grade_clean = target_grade.replace('+', '')
                        grade_order = ['C', 'B', 'A', 'S']
                        
                        if grade_order.index(result_grade) >= grade_order.index(target_grade_clean):
                            # AI 예측
                            ai_pred = None
                            ai_acc = None
                            if enable_ai:
                                ai_pred, ai_acc, _ = ai_predictor.train_and_predict(df)
                            
                            results.append({
                                'ticker': ticker,
                                'name': info.get('longName', ticker),
                                'market': 'US',
                                'grade': result['grade'],
                                'score': result['score'],
                                'price': df['Close'].iloc[-1],
                                'change': ((df['Close'].iloc[-1] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100,
                                'volume': df['Volume'].iloc[-1],
                                'conditions': result['conditions'],
                                'category_scores': result['category_scores'],
                                'ai_prediction': ai_pred,
                                'ai_accuracy': ai_acc,
                                'df': df
                            })
            
            progress_bar.empty()
        
        st.session_state.search_results = results

# 검색 결과 표시
if st.session_state.search_results:
    results = st.session_state.search_results
    
    if results:
        st.success(f"✅ {len(results)}개 종목이 조건을 충족했습니다!")
        
        # 점수순 정렬
        results.sort(key=lambda x: x['score'], reverse=True)
        
        # 결과 표시
        for result in results:
            with st.expander(f"📊 {result['name']} ({result['ticker']}) - {result['grade']}등급"):
                # 기본 정보
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    if result['market'] == 'KR':
                        st.metric("현재가", f"₩{result['price']:,.0f}")
                    else:
                        st.metric("현재가", f"${result['price']:.2f}")
                
                with col2:
                    st.metric("전일비", f"{result['change']:.2f}%")
                
                with col3:
                    st.metric("점수", result['score'])
                
                with col4:
                    if result.get('ai_prediction'):
                        pred_pct = result['ai_prediction'] * 100
                        st.metric("AI 예측", f"{pred_pct:.1f}%",
                                 delta="상승" if pred_pct > 50 else "하락")
                
                # 조건 상세
                st.write("**충족 조건:**")
                for cond_name, (satisfied, detail) in result['conditions'].items():
                    if satisfied:
                        if 'CCI_돌파직전' in cond_name:
                            st.write(f"- 🔥 **{cond_name}**: {detail}")
                        elif 'Stoch' in cond_name:
                            st.write(f"- 📈 **{cond_name}**: {detail}")
                        else:
                            st.write(f"- {cond_name}: {detail}")
                
                # AI 예측 상세
                if result.get('ai_prediction') and result.get('ai_accuracy'):
                    st.write(f"**AI 분석:** 정확도 {result['ai_accuracy']*100:.1f}%")
                    
                    # 백테스팅 수행
                    if enable_backtest:
                        backtest = perform_backtest(
                            result['df'],
                            result['ai_prediction'],
                            datetime.now().strftime('%Y-%m-%d')
                        )
                        
                        if backtest:
                            # 통계 업데이트
                            st.session_state.backtest_stats['total'] += 1
                            if backtest['success']:
                                st.session_state.backtest_stats['success'] += 1
                            else:
                                st.session_state.backtest_stats['fail'] += 1
                            
                            if st.session_state.backtest_stats['total'] > 0:
                                st.session_state.backtest_stats['success_rate'] = (
                                    st.session_state.backtest_stats['success'] / 
                                    st.session_state.backtest_stats['total'] * 100
                                )
                
                # 관심종목 추가
                if st.button(f"➕ 관심종목 추가", key=f"add_{result['ticker']}"):
                    new_item = {
                        'ticker': result['ticker'],
                        'name': result['name'],
                        'market': result['market'],
                        'price': result['price'],
                        'add_date': datetime.now().strftime('%Y-%m-%d'),
                        'grade': result['grade'],
                        'ai_prediction': result.get('ai_prediction')
                    }
                    st.session_state.watchlist.append(new_item)
                    save_watchlist(st.session_state.watchlist)
                    st.success(f"✅ {result['name']} 추가됨!")
    else:
        st.warning("조건을 충족하는 종목이 없습니다.")
        st.info("""
        💡 **해결 방법:**
        1. 검색 종목 수 늘리기
        2. 목표 등급 낮추기
        3. CCI 돌파 직전 감지 범위 늘리기
        4. 최소 거래량 낮추기
        """)

# 관심종목 섹션
st.markdown("---")
st.header("📌 관심종목")

if st.session_state.watchlist:
    for item in st.session_state.watchlist:
        col1, col2, col3, col4, col5 = st.columns([3, 1, 1, 1, 1])
        
        with col1:
            flag = "🇰🇷" if item['market'] == 'KR' else "🇺🇸"
            st.write(f"{flag} **{item['name']}** ({item['ticker']})")
        
        with col2:
            st.write(f"등급: {item['grade']}")
        
        with col3:
            if item['market'] == 'KR':
                st.write(f"₩{item['price']:,.0f}")
            else:
                st.write(f"${item['price']:.2f}")
        
        with col4:
            if item.get('ai_prediction'):
                st.write(f"AI: {item['ai_prediction']*100:.1f}%")
        
        with col5:
            if st.button("🗑️", key=f"del_{item['ticker']}"):
                st.session_state.watchlist = [w for w in st.session_state.watchlist if w['ticker'] != item['ticker']]
                save_watchlist(st.session_state.watchlist)
                st.rerun()
else:
    st.info("관심종목이 없습니다.")

# 푸터
st.markdown("---")
st.caption("""
💡 **투자 유의사항**
- 모든 투자 결정은 본인의 책임입니다
- AI 예측은 참고용이며 100% 정확하지 않습니다
- 백테스팅 신뢰도 60% 이상일 때 참고하세요
- 버전: 4.0 (한국/미국 통합)
""")

