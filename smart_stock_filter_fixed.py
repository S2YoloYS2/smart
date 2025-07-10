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
from pykrx import stock
from datetime import datetime, timedelta
import pandas_market_calendars as mcal
import os
import mplfinance as mpf
import matplotlib.pyplot as plt
import concurrent.futures
import warnings
import json
from pathlib import Path
import time
import requests
from bs4 import BeautifulSoup
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import urllib.parse

warnings.filterwarnings('ignore')

# 페이지 설정
st.set_page_config(page_title="🔍 스마트 주식 필터 Pro", layout="wide")
st.title("🔍 스마트 주식 필터 Pro v3.0 - AI 예측 & 백테스팅")

# 섹터 분류 데이터 (한국 주식시장 기준)
SECTOR_MAPPING = {
    '반도체': ['삼성전자', 'SK하이닉스', 'DB하이텍', '리노공업', '하나마이크론', '원익IPS', '테스', '케이아이엔엑스', '유니테스트', '에스엠코어'],
    '배터리/2차전지': ['LG에너지솔루션', '삼성SDI', 'LG화학', '에코프로비엠', '에코프로', '포스코퓨처엠', '엘앤에프', '천보', '코스모신소재', '일진머티리얼즈'],
    '자동차': ['현대차', '기아', '현대모비스', '만도', '화신', '평화산업', '에스엘', '모토닉', '대원강업', '서연이화'],
    '조선': ['한국조선해양', '삼성중공업', '대우조선해양', 'STX조선해양', '현대미포조선'],
    '철강': ['포스코홀딩스', '현대제철', '동국제강', '세아제강', '고려제강', '대한제강', '동부제철', 'KG스틸'],
    '화학': ['LG화학', '롯데케미칼', '한화솔루션', '금호석유', 'SKC', '효성화학', '코오롱인더', '한국석유', '대한유화'],
    '정유': ['SK이노베이션', 'S-Oil', 'GS', '현대오일뱅크'],
    '건설': ['삼성물산', '현대건설', 'GS건설', '대우건설', '대림산업', 'DL이앤씨', '호반건설', '금호건설', '계룡건설'],
    '유통': ['신세계', '롯데쇼핑', '이마트', '현대백화점', 'BGF리테일', 'GS리테일', '이랜드', '하이마트'],
    '금융': ['KB금융', '신한지주', '하나금융지주', '우리금융지주', '삼성생명', '삼성화재', '현대해상', 'DB손해보험', '한화생명'],
    '통신': ['SK텔레콤', 'KT', 'LG유플러스', '케이티엠모바일', '티브로드'],
    '인터넷/게임': ['네이버', '카카오', '엔씨소프트', '넷마블', '크래프톤', '펄어비스', '컴투스', '위메이드', '넥슨게임즈'],
    '바이오/제약': ['삼성바이오로직스', '셀트리온', '한미약품', '유한양행', '대웅제약', '종근당', '녹십자', '동아에스티', 'SK바이오팜'],
    '엔터테인먼트': ['하이브', 'SM', 'YG', 'JYP', 'CJ ENM', '스튜디오드래곤', '초이스엔터테인먼트'],
    '식품/음료': ['CJ제일제당', '오리온', '농심', '롯데제과', '하이트진로', '빙그레', '삼양식품', '동원F&B', '대상'],
    '화장품': ['아모레퍼시픽', 'LG생활건강', '코스맥스', '한국콜마', '에이블씨엔씨', '토니모리'],
    '항공': ['대한항공', '아시아나항공', '제주항공', '진에어', '티웨이항공', '에어부산'],
    '호텔/여행': ['호텔신라', '롯데관광개발', '하나투어', '모두투어', '참좋은여행'],
    '방송/미디어': ['CJ ENM', 'SBS', 'JTBC스튜디오', '스튜디오드래곤', '키이스트'],
    '에너지/전력': ['한국전력', '두산에너빌리티', '포스코에너지', 'GS EPS', '지역난방공사']
}

# 종목별 섹터 찾기 함수
def get_stock_sector(stock_name):
    """종목명으로 섹터 찾기"""
    for sector, stocks in SECTOR_MAPPING.items():
        for stock in stocks:
            if stock in stock_name or stock_name in stock:
                return sector
    
    # 키워드 기반 섹터 분류
    if any(keyword in stock_name for keyword in ['전자', '반도체', '디스플레이', 'OLED']):
        return '반도체'
    elif any(keyword in stock_name for keyword in ['배터리', '2차전지', '에너지솔루션', '전지']):
        return '배터리/2차전지'
    elif any(keyword in stock_name for keyword in ['자동차', '모빌리티', '부품']):
        return '자동차'
    elif any(keyword in stock_name for keyword in ['조선', '중공업', '해양']):
        return '조선'
    elif any(keyword in stock_name for keyword in ['제철', '철강', '스틸']):
        return '철강'
    elif any(keyword in stock_name for keyword in ['화학', '케미칼', '석유화학']):
        return '화학'
    elif any(keyword in stock_name for keyword in ['정유', '오일', '에너지']):
        return '정유'
    elif any(keyword in stock_name for keyword in ['건설', '건축', '토목', '산업']):
        return '건설'
    elif any(keyword in stock_name for keyword in ['유통', '리테일', '마트', '백화점']):
        return '유통'
    elif any(keyword in stock_name for keyword in ['금융', '은행', '증권', '보험', '캐피탈']):
        return '금융'
    elif any(keyword in stock_name for keyword in ['통신', '텔레콤', '모바일']):
        return '통신'
    elif any(keyword in stock_name for keyword in ['IT', '소프트웨어', '게임', '인터넷']):
        return '인터넷/게임'
    elif any(keyword in stock_name for keyword in ['바이오', '제약', '신약', '헬스케어']):
        return '바이오/제약'
    elif any(keyword in stock_name for keyword in ['엔터', '연예', '방송', '미디어']):
        return '엔터테인먼트'
    elif any(keyword in stock_name for keyword in ['식품', '음료', '제과', 'F&B']):
        return '식품/음료'
    elif any(keyword in stock_name for keyword in ['화장품', '뷰티', '코스메틱']):
        return '화장품'
    elif any(keyword in stock_name for keyword in ['항공', '에어']):
        return '항공'
    elif any(keyword in stock_name for keyword in ['호텔', '리조트', '관광', '여행']):
        return '호텔/여행'
    else:
        return '기타'

# 세션 상태 초기화
if 'investment_mode' not in st.session_state:
    st.session_state.investment_mode = "중급자 (균형형)"

if 'watchlist' not in st.session_state:
    st.session_state.watchlist = []

if 'search_results' not in st.session_state:
    st.session_state.search_results = None

if 'show_results' not in st.session_state:
    st.session_state.show_results = False

# 파일 경로 설정 (관심종목 저장용)
WATCHLIST_FILE = "watchlist.json"

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

# 세션 시작 시 관심종목 로드
if 'watchlist_loaded' not in st.session_state:
    st.session_state.watchlist = load_watchlist()
    st.session_state.watchlist_loaded = True
# --- 기존 지표 계산 함수들 ---
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
def compute_obv(close, volume):
    obv = pd.Series(0, index=close.index)
    obv[1:] = (volume[1:] * np.sign(close[1:].diff())).cumsum() + obv[0]
    return obv

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

# --- 추가 지표 함수들 ---
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
    prev_high = df['High'].iloc[-2]
    prev_low = df['Low'].iloc[-2]
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
    
    # 4. 모닝스타 (Morning Star)
    if len(df) >= 3:
        if (df['Open'].iloc[-3] > df['Close'].iloc[-3] and  # 첫날 음봉
            abs(df['Close'].iloc[-2] - df['Open'].iloc[-2]) < body * 0.3 and  # 둘째날 작은 몸통
            df['Close'].iloc[-1] > df['Open'].iloc[-1] and  # 셋째날 양봉
            df['Close'].iloc[-1] > (df['Open'].iloc[-3] + df['Close'].iloc[-3]) / 2):  # 첫날 몸통 50% 이상 회복
            patterns['모닝스타'] = True
    
    return patterns

# --- 유틸리티 함수들 ---
@st.cache_data(ttl=3600)
def get_most_recent_trading_day():
    today = datetime.today()
    
    # 오늘이 새벽 시간대면 어제로 설정
    if today.hour < 6:  # 새벽 6시 이전이면
        today = today - timedelta(days=1)
    
    # 주말인 경우 금요일로 조정
    if today.weekday() >= 5:  # 토요일(5) 또는 일요일(6)
        days_to_subtract = today.weekday() - 4
        today = today - timedelta(days=days_to_subtract)
    
    # 최근 거래일 확인 (더 많은 날짜 확인)
    for i in range(30):  # 7 → 30으로 증가
        check_date = (today - timedelta(days=i)).strftime('%Y%m%d')
        try:
            # 해당 날짜에 데이터가 있는지 확인
            test_data = stock.get_market_ticker_list(check_date, market="KOSPI")
            if test_data and len(test_data) > 0:
                print(f"데이터 확인된 날짜: {check_date}")  # 디버깅용
                return check_date
        except Exception as e:
            print(f"날짜 {check_date} 오류: {str(e)}")  # 디버깅용
            continue
    
    # 모두 실패 시 어제 날짜 강제 반환
    yesterday = datetime.today() - timedelta(days=1)
    if yesterday.weekday() >= 5:  # 주말이면 금요일로
        days_to_subtract = yesterday.weekday() - 4
        yesterday = yesterday - timedelta(days=days_to_subtract)
    return yesterday.strftime('%Y%m%d')

@st.cache_data(ttl=3600)
def get_name_code_map():
    today_str = get_most_recent_trading_day()
    if today_str is None:
        return {}, {}
    
    name_code = {}
    code_name = {}
    
    try:
        for market in ['KOSPI', 'KOSDAQ']:
            try:
                tickers = stock.get_market_ticker_list(today_str, market=market)
                for code in tickers:
                    try:
                        name = stock.get_market_ticker_name(code)
                        if name:
                            name_code[name] = code
                            code_name[code] = name
                    except:
                        continue
            except:
                continue
        
        return name_code, code_name
    except:
        return {}, {}

@st.cache_data(ttl=3600)
def get_ohlcv_df(ticker, start_date_str, end_date_str):
    try:
        df_ohlcv = stock.get_market_ohlcv_by_date(start_date_str, end_date_str, ticker)
        if df_ohlcv.empty:
            return pd.DataFrame()
        
        df_ohlcv.index = pd.to_datetime(df_ohlcv.index)
        df_ohlcv.columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Change']
        
        try:
            df_marcap = stock.get_market_cap_by_date(start_date_str, end_date_str, ticker)
            if not df_marcap.empty:
                df_marcap.index = pd.to_datetime(df_marcap.index)
                df_ohlcv = df_ohlcv.join(df_marcap[['시가총액']], how='left')
                df_ohlcv.rename(columns={'시가총액': 'MarketCap'}, inplace=True)
            else:
                df_ohlcv['MarketCap'] = np.nan
        except:
            df_ohlcv['MarketCap'] = np.nan

        df_ohlcv['TradingValue'] = df_ohlcv['Close'] * df_ohlcv['Volume']
        
        return df_ohlcv
    except:
        return pd.DataFrame()

# --- 빠른 필터링을 위한 함수 (수정됨) ---
@st.cache_data(ttl=3600)
def get_top_volume_stocks(today_str, top_n=200):
    """거래대금 상위 종목 빠르게 가져오기"""
    try:
        volume_data = []
        
        # KOSPI 상위 종목만 (속도 개선)
        df_kospi = stock.get_market_ohlcv_by_ticker(today_str, market="KOSPI")
        
        if not df_kospi.empty:
            # 거래대금 계산
            df_kospi['거래대금'] = df_kospi['거래량'] * df_kospi['종가']
            # 거래대금 상위 종목만 선택
            df_kospi = df_kospi.nlargest(min(top_n, len(df_kospi)), '거래대금')
            
            return df_kospi.index.tolist()
        else:
            return []
            
    except Exception as e:
        print(f"거래대금 상위 종목 가져오기 실패: {str(e)}")
        return []

# --- 뉴스 크롤링 함수 ---
@st.cache_data(ttl=3600)
def get_stock_news(stock_name, limit=5):
    """네이버 금융에서 종목 뉴스 크롤링"""
    try:
        # URL 인코딩
        encoded_name = urllib.parse.quote(stock_name)
        url = f"https://search.naver.com/search.naver?where=news&query={encoded_name}+주가"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        news_items = []
        news_list = soup.select('div.news_area')[:limit]
        
        for item in news_list:
            try:
                title_elem = item.select_one('a.news_tit')
                if title_elem:
                    title = title_elem.text.strip()
                    link = title_elem.get('href', '')
                    
                    # 날짜 추출
                    date_elem = item.select_one('span.info')
                    date = date_elem.text.strip() if date_elem else '날짜 없음'
                    
                    # 요약 추출
                    desc_elem = item.select_one('div.dsc_txt')
                    description = desc_elem.text.strip()[:100] + '...' if desc_elem else ''
                    
                    news_items.append({
                        'title': title,
                        'link': link,
                        'date': date,
                        'description': description
                    })
            except:
                continue
        
        return news_items
    except Exception as e:
        print(f"뉴스 크롤링 실패: {str(e)}")
        return []

# --- AI 예측 모델 함수 ---
@st.cache_data
def prepare_features_for_ml(df):
    """머신러닝을 위한 특징 준비"""
    features = pd.DataFrame(index=df.index)
    
    # 가격 변화율
    features['returns_1d'] = df['Close'].pct_change(1)
    features['returns_5d'] = df['Close'].pct_change(5)
    features['returns_20d'] = df['Close'].pct_change(20)
    
    # 이동평균
    features['ma5'] = df['Close'].rolling(5).mean() / df['Close'] - 1
    features['ma20'] = df['Close'].rolling(20).mean() / df['Close'] - 1
    features['ma60'] = df['Close'].rolling(60).mean() / df['Close'] - 1
    
    # RSI
    features['rsi'] = compute_rsi(df['Close'])
    
    # 볼린저 밴드
    _, upper, lower, _ = compute_bollinger_bands(df['Close'])
    features['bb_position'] = (df['Close'] - lower) / (upper - lower + 1e-9)
    
    # 거래량
    features['volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
    
    # CCI
    features['cci'] = compute_cci(df['High'], df['Low'], df['Close'])
    
    # 타겟: 5일 후 상승 여부
    features['target'] = (df['Close'].shift(-5) > df['Close']).astype(int)
    
    return features.dropna()

@st.cache_data
def train_ai_model(df):
    """AI 예측 모델 학습"""
    features = prepare_features_for_ml(df)
    
    if len(features) < 100:
        return None, None
    
    # 특징과 타겟 분리
    feature_cols = ['returns_1d', 'returns_5d', 'returns_20d', 'ma5', 'ma20', 
                    'ma60', 'rsi', 'bb_position', 'volume_ratio', 'cci']
    
    X = features[feature_cols]
    y = features['target']
    
    # 학습/테스트 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X[:-1], y[:-1], test_size=0.2, random_state=42, shuffle=False
    )
    
    # 스케일링
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 모델 학습
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # 정확도
    accuracy = model.score(X_test_scaled, y_test)
    
    # 현재 예측
    current_features = scaler.transform(X.iloc[-1:])
    prediction_proba = model.predict_proba(current_features)[0][1]
    
    return prediction_proba, accuracy

# --- 백테스팅 함수 ---
@st.cache_data
def backtest_strategy(df, conditions_met_dates):
    """전략 백테스팅"""
    results = []
    
    for date in conditions_met_dates:
        if date in df.index:
            entry_price = df.loc[date, 'Close']
            entry_idx = df.index.get_loc(date)
            
            # 5일, 10일, 20일 후 수익률 계산
            for days in [5, 10, 20]:
                if entry_idx + days < len(df):
                    exit_date = df.index[entry_idx + days]
                    exit_price = df.iloc[entry_idx + days]['Close']
                    returns = (exit_price - entry_price) / entry_price * 100
                    
                    results.append({
                        'entry_date': date,
                        'exit_date': exit_date,
                        'holding_days': days,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'returns': returns
                    })
    
    return pd.DataFrame(results)
# --- 메인 UI ---
st.markdown("""
### 📊 스마트 필터 시스템 v3.0
- **중급자 모드**: CCI 또는 거래량 조건, 캔들패턴 + 추세지표 가산점
- **고급자 모드**: CCI 또는 캔들패턴 필수, 모든 지표 활용
- **NEW**: 🤖 AI 예측, 📈 백테스팅, 📰 뉴스 분석 기능 추가
""")

# 사이드바
with st.sidebar:
    st.header("⚙️ 설정")
    st.write("테스트 버전입니다.")

# --- 관심종목 성과 계산 함수 ---
def calculate_watchlist_performance():
    """관심종목의 7일 성과 계산"""
    today_str = get_most_recent_trading_day()
    if not today_str:
        return []
    
    updated_watchlist = []
    
    for item in st.session_state.watchlist:
        try:
            # 현재 가격 조회
            current_date = datetime.strptime(today_str, '%Y%m%d')
            add_date = datetime.strptime(item['add_date'], '%Y-%m-%d')
            days_passed = (current_date - add_date).days
            
            # 7일 이상 경과한 종목만 성과 계산
            if days_passed >= 7 and item.get('status') == 'watching':
                # 7일간의 데이터 조회 (최저가 확인용)
                start_date_str = add_date.strftime('%Y%m%d')
                df = get_ohlcv_df(item['code'], start_date_str, today_str)
                
                if not df.empty:
                    current_price = df['Close'].iloc[-1]
                    
                    # 기간 중 최저가 찾기
                    lowest_price = df['Low'].min()
                    
                    # 매수가 대비 수익률
                    return_rate = ((current_price - item['price']) / item['price']) * 100
                    
                    # 최저가 대비 상승률
                    rise_from_low = ((current_price - lowest_price) / lowest_price) * 100
                    
                    item['current_price'] = current_price
                    item['return_rate'] = return_rate
                    item['lowest_price'] = lowest_price
                    item['rise_from_low'] = rise_from_low
                    item['days_passed'] = days_passed
                    
                    # 성공 조건: 매수가 대비 5% 이상 또는 최저가 대비 5% 이상 상승
                    if return_rate >= 5 or rise_from_low >= 5:
                        item['status'] = 'success'
                        item['success_reason'] = '매수가 대비' if return_rate >= 5 else '최저가 대비'
                    elif days_passed > 7:
                        item['status'] = 'expired'
        except:
            pass
        
        updated_watchlist.append(item)
    
    return updated_watchlist

# --- 매수 추천 분석 함수 ---
def analyze_buy_recommendation(result, stock_name):
    """종목의 매수 추천 분석"""
    recommendation = {
        'buy_score': 0,  # 매수 점수 (0~100)
        'recommendation': '',  # 추천 내용
        'reasons': [],  # 매수 이유
        'risks': [],  # 리스크 요인
        'strategy': ''  # 매수 전략
    }
    
    # 등급별 기본 점수
    grade_scores = {
        'S+': 95, 'S': 90,
        'A+': 85, 'A': 80,
        'B+': 70, 'B': 60,
        'C': 40
    }
    
    base_score = grade_scores.get(result['grade'], 50)
    recommendation['buy_score'] = base_score
    
    # 조건별 분석
    category_scores = result['category_scores']
    
    # 1. CCI 조건 분석
    if category_scores['CCI_조건']['count'] > 0:
        if 'CCI_교차' in str(result['conditions']):
            recommendation['reasons'].append("✅ CCI 골든크로스 발생 - 매수 타이밍 우수")
            recommendation['buy_score'] += 5
        elif 'CCI_접근' in str(result['conditions']):
            recommendation['reasons'].append("📈 CCI가 평균선 접근 중 - 반등 예상")
            recommendation['buy_score'] += 3
    
    # 2. 캔들 패턴 분석
    if category_scores['캔들_패턴']['count'] > 0:
        recommendation['reasons'].append("🕯️ 상승 반전 캔들 패턴 출현")
        recommendation['buy_score'] += 3
    
    # 3. 추세 지표 분석
    if category_scores['추세_지표']['count'] >= 2:
        recommendation['reasons'].append("📊 다수의 추세 지표가 상승 신호")
        recommendation['buy_score'] += 5
    
    # 4. 거래량 지표 분석
    if category_scores['거래량_지표']['count'] > 0:
        if '거래량_증가' in str(result['conditions']):
            recommendation['reasons'].append("💹 거래량 급증 - 세력 개입 가능성")
            recommendation['buy_score'] += 3
    
    # 매수 추천 결정
    if recommendation['buy_score'] >= 85:
        recommendation['recommendation'] = "🔥 적극 매수"
        recommendation['strategy'] = "즉시 매수 또는 분할 매수 시작"
    elif recommendation['buy_score'] >= 75:
        recommendation['recommendation'] = "✅ 매수 추천"
        recommendation['strategy'] = "소량 매수 후 추가 매수 대기"
    elif recommendation['buy_score'] >= 65:
        recommendation['recommendation'] = "👀 관심 종목"
        recommendation['strategy'] = "추가 신호 확인 후 매수"
    else:
        recommendation['recommendation'] = "⏸️ 대기"
        recommendation['strategy'] = "더 명확한 신호를 기다리세요"
    
    # 리스크 요인 체크
    if result['score'] < 50:
        recommendation['risks'].append("⚠️ 전체 점수가 낮음 - 신중한 접근 필요")
    
    if category_scores['모멘텀_지표']['count'] == 0:
        recommendation['risks'].append("📉 모멘텀 지표 미충족 - 단기 조정 가능")
    
    return recommendation

# --- 스마트 조건 평가 시스템 ---
class SmartStockFilter:
    """중급자/고급자용 스마트 필터"""
    
    def __init__(self, mode='intermediate'):
        self.mode = mode  # 'intermediate' or 'advanced'
        
    def evaluate_stock(self, df, min_volume, min_market_cap):
        """종목 평가 - 조건 완화 버전"""
        
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
        
        # 1. CCI 조건 체크 ― ★ near-cross(직전 교차) 로직 추가 ★
    try:
        cci = compute_cci(df['High'], df['Low'], df['Close'])
        cci_ma = compute_cci_ma(cci)

        if len(cci) >= 2:
            curr_cci, prev_cci = cci.iloc[-1], cci.iloc[-2]
            curr_ma,  prev_ma  = cci_ma.iloc[-1], cci_ma.iloc[-2]
            diff = curr_ma - curr_cci      # MA – CCI (+면 CCI가 아래)

            # 1-A. 직전 교차(near-cross): 아직 교차 전, 간격 ≤ 5pt, CCI 상승중
            if (prev_cci < prev_ma and       # 이전 봉: CCI < MA
                curr_cci < curr_ma and       # 아직 교차하지 않음
                0 < diff <= 5 and            # 간격이 5포인트 이하
                curr_cci > prev_cci):        # CCI 상승중
                score += 40
                conditions['CCI_직전교차'] = (True,
                    f\"CCI {curr_cci:.1f}, MA {curr_ma:.1f} Δ={diff:.1f} 직전 교차\")
                category_scores['CCI_조건']['score'] += 40
                category_scores['CCI_조건']['count'] += 1
                category_scores['CCI_조건']['conditions'].append('CCI_직전교차')

            # 1-B. 골든크로스(완료)
            elif prev_cci < prev_ma and curr_cci >= curr_ma and curr_ma < 0:
                score += 35
                conditions['CCI_교차'] = (True,
                    f\"CCI({prev_cci:.1f}→{curr_cci:.1f}) 골든크로스\")
                category_scores['CCI_조건']['score'] += 35
                category_scores['CCI_조건']['count'] += 1
                category_scores['CCI_조건']['conditions'].append('CCI_교차')

            # 1-C. 접근(완화: -60까지)
            elif (curr_cci < curr_ma and curr_cci > prev_cci and
                  curr_ma < 0 and curr_cci >= -60):
                score += 30
                conditions['CCI_접근'] = (True,
                    f\"CCI({curr_cci:.1f}) MA 접근중\")
                category_scores['CCI_조건']['score'] += 30
                category_scores['CCI_조건']['count'] += 1
                category_scores['CCI_조건']['conditions'].append('CCI_접근')

            # 1-D. 상승 전환
            elif prev_cci < -50 and curr_cci > prev_cci and (curr_cci - prev_cci) > 5:
                score += 25
                conditions['CCI_상승전환'] = (True,
                    f\"CCI({prev_cci:.1f}→{curr_cci:.1f}) 상승 전환\")
                category_scores['CCI_조건']['score'] += 25
                category_scores['CCI_조건']['count'] += 1
                category_scores['CCI_조건']['conditions'].append('CCI_상승전환')

            # 1-E. 과매도
            elif curr_cci < -50:
                score += 15
                conditions['CCI_과매도'] = (True,
                    f\"CCI({curr_cci:.1f}) 과매도 구간\")
                category_scores['CCI_조건']['score'] += 15
                category_scores['CCI_조건']['count'] += 1
                category_scores['CCI_조건']['conditions'].append('CCI_과매도')

    except Exception:
        pass
        
        # 2. 캔들 패턴 체크
        try:
            patterns = detect_candle_patterns(df)
            pattern_scores = {
                '망치형': 15,
                '상승장악형': 15,
                '적삼병': 20,
                '모닝스타': 18
            }
            
            for pattern, pattern_score in pattern_scores.items():
                if patterns.get(pattern, False):
                    score += pattern_score
                    conditions[f'캔들_{pattern}'] = (True, f"{pattern} 패턴 감지")
                    category_scores['캔들_패턴']['score'] += pattern_score
                    category_scores['캔들_패턴']['count'] += 1
                    category_scores['캔들_패턴']['conditions'].append(pattern)
        except:
            pass
        
        # 3. 거래량 조건 (완화)
        try:
            # 거래량 충족 (원래 조건의 70%만 만족해도 OK)
            if df['Volume'].iloc[-1] >= min_volume * 0.7:
                score += 5
                conditions['거래량_충족'] = (True, f"{df['Volume'].iloc[-1]:,}")
                category_scores['거래량_지표']['score'] += 5
                category_scores['거래량_지표']['count'] += 1
            
            # 거래량 증가 (20일 평균의 2배 이상으로 완화)
            avg_vol_20 = df['Volume'].rolling(20).mean().iloc[-1]
            if df['Volume'].iloc[-1] >= avg_vol_20 * 2:  # 3배 → 2배
                score += 15
                conditions['거래량_증가'] = (True, f"20일 평균의 {df['Volume'].iloc[-1]/avg_vol_20:.1f}배")
                category_scores['거래량_지표']['score'] += 15
                category_scores['거래량_지표']['count'] += 1
        except:
            pass
        
        # 중급자 모드: CCI + 거래량 중 하나라도 있으면 OK (완화)
        if self.mode == 'intermediate':
            if category_scores['CCI_조건']['count'] == 0 and category_scores['거래량_지표']['count'] == 0:
                return None
        
        # 고급자 모드: CCI 또는 캔들패턴 중 하나
        elif self.mode == 'advanced':
            if category_scores['CCI_조건']['count'] == 0 and category_scores['캔들_패턴']['count'] == 0:
                return None
        
        # 4. 추세 지표
        try:
            # MA 골든크로스
            ma5 = df['Close'].rolling(5).mean()
            ma20 = df['Close'].rolling(20).mean()
            ma60 = df['Close'].rolling(60).mean()
            
            if len(ma5) >= 2 and len(ma20) >= 2:
                if ma5.iloc[-1] > ma20.iloc[-1] and ma5.iloc[-2] <= ma20.iloc[-2]:
                    score += 15
                    conditions['MA_골든크로스'] = (True, f"5일선이 20일선 상향돌파")
                    category_scores['추세_지표']['score'] += 15
                    category_scores['추세_지표']['count'] += 1
            
            # VWAP 돌파
            vwap = compute_vwap(df)
            if len(vwap) > 0:
                if df['Close'].iloc[-1] > vwap.iloc[-1]:
                    score += 10
                    conditions['VWAP_돌파'] = (True, f"현재가 > VWAP({vwap.iloc[-1]:.0f})")
                    category_scores['추세_지표']['score'] += 10
                    category_scores['추세_지표']['count'] += 1
            
            # ADX 추세
            adx, plus_di, minus_di = compute_adx(df)
            if len(adx) > 0:
                if adx.iloc[-1] > 25 and plus_di.iloc[-1] > minus_di.iloc[-1]:
                    score += 15
                    conditions['ADX_강한추세'] = (True, f"ADX({adx.iloc[-1]:.1f}) 강한 상승추세")
                    category_scores['추세_지표']['score'] += 15
                    category_scores['추세_지표']['count'] += 1
            
            # 52주 신고가
            if len(df) >= 252:
                high_52w = df['High'].rolling(252).max().iloc[-1]
                if df['Close'].iloc[-1] >= high_52w * 0.90:  # 0.95 → 0.90으로 완화
                    score += 20
                    conditions['52주_신고가'] = (True, f"52주 최고가의 {df['Close'].iloc[-1]/high_52w*100:.1f}%")
                    category_scores['추세_지표']['score'] += 20
                    category_scores['추세_지표']['count'] += 1
        except:
            pass
        
        # 5. 모멘텀 지표 (고급자 모드에서만)
        if self.mode == 'advanced':
            try:
                # RSI
                rsi = compute_rsi(df['Close'])
                if len(rsi) >= 2:
                    # RSI 조건 완화
                    if 25 < rsi.iloc[-1] < 75 and rsi.iloc[-1] > rsi.iloc[-2]:
                        score += 10
                        conditions['RSI_상승'] = (True, f"RSI({rsi.iloc[-1]:.1f}) 상승중")
                        category_scores['모멘텀_지표']['score'] += 10
                        category_scores['모멘텀_지표']['count'] += 1
        
                # 스토캐스틱 조건 (완화)
                stoch_k5, stoch_ema5 = compute_stoch_mtm(df['Close'], k_length=5)
                if len(stoch_ema5) >= 2:
                    current_stoch_ema5 = stoch_ema5.iloc[-1]
                    prev_stoch_ema5 = stoch_ema5.iloc[-2]
                    
                    # -40 → -30으로 완화
                    if current_stoch_ema5 < -30 and current_stoch_ema5 > prev_stoch_ema5:
                        score += 10
                        conditions['Stoch_과매도반등'] = (True, f"Stoch({current_stoch_ema5:.1f}) 반등")
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
                
                # 볼린저 밴드
                _, upper, lower, _ = compute_bollinger_bands(df['Close'])
                if len(upper) > 0:
                    if df['Close'].iloc[-1] > upper.iloc[-1]:
                        score += 15
                        conditions['BB_상단돌파'] = (True, "볼린저밴드 상단 돌파")
                        category_scores['모멘텀_지표']['score'] += 15
                        category_scores['모멘텀_지표']['count'] += 1
            except:
                pass
        
        # 6. 기관/외국인 수급 (간단 버전)
            try:
                # 최근 5일 수급 데이터
                recent_date = df.index[-1].strftime('%Y%m%d')
                df_trading = stock.get_market_trading_value_by_date(recent_date, recent_date, code)
    
                if not df_trading.empty:
                    inst_net = df_trading['기관합계'].iloc[-1]
                    foreign_net = df_trading['외국인합계'].iloc[-1]
        
                if inst_net > 0 and foreign_net > 0:
                    score += 20
                    conditions['수급_동시매수'] = (True, "기관/외인 동시 순매수")
                    category_scores['거래량_지표']['score'] += 20
                    category_scores['거래량_지표']['count'] += 1
            except:
                pass  # 오류 무시
            
        # 등급 계산
        grade = self.calculate_grade(score, category_scores)
        
        return {
            'score': score,
            'grade': grade,
            'conditions': conditions,
            'category_scores': category_scores
        }
    
    def calculate_grade(self, score, category_scores):
        """점수와 카테고리별 충족도를 고려한 등급"""
        if self.mode == 'intermediate':
            # 중급자: 점수 기준 완화
            if score >= 80:  # 100 → 80
                base_grade = 'A'
            elif score >= 50:  # 70 → 50
                base_grade = 'B'
            else:
                base_grade = 'C'
            
            # 캔들패턴 + 추세지표 중 하나라도 있으면 + (완화)
            if category_scores['캔들_패턴']['count'] > 0 or category_scores['추세_지표']['count'] >= 1:
                base_grade += '+'
        
        else:  # advanced
            # 고급자: 점수 기준 완화
            if score >= 120:  # 150 → 120
                base_grade = 'S'
            elif score >= 80:  # 120 → 80
                base_grade = 'A'
            else:
                base_grade = 'B'
            
            # 2개 이상 카테고리에서 점수가 있으면 + (완화)
            high_categories = sum(1 for cat in category_scores.values() if cat['score'] >= 10)
            if high_categories >= 2:
                base_grade += '+'
        
        return base_grade

# 사이드바 계속
with st.sidebar:
    # 모드 선택 - key 추가로 경고 해결
    mode = st.radio(
        "🎯 투자 스타일",
        ["중급자 (균형형)", "고급자 (공격형)"],
        help="중급자는 안정성, 고급자는 수익성 중심",
        key="investment_mode"
    )
    
    filter_mode = 'intermediate' if "중급자" in mode else 'advanced'
    
    st.markdown("---")
    
    # 조건 설정
    min_volume = st.number_input(
        "📊 최소 거래량", 
        value=300000 if filter_mode == 'intermediate' else 200000,  # 더 낮춤
        step=50000
    )
    
    min_market_cap = st.number_input(
        "💰 최소 시가총액 (억원)", 
        value=300 if filter_mode == 'intermediate' else 200,  # 더 낮춤
        step=100
    ) * 100_000_000
    
    # 검색 종목 수
    search_limit = st.slider(
        "🔍 검색 종목 수",
        50, 500, 
        value=150,  # 기본값 상향
        step=50,
        help="많을수록 정확하지만 느려집니다"
    )
    
    # 목표 등급
    if filter_mode == 'intermediate':
        target_grade = st.select_slider(
            "🎖️ 목표 등급",
            options=['C', 'B', 'B+', 'A', 'A+'],
            value='B'  # 기본값을 B로 낮춤
        )
    else:
        target_grade = st.select_slider(
            "🎖️ 목표 등급",
            options=['B', 'B+', 'A', 'A+', 'S', 'S+'],
            value='A'  # 기본값을 A로 낮춤
        )
    
    # 빠른 검색 옵션
    st.markdown("---")
    quick_search = st.checkbox("⚡ 빠른 검색 모드", value=True, help="KOSPI 상위 종목만 검색")
    
    # 조건 엄격도
    st.markdown("---")
    condition_strictness = st.radio(
        "📏 조건 엄격도",
        ["느슨함", "보통", "엄격함"],
        index=0,  # 기본값: 느슨함
        help="느슨함: 더 많은 종목 검색, 엄격함: 정확한 조건만"
    )
    
    # 섹터 필터 추가
    st.markdown("---")
    st.subheader("🏢 섹터 필터")
    selected_sectors = st.multiselect(
        "특정 섹터만 검색",
        options=list(SECTOR_MAPPING.keys()),
        default=[],
        help="선택하지 않으면 전체 섹터 검색"
    )
    
    # AI 기능 활성화
    st.markdown("---")
    st.subheader("🤖 AI 기능")
    enable_ai = st.checkbox("AI 예측 활성화", value=True)
    enable_backtest = st.checkbox("백테스팅 활성화", value=True)
    enable_news = st.checkbox("뉴스 분석 활성화", value=True)

# 검색 버튼
if st.button("🔍 스마트 검색 실행", type="primary"):
    st.session_state.show_results = True
    with st.spinner("종목 분석 중..."):
        # 데이터 로딩
        today_str = get_most_recent_trading_day()
        if not today_str:
            st.error("개장일 정보를 가져올 수 없습니다.")
            st.stop()
        
        st.info(f"📅 기준일: {today_str[:4]}-{today_str[4:6]}-{today_str[6:]}")
        
        # 종목 리스트 가져오기
        if quick_search:
            # 빠른 검색: 거래대금 상위 종목만
            with st.spinner("거래대금 상위 종목을 가져오는 중..."):
                top_volume_codes = get_top_volume_stocks(today_str, search_limit)
                
                if not top_volume_codes:
                    st.warning("거래대금 상위 종목을 가져올 수 없습니다. 전체 검색으로 전환합니다.")
                    name_code_map, code_name_map = get_name_code_map()
                    top_volume_codes = list(code_name_map.keys())[:search_limit]
                else:
                    _, code_name_map = get_name_code_map()
        else:
            # 전체 검색
            name_code_map, code_name_map = get_name_code_map()
            if not name_code_map:
                st.error("종목 정보를 가져올 수 없습니다.")
                st.stop()
            
            top_volume_codes = list(code_name_map.keys())[:search_limit]
        
        # 스마트 필터 실행
        smart_filter = SmartStockFilter(mode=filter_mode)
        results = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # 배치 처리
        batch_size = 10
        total_codes = len(top_volume_codes)
        
        # 조건 엄격도에 따른 최소 점수 조정
        min_score_map = {
            "느슨함": {"intermediate": 30, "advanced": 40},
            "보통": {"intermediate": 50, "advanced": 60},
            "엄격함": {"intermediate": 70, "advanced": 80}
        }
        min_score = min_score_map[condition_strictness][filter_mode]
        
        for batch_idx in range(0, total_codes, batch_size):
            batch_codes = top_volume_codes[batch_idx:batch_idx + batch_size]
            
            for idx, code in enumerate(batch_codes):
                current_idx = batch_idx + idx
                progress_bar.progress(
                    current_idx / total_codes
                )
                status_text.text(f"분석 중... {current_idx}/{total_codes} - {code_name_map.get(code, code)}")
                
                try:
                    # 90일 데이터만 가져오기 (속도 개선)
                    start_date = (datetime.strptime(today_str, '%Y%m%d') - timedelta(days=90)).strftime('%Y%m%d')
                    df = get_ohlcv_df(code, start_date, today_str)
                    
                    if df.empty or len(df) < 60:
                        continue
                    
                    # 빠른 필터링: 거래량이 너무 적으면 스킵
                    if df['Volume'].iloc[-1] < min_volume * 0.3:  # 더 낮춤
                        continue
                    
                    result = smart_filter.evaluate_stock(df, min_volume, min_market_cap)
                    
                    if result and result['score'] >= min_score:  # 최소 점수 체크
                        # 목표 등급 확인 (더 유연하게)
                        result_grade = result['grade'].replace('+', '')  # A+ → A로 변환
                        target_grade_clean = target_grade.replace('+', '')
                        
                        grade_order = ['C', 'B', 'A', 'S']
                        
                        if grade_order.index(result_grade) >= grade_order.index(target_grade_clean):
                            # 현재가 정보 추가
                            current_price = df['Close'].iloc[-1]
                            prev_close = df['Close'].iloc[-2]
                            change_pct = (current_price - prev_close) / prev_close * 100
                            
                            # 섹터 정보 추가
                            stock_sector = get_stock_sector(code_name_map.get(code, f"Unknown({code})"))
                            
                            # 섹터 필터링
                            if selected_sectors and stock_sector not in selected_sectors:
                                continue
                            
                            # AI 예측 추가
                            ai_prediction = None
                            ai_accuracy = None
                            if enable_ai:
                                try:
                                    # 더 긴 기간 데이터로 AI 학습
                                    long_start = (datetime.strptime(today_str, '%Y%m%d') - timedelta(days=365)).strftime('%Y%m%d')
                                    long_df = get_ohlcv_df(code, long_start, today_str)
                                    if len(long_df) >= 100:
                                        ai_prediction, ai_accuracy = train_ai_model(long_df)
                                except:
                                    pass
                            
                            results.append({
                                'code': code,
                                'name': code_name_map.get(code, f"Unknown({code})"),
                                'sector': stock_sector,
                                'grade': result['grade'],
                                'score': result['score'],
                                'price': current_price,
                                'change': change_pct,
                                'volume': df['Volume'].iloc[-1],
                                'conditions': result['conditions'],
                                'category_scores': result['category_scores'],
                                'ai_prediction': ai_prediction,
                                'ai_accuracy': ai_accuracy,
                                'df': df  # 백테스팅용
                            })
                            
                            # 목표 개수 도달 시 조기 종료
                            if filter_mode == 'intermediate' and len(results) >= 20:  # 10 → 20
                                break
                            elif filter_mode == 'advanced' and len(results) >= 30:  # 20 → 30
                                break
                
                except Exception as e:
                    # 오류 발생 시 스킵
                    continue
            
            # 충분한 결과가 나왔으면 중단
            if (filter_mode == 'intermediate' and len(results) >= 20) or \
               (filter_mode == 'advanced' and len(results) >= 30):
                break
        
        progress_bar.empty()
        status_text.empty()
        
        # 결과를 세션 상태에 저장
        st.session_state.search_results = results
# 검색 결과 표시 (세션 상태에서 가져오기)
if st.session_state.show_results and st.session_state.search_results is not None:
    results = st.session_state.search_results
    
    if results:
        st.success(f"✅ {len(results)}개 종목이 조건을 충족했습니다!")
        
        # 등급별 정렬 (수정: 같은 등급 내에서 점수 순으로 정렬)
        results.sort(key=lambda x: (x['grade'], x['score']), reverse=True)
        
        # 등급별 그룹화
        grade_groups = {}
        for result in results:
            grade = result['grade']
            if grade not in grade_groups:
                grade_groups[grade] = []
            grade_groups[grade].append(result)
        
        # 등급별 표시
        for grade in sorted(grade_groups.keys(), reverse=True):
            stocks = grade_groups[grade]
            
            # 같은 등급 내에서 점수 순으로 정렬
            stocks.sort(key=lambda x: x['score'], reverse=True)
            
            st.subheader(f"🏆 {grade}등급 ({len(stocks)}개)")
            
            # 요약 테이블
            summary_data = []
            for stock in stocks[:10]:  # 각 등급당 최대 10개
                # 주요 충족 조건 요약
                main_conditions = []
                for cond_name, (satisfied, _) in stock['conditions'].items():
                    if satisfied and any(key in cond_name for key in ['CCI', '캔들', 'MA', '52주']):
                        main_conditions.append(cond_name.split('_')[0])
                
                # 매수 추천 분석
                buy_rec = analyze_buy_recommendation(stock, stock['name'])
                
                # AI 예측 포맷팅
                ai_pred_str = "-"
                if stock.get('ai_prediction') is not None:
                    ai_pred_str = f"{stock['ai_prediction']*100:.1f}%"
                    if stock.get('ai_accuracy'):
                        ai_pred_str += f" (정확도: {stock['ai_accuracy']*100:.1f}%)"
                
                summary_data.append({
                    '종목명': stock['name'],
                    '섹터': stock['sector'],
                    '코드': stock['code'],
                    '현재가': f"{stock['price']:,.0f}",
                    '전일비': f"{stock['change']:+.2f}%",
                    '거래량': f"{stock['volume']:,}",
                    '점수': stock['score'],
                    '매수추천': buy_rec['recommendation'],
                    'AI예측': ai_pred_str,
                    '주요신호': ', '.join(main_conditions[:3])  # 상위 3개만
                })
            
            df_summary = pd.DataFrame(summary_data)
            
            # 관심종목 추가 버튼을 각 행에 추가
            # 폼을 사용한 관심종목 추가
            for idx, row in df_summary.iterrows():
                with st.form(key=f"form_{row['코드']}_{grade}_{idx}"):
                    col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11 = st.columns([2, 1.5, 1, 1.5, 1, 1.5, 1, 1.5, 2, 2, 1])
                    
                    with col1:
                        st.write(row['종목명'])
                    with col2:
                        st.write(row['섹터'])
                    with col3:
                        st.write(row['코드'])
                    with col4:
                        st.write(row['현재가'])
                    with col5:
                        st.write(row['전일비'])
                    with col6:
                        st.write(row['거래량'])
                    with col7:
                        st.write(row['점수'])
                    with col8:
                        st.write(row['매수추천'])
                    with col9:
                        st.write(row['AI예측'])
                    with col10:
                        st.write(row['주요신호'])
                    with col11:
                        # 이미 관심종목에 있는지 확인
                        is_in_watchlist = any(item['code'] == row['코드'] for item in st.session_state.watchlist)
                        
                        if not is_in_watchlist:
                            if st.form_submit_button("➕", help="관심종목 추가"):
                                # 관심종목에 추가
                                new_stock = {
                                    'code': row['코드'],
                                    'name': row['종목명'],
                                    'sector': row['섹터'],
                                    'price': float(row['현재가'].replace(',', '')),
                                    'add_date': datetime.now().strftime('%Y-%m-%d'),
                                    'grade': grade,
                                    'score': stocks[idx]['score'],
                                    'status': 'watching'
                                }
                                st.session_state.watchlist.append(new_stock)
                                save_watchlist(st.session_state.watchlist)
                                st.success(f"✅ {row['종목명']} 관심종목에 추가됨!")
                        else:
                            st.write("✅")
                        
            # 상세 정보 (확장 가능)
            with st.expander("📋 상세 분석 보기"):
                for stock in stocks[:5]:
                    st.markdown(f"### {stock['name']} ({stock['code']}) - {stock['sector']}")
                    
                    # 탭으로 구성
                    tab1, tab2, tab3, tab4 = st.tabs(["📊 기술적 분석", "🤖 AI 예측", "📈 백테스팅", "📰 최신 뉴스"])
                    
                    with tab1:
                        # 매수 추천 분석 상세
                        buy_rec = analyze_buy_recommendation(stock, stock['name'])
                        
                        # 매수 추천 박스
                        if buy_rec['buy_score'] >= 85:
                            st.success(f"### {buy_rec['recommendation']} (점수: {buy_rec['buy_score']}점)")
                        elif buy_rec['buy_score'] >= 75:
                            st.info(f"### {buy_rec['recommendation']} (점수: {buy_rec['buy_score']}점)")
                        elif buy_rec['buy_score'] >= 65:
                            st.warning(f"### {buy_rec['recommendation']} (점수: {buy_rec['buy_score']}점)")
                        else:
                            st.error(f"### {buy_rec['recommendation']} (점수: {buy_rec['buy_score']}점)")
                        
                        # 매수 전략
                        st.write(f"**📊 매수 전략**: {buy_rec['strategy']}")
                        
                        # 카테고리별 점수
                        col1, col2, col3, col4, col5 = st.columns(5)
                        
                        with col1:
                            cat = stock['category_scores']['CCI_조건']
                            st.metric("CCI", f"{cat['count']}개", f"{cat['score']}점")
                        
                        with col2:
                            cat = stock['category_scores']['캔들_패턴']
                            st.metric("캔들", f"{cat['count']}개", f"{cat['score']}점")
                        
                        with col3:
                            cat = stock['category_scores']['추세_지표']
                            st.metric("추세", f"{cat['count']}개", f"{cat['score']}점")
                        
                        with col4:
                            cat = stock['category_scores']['모멘텀_지표']
                            st.metric("모멘텀", f"{cat['count']}개", f"{cat['score']}점")
                        
                        with col5:
                            cat = stock['category_scores']['거래량_지표']
                            st.metric("거래량", f"{cat['count']}개", f"{cat['score']}점")
                        
                        # 매수 이유
                        if buy_rec['reasons']:
                            st.write("**✅ 매수 이유:**")
                            for reason in buy_rec['reasons']:
                                st.write(f"  {reason}")
                        
                        # 리스크 요인
                        if buy_rec['risks']:
                            st.write("**⚠️ 리스크 요인:**")
                            for risk in buy_rec['risks']:
                                st.write(f"  {risk}")
                        
                        # 충족 조건 상세
                        st.write("\n**📊 충족 조건 상세:**")
                        for cond_name, (satisfied, detail) in stock['conditions'].items():
                            if satisfied:
                                st.write(f"- {cond_name}: {detail}")
                    
                    with tab2:
                        if enable_ai and stock.get('ai_prediction') is not None:
                            st.subheader("🤖 AI 예측 분석")
                            
                            # AI 예측 결과
                            pred_prob = stock['ai_prediction']
                            accuracy = stock.get('ai_accuracy', 0)
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("5일 후 상승 확률", f"{pred_prob*100:.1f}%")
                            with col2:
                                st.metric("모델 정확도", f"{accuracy*100:.1f}%")
                            
                            # 예측 해석
                            if pred_prob >= 0.7:
                                st.success("🚀 AI가 강한 상승을 예측합니다!")
                            elif pred_prob >= 0.6:
                                st.info("📈 AI가 상승 가능성이 높다고 판단합니다.")
                            elif pred_prob >= 0.5:
                                st.warning("📊 AI가 약간의 상승 가능성을 봅니다.")
                            else:
                                st.error("📉 AI가 하락 가능성이 높다고 판단합니다.")
                            
                            st.caption("* AI 예측은 참고용입니다. 과거 데이터 기반으로 학습했습니다.")
                        else:
                            st.info("AI 예측을 위한 충분한 데이터가 없습니다.")
                    
                    with tab3:
                        if enable_backtest and 'df' in stock:
                            st.subheader("📈 백테스팅 결과")
                            
                            # 과거에 동일한 조건을 만족했던 날짜 찾기
                            df = stock['df']
                            backtest_dates = []
                            
                            # 간단한 백테스팅: CCI 골든크로스 날짜 찾기
                            try:
                                cci = compute_cci(df['High'], df['Low'], df['Close'])
                                cci_ma = compute_cci_ma(cci)
                                
                                for i in range(1, len(cci)-20):  # 최근 20일 제외
                                    if cci.iloc[i-1] < cci_ma.iloc[i-1] and cci.iloc[i] >= cci_ma.iloc[i]:
                                        backtest_dates.append(df.index[i])
                                
                                if backtest_dates:
                                    # 백테스트 실행
                                    backtest_results = backtest_strategy(df, backtest_dates[-10:])  # 최근 10개만
                                    
                                    if not backtest_results.empty:
                                        # 수익률 통계
                                        for days in [5, 10, 20]:
                                            day_results = backtest_results[backtest_results['holding_days'] == days]
                                            if not day_results.empty:
                                                avg_return = day_results['returns'].mean()
                                                win_rate = (day_results['returns'] > 0).mean() * 100
                                                
                                                col1, col2 = st.columns(2)
                                                with col1:
                                                    st.metric(f"{days}일 평균 수익률", f"{avg_return:.2f}%")
                                                with col2:
                                                    st.metric(f"{days}일 승률", f"{win_rate:.1f}%")
                                        
                                        st.caption(f"* 최근 {len(backtest_dates)}번의 신호 중 마지막 10개 분석")
                                    else:
                                        st.info("백테스팅 결과가 없습니다.")
                                else:
                                    st.info("과거에 유사한 신호가 발견되지 않았습니다.")
                            except:
                                st.error("백테스팅 중 오류가 발생했습니다.")
                        else:
                            st.info("백테스팅이 비활성화되어 있습니다.")
                    
                    with tab4:
                        if enable_news:
                            st.subheader("📰 최신 뉴스")
                            
                            with st.spinner("뉴스를 가져오는 중..."):
                                news_items = get_stock_news(stock['name'])
                                
                                if news_items:
                                    for news in news_items[:3]:  # 상위 3개만
                                        st.markdown(f"**[{news['title']}]({news['link']})**")
                                        st.caption(f"{news['date']}")
                                        if news['description']:
                                            st.write(news['description'])
                                        st.markdown("---")
                                else:
                                    st.info("최신 뉴스를 찾을 수 없습니다.")
                        else:
                            st.info("뉴스 분석이 비활성화되어 있습니다.")
                    
                    st.markdown("---")
    
    else:
        st.warning("조건을 충족하는 종목이 없습니다.")
        st.info("""
        💡 **해결 방법:**
        1. 조건 엄격도를 '느슨함'으로 설정
        2. 최소 거래량/시가총액 낮추기
        3. 검색 종목 수 늘리기 (200~300개)
        4. 목표 등급 낮추기 (B 또는 C)
        5. 고급자 모드 시도
        """)

# 섹터별 분석 섹션 추가
st.markdown("---")
st.header("🏢 섹터별 분석")

if st.button("📊 섹터별 종목 현황 보기", key="sector_analysis"):
    with st.spinner("섹터별 종목 분석 중..."):
        today_str = get_most_recent_trading_day()
        if today_str:
            name_code_map, code_name_map = get_name_code_map()
            
            # 섹터별 종목 분류
            sector_stocks = {}
            for code, name in code_name_map.items():
                sector = get_stock_sector(name)
                if sector not in sector_stocks:
                    sector_stocks[sector] = []
                sector_stocks[sector].append({'code': code, 'name': name})
            
            # 섹터별 통계 표시
            st.subheader("📈 섹터별 종목 분포")
            
            # 섹터별 종목 수 계산
            sector_counts = {sector: len(stocks) for sector, stocks in sector_stocks.items()}
            sector_df = pd.DataFrame(list(sector_counts.items()), columns=['섹터', '종목수'])
            sector_df = sector_df.sort_values('종목수', ascending=False)
            
            # 차트로 표시
            col1, col2 = st.columns([2, 1])
            with col1:
                st.bar_chart(sector_df.set_index('섹터')['종목수'])
            with col2:
                st.dataframe(sector_df, height=400)
            
            # 섹터별 상세 정보
            selected_sector = st.selectbox("섹터 선택", sector_df['섹터'].tolist())
            
            if selected_sector:
                st.subheader(f"📌 {selected_sector} 섹터 종목 목록")
                sector_stock_list = sector_stocks[selected_sector]
                
                # 종목 리스트를 테이블로 표시
                display_data = []
                for stock in sector_stock_list[:20]:  # 상위 20개만 표시
                    display_data.append({
                        '종목명': stock['name'],
                        '종목코드': stock['code']
                    })
                
                st.dataframe(pd.DataFrame(display_data), height=400)
                st.info(f"💡 {selected_sector} 섹터 총 {len(sector_stock_list)}개 종목")

# 관심종목 추적 섹션
st.markdown("---")
st.header("📌 관심종목 추적")

# 관심종목 성과 업데이트
if st.session_state.watchlist:
    st.session_state.watchlist = calculate_watchlist_performance()
    save_watchlist(st.session_state.watchlist)

# 관심종목 표시
if st.session_state.watchlist:
    # 탭 생성
    tab1, tab2, tab3 = st.tabs(["🎯 관찰 중", "✅ 성공", "📊 전체 현황"])
    
    with tab1:
        watching_stocks = [s for s in st.session_state.watchlist if s['status'] == 'watching']
        if watching_stocks:
            st.write(f"**관찰 중인 종목: {len(watching_stocks)}개**")
            
            for stock in watching_stocks:
                col1, col2, col3, col4, col5, col6, col7 = st.columns([2, 1.5, 1, 1, 1, 1, 1])
                
                with col1:
                    st.write(f"**{stock['name']}** ({stock['code']})")
                with col2:
                    st.write(f"섹터: {stock.get('sector', '기타')}")
                with col3:
                    st.write(f"등급: {stock['grade']}")
                with col4:
                    st.write(f"매수가: {stock['price']:,.0f}")
                with col5:
                    add_date = datetime.strptime(stock['add_date'], '%Y-%m-%d')
                    days_passed = (datetime.now() - add_date).days
                    st.write(f"D+{days_passed}")
                with col6:
                    if 'current_price' in stock:
                        return_rate = stock.get('return_rate', 0)
                        rise_from_low = stock.get('rise_from_low', 0)
                        
                        # 더 높은 수익률 표시
                        display_rate = max(return_rate, rise_from_low)
                        color = "green" if display_rate > 0 else "red"
                        
                        # 어떤 기준인지 표시
                        if rise_from_low > return_rate and rise_from_low >= 5:
                            st.markdown(f"<span style='color:{color}'>{rise_from_low:+.2f}% (최저가↑)</span>", unsafe_allow_html=True)
                        else:
                            st.markdown(f"<span style='color:{color}'>{return_rate:+.2f}%</span>", unsafe_allow_html=True)
                    else:
                        st.write("-")
                with col7:
                    if st.button("🗑️", key=f"del_{stock['code']}", help="삭제"):
                        st.session_state.watchlist = [s for s in st.session_state.watchlist if s['code'] != stock['code']]
                        save_watchlist(st.session_state.watchlist)
                        st.rerun()
        else:
            st.info("관찰 중인 종목이 없습니다.")
    
    with tab2:
        success_stocks = [s for s in st.session_state.watchlist if s['status'] == 'success']
        if success_stocks:
            st.write(f"**성공한 종목: {len(success_stocks)}개**")
            
            for stock in success_stocks:
                col1, col2, col3, col4, col5 = st.columns([2, 1.5, 1, 1, 1])
                
                with col1:
                    st.write(f"**{stock['name']}** ({stock['code']})")
                with col2:
                    st.write(f"섹터: {stock.get('sector', '기타')}")
                with col3:
                    st.write(f"매수가: {stock['price']:,.0f}")
                with col4:
                    if stock.get('success_reason') == '매수가 대비':
                        st.write(f"수익률: {stock.get('return_rate', 0):+.2f}%")
                    else:
                        st.write(f"최저가 대비: {stock.get('rise_from_low', 0):+.2f}%")
                with col5:
                    st.write(f"✅ {stock.get('success_reason', '성공')}")
        else:
            st.info("아직 성공한 종목이 없습니다.")
    
    with tab3:
        # 전체 통계
        total_stocks = len(st.session_state.watchlist)
        watching = len([s for s in st.session_state.watchlist if s['status'] == 'watching'])
        success = len([s for s in st.session_state.watchlist if s['status'] == 'success'])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("전체 종목", total_stocks)
        with col2:
            st.metric("관찰 중", watching)
        with col3:
            st.metric("성공", success)
        
        # 섹터별 분포
        if st.session_state.watchlist:
            st.subheader("📊 관심종목 섹터별 분포")
            sector_dist = {}
            for stock in st.session_state.watchlist:
                sector = stock.get('sector', '기타')
                sector_dist[sector] = sector_dist.get(sector, 0) + 1
            
            sector_dist_df = pd.DataFrame(list(sector_dist.items()), columns=['섹터', '종목수'])
            st.bar_chart(sector_dist_df.set_index('섹터')['종목수'])
else:
    st.info("관심종목이 없습니다. 종목 검색 후 ➕ 버튼을 눌러 추가하세요.")

# 푸터
st.markdown("---")
st.caption("""
💡 **투자 유의사항**
- 모든 투자 결정은 본인의 책임입니다.
- AI 예측과 백테스팅은 참고용입니다.
- 프로그램 버전: 3.0 (AI 예측, 백테스팅, 뉴스 분석 추가)
- 개발자: AI Assistant
""")
