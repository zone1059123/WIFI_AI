import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor

# --- 1. 頁面配置與專業樣式 ---
st.set_page_config(page_title="Antenna AI Multi-Model Lab", page_icon="🔬", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 20px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    </style>
    """, unsafe_allow_html=True)

# --- 2. 核心邏輯：動態模型訓練與欄位模糊比對 ---
def train_all_models(df):
    try:
        # 清洗所有欄位名稱：移除前後空格、換行、統一轉成字串
        df.columns = [str(col).strip() for col in df.columns]
        
        # 定義我們「想要」的目標欄位名稱
        orig_features = ['Dist', 'L_ant2', 'W_ant2']
        orig_targets = [
            'Tables\\0D Results\\S1,1_2.45GHz', 
            'Tables\\0D Results\\S1,1_5.5GHz', 
            'Tables\\0D Results\\S1,1_6.5GHz'
        ]

        # 自動模糊對應欄位 (不計大小寫、空格、反斜線符號)
        def clean_str(s):
            return s.lower().replace(" ", "").replace("\\", "").replace(",", "")

        feature_map = {}
        for goal in orig_features + orig_targets:
            for actual in df.columns:
                if clean_str(goal) == clean_str(actual):
                    feature_map[goal] = actual
                    break
        
        # 檢查是否所有必要的欄位都找到了
        found_features = [feature_map.get(f) for f in orig_features if f in feature_map]
        found_targets = [feature_map.get(t) for t in orig_targets if t in feature_map]

        if len(found_features) < 3 or len(found_targets) < 3:
            st.error("❌ 檔案欄位匹配失敗！請確認 CSV 內容。")
            st.write("🔎 系統嘗試尋找：", orig_features + orig_targets)
            st.write("📄 CSV 目前偵測到：", df.columns.tolist())
            return None
            
        # 數據清洗：移除空值
        df = df.dropna(subset=found_features + found_targets)
        X = df[found_features]
        y = df[found_targets]
        
        # 標準化與訓練
        scaler = StandardScaler().fit(X)
        X_scaled = scaler.transform(X)
        
        m_lr = LinearRegression().fit(X_scaled, y)
        m_rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_scaled, y)
        # SVR 需包裝以支援多輸出
        m_svr = MultiOutputRegressor(SVR(kernel='rbf', C=100, gamma=0.1)).fit(X_scaled, y)
        
        return scaler, m_lr, m_rf, m_svr, found_features, len(df)
        
    except Exception as e:
        st.error(f"🚨 訓練過程出錯: {e}")
        return None

# --- 3. 側邊欄：數據來源與參數調整 ---
st.sidebar.title("📥 數據與設計參數")

uploaded_file = st.sidebar.file_uploader("1. 上傳天線數據 (CSV)", type=["csv"])

if uploaded_file is not None:
    raw_df = pd.read_csv(uploaded_file)
    st.sidebar.success("✅ 已載入上傳數據")
else:
    try:
        raw_df = pd.read_csv('antenna_data.csv')
        st.sidebar.info("ℹ️ 使用系統預設數據 (antenna_data.csv)")
    except:
        st.sidebar.warning("⚠️ 找不到預設檔案，請上傳 CSV。")
        st.stop()

# 執行模型初始化/重訓
training_res = train_all_models(raw_df)

if training_res:
    scaler, m_lr, m_rf, m_svr, feat_cols, n_samples = training_res
else:
    st.stop()

st.sidebar.divider()
st.sidebar.write("2. 調整幾何尺寸 (mm)")
# 動態生成滑桿
user_vals = []
for f in feat_cols:
    val = st.sidebar.slider(f"{f}", 0.0, 40.0, 15.0, 0.05)
    user_vals.append(val)