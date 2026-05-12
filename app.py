import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor

# --- 1. 頁面配置 ---
st.set_page_config(page_title="Antenna AI Multi-Model Lab", page_icon="🔬", layout="wide")

# --- 2. 核心功能：模型訓練函數 (改為動態接收 DataFrame) ---
def train_all_models(df):
    try:
        df.columns = [col.strip() for col in df.columns]
        features = ['Dist', 'L_ant2', 'W_ant2']
        targets = [
            'Tables\\0D Results\\S1,1_2.45GHz', 
            'Tables\\0D Results\\S1,1_5.5GHz', 
            'Tables\\0D Results\\S1,1_6.5GHz'
        ]
        
        # 檢查欄位是否存在
        missing = [f for f in features + targets if f not in df.columns]
        if missing:
            st.error(f"❌ 上傳的檔案格式不符，缺少欄位: {missing}")
            return None
            
        df = df.dropna(subset=features + targets)
        X = df[features]
        y = df[targets]
        
        scaler = StandardScaler().fit(X)
        X_scaled = scaler.transform(X)
        
        # 訓練三模型
        m_lr = LinearRegression().fit(X_scaled, y)
        m_rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_scaled, y)
        m_svr = MultiOutputRegressor(SVR(kernel='rbf', C=100)).fit(X_scaled, y)
        
        return scaler, m_lr, m_rf, m_svr, features, len(df)
    except Exception as e:
        st.error(f"訓練過程出錯: {e}")
        return None

# --- 3. 側邊欄：檔案上傳與參數調整 ---
st.sidebar.title("📥 數據與參數")

# 上傳組件
uploaded_file = st.sidebar.file_uploader("上傳自定義天線數據 (CSV)", type=["csv"])

# 決定數據源
if uploaded_file is not None:
    raw_df = pd.read_csv(uploaded_file)
    st.sidebar.success("已使用上傳數據！")
else:
    # 預設讀取你的原始檔
    try:
        raw_df = pd.read_csv('antenna_data.csv')
        st.sidebar.info("使用系統預設數據 (antenna_data.csv)")
    except:
        st.sidebar.warning("找不到預設數據，請上傳 CSV 檔案。")
        st.stop()

# 執行訓練
result = train_all_models(raw_df)
if result:
    scaler, m_lr, m_rf, m_svr, feat_cols, sample_size = result
    st.sidebar.write(f"📊 當前樣本數: {sample_size}")
else:
    st.stop()

# 參數調整滑桿
st.sidebar.divider()
st.sidebar.write("🔍 調整預測參數 (mm)")
inputs = [st.sidebar.slider(f"{f}", 0.0, 40.0, 15.0, 0.05) for f in feat_cols]
input_scaled = scaler.transform([inputs])

# --- 4. 預測與比對邏輯 ---
pred_lr = m_lr.predict(input_scaled)[0]
pred_rf = m_rf.predict(input_scaled)[0]
pred_svr = m_svr.predict(input_scaled)[0]

all_preds = np.array([pred_lr, pred_rf, pred_svr])
consensus_std = np.mean(np.std(all_preds, axis=0))
confidence_score = max(0, min(100, 100 - (consensus_std * 20)))

# --- 5. 主介面展示 (與之前相同) ---
st.title("🔬 多模型交叉驗證與動態數據分析系統")
st.markdown(f"目前訓練基數：**{sample_size}** 筆模擬數據")

# 數據表格
st.subheader("📋 三模型預測數值比對 (dB)")
res_df = pd.DataFrame({
    "頻率 (GHz)": ["2.45", "5.5", "6.5"],
    "線性回歸 (LR)": pred_lr,
    "隨機森林 (RF)": pred_rf,
    "支持向量機 (SVR)": pred_svr
}).set_index("頻率 (GHz)")
st.table(res_df.style.highlight_min(axis=1, color='#d4edda'))

# 圖表
c_plot, c_conf = st.columns([2, 1])
with c_plot:
    st.subheader("📈 預測曲線交叉比對")
    fig, ax = plt.subplots(figsize=(10, 5))
    freqs = [2.45, 5.5, 6.5]
    ax.plot(freqs, pred_lr, 'o-', label='Linear Regression')
    ax.plot(freqs, pred_rf, 's--', label='Random Forest')
    ax.plot(freqs, pred_svr, '^-.', label='SVR')
    ax.axhline(-10, color='red', linestyle=':', label='-10dB')
    ax.set_ylim(-35, 0)
    ax.legend()
    st.pyplot(fig)

with c_conf:
    st.subheader("🛡️ 多模型共識分析")
    st.metric("一致性得分", f"{confidence_score:.1f}%")
    st.progress(int(confidence_score))
    
    if confidence_score > 80:
        st.success("🎯 **高度一致**：結果具備極高參考價值。")
    else:
        st.warning("⚖️ **存在偏差**：請注意不同模型間的數值差異。")
    
    st.write("---")
    st.write("**模型偏差細節 (Std Dev):**")
    for f_idx, f_name in enumerate(freqs):
        st.write(f"- {f_name}GHz: {np.std(all_preds[:,f_idx]):.4f}")