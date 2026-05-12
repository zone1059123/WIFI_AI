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

# --- 2. 核心邏輯：多模型訓練 ---
@st.cache_resource
def train_all_models():
    df = pd.read_csv('antenna_data.csv', sep=None, engine='python')
    df.columns = [col.strip() for col in df.columns]
    
    features = ['Dist', 'L_ant2', 'W_ant2']
    targets = [
        'Tables\\0D Results\\S1,1_2.45GHz', 
        'Tables\\0D Results\\S1,1_5.5GHz', 
        'Tables\\0D Results\\S1,1_6.5GHz'
    ]
    
    df = df.dropna(subset=features + targets)
    X = df[features]
    y = df[targets]
    
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    
    # 模型 A: 線性回歸
    model_lr = LinearRegression().fit(X_scaled, y)
    
    # 模型 B: 隨機森林
    model_rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_scaled, y)
    
    # 模型 C: SVR (支持向量機) - 因為 SVR 原生不支援多輸出，需包裝
    model_svr = MultiOutputRegressor(SVR(kernel='rbf', C=100, gamma=0.1)).fit(X_scaled, y)
    
    return scaler, model_lr, model_rf, model_svr, features

scaler, m_lr, m_rf, m_svr, feat_cols = train_all_models()

# --- 3. 側邊欄控制 ---
st.sidebar.title("🛠️ 設計參數")
inputs = [st.sidebar.slider(f"{f} (mm)", 0.0, 40.0, 15.0, 0.05) for f in feat_cols]
input_scaled = scaler.transform([inputs])

# --- 4. 交叉比對計算 ---
pred_lr = m_lr.predict(input_scaled)[0]
pred_rf = m_rf.predict(input_scaled)[0]
pred_svr = m_svr.predict(input_scaled)[0]

# 計算共識度 (Confidence)
# 原理：計算三個模型預測值之間的平均標準差，標準差越小，代表模型們「意見一致」
all_preds = np.array([pred_lr, pred_rf, pred_svr])
consensus_std = np.mean(np.std(all_preds, axis=0))
confidence_score = max(0, min(100, 100 - (consensus_std * 20)))

# --- 5. 主介面展示 ---
st.title("🔬 多模型交叉驗證天線預測系統")
st.info("本系統同時調用三種不同演算法進行即時演算，透過模型間的共識程度評估設計風險。")

# 第一排：模型預測結果對照表
st.subheader("📋 模型預測數據對照 (dB)")
res_df = pd.DataFrame({
    "頻率 (GHz)": ["2.45", "5.5", "6.5"],
    "線性回歸 (LR)": pred_lr,
    "隨機森林 (RF)": pred_rf,
    "支持向量機 (SVR)": pred_svr
}).set_index("頻率 (GHz)")

st.table(res_df.style.highlight_min(axis=1, color='#d4edda')) # 自動標示出表現最好(dB最低)的值

# 第二排：圖表比對與信心區塊
col_plot, col_conf = st.columns([2, 1])

with col_plot:
    st.subheader("📈 交叉比對頻譜圖")
    fig, ax = plt.subplots(figsize=(10, 5))
    freqs = [2.45, 5.5, 6.5]
    
    ax.plot(freqs, pred_lr, 'o-', label='Linear Regression', alpha=0.8)
    ax.plot(freqs, pred_rf, 's--', label='Random Forest', alpha=0.8)
    ax.plot(freqs, pred_svr, '^-.', label='SVR', alpha=0.8)
    
    ax.axhline(-10, color='red', linestyle=':', label='-10dB Threshold')
    ax.set_ylim(-35, 0)
    ax.set_ylabel("S11 (dB)")
    ax.set_xlabel("Frequency (GHz)")
    ax.legend()
    st.pyplot(fig)

with col_conf:
    st.subheader("🛡️ 多模型共識診斷")
    st.metric("模型一致性 (Confidence)", f"{confidence_score:.1f}%")
    st.progress(int(confidence_score))
    
    if confidence_score > 85:
        st.success("🎯 **高度共識**：三種演算法結果趨於一致，數據極為可信。")
    elif confidence_score > 60:
        st.warning("⚖️ **存在分歧**：模型間出現偏差，建議以隨機森林(RF)結果為主要參考。")
    else:
        st.error("⚠️ **低度共識**：演算法預測分歧巨大，請重新檢查參數是否超出訓練範圍。")
    
    # 這裡加入你要求的信心區塊比對邏輯
    st.write("---")
    st.write("**模型偏差分析 (Std Dev):**")
    st.write(f"- 2.45GHz 偏差: {np.std(all_preds[:,0]):.4f}")
    st.write(f"- 5.50GHz 偏差: {np.std(all_preds[:,1]):.4f}")
    st.write(f"- 6.50GHz 偏差: {np.std(all_preds[:,2]):.4f}")