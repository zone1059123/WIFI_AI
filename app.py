import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error

# --- 1. 頁面配置與校徽 ---
st.set_page_config(page_title="CYCU Antenna AI Master", page_icon="📡", layout="wide")

# --- 2. 核心訓練函數 ---
@st.cache_resource
def train_hyper_models(df):
    try:
        df.columns = [str(col).strip() for col in df.columns]
        orig_features = ['Dist', 'L_ant2', 'W_ant2']
        orig_targets = ['Tables\\0D Results\\S1,1_2.45GHz', 'Tables\\0D Results\\S1,1_5.5GHz', 'Tables\\0D Results\\S1,1_6.5GHz']

        def clean_str(s): return s.lower().replace(" ", "").replace("\\", "").replace(",", "").replace("\t", "")
        f_map = {goal: actual for goal in orig_features + orig_targets for actual in df.columns if clean_str(goal) in clean_str(actual)}
        
        found_f = [f_map.get(f) for f in orig_features if f in f_map]
        found_t = [f_map.get(t) for t in orig_targets if t in f_map]

        if len(found_f) < 3 or len(found_t) < 3: return None
        df = df.dropna(subset=found_f + found_t)
        X, y = df[found_f], df[found_t]
        scaler = StandardScaler().fit(X)
        X_s = scaler.transform(X)

        models = {
            "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
            "SVR": MultiOutputRegressor(SVR(kernel='rbf', C=10)),
            "GBM": MultiOutputRegressor(GradientBoostingRegressor(n_estimators=50, random_state=42)),
            "Linear": MultiOutputRegressor(LinearRegression()),
            "KNN": KNeighborsRegressor(n_neighbors=5)
        }
        trained = {name: m.fit(X_s, y) for name, m in models.items()}
        return scaler, trained, found_f, len(df), df[found_f].min(), df[found_f].max()
    except: return None

# --- 3. 介面與數據載入 ---
st.sidebar.image("https://upload.wikimedia.org/wikipedia/zh/thumb/4/4e/Chung_Yuan_Christian_University_Logo.svg/1200px-Chung_Yuan_Christian_University_Logo.svg.png", width=80)
st.sidebar.title("中原電機天線 AI 站")
st.sidebar.caption("作者：張宇宸 | 指導：黃崇豪 教授")

raw_df = pd.read_csv('antenna_data.csv', sep=None, engine='python')
res = train_hyper_models(raw_df)
scaler, m_dict, feat_cols, n_samples, f_min, f_max = res

# --- 4. 側邊欄控制：參數輸入 ---
st.sidebar.subheader("🛠️ 參數調試")
u_vals = [st.sidebar.number_input(f"{f}", value=float(raw_df[f].mean()), step=0.01, format="%.2f") for f in feat_cols]

# 功能 3: 鎖定目前結果進行比較
st.sidebar.divider()
if 'locked_pred' not in st.session_state: st.session_state.locked_pred = None
if 'locked_vals' not in st.session_state: st.session_state.locked_vals = None

if st.sidebar.button("🔒 鎖定目前曲線作為對照"):
    input_s = scaler.transform([u_vals])
    st.session_state.locked_pred = m_dict["Random Forest"].predict(input_s)[0]
    st.session_state.locked_vals = u_vals
    st.sidebar.success("已鎖定！調整參數即可看見對比。")

if st.sidebar.button("🔓 清除對照"):
    st.session_state.locked_pred = None
    st.sidebar.info("對照已清除")

# --- 5. 主視覺顯示 ---
st.title("📡 WiFi 6E 天線設計：AI 實時優化工作站")

# 進行當前預測
input_cur = scaler.transform([u_vals])
p_cur = m_dict["Random Forest"].predict(input_cur)[0]
freq_names = ["2.45 GHz", "5.5 GHz", "6.5 GHz"]

c1, c2 = st.columns([2, 1])

with c1:
    st.subheader("📈 頻譜比較與趨勢分析")
    fig, ax = plt.subplots(figsize=(10, 5))
    freq_pts = [2.45, 5.5, 6.5]
    
    # 繪製當前曲線
    ax.plot(freq_pts, p_cur, 'o-', color='#1f77b4', linewidth=3, label='Current Design')
    
    # 繪製鎖定曲線 (功能 3)
    if st.session_state.locked_pred is not None:
        ax.plot(freq_pts, st.session_state.locked_pred, 'o--', color='#ff7f0e', alpha=0.6, label='Locked Comparison')
        ax.fill_between(freq_pts, p_cur, st.session_state.locked_pred, color='gray', alpha=0.1)
    
    ax.axhline(-10, color='red', linestyle=':', label='-10dB Threshold')
    ax.set_ylim(-35, 0)
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("S11 (dB)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

with c2:
    st.subheader("🎯 當前設計狀態")
    for i in range(3):
        delta = p_cur[i] - (st.session_state.locked_pred[i] if st.session_state.locked_pred is not None else p_cur[i])
        st.metric(f"S11 @ {freq_names[i]}", f"{p_cur[i]:.2f} dB", delta=f"{delta:.2f} vs Locked" if st.session_state.locked_pred is not None else None, delta_color="inverse")

# --- 6. 功能 4: 反向設計推薦系統 ---
st.divider()
st.subheader("🤖 AI 反向設計導航員 (Dimension Recommender)")
col_rec1, col_rec2 = st.columns([1, 2])

with col_rec1:
    target_s11 = st.slider("目標 S11 門檻 (dB)", -25.0, -10.0, -15.0)
    if st.button("🚀 開始尋找最佳尺寸"):
        with st.spinner("AI 正在模擬 500 組組合..."):
            # 隨機生成 500 組範圍內的尺寸
            samples = np.random.uniform(f_min, f_max, (500, 3))
            s_scaled = scaler.transform(samples)
            s_preds = m_dict["Random Forest"].predict(s_scaled)
            
            # 評分機制：三個頻段都低於目標值，且平均值最小的
            score = np.mean(s_preds, axis=1)
            valid_mask = np.all(s_preds < target_s11, axis=1)
            
            if np.any(valid_mask):
                best_idx = np.argmin(score[valid_mask])
                best_dims = samples[valid_mask][best_idx]
                st.session_state.rec_dims = best_dims
                st.success("🎯 找到理想組合！")
            else:
                st.error("無法在目前範圍內找到符合全頻段目標的組合，請放寬門檻。")

with col_rec2:
    if 'rec_dims' in st.session_state:
        st.write("✨ **AI 推薦尺寸建議：**")
        d_cols = st.columns(3)
        for i, f in enumerate(feat_cols):
            d_cols[i].info(f"**{f}**\n\n{st.session_state.rec_dims[i]:.2f} mm")
        st.caption("請將左側調試參數設為上述數值，以驗證預測頻譜。")