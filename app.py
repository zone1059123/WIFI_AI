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
st.set_page_config(page_title="WiFi 6E Antenna Mini-Data Lab", page_icon="📡", layout="wide")

# --- 2. 核心訓練函數 (針對小樣本優化) ---
def train_all_models(df):
    try:
        df.columns = [str(col).strip() for col in df.columns]
        orig_features = ['Dist', 'L_ant2', 'W_ant2']
        orig_targets = [
            'Tables\\0D Results\\S1,1_2.45GHz', 
            'Tables\\0D Results\\S1,1_5.5GHz', 
            'Tables\\0D Results\\S1,1_6.5GHz'
        ]

        # 模糊匹配函數
        def clean_str(s):
            return s.lower().replace(" ", "").replace("\\", "").replace(",", "").replace("\t", "")

        feature_map = {}
        for goal in orig_features + orig_targets:
            for actual in df.columns:
                if clean_str(goal) in clean_str(actual):
                    feature_map[goal] = actual
                    break
        
        found_features = [feature_map.get(f) for f in orig_features if f in feature_map]
        found_targets = [feature_map.get(t) for t in orig_targets if t in feature_map]

        if len(found_features) < 3 or len(found_targets) < 3:
            st.error("❌ 欄位匹配失敗")
            return None
            
        df = df.dropna(subset=found_features + found_targets)
        X = df[found_features]
        y = df[found_targets]
        
        scaler = StandardScaler().fit(X)
        X_scaled = scaler.transform(X)
        
        # --- 小樣本優化模型組合 ---
        # 1. Linear Regression (Low Variance, High Bias) - 捕捉大趨勢
        m_lr = LinearRegression().fit(X_scaled, y)
        # 2. Random Forest (Reduced Overfitting via Ensemble) - 處理非線性
        m_rf = RandomForestRegressor(n_estimators=50, max_depth=7, random_state=42).fit(X_scaled, y)
        # 3. SVR (Robust to Outliers) - 適合極小樣本
        m_svr = MultiOutputRegressor(SVR(kernel='rbf', C=10, epsilon=0.1)).fit(X_scaled, y)
        
        return scaler, m_lr, m_rf, m_svr, found_features, len(df)
    except Exception as e:
        st.error(f"🚨 訓練出錯: {e}")
        return None

# --- 3. 側邊欄：模式切換與數值輸入 ---
st.sidebar.title("🔬 小樣本 AI 實驗室")

data_mode = st.sidebar.radio(
    "1. 數據源選擇：",
    ("內建小樣本數據 (239筆)", "上傳新數據 (CSV/TSV)")
)

raw_df = None
if data_mode == "內建小樣本數據 (239筆)":
    try:
        raw_df = pd.read_csv('antenna_data.csv', sep=None, engine='python')
    except:
        st.sidebar.error("找不到預設檔案。")
else:
    uploaded_file = st.sidebar.file_uploader("上傳 CSV 檔案", type=["csv", "txt"])
    if uploaded_file:
        raw_df = pd.read_csv(uploaded_file, sep=None, engine='python')
    else:
        st.stop()

if raw_df is not None:
    res = train_all_models(raw_df)
    if res:
        scaler, m_lr, m_rf, m_svr, feat_cols, n_samples = res
    else:
        st.stop()

st.sidebar.divider()
st.sidebar.write("2. 設計參數精確輸入 (mm)")

user_vals = []
default_vals = [5.0, 15.0, 2.5] 
for i, f in enumerate(feat_cols):
    val = st.sidebar.number_input(f"{f}", value=default_vals[i], step=0.01, format="%.2f")
    user_vals.append(val)

# --- 4. 預測與交叉比對 ---
input_scaled = scaler.transform([user_vals])
p_lr = m_lr.predict(input_scaled)[0]
p_rf = m_rf.predict(input_scaled)[0]
p_svr = m_svr.predict(input_scaled)[0]

# 計算模型一致性 (Confidence)
all_p = np.array([p_lr, p_rf, p_svr])
std_err = np.mean(np.std(all_p, axis=0))
confidence = max(0, min(100, 100 - (std_err * 20)))

# --- 5. 主介面：強調小樣本分析 ---
st.title("📡 WiFi 6E 天線：小樣本多模型交叉驗證系統")
st.info(f"💡 當前處於「小樣本模式」：已載入 {n_samples} 筆高品質模擬數據，並啟動三算法互校機制。")

# 數據比對表
res_table = pd.DataFrame({
    "頻率 (GHz)": ["2.45", "5.5", "6.5"],
    "Linear (趨勢捕捉)": p_lr, 
    "RandomForest (非線性)": p_rf, 
    "SVR (魯棒擬合)": p_svr
}).set_index("頻率 (GHz)")

st.subheader("📋 跨算法預測對照 (dB)")
st.table(res_table.style.highlight_min(axis=1, color='#d4edda'))

c_plot, c_conf = st.columns([2, 1])

with c_plot:
    st.subheader("📈 多模型預測頻譜")
    fig, ax = plt.subplots(figsize=(10, 5))
    pts = [2.45, 5.5, 6.5]
    ax.plot(pts, p_lr, 'o-', label='LR Model (Trend)')
    ax.plot(pts, p_rf, 's--', label='RF Model (Complexity)')
    ax.plot(pts, p_svr, '^-.', label='SVR Model (Robust)')
    ax.axhline(-10, color='red', linestyle=':', label='-10dB Limit')
    ax.set_ylim(-35, 0)
    ax.set_ylabel("S11 (dB)")
    ax.set_xlabel("Frequency (GHz)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

with c_conf:
    st.subheader("🛡️ 小樣本預測可靠度")
    st.metric("模型一致性評分", f"{confidence:.1f}%")
    st.progress(int(confidence))
    
    if confidence > 80:
        st.success("🎯 **高度可靠**：不同性質演算法結論一致。")
    elif confidence > 50:
        st.warning("⚖️ **建議校驗**：算法間存在微小分歧。")
    else:
        st.error("🚨 **數據外延警告**：建議重新進行 CST 模擬。")
    
    st.write("---")
    st.caption("**為什麼需要多模型比對？**\n在小樣本場景下，單一模型容易產生偏差。我們透過線性、樹狀與核函數三種完全不同邏輯的算法進行「投票」，只有當三者意見接近時，預測才具備工程意義。")