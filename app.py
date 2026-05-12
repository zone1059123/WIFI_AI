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
st.set_page_config(page_title="Antenna AI Lab", page_icon="📡", layout="wide")

# --- 2. 核心訓練函數 (保持強大的格式偵測與模糊比對) ---
def train_all_models(df):
    try:
        df.columns = [str(col).strip() for col in df.columns]
        orig_features = ['Dist', 'L_ant2', 'W_ant2']
        orig_targets = [
            'Tables\\0D Results\\S1,1_2.45GHz', 
            'Tables\\0D Results\\S1,1_5.5GHz', 
            'Tables\\0D Results\\S1,1_6.5GHz'
        ]

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
            st.error("❌ 檔案欄位匹配失敗！")
            st.write("🔍 系統預期：", orig_features + orig_targets)
            st.write("📄 CSV 實際偵測：", df.columns.tolist())
            return None
            
        df = df.dropna(subset=found_features + found_targets)
        X = df[found_features]
        y = df[found_targets]
        
        scaler = StandardScaler().fit(X)
        X_scaled = scaler.transform(X)
        
        m_lr = LinearRegression().fit(X_scaled, y)
        m_rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_scaled, y)
        m_svr = MultiOutputRegressor(SVR(kernel='rbf', C=100)).fit(X_scaled, y)
        
        return scaler, m_lr, m_rf, m_svr, found_features, len(df)
    except Exception as e:
        st.error(f"🚨 訓練出錯: {e}")
        return None

# --- 3. 側邊欄：模式切換與數值輸入 ---
st.sidebar.title("📥 數據與參數管理")

data_mode = st.sidebar.radio(
    "1. 選擇數據來源：",
    ("使用系統內建數據 (antenna_data.csv)", "上傳自己的數據檔案 (CSV/TSV)")
)

raw_df = None
if data_mode == "使用系統內建數據 (antenna_data.csv)":
    try:
        raw_df = pd.read_csv('antenna_data.csv', sep=None, engine='python')
        st.sidebar.success("✅ 已載入系統預設數據")
    except:
        st.sidebar.error("找不到預設檔案。")
else:
    uploaded_file = st.sidebar.file_uploader("上傳天線數據", type=["csv", "txt"])
    if uploaded_file:
        raw_df = pd.read_csv(uploaded_file, sep=None, engine='python')
    else:
        st.info("請上傳 CSV 檔案。")
        st.stop()

if raw_df is not None:
    res = train_all_models(raw_df)
    if res:
        scaler, m_lr, m_rf, m_svr, feat_cols, n_samples = res
    else:
        st.stop()

st.sidebar.divider()
st.sidebar.write("2. 精確輸入設計參數 (mm)")

# --- 改動重點：將 Slider 改為 Number Input ---
user_vals = []
# 預設值參考
default_vals = [5.0, 15.0, 2.5] 

for i, f in enumerate(feat_cols):
    # 使用 number_input 允許精確打字
    val = st.sidebar.number_input(
        label=f"輸入 {f}", 
        min_value=0.0, 
        max_value=100.0, 
        value=default_vals[i] if i < len(default_vals) else 10.0,
        step=0.01,   # 設定調整步進值
        format="%.2f" # 顯示到小數點後兩位
    )
    user_vals.append(val)

# --- 4. 預測與展示邏輯 ---
input_scaled = scaler.transform([user_vals])
p_lr = m_lr.predict(input_scaled)[0]
p_rf = m_rf.predict(input_scaled)[0]
p_svr = m_svr.predict(input_scaled)[0]

all_p = np.array([p_lr, p_rf, p_svr])
std_err = np.mean(np.std(all_p, axis=0))
confidence = max(0, min(100, 100 - (std_err * 25)))

# --- 5. 主介面展示 ---
st.title("🔬 WiFi 6E 多模型交叉驗證系統 (精確輸入版)")
st.caption(f"當前訓練樣本數：{n_samples} 筆 | 目前參數：{dict(zip(feat_cols, user_vals))}")

res_table = pd.DataFrame({
    "頻率 (GHz)": ["2.45", "5.5", "6.5"],
    "線性回歸 (LR)": p_lr, 
    "隨機森林 (RF)": p_rf, 
    "支持向量機 (SVR)": p_svr
}).set_index("頻率 (GHz)")

st.subheader("📋 三模型預測數值比對 (dB)")
st.table(res_table.style.highlight_min(axis=1, color='#d4edda'))

col_plot, col_conf = st.columns([2, 1])

with col_plot:
    st.subheader("📈 交叉比對頻譜圖")
    fig, ax = plt.subplots(figsize=(10, 5))
    pts = [2.45, 5.5, 6.5]
    ax.plot(pts, p_lr, 'o-', label='LR', linewidth=2)
    ax.plot(pts, p_rf, 's--', label='RF', linewidth=2)
    ax.plot(pts, p_svr, '^-.', label='SVR', linewidth=2)
    ax.axhline(-10, color='red', linestyle=':', label='Matched Threshold')
    ax.set_ylim(-35, 0)
    ax.set_ylabel("S11 (dB)")
    ax.set_xlabel("Frequency (GHz)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    st.pyplot(fig)

with col_conf:
    st.subheader("🛡️ 多模型共識分析")
    st.metric("一致性得分", f"{confidence:.1f}%")
    st.progress(int(confidence))
    if confidence > 80:
        st.success("🎯 **高度一致**：三模型結論相近。")
    elif confidence > 50:
        st.warning("⚖️ **存在分歧**：建議參考 RF 或 SVR 結果。")
    else:
        st.error("⚠️ **數據外推**：預測分歧大，請回歸電磁模擬。")