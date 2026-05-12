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

# --- 2. 核心訓練函數 (強化格式偵測) ---
def train_all_models(df):
    try:
        # 清洗欄位：移除前後空格、換行，並處理可能的 Tab 字元
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
                if clean_str(goal) in clean_str(actual): # 改用包含關係，增加容錯
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

# --- 3. 側邊欄：模式切換 ---
st.sidebar.title("📥 數據管理中心")

# 【新功能】讓用戶選擇數據來源
data_mode = st.sidebar.radio(
    "選擇數據來源：",
    ("使用系統內建數據 (antenna_data.csv)", "上傳自己的數據檔案 (CSV/TSV)")
)

raw_df = None

if data_mode == "使用系統內建數據 (antenna_data.csv)":
    try:
        raw_df = pd.read_csv('antenna_data.csv', sep=None, engine='python')
        st.sidebar.success("✅ 已載入系統預設數據")
    except Exception as e:
        st.sidebar.error(f"找不到預設檔案: {e}")
else:
    uploaded_file = st.sidebar.file_uploader("上傳天線數據", type=["csv", "txt"])
    if uploaded_file is not None:
        # 【強化】自動嘗試逗號或 Tab 分隔
        raw_df = pd.read_csv(uploaded_file, sep=None, engine='python')
        st.sidebar.success("✅ 已載入上傳數據")
    else:
        st.info("請在左側上傳 CSV 檔案。")
        st.stop()

# 執行訓練
if raw_df is not None:
    res = train_all_models(raw_df)
    if res:
        scaler, m_lr, m_rf, m_svr, feat_cols, n_samples = res
    else:
        st.stop()

# 參數調整
st.sidebar.divider()
st.sidebar.write("🔍 調整幾何尺寸 (mm)")
user_vals = [st.sidebar.slider(f"{f}", 0.0, 40.0, 15.0, 0.05) for f in feat_cols]

# --- 4. 預測與展示 (後續與之前相同) ---
input_scaled = scaler.transform([user_vals])
p_lr = m_lr.predict(input_scaled)[0]
p_rf = m_rf.predict(input_scaled)[0]
p_svr = m_svr.predict(input_scaled)[0]

all_p = np.array([p_lr, p_rf, p_svr])
std_err = np.mean(np.std(all_p, axis=0))
confidence = max(0, min(100, 100 - (std_err * 25)))

st.title("🔬 WiFi 6E 多模型交叉驗證系統")
st.write(f"當前樣本數：`{n_samples}` 筆")

res_table = pd.DataFrame({
    "頻率 (GHz)": ["2.45", "5.5", "6.5"],
    "Linear Regression": p_lr, "Random Forest": p_rf, "SVR": p_svr
}).set_index("頻率 (GHz)")
st.table(res_table.style.highlight_min(axis=1, color='#d4edda'))

c_plot, c_conf = st.columns([2, 1])
with c_plot:
    fig, ax = plt.subplots(figsize=(10, 5))
    pts = [2.45, 5.5, 6.5]
    ax.plot(pts, p_lr, 'o-', label='LR')
    ax.plot(pts, p_rf, 's--', label='RF')
    ax.plot(pts, p_svr, '^-.', label='SVR')
    ax.axhline(-10, color='red', linestyle=':')
    ax.set_ylim(-35, 0)
    ax.legend()
    st.pyplot(fig)

with c_conf:
    st.metric("一致性得分", f"{confidence:.1f}%")
    st.progress(int(confidence))
    if confidence > 80: st.success("🎯 高度一致")
    else: st.warning("⚖️ 存在偏差")