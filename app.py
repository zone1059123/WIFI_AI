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

# --- 1. 專業 UI 配置 ---
st.set_page_config(page_title="CYCU Antenna AI Lab", page_icon="📡", layout="wide")

st.markdown("""
    <style>
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); border-left: 5px solid #004488; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. 核心訓練函數 (包含五大模型與效率比對) ---
@st.cache_resource
def train_full_suite(df):
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

        # 五大模型定義
        models_def = {
            "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
            "SVR": MultiOutputRegressor(SVR(kernel='rbf', C=10)),
            "Gradient Boosting": MultiOutputRegressor(GradientBoostingRegressor(n_estimators=50, random_state=42)),
            "Linear Regression": MultiOutputRegressor(LinearRegression()),
            "KNN": KNeighborsRegressor(n_neighbors=5)
        }
        
        trained_models = {name: m.fit(X_s, y) for name, m in models_def.items()}

        # 數據效率分析 (20%, 50%, 100%)
        eff_list = []
        for frac in [0.2, 0.5, 1.0]:
            size = max(5, int(len(df) * frac))
            for name, m in models_def.items():
                m.fit(X_s[:size], y[:size])
                mae = mean_absolute_error(y, m.predict(X_s))
                eff_list.append({"模型": name, "數據量": f"{int(frac*100)}%", "MAE": round(mae, 4)})
        
        return scaler, trained_models, found_f, len(df), pd.DataFrame(eff_list), df[found_f].min(), df[found_f].max()
    except: return None

# --- 3. 側邊欄與資料載入 (完全體：支援上傳 + 內建) ---
st.sidebar.image("https://upload.wikimedia.org/wikipedia/zh/thumb/4/4e/Chung_Yuan_Christian_University_Logo.svg/1200px-Chung_Yuan_Christian_University_Logo.svg.png", width=80)
st.sidebar.title("中原電機天線 AI 站")
st.sidebar.caption("作者：張宇宸 | 指導：黃崇豪 教授")

st.sidebar.divider()
# 加入數據源切換選單
data_mode = st.sidebar.radio("📁 數據源選擇", ("系統內建 (239筆)", "手動上傳新數據"))

if data_mode == "系統內建 (239筆)":
    # 讀取原本的檔案
    raw_df = pd.read_csv('antenna_data.csv', sep=None, engine='python')
else:
    # 顯示上傳元件
    uploaded_file = st.sidebar.file_uploader("請上傳天線數據 CSV", type=["csv"])
    if not uploaded_file:
        st.info("💡 提示：請上傳 CSV 檔案以開始分析")
        st.stop() # 沒上傳檔案前，程式會停在這裡，不會報錯
    raw_df = pd.read_csv(uploaded_file, sep=None, engine='python')

# 訓練模型並獲取結果
res = train_full_suite(raw_df)
if not res: 
    st.error("數據格式不符，請檢查 CSV 是否包含 Dist, L_ant2, W_ant2 等欄位")
    st.stop()

scaler, m_dict, feat_cols, n_samples, eff_df, f_min, f_max = res

st.sidebar.divider()
st.sidebar.subheader("📐 幾何參數調試")
# 注意：這裡的 value 改用 raw_df 的平均值，這樣你上傳新檔案時，滑桿初始值會自動跟著新檔案跑
u_vals = [st.sidebar.number_input(f"{f}", value=float(raw_df[f].mean()), step=0.01, format="%.2f") for f in feat_cols]

# 鎖定功能 (Session State)
if 'locked_pred' not in st.session_state: st.session_state.locked_pred = None
if st.sidebar.button("🔒 鎖定目前曲線"):
    st.session_state.locked_pred = m_dict["Random Forest"].predict(scaler.transform([u_vals]))[0]
if st.sidebar.button("🔓 清除鎖定"):
    st.session_state.locked_pred = None

# --- 4. 主介面預測與 KPI ---
input_cur = scaler.transform([u_vals])
# 獲取五模型預測結果
all_preds = {name: m.predict(input_cur)[0] for name, m in m_dict.items()}
consensus_pred = all_preds["Random Forest"] # 以 RF 作為主視覺基準

st.title("📡 WiFi 6E 天線：全功能 AI 實驗室")
st.caption(f"訓練樣本: {n_samples} | 狀態: 🟢 線上運行中")

# KPI 儀表板
kpi_cols = st.columns(3)
freq_names = ["2.45 GHz", "5.5 GHz", "6.5 GHz"]
for i, col in enumerate(kpi_cols):
    val = consensus_pred[i]
    delta = val - st.session_state.locked_pred[i] if st.session_state.locked_pred is not None else None
    col.metric(f"S11 @ {freq_names[i]}", f"{val:.2f} dB", delta=f"{delta:.2f}" if delta else None, delta_color="inverse")

st.divider()

# --- 5. 圖表與比對分析 ---
left, right = st.columns([2, 1])

with left:
    st.subheader("📈 頻譜對照圖")
    fig, ax = plt.subplots(figsize=(10, 5))
    freq_pts = [2.45, 5.5, 6.5]
    ax.plot(freq_pts, consensus_pred, 'o-', linewidth=3, label='Current')
    if st.session_state.locked_pred is not None:
        ax.plot(freq_pts, st.session_state.locked_pred, 'o--', alpha=0.5, label='Locked')
    ax.axhline(-10, color='red', linestyle=':')
    ax.set_ylim(-35, 0)
    ax.legend()
    st.pyplot(fig)

with right:
    st.subheader("📋 五模型交叉比對")
    res_table = pd.DataFrame(all_preds, index=freq_names).T
    st.dataframe(res_table.style.highlight_min(axis=0, color='#d4edda'))
    csv = res_table.to_csv().encode('utf-8')
    st.download_button("💾 下載預測報告", csv, "report.csv", "text/csv")

st.divider()

# --- 6. 敏感度與效率分析 ---
low1, low2 = st.columns(2)

with low1:
    st.subheader("🎯 幾何敏感度分析")
    importances = m_dict["Random Forest"].feature_importances_
    imp_df = pd.DataFrame({"參數": feat_cols, "權重": importances}).sort_values(by="權重")
    fig_imp, ax_imp = plt.subplots()
    ax_imp.barh(imp_df["參數"], imp_df["權重"], color="#004488")
    st.pyplot(fig_imp)

with low2:
    st.subheader("📉 數據量學習效率矩陣")
    eff_pivot = eff_df.pivot(index="模型", columns="數據量", values="MAE")
    st.dataframe(eff_pivot, use_container_width=True)

# --- 7. 反向設計功能 ---
st.divider()
st.subheader("🤖 AI 反向設計導航員")
rec_c1, rec_c2 = st.columns([1, 2])

with rec_c1:
    target = st.slider("目標 S11 (dB)", -25.0, -10.0, -15.0)
    if st.button("🚀 搜尋最佳尺寸"):
        samples = np.random.uniform(f_min, f_max, (500, 3))
        preds = m_dict["Random Forest"].predict(scaler.transform(samples))
        valid = np.all(preds < target, axis=1)
        if np.any(valid):
            best = samples[valid][np.argmin(np.mean(preds[valid], axis=1))]
            st.session_state.best_dim = best
            st.success("找到合適尺寸！")
        else: st.error("未找到符合尺寸，請放寬目標。")

with rec_c2:
    if 'best_dim' in st.session_state:
        st.write("✨ **AI 推薦組合：**")
        for i, f in enumerate(feat_cols):
            st.code(f"{f}: {st.session_state.best_dim[i]:.2f} mm")