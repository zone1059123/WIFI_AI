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

# --- 1. 專業 UI 配置與 CSS ---
st.set_page_config(page_title="CYCU Antenna AI Lab", page_icon="📡", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    .stMetric { 
        background-color: #ffffff; padding: 15px; border-radius: 10px; 
        box-shadow: 0 4px 6px rgba(0,0,0,0.05); border-left: 5px solid #004488;
    }
    div[data-testid="stExpander"] { background-color: #ffffff; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. 核心算法與數據清洗 ---
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
            "Linear Regression": MultiOutputRegressor(LinearRegression()),
            "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42),
            "SVR": MultiOutputRegressor(SVR(kernel='rbf', C=10)),
            "Gradient Boosting": MultiOutputRegressor(GradientBoostingRegressor(n_estimators=50, random_state=42)),
            "KNN": KNeighborsRegressor(n_neighbors=5)
        }

        trained_models = {name: m.fit(X_s, y) for name, m in models.items()}
        
        # 數據量演化數據
        eff_list = []
        for frac in [0.2, 0.5, 1.0]:
            size = max(5, int(len(df) * frac))
            for name, m in models.items():
                m.fit(X_s[:size], y[:size])
                mae = mean_absolute_error(y, m.predict(X_s))
                eff_list.append({"模型": name, "數據量": f"{int(frac*100)}%", "MAE": round(mae, 4)})
        
        return scaler, trained_models, found_f, len(df), pd.DataFrame(eff_list)
    except: return None

# --- 3. 側邊欄與數據載入 ---
st.sidebar.image("https://upload.wikimedia.org/wikipedia/zh/thumb/4/4e/Chung_Yuan_Christian_University_Logo.svg/1200px-Chung_Yuan_Christian_University_Logo.svg.png", width=100)
st.sidebar.title("中原大學電機系\n天線 AI 研究室")
st.sidebar.write(f"作者：張宇宸 | 指導：黃崇豪 教授")

data_mode = st.sidebar.radio("數據源：", ("系統內建", "手動上傳"))
if data_mode == "系統內建":
    raw_df = pd.read_csv('antenna_data.csv', sep=None, engine='python')
else:
    uploaded_file = st.sidebar.file_uploader("上傳 CSV", type=["csv"])
    if not uploaded_file: st.stop()
    raw_df = pd.read_csv(uploaded_file, sep=None, engine='python')

res = train_hyper_models(raw_df)
if not res: st.stop()
scaler, m_dict, feat_cols, n_samples, eff_df = res

st.sidebar.divider()
st.sidebar.subheader("📐 設計尺寸輸入 (mm)")
u_vals = [st.sidebar.number_input(f"{f}", value=raw_df[f].mean(), step=0.01, format="%.2f") for f in feat_cols]

# --- 4. 預測與分析邏輯 ---
input_s = scaler.transform([u_vals])
all_preds = {name: m.predict(input_s)[0] for name, m in m_dict.items()}
consensus_pred = np.mean(list(all_preds.values()), axis=0)

# --- 5. 主介面展示 ---
st.title("🔬 WiFi 6E 天線多模型設計實驗室")
st.caption(f"Status: 🟢 系統運行中 | 訓練樣本: {n_samples} | 演算法: 五大混合模型")

# KPI 儀表板
cols = st.columns(3)
freq_names = ["2.45 GHz", "5.5 GHz", "6.5 GHz"]
for i, col in enumerate(cols):
    val = consensus_pred[i]
    status = "✅ PASS" if val < -10 else "❌ FAIL"
    col.metric(f"S11 @ {freq_names[i]}", f"{val:.2f} dB", delta=status, delta_color="normal" if val < -10 else "inverse")

st.divider()

# 交叉比對表格與特徵重要性
left, right = st.columns([2, 1])

with left:
    st.subheader("📋 五大模型即時預測比對")
    res_table = pd.DataFrame(all_preds, index=freq_names).T
    st.table(res_table.style.highlight_min(axis=0, color='#d4edda'))
    
    # 下載按鈕
    csv = res_table.to_csv().encode('utf-8')
    st.download_button("💾 導出預測報告 (CSV)", csv, "antenna_report.csv", "text/csv")

with right:
    st.subheader("🎯 特徵敏感度分析")
    # 從隨機森林提取重要性
    importances = m_dict["Random Forest"].feature_importances_
    imp_df = pd.DataFrame({"參數": feat_cols, "影響權重": importances}).sort_values(by="影響權重", ascending=True)
    fig_imp, ax_imp = plt.subplots(figsize=(5, 4))
    ax_imp.barh(imp_df["參數"], imp_df["影響權重"], color="#004488")
    ax_imp.set_title("幾何尺寸對頻率的影響程度")
    st.pyplot(fig_imp)

# 底部進階功能
st.divider()
exp1, exp2 = st.columns(2)

with exp1:
    with st.expander("📉 查看數據量學習效率矩陣"):
        st.write("展示各模型在 20% vs 100% 數據量下的誤差演化 (MAE)")
        eff_pivot = eff_df.pivot(index="模型", columns="數據量", values="MAE")
        st.dataframe(eff_pivot, use_container_width=True)

with exp2:
    with st.expander("🤖 反向設計建議 (Dimensions Recommendation)"):
        st.write("系統自動模擬 100 組隨機尺寸，尋找 S11 最優組合：")
        # 簡單隨機優化器
        random_samples = np.random.uniform(raw_df[feat_cols].min(), raw_df[feat_cols].max(), (100, 3))
        rs_scaled = scaler.transform(random_samples)
        rs_preds = m_dict["Random Forest"].predict(rs_scaled)
        best_idx = np.argmin(np.mean(rs_preds, axis=1))
        best_dim = random_samples[best_idx]
        st.success(f"推薦尺寸：Dist={best_dim[0]:.2f}, L={best_dim[1]:.2f}, W={best_dim[2]:.2f}")