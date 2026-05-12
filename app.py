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

# --- 1. 頁面配置 ---
st.set_page_config(page_title="Antenna AI Hyper-Lab", page_icon="🧬", layout="wide")

# --- 2. 核心訓練函數 (五大模型 + 效率評估) ---
def train_hyper_models(df):
    try:
        df.columns = [str(col).strip() for col in df.columns]
        orig_features = ['Dist', 'L_ant2', 'W_ant2']
        orig_targets = ['Tables\\0D Results\\S1,1_2.45GHz', 'Tables\\0D Results\\S1,1_5.5GHz', 'Tables\\0D Results\\S1,1_6.5GHz']

        def clean_str(s):
            return s.lower().replace(" ", "").replace("\\", "").replace(",", "").replace("\t", "")

        feature_map = {goal: actual for goal in orig_features + orig_targets for actual in df.columns if clean_str(goal) in clean_str(actual)}
        
        found_f = [feature_map.get(f) for f in orig_features if f in feature_map]
        found_t = [feature_map.get(t) for t in orig_targets if t in feature_map]

        if len(found_f) < 3 or len(found_t) < 3: return None
            
        df = df.dropna(subset=found_f + found_t)
        X, y = df[found_f], df[found_t]
        scaler = StandardScaler().fit(X)
        X_s = scaler.transform(X)

        # 定義五大模型
        models = {
            "Linear Regression": MultiOutputRegressor(LinearRegression()),
            "Random Forest": RandomForestRegressor(n_estimators=50, max_depth=7, random_state=42),
            "SVR": MultiOutputRegressor(SVR(kernel='rbf', C=10)),
            "Gradient Boosting": MultiOutputRegressor(GradientBoostingRegressor(n_estimators=50, random_state=42)),
            "KNN": KNeighborsRegressor(n_neighbors=5)
        }

        trained_models = {}
        for name, m in models.items():
            trained_models[name] = m.fit(X_s, y)

        # --- 數據量效率比對邏輯 ---
        efficiency_data = []
        # 定義測試的數據量比例 (例如 20%, 50%, 100%)
        fractions = [0.2, 0.5, 1.0]
        
        for frac in fractions:
            sample_size = max(5, int(len(df) * frac))
            X_sub, y_sub = X_s[:sample_size], y[:sample_size]
            
            for name, m in models.items():
                start_t = time.time()
                m.fit(X_sub, y_sub)
                duration = (time.time() - start_t) * 1000 # 毫秒
                
                # 預測全部數據來計算誤差 (MAE)
                preds = m.predict(X_s)
                error = mean_absolute_error(y, preds)
                
                efficiency_data.append({
                    "模型": name,
                    "數據量": f"{int(frac*100)}% ({sample_size}筆)",
                    "訓練耗時(ms)": round(duration, 2),
                    "全局預測誤差(MAE)": round(error, 4)
                })

        return scaler, trained_models, found_f, len(df), pd.DataFrame(efficiency_data)
    except Exception as e:
        st.error(f"訓練錯誤: {e}")
        return None

# --- 3. 介面控制 ---
st.sidebar.title("🧬 五大模型超級實驗室")
data_mode = st.sidebar.radio("數據源：", ("內建數據", "上傳數據"))

if data_mode == "內建數據":
    raw_df = pd.read_csv('antenna_data.csv', sep=None, engine='python')
else:
    uploaded_file = st.sidebar.file_uploader("上傳 CSV", type=["csv", "txt"])
    if not uploaded_file: st.stop()
    raw_df = pd.read_csv(uploaded_file, sep=None, engine='python')

res = train_hyper_models(raw_df)
if not res: st.stop()
scaler, m_dict, feat_cols, n_samples, eff_df = res

st.sidebar.divider()
st.sidebar.write("📏 精確參數輸入 (mm)")
u_vals = [st.sidebar.number_input(f"{f}", value=10.0, step=0.01, format="%.2f") for f in feat_cols]

# --- 4. 預測與展示 ---
input_s = scaler.transform([u_vals])
all_preds = {name: m.predict(input_s)[0] for name, m in m_dict.items()}

st.title("🧬 WiFi 6E 天線：五大模型超級驗證系統")
st.markdown(f"**當前總樣本數：** `{n_samples}` 筆 | **分析模式：** 小數據高效能比對")

# 表格一：原本的五模型交叉比對
st.subheader("📋 表格一：五大模型即時預測比對 (dB)")
res_table = pd.DataFrame(all_preds, index=["2.45GHz", "5.5GHz", "6.5GHz"]).T
st.table(res_table.style.highlight_min(axis=0, color='#d4edda'))

# 圖表展示
fig, ax = plt.subplots(figsize=(10, 4))
freqs = [2.45, 5.5, 6.5]
for name, p in all_preds.items():
    ax.plot(freqs, p, 'o-', label=name, alpha=0.7)
ax.axhline(-10, color='red', linestyle=':')
ax.set_ylim(-35, 0)
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
st.pyplot(fig)

st.divider()

# 表格二：數據量與效率比對 (你的新需求)
st.subheader("📉 表格二：不同數據量級下的模型效率與精確度演化")
st.write("此表格展示了模型在僅使用 20%、50% 數據量時，對比 100% 全量數據的學習效率與誤差表現。")

# 整理表格二的顯示格式
eff_pivot = eff_df.pivot(index="模型", columns="數據量", values=["全局預測誤差(MAE)", "訓練耗時(ms)"])
st.dataframe(eff_pivot, use_container_width=True)

# 信心診斷
std_consensus = np.std(list(all_preds.values()), axis=0).mean()
conf_score = max(0, min(100, 100 - (std_consensus * 20)))
st.metric("五模型共識可靠度", f"{conf_score:.1f}%")