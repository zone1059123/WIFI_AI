import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
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

# --- 2. 核心訓練函數 (動態偵測欄位與四變數最佳化) ---
@st.cache_resource
def train_full_suite(df):
    try:
        # 清洗欄位名稱，移除前後空格
        df.columns = [str(col).strip() for col in df.columns]
        
        # 鎖定最新的「四個幾何變數」特徵
        target_features = ['L_p', 'L_slot', 'W_slot', 'W_slot2']
        found_f = [col for col in df.columns if any(p.lower() == col.lower() for p in target_features)]
        
        # 尋找目標 S11 欄位 (相容 CST 各種導出命名)
        found_t = [col for col in df.columns if "s1,1" in col.lower() or "s11" in col.lower() or "0d" in col.lower()]
        
        # 排序確保頻段對齊 (2.45 -> 5.5 -> 6.5)
        found_t = sorted(found_t, key=lambda x: [float(s) for s in ["2.45", "5.5", "6.5"] if s in x] or [0])

        if len(found_f) < 4 or len(found_t) < 3: 
            return None
            
        # 關鍵：剔除沒有 S11 標籤的資料（例如空殼的 581 檔），只留完整可訓練的數據
        df_clean = df.dropna(subset=found_f + found_t)
        if len(df_clean) < 10: return None
        
        X, y = df_clean[found_f], df_clean[found_t]
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

        # 數據效率分析
        eff_list = []
        for frac in [0.2, 0.5, 1.0]:
            size = max(5, int(len(df_clean) * frac))
            for name, m in models_def.items():
                m.fit(X_s[:size], y[:size])
                mae = mean_absolute_error(y, m.predict(X_s))
                eff_list.append({"模型": name, "數據量": f"{int(frac*100)}%", "MAE": round(mae, 4)})
        
        return scaler, trained_models, found_f, len(df_clean), pd.DataFrame(eff_list), df_clean[found_f].min(), df_clean[found_f].max()
    except Exception as e:
        return None

# --- 3. ⚙️ 自動掃描資料夾內所有 CSV 檔案並合併 ---
def load_and_combine_folder(folder_name="data_folder"):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    
    # 抓取資料夾內所有 .csv 檔案
    csv_files = glob.glob(os.path.join(folder_name, "*.csv"))
    
    if not csv_files:
        return None, []
    
    combined_list = []
    file_names = []
    
    for file in csv_files:
        try:
            # 支援自動辨識 Tab 鍵或逗號分隔
            tmp_df = pd.read_csv(file, sep=None, engine='python')
            tmp_df.columns = [str(col).strip() for col in tmp_df.columns]
            combined_list.append(tmp_df)
            file_names.append(os.path.basename(file))
        except:
            continue
            
    if not combined_list:
        return None, []
        
    # 將所有讀取到的 CSV 合併成一個大型 DataFrame
    # 使用 join='outer' 可以確保就算有些檔案欄位不完全相同，也能完整保留並自動對齊
    combined_df = pd.concat(combined_list, axis=0, ignore_index=True, join='outer')
    return combined_df, file_names

# --- 4. 側邊欄與資料載入 ---
st.sidebar.image("https://upload.wikimedia.org/wikipedia/zh/thumb/4/4e/Chung_Yuan_Christian_University_Logo.svg/1200px-Chung_Yuan_Christian_University_Logo.svg.png", width=80)
st.sidebar.title("中原電機天線 AI 站")
st.sidebar.caption("作者：張宇宸 | 指導：黃崇豪 教授")

st.sidebar.divider()
st.sidebar.subheader("📂 自動化資料庫狀態")

# 執行自動資料夾掃描
folder_to_scan = "data_folder"
raw_df, detected_files = load_and_combine_folder(folder_to_scan)

if raw_df is None:
    st.sidebar.error(f"❌ 在 `{folder_to_scan}` 資料夾中找不到任何 CSV 檔案！")
    st.sidebar.info("💡 請在專案中建立 `data_folder` 資料夾並放入天線 CSV 數據檔案。")
    st.stop()
else:
    st.sidebar.success(f"🟢 成功自動偵測並合併 {len(detected_files)} 個檔案：")
    for f_name in detected_files:
        st.sidebar.caption(f"📄 {f_name}")

# 執行訓練
res = train_full_suite(raw_df)
if not res: 
    st.error("錯誤：無法從合併後的檔案中辨識出『四個幾何變數（L_p, L_slot, W_slot, W_slot2）』或 S11 目標欄位，請檢查檔案標頭名稱！")
    st.stop()

scaler, m_dict, feat_cols, n_samples, eff_df, f_min, f_max = res

st.sidebar.divider()
st.sidebar.subheader("📐 四變數幾何參數調試")
# 根據自動讀取到的四變數，自動產出對應的輸入框
u_vals = [st.sidebar.number_input(f"{f} (mm)", value=float(raw_df[f].dropna().mean()), step=0.01, format="%.2f") for f in feat_cols]

# 🔒 鎖定功能
if 'locked_pred' not in st.session_state: st.session_state.locked_pred = None
if st.sidebar.button("🔒 鎖定目前曲線"):
    st.session_state.locked_pred = m_dict["Random Forest"].predict(scaler.transform([u_vals]))[0]
if st.sidebar.button("🔓 清除鎖定"):
    st.session_state.locked_pred = None

# --- 5. 主介面預測與 KPI ---
input_cur = scaler.transform([u_vals])
all_preds = {name: m.predict(input_cur)[0] for name, m in m_dict.items()}
consensus_pred = all_preds["Random Forest"] 

st.title("📡 WiFi 6E 天線：自動化大數據 AI 實驗室")
st.caption(f"📊 資料庫總計有效訓練樣本數: {n_samples} 筆 | 狀態: 🟢 多檔案自動鏈結中")

kpi_cols = st.columns(3)
freq_names = ["2.45 GHz", "5.5 GHz", "6.5 GHz"]
for i, col in enumerate(kpi_cols):
    val = consensus_pred[i]
    delta = val - st.session_state.locked_pred[i] if st.session_state.locked_pred is not None else None
    col.metric(f"S11 @ {freq_names[i]}", f"{val:.2f} dB", delta=f"{delta:.2f}" if delta else None, delta_color="inverse")

st.divider()

# --- 6. 圖表與比對分析 ---
left, right = st.columns([2, 1])

with left:
    st.subheader("📈 頻譜對照圖 (隨機森林基準)")
    fig, ax = plt.subplots(figsize=(10, 5))
    freq_pts = [2.45, 5.5, 6.5]
    ax.plot(freq_pts, consensus_pred, 'o-', linewidth=3, color='#1f77b4', label='Current Design')
    if st.session_state.locked_pred is not None:
        ax.plot(freq_pts, st.session_state.locked_pred, 'o--', color='#ff7f0e', alpha=0.5, label='Locked Design')
    ax.axhline(-10, color='red', linestyle=':', label='-10dB Spec')
    ax.set_ylim(-35, 0)
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("S11 (dB)")
    ax.legend()
    st.pyplot(fig)

with right:
    st.subheader("📋 五模型交叉比對矩陣")
    res_table = pd.DataFrame(all_preds, index=freq_names).T
    st.dataframe(res_table.style.highlight_min(axis=0, color='#d4edda'))
    csv = res_table.to_csv().encode('utf-8')
    st.download_button("💾 下載本次預測報告", csv, "antenna_report.csv", "text/csv")

st.divider()

# --- 7. 敏感度與效率分析 ---
low1, low2 = st.columns(2)

with low1:
    st.subheader("🎯 四變數幾何敏感度分析 (Feature Importance)")
    importances = m_dict["Random Forest"].feature_importances_
    imp_df = pd.DataFrame({"參數": feat_cols, "權重": importances}).sort_values(by="權重")
    fig_imp, ax_imp = plt.subplots()
    ax_imp.barh(imp_df["參數"], imp_df["權重"], color="#004488")
    st.pyplot(fig_imp)

with low2:
    st.subheader("📉 數據量學習效率矩陣 (MAE)")
    eff_pivot = eff_df.pivot(index="模型", columns="數據量", values="MAE")
    st.dataframe(eff_pivot, use_container_width=True)

# --- 8. 反向設計功能 ---
st.divider()
st.subheader("🤖 AI 四維反向設計導航員")
rec_c1, rec_c2 = st.columns([1, 2])

with rec_c1:
    target = st.slider("目標 S11 門檻 (dB)", -25.0, -10.0, -15.0)
    if st.button("🚀 搜尋最佳尺寸"):
        # 在四個變數的上下限空間內進行 500 組蒙地卡羅抽樣
        samples = np.random.uniform(f_min, f_max, (500, len(feat_cols)))
        preds = m_dict["Random Forest"].predict(scaler.transform(samples))
        valid = np.all(preds < target, axis=1)
        if np.any(valid):
            best = samples[valid][np.argmin(np.mean(preds[valid], axis=1))]
            st.session_state.best_dim = best
            st.success("🎯 找到完美符合四變數之天線尺寸！")
        else: 
            st.error("未找到符合尺寸，請放寬目標 dB 門檻。")

with rec_c2:
    if 'best_dim' in st.session_state:
        st.write("✨ **AI 推薦之四維最佳尺寸組合：**")
        for i, f in enumerate(feat_cols):
            st.code(f"{f}: {st.session_state.best_dim[i]:.2f} mm")