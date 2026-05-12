# 🔬 WiFi 6E 天線設計多模型交叉驗證系統
### Antenna Design AI-Assistant: Multi-Model Cross-Validation Lab

[![Link](https://img.shields.io/badge/Live-Web%20App-ff4b4b?style=for-the-badge&logo=streamlit)](https://wifiai-gzraaanftudokcykdupzyx.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Streamlit-white?style=for-the-badge&logo=streamlit)](https://streamlit.io/)

## 📖 專案概述
本專案開發了一套針對 WiFi 6E（2.45GHz, 5.5GHz, 6.5GHz）天線設計的智能輔助決策系統。傳統電磁模擬軟體（如 CST）在進行參數優化時極為耗時，本系統透過 **機器學習 (Machine Learning)** 技術，實現了秒級的 S11 參數預測，並導入「多模型交叉驗證」機制，確保預測結果的工程參考價值。

🔗 **即時展示網址**：[點此進入系統](https://wifiai-gzraaanftudokcykdupzyx.streamlit.app/)

## 🚀 核心功能
- **精確參數輸入**：捨棄傳統滑桿，採用 `Number Input` 模式，支援小數點後兩位的精確幾何尺寸調整。
- **三模型並行演算**：系統同時調用三種不同性質的演算法：
  1. **線性回歸 (Linear Regression)**：捕捉參數間的全局線性趨勢。
  2. **隨機森林 (Random Forest)**：捕捉複雜的非線性特徵。
  3. **支持向量機 (SVR)**：處理小樣本數據下的穩定擬合。
- **模型共識度診斷 (Confidence Score)**：透過計算三模型預測值之間的偏差（標準差），自動產生一致性得分。得分越高，代表該設計組合越可靠。
- **動態數據管理**：
  - **內建數據**：預設搭載 239 筆 CST 精確模擬數據。
  - **自定義上傳**：支援用戶上傳 CSV/TSV 格式的數據集，系統會即時「重訓練」模型以適應新的天線架構。

## 🛠️ 技術規格
- **數據清洗**：具備欄位模糊匹配功能，能自動識別包含空格、Tab 或符號的 CSV 標頭。
- **前端框架**：Streamlit Cloud (實現 24/7 持續運行)。
- **機器學習庫**：Scikit-learn (StandardScaler, MultiOutputRegressor)。
- **繪圖引擎**：Matplotlib (產出符合學術論文規格的 S11 頻譜圖)。

## 📂 檔案結構
- `app.py`: 核心應用程式，包含 UI、多模型訓練與交叉驗證邏輯。
- `antenna_data.csv`: 系統預設訓練數據集。
- `requirements.txt`: 雲端部署所需之環境套件清單。
- `README.md`: 專案說明文件。

## 📊 如何開始？
1. **進入網頁**：開啟[系統網址](https://wifiai-gzraaanftudokcykdupzyx.streamlit.app/)。
2. **數據選擇**：於左側選擇使用「系統內建數據」或「上傳自定義 CSV」。
3. **輸入參數**：在左側欄位精確輸入天線的 `Dist`, `L_ant2`, `W_ant2` 尺寸。
4. **分析結果**：
   - 觀察下方表格中三種模型的數值對照（綠色高亮代表最佳性能預測）。
   - 查看右側的「一致性得分」，判斷是否需回歸 CST 進行最終驗證。

---
**程式作者**：[張宇宸]  
**專題指導**：[黃崇豪教授]  
**所屬單位**：[中原大學電機工程系三甲]