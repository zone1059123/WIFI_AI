# WIFI_AI# 📡 WiFi 6E 天線設計 AI 輔助決策系統
### WiFi 6E Antenna Design AI-Assistant System

![Status](https://img.shields.io/badge/Status-Live-brightgreen)
![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Framework](https://img.shields.io/badge/Framework-Streamlit-ff4b4b)

## 📖 專案簡介
本專案為大學專題研究成果，旨在解決傳統天線電磁模擬（如 CST）耗時過長的問題。我們透過 **機器學習 (Machine Learning)** 技術，訓練了一個能針對 WiFi 6E 三頻（2.45GHz, 5.5GHz, 6.5GHz）進行 S11 參數預測的輕量化模型，並封裝成易於使用的 Web APP。

## 🚀 核心亮點
- **即時預測**：將原本需要 15-20 分鐘的 CST 模擬縮短至 **< 0.1 秒**。
- **小樣本優化**：利用 239 筆精確模擬數據，透過線性回歸與隨機森林實現高泛化性。
- **不確定性診斷**：內建「信心指數」分析，提醒工程師何時需回歸傳統模擬驗證。
- **跨平台展示**：基於 Streamlit 開發，支援手機與網頁即時操作。

## 🛠️ 技術棧
- **開發語言**：Python
- **數據處理**：Pandas, NumPy
- **機器學習**：Scikit-learn (Linear Regression, Random Forest)
- **視覺化**：Matplotlib
- **部署**：Streamlit Cloud / GitHub Codespaces

## 📂 檔案結構
- `app.py`: 系統核心程式碼（包含 UI 介面與模型推理邏輯）。
- `antenna_data.csv`: 經過 CST 模擬生成的 239 筆天線幾何參數與 S11 數據。
- `requirements.txt`: 專案執行環境依賴清單。

## 📊 使用說明
1. 開啟 [專案網頁連結] (在此處貼上你的 Streamlit 網址)。
2. 在左側側邊欄調整天線參數：`Dist` (間距), `L_ant2` (長度), `W_ant2` (寬度)。
3. 系統將即時產出三個頻段的 S11 預測值與頻譜圖。
4. 參考「預測穩定度分析」判斷結果之工程參考價值。

---
**專題作者**：[張宇宸]  
**指導老師**：[黃崇豪]  
**學術單位**：[中原大學 電機三甲]