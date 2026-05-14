# 📡 CYCU Antenna AI Master: 智慧化天線設計與反向導航系統
### AI-Powered Antenna Design: Real-time Debugging & Inverse Navigation Station

[![Live Demo](https://img.shields.io/badge/Live-Web%20App-ff4b4b?style=for-the-badge&logo=streamlit)](https://wifiai-gzraaanftudokcykdupzyx.streamlit.app/)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![School](https://img.shields.io/badge/CYCU-電機工程學系-red)](https://ee.cycu.edu.tw/)

## 🎓 作者資訊
* **學校單位**：中原大學 電機工程學系 三甲
* **專題作者**：張宇宸 (Yu-Chen Chang)
* **指導教授**：黃崇豪 教授
* **專案定位**：電磁模擬輔助設計 (AI-assisted EM Design)

---

## 🎯 專案核心目標
本系統專門為 WiFi 6E 三頻天線設計打造，旨在解決傳統 CST 模擬耗時過長的痛點。透過五大機器學習模型，我們不僅實現了「秒級預測」，更導入了**動態對照**與**自動化搜尋**功能，將 AI 從單純的「預測工具」提升為「設計決策導航」。

## 🚀 重大更新亮點 (Latest Updates)

### 1. 🔒 鎖定對照功能 (Design Comparison)
* **功能描述**：支援「一鍵鎖定」當前預測頻譜，當使用者微調幾何參數時，系統會同時顯示舊設計（對照組）與新設計（實驗組）的曲線位移。
* **技術來源**：參考 **Streamlit Session State** 狀態管理機制，實現動態緩存對比。
* **工程價值**：方便工程師直觀觀察參數（如 L, W, Dist）對頻譜位移的敏感度。

### 2. 🤖 AI 反向設計導航 (Inverse Dimension Recommender)
* **功能描述**：使用者設定目標 S11 門檻（如 -15dB），系統自動透過 **Monte Carlo Sampling (蒙地卡羅抽樣)** 在背景模擬 500 組組合。
* **技術來源**：參考工業界常用的 **Heuristic Optimization (啟發式優化)** 策略。
* **工程價值**：解決「給定性能要求，反求幾何尺寸」的逆向工程難題。

### 3. 五大模型集成與數據演化
* **技術架構**：整合 Random Forest, SVR, GBM, Linear, KNN。
* **學術引用**：
    * **Random Forest**: 參考 Breiman (2001) 之隨機森林理論處理非線性特徵。
    * **Efficiency Matrix**: 參考 Vapnik-Chervonenkis (VC) 理論分析小樣本收斂性。

## 📊 數據說明 (Data Statistics)
* **來源**：由 CST Studio Suite 模擬產出之 239 筆 WiFi 6E 結構化數據。
* **特徵**：Dist (間距), L_ant2 (長度), W_ant2 (寬度)。
* **標籤**：2.45GHz, 5.5GHz, 6.5GHz 之 S11 反射損耗 (dB)。

## 🛠️ 環境配置
```text
streamlit
pandas
numpy
scikit-learn
matplotlib