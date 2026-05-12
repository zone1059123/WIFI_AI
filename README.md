# 🧬 WiFi 6E 天線設計：五大模型超級驗證與小數據學習實驗室
### AI-Antenna Lab: Hyper-Model Cross-Validation & Data-Efficiency Analysis

[![Live Demo](https://img.shields.io/badge/Live-Web%20App-ff4b4b?style=for-the-badge&logo=streamlit)](https://wifiai-gzraaanftudokcykdupzyx.streamlit.app/)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![School](https://img.shields.io/badge/CYCU-電機工程學系-red)](https://ee.cycu.edu.tw/)

## 🎓 專案作者與指導
* **學校單位**：中原大學 電機工程學系 三甲
* **專題作者**：張宇宸 (Yu-Chen Chang)
* **指導教授**：黃崇豪 教授
* **專案網址**：[CYCU Antenna AI Lab](https://wifiai-gzraaanftudokcykdupzyx.streamlit.app/)

---

## 🎯 專案目的 (Project Objective)
本專案針對 WiFi 6E 天線開發流程中「模擬成本過高」的問題，提出一套基於機器學習的輔助系統。我們成功將 239 筆高品質 CST 數據轉化為可即時運算的預測模型，並導入**敏感度分析**與**反向設計建議**，優化研發效率。

## 🚀 核心功能與技術來源 (Technical Citations)

本系統之實作參考了多項工業界與學術界的機器學習標準規範，具體參考來源標註如下：

### 1. 五大模型交叉驗證 (Cross-Validation Framework)
* **技術實作**：同時運行 LR, RF, SVR, GBM, KNN 五種演算法。
* **參考來源**：參考自 **Scikit-learn (sklearn)** 官方文件的 *Ensemble Methods* 章節。
* **應用點**：使用多輸出回歸 (Multi-output Regression) 處理三頻段同步預測。

### 2. 特徵敏感度分析 (Sensitivity Analysis)
* **技術實作**：利用隨機森林的 `feature_importances_` 屬性提取幾何參數權重。
* **參考來源**：參考自 **Breiman (2001)** 的隨機森林理論。
* **應用點**：提供工程師直觀的尺寸影響圖表，判斷 L, W, Dist 哪個參數最為關鍵。

### 3. 小樣本演化效率分析 (Efficiency Matrix)
* **技術實作**：對比 20%、50%、100% 數據量級下的 MAE 誤差收斂。
* **參考來源**：參考自 **Vapnik-Chervonenkis (VC) Theory** 關於結構風險最小化之討論。
* **應用點**：證明系統在極小數據量（僅 40 餘筆）時即可達到穩定趨勢。

### 4. 反向設計優化 (Inverse Design Recommendation)
* **技術實作**：使用 Monte Carlo 方法生成 100 組隨機維度組合，經由預測模型選取最優解。
* **參考來源**：參考自工業界常見的 **Grid Search (網格搜索)** 與 **Heuristic Optimization** 策略。
* **應用點**：輸入目標性能，由 AI 建議最佳幾何尺寸。

## 📂 檔案架構
* `app.py`: 五大模型核心邏輯、敏感度分析圖表與 UI 實作。
* `antenna_data.csv`: 經由 CST Studio Suite 模擬產出之 239 筆結構化數據。
* `requirements.txt`: 包含 NumPy, Pandas, Scikit-learn, Matplotlib, Streamlit 等依賴。

## 💡 如何使用
1. **設定參數**：在側邊欄輸入天線尺寸（精確至 0.01mm）。
2. **檢視指標**：觀察頂部儀表板，確認 S11 是否低於 -10dB (PASS)。
3. **數據分析**：向下滾動查看各模型偏差、參數影響權重及數據量演化表。
4. **報告導出**：點擊「導出預測報告 (CSV)」按鈕，獲取當前設計數據。

---
### 📚 技術文獻引用 (References)
1. **Scikit-learn Developers (2024)**. *Scikit-learn User Guide*. [Online]. 
2. **Pedregosa, F., et al. (2011)**. "Scikit-learn: Machine Learning in Python," *Journal of Machine Learning Research*.
3. **Streamlit, Inc.** *API Reference: Metrics, Sliders, and Dataframe visualization*. [Online].
4. **CST Studio Suite**. *Post-processing and 0D Result Exporting Guidelines*.

---
© 2026 張宇宸 | 中原大學電機工程學系 | 指導教授：黃崇豪