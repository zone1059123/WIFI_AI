# 📡 CYCU Antenna AI Lab: 全功能天線智慧設計工作站
### Integrated Antenna AI Lab: Multi-Model Analysis, Comparison & Inverse Navigation

[![Live Demo](https://img.shields.io/badge/Live-Web%20App-ff4b4b?style=for-the-badge&logo=streamlit)](https://wifiai-gzraaanftudokcykdupzyx.streamlit.app/)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![School](https://img.shields.io/badge/CYCU-電機工程學系-red)](https://ee.cycu.edu.tw/)

## 🎓 專案作者與指導
* **學校單位**：中原大學 電機工程學系 三甲
* **專題作者**：張宇宸 (Yu-Chen Chang)
* **指導教授**：黃崇豪 教授
* **應用領域**：WiFi 6E 三頻天線 (2.45/5.5/6.5 GHz) 優化設計

---

## 🚀 系統核心功能 (Full Feature Suite)

本系統整合了從數據導入、模型訓練到設計優化的完整工作流：

### 1. 📂 多元數據管理 (Data Management)
* **雙模式載入**：支援「系統內建數據」快速演示，以及「手動上傳 CSV」進行自定義天線分析。
* **自動數據清洗**：針對 CST 導出的複雜表頭進行自動模糊對應與標準化處理。

### 2. 🧠 五大 AI 模型集成預測 (Multi-Model Engine)
* **並行運算**：同時調用 Random Forest, SVR, Gradient Boosting, Linear Regression 與 KNN。
* **交叉比對矩陣**：即時呈現各模型在三頻段下的預測數值，確保預測結果的穩健性。

### 3. 🔒 實時調試與鎖定比較 (Comparison Workstation)
* **動態對照**：支援鎖定特定設計點，當微調幾何尺寸時，圖表會同步顯示「新舊設計」的頻譜位移差。
* **KPI 儀表板**：直觀顯示 S11 是否達標 (-10dB)，並計算與鎖定值之間的效能增益。

### 4. 🤖 AI 反向設計導航 (Inverse Design Recommender)
* **自動優化**：使用者設定目標性能門檻，系統透過蒙地卡羅模擬 (Monte Carlo) 自動在數百組組合中尋找最優幾何尺寸。

### 5. 📊 深度學術分析 (Academic Analytics)
* **敏感度分析**：以隨機森林權重導出幾何參數（Dist, L, W）對頻率影響的貢獻度。
* **學習效率矩陣**：對比不同數據規模（20% vs 100%）下的誤差演化，驗證小樣本學習之可靠性。

---

## 🛠️ 技術棧與參考來源 (Technical Stack & Citations)
* **前端框架**：Streamlit (UI/UX 實作)
* **核心算法**：Scikit-learn (Multi-output Regression 模型)
* **數據處理**：Pandas, NumPy
* **視覺化**：Matplotlib (動態頻譜與分析圖表)

**技術引用說明：**
* 隨機森林特徵重要性分析參考自 **Breiman (2001)**。
* 數據標準化與多輸出架構參考自 **Pedregosa et al. (2011)**。

## 📖 使用步驟
1. **選擇數據源**：於側邊欄選擇內建數據或上傳您的 CST 模擬 CSV。
2. **調整參數**：移動幾何尺寸數值，觀察 KPI 儀表板與頻譜圖。
3. **鎖定比對**：點擊「鎖定目前曲線」後進行微調，觀察性能優化軌跡。
4. **獲取建議**：設定目標 dB 值，由 AI 推薦最佳尺寸組合。

---
© 2026 張宇宸 | 中原大學電機工程學系 | 指導教授：黃崇豪