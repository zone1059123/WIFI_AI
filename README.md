# 🧬 WiFi 6E 天線設計：五大模型超級驗證與小數據學習實驗室
### AI-Antenna Lab: Hyper-Model Cross-Validation & Data-Efficiency Analysis

[![Live Demo](https://img.shields.io/badge/Live-Web%20App-ff4b4b?style=for-the-badge&logo=streamlit)](https://wifiai-gzraaanftudokcykdupzyx.streamlit.app/)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![School](https://img.shields.io/badge/CYCU-電機工程學系-red)](https://eeweb.cycu.edu.tw/)

## 🎓 專案背景與作者資料
* **學校單位**：中原大學 電機工程學系 三甲
* **專題作者**：張宇宸 (Yu-Chen Chang)
* **指導教授**：黃崇豪 教授
* **專案網址**：[https://wifiai-gzraaanftudokcykdupzyx.streamlit.app/](https://wifiai-gzraaanftudokcykdupzyx.streamlit.app/)

## 🎯 專案目的 (Project Objective)
本專案旨在解決天線設計過程中，CST 電磁模擬極度耗時的痛點。我們提出一套基於 **「小樣本學習 (Small-Data Learning)」** 的 AI 預測框架，透過多模型互校技術，在僅有 200 餘筆數據的情況下，實現高可靠度的 S11 參數即時預測。

## 🚀 核心技術亮點 (Key Features)

### 1. 五大 AI 模型並行演算 (Hyper-Model Suite)
系統同時調用五種不同數學邏輯的演算法進行交叉比對，確保預測結果非單一模型的極端偏差：
* **Linear Regression (線性回歸)**：掌握物理參數的全局線性趨勢。
* **Random Forest (隨機森林)**：處理參數間的非線性耦合關係。
* **SVR (支持向量機)**：確保在數據稀疏區域的擬合魯棒性。
* **Gradient Boosting (梯度提升)**：精確捕捉頻譜中的殘差特徵。
* **KNN (最近鄰演算法)**：利用幾何空間相似性進行直覺式預測。

### 2. 數據效率與學習演化分析 (Efficiency Analysis)
這是本專案最具學術價值的模組。系統會自動分析並展示：
* **學習曲線對照**：對比模型在 **20%、50%、100%** 不同數據量級下的誤差 (MAE) 演化。
* **訓練效能比對**：展示各演算法在毫秒級別的計算效率，證明 AI 輔助設計的即時性優勢。

### 3. 工程級精確輸入
* 支援 **0.01mm** 等級的幾何尺寸手動輸入，符合工程實務中的精確設計規範。
* 自動容錯機制：支援 CSV 與 TSV (Tab 分隔) 格式自動識別，降低數據導入門檻。

## 🛠️ 開發工具與環境
* **數據來源**：CST Studio Suite 模擬生成之 WiFi 6E 天線數據。
* **後端演算法**：Scikit-learn, Numpy, Pandas。
* **前端展示**：Streamlit Cloud。
* **版本控制**：GitHub Codespaces (CI/CD 流程)。

## 📂 檔案架構
* `app.py`: 五大模型核心邏輯、預測演算法與數據效率分析模組。
* `antenna_data.csv`: 內建之 239 筆高品質天線訓練樣本。
* `requirements.txt`: 系統運行所需之 Python 套件環境配置。

---
© 2026 張宇宸 | 中原大學電機工程學系 | 指導教授：黃崇豪