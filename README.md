# 🔬 WiFi 6E 天線設計：小樣本多模型交叉驗證 AI 實驗室
### AI-Assisted Antenna Design: Small-Data Multi-Model Cross-Validation System

[![Live Demo](https://img.shields.io/badge/Live-Web%20App-ff4b4b?style=for-the-badge&logo=streamlit)](https://wifiai-gzraaanftudokcykdupzyx.streamlit.app/)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

## 🎯 專案目的 (Project Objective)
在天線工程領域，傳統電磁模擬軟體（如 CST Studio Suite）雖然精確，但完成單次模擬通常需耗費 15-30 分鐘，這在參數優化階段（需嘗試數百種尺寸組合）會造成極大的時間成本。

**本專案核心目標：**
1. **秒級預測**：將模擬時間從數十分鐘縮短至 **0.1 秒** 內。
2. **小樣本解決方案**：針對工程初期難以取得大量標註數據的痛點，利用僅 **239 筆** 模擬數據實現高精度擬合。
3. **降低決策風險**：導入多演算法交叉驗證，解決單一 AI 模型可能產生的「幻覺」或極端偏差。

## 🛠️ 開發過程 (Development Process)
專案遵循標準數據科學流程（CRISP-DM）開發：
1. **數據獲取**：透過 CST 模擬 WiFi 6E 三頻天線，獲取幾何參數（Dist, L_ant2, W_ant2）對應之 S11（2.45GHz, 5.5GHz, 6.5GHz）回傳損耗數值。
2. **預處理優化**：開發模糊匹配算法，自動修正 CSV 分隔符（Tab/Comma）與標頭異常，確保系統魯棒性。
3. **模型建模**：
   - **線性回歸 (LR)**：負責捕捉參數與頻率間的基本趨勢。
   - **隨機森林 (RF)**：處理複雜的非線性干涉特徵（限制深度以防止小樣本過擬合）。
   - **支持向量機 (SVR)**：在數據邊界區域提供更平滑、更魯棒的預測。
4. **系統整合**：利用 Streamlit 構建前端介面，並部署於雲端平台實現 24/7 持續服務。

## 📊 數據來源 (Data Source)
- **數據集**：`antenna_data.csv`
- **樣本規模**：239 筆高品質模擬數據。
- **特徵維度**：
  - `Dist` (間距)
  - `L_ant2` (天線長度)
  - `W_ant2` (天線寬度)
- **目標值**：WiFi 6E 三大核心頻段之 S11 參數 (dB)。

## 🚀 核心功能亮點
- **精確數值輸入**：支援手動輸入精確到 **0.01mm** 的尺寸，符合工程製圖需求。
- **三模型互校機制**：同時產出三種演算法預測結果，並高亮顯示最佳預估值。
- **可靠度診斷**：
  - **一致性得分**：當三模型預測接近，得分上升；分歧較大時，系統會發出警示。
  - **動態重訓**：支持用戶上傳新的模擬數據，實現模型的持續優化與進化。

## 📂 檔案結構說明
- `app.py`: 系統核心，包含小樣本優化演算法、交叉驗證邏輯與 Streamlit UI。
- `antenna_data.csv`: 核心訓練數據集。
- `requirements.txt`: 雲端環境依賴配置（Pandas, Scikit-learn, Matplotlib 等）。

## 💡 使用指南
1. 前往 [線上演示網址](https://wifiai-gzraaanftudokcykdupzyx.streamlit.app/)。
2. 選擇數據來源：可使用內建的 239 筆數據，或上傳自己的 CSV。
3. 輸入欲測試的天線尺寸。
4. 參考「一致性得分」與「交叉比對圖表」評估該設計方案之可行性（S11 < -10dB 為達標）。

---
**專題作者**：張宇宸 
**指導教授**：黃崇豪
**開發時間**：2024年 (或 2026年)  
**單位**：中原大學 電機三甲