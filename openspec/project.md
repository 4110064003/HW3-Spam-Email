# Project Context

## Purpose
本專案旨在開發一個智慧物聯網 (AIOT) 應用系統，結合人工智慧與物聯網技術，實現智慧感測、資料分析與自動化控制功能。同時也包含了基於機器學習的智慧應用，如垃圾郵件分類系統。

## Tech Stack
- Python: 主要開發語言
- scikit-learn: 機器學習模型開發（垃圾郵件分類）
- TensorFlow/PyTorch: 深度學習模型開發
- MQTT: 物聯網通訊協議
- FastAPI: RESTful API 開發
- MongoDB: 資料儲存
- Docker: 容器化部署

## Project Conventions

### Code Style
- 遵循 PEP 8 Python 程式碼風格指南
- 使用 Black 作為程式碼格式化工具
- 變數命名使用蛇形命名法 (snake_case)
- 類別使用駝峰命名法 (CamelCase)
- 必要的程式碼註解使用中文，文件字串使用英文

### Architecture Patterns
- 微服務架構設計
- 事件驅動架構
- Repository 模式進行資料存取
- Factory 模式創建物件
- 依賴注入原則

### Testing Strategy
- 單元測試：使用 pytest
- 整合測試：實際硬體與模擬器測試
- 效能測試：使用 locust
- 程式碼覆蓋率要求：80% 以上
- CI/CD 整合測試自動化

### Git Workflow
- 主分支：main
- 開發分支：develop
- 功能分支：feature/*
- 修復分支：hotfix/*
- Commit 訊息格式：<type>(<scope>): <subject>
  - type: feat, fix, docs, style, refactor, test, chore
  - scope: 影響範圍
  - subject: 簡短描述

## Domain Context
- IoT 感測器與執行器整合
- 機器學習模型訓練與部署
- 即時資料處理與分析
- 邊緣運算
- 自動化控制系統
- 資料可視化

## Important Constraints
- 硬體相容性要求
- 低延遲響應時間 (<100ms)
- 資料安全性與隱私保護
- 電源管理與節能考量
- 網路穩定性要求
- 成本效益平衡

## External Dependencies
- MQTT Broker (Mosquitto)
- 雲端服務平台 (AWS/Azure)
- 感測器硬體供應商 API
- 天氣資訊 API
- 時間序列資料庫
- 監控系統 (Grafana)
