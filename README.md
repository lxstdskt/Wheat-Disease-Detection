# 🌾 智农·基于 Wheat-YOLO 与 PyQt5 的小麦病害智能评估系统

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![YOLOv8](https://img.shields.io/badge/YOLO-v8-yellow.svg)
![PyQt5](https://img.shields.io/badge/PyQt-5.15-green.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8-red.svg)

## 📖 项目简介
本项目是一个面向智慧农业的桌面级端侧目标检测与数据分析系统。针对大规模农田病害（如黄矮病、条锈病、白粉病等）排查效率低下、人工统计繁琐等痛点，提供了一套集**“实时流媒体检测 - 大规模批量评估 - 动态数据可视化”**为一体的完整解决方案。

## ✨ 核心特性

### 1. 🔍 全场景目标检测
- **单图精细识别**：支持极速导入单张病害图像，输出带有置信度、边界框的渲染图。
- **流媒体实时推理**：原生支持本地视频文件导入及 PC 摄像头实时画面捕获，帧率稳定，实时反馈。
- **多模型无缝切换**：支持“通用模型检测”与“小麦专属病害模型”的一键切换，适应多场景需求。

### 2. 🚀 高并发批量评估引擎
- **工业级文件夹处理**：支持一键选中包含数千张图片的庞大数据集。
- **异步非阻塞设计**：采用 `QThread` 多线程架构重构批量检测逻辑。后台 `BatchWorker` 专注显卡推理，前台 UI 事件循环保持极速响应，彻底消灭“界面假死”。
- **智能诊断报告**：完成分析后，自动计算系统级“侵染指数”，生成包含综合评级与防治建议的弹窗报告。

### 3. 📊 数据闭环与可视化大屏
- **本地持久化管道**：检测产出的结构化数据（病害种类、检测总数）自动沉淀至轻量级本地 CSV 数据库。
- **动态统计看板**：利用 `Matplotlib` 引擎，动态渲染近 15 天检测趋势折线图及病害分布柱状图，为农业决策提供直观的数据支撑。

## 🛠️ 技术架构与环境栈

- **前端展示层**：`PyQt5` (结合自定义 QSS 样式表进行 UI 美化)
- **AI 算法层**：`Ultralytics YOLO` (深度学习推理框架)
- **视觉处理层**：`OpenCV` (矩阵运算、图像绘制与流媒体解码)
- **数据呈现层**：`Matplotlib` + 原生 CSV 数据管道

## 🚀 快速开始

### 1. 环境安装
强烈建议使用 Python 虚拟环境（Virtualenv 或 Conda）运行本项目。
```bash
# 克隆仓库
git clone [https://github.com/你的用户名/Wheat-Disease-Detection.git](https://github.com/你的用户名/Wheat-Disease-Detection.git)
cd Wheat-Disease-Detection

# 安装核心依赖
pip install -r requirements.txt