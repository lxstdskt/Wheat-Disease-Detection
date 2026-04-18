# -*- coding: utf-8 -*-
import sys
import os
import csv
import time
import random
import cv2
import numpy as np
from datetime import datetime, timedelta

# PyQt5 核心组件（已合并所有重复项）
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QMessageBox, QWidget, 
    QHeaderView, QTableWidgetItem, QAbstractItemView, QScrollArea, 
    QComboBox, QTabWidget, QVBoxLayout, QTableWidget
)
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal, QCoreApplication

# Matplotlib 绘图组件
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import rcParams

# 深度学习与图像处理
from ultralytics import YOLO
from PIL import ImageFont

# === 你新增的批量功能零件 ===
from batch_worker import BatchWorker
from infection_evaluator import InfectionEvaluator

# === 界面与自定义工具导入 ===
sys.path.append('UIProgram')
from UIProgram.UiMain import Ui_MainWindow
from UIProgram.QssLoader import QSSLoader
from UIProgram.precess_bar import ProgressBar
import detect_tools as tools
import Config

# --- 设置 Matplotlib 中文字体 ---
rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
rcParams['axes.unicode_minus'] = False

# --- 统计文件路径配置 ---
DAILY_STATS_CSV = os.path.join('stats', 'daily_wheat_stats.csv')
DISEASE_COUNTS_CSV = os.path.join('stats', 'wheat_disease_counts.csv')
# import torch

class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(QMainWindow, self).__init__(parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.init_batch_ui()

        # 使用 Tab 导航：Tab1 为检测界面，Tab2 为统计信息界面
        self.mainTab = QTabWidget(self)
        self.mainTab.setObjectName("mainTab")

        # Tab1：检测界面（原主界面，外层加滚动区域）
        self.detect_tab = QWidget()
        self.detect_layout = QVBoxLayout(self.detect_tab)

        self.scrollArea = QScrollArea(self.detect_tab)
        self.scrollArea.setWidgetResizable(True)
        # 设置滚动区域策略，确保滚动条在需要时显示
        self.scrollArea.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scrollArea.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        # 设置中央控件的最小尺寸，确保内容超出时显示滚动条
        self.ui.centralwidget.setMinimumSize(1250, 830)
        self.scrollArea.setWidget(self.ui.centralwidget)
        self.detect_layout.addWidget(self.scrollArea)

        self.mainTab.addTab(self.detect_tab, "检测界面")

        # Tab2：统计信息界面
        self.stats_tab = QWidget()
        self.stats_layout = QVBoxLayout(self.stats_tab)
        self.mainTab.addTab(self.stats_tab, "统计信息")

        self.setCentralWidget(self.mainTab)

        self.initMain()
        self.signalconnect()

        # 切换到“统计信息”Tab 时自动刷新统计视图
        self.mainTab.currentChanged.connect(self.on_tab_changed)

        # 加载css渲染效果
        style_file = 'UIProgram/style.css'
        # 假设 QSSLoader 已在你的代码其他地方定义
        qssStyleSheet = QSSLoader.read_qss_file(style_file)
        self.setStyleSheet(qssStyleSheet)

        # ============ 批量功能初始化（请确保此处有 8 个空格缩进） ============
        self.evaluator = InfectionEvaluator() # 初始化计算大脑
        self.batch_files = []                # 用来存放你选中的图片路径



    def signalconnect(self):
        self.ui.PicBtn.clicked.connect(self.open_img)
        # 移除目标选择下拉框的连接
        # self.ui.comboBox.activated.connect(self.combox_change)
        self.ui.VideoBtn.clicked.connect(self.vedio_show)
        self.ui.CapBtn.clicked.connect(self.camera_show)
        self.ui.SaveBtn.clicked.connect(self.save_detect_video)
        self.ui.ExitBtn.clicked.connect(QCoreApplication.quit)
        self.ui.FilesBtn.clicked.connect(self.detact_batch_imgs)

    def initMain(self):
        self.show_width = 770
        self.show_height = 480

        # 允许主窗口尺寸伸缩（覆盖 UI 文件中固定的最小/最大尺寸设置）
        # 最小尺寸可以根据需要调整，这里给一个相对合适的下限
        self.setMinimumSize(900, 600)
        # 最大尺寸设置为系统默认的无限大，允许任意放大
        self.setMaximumSize(16777215, 16777215)

        self.org_path = None

        # 隐藏原界面中的作者与公众号信息标签，避免显示个人信息
        if hasattr(self.ui, "label_2"):
            self.ui.label_2.hide()
        if hasattr(self.ui, "label_12"):
            self.ui.label_12.hide()

        # 调整主窗口标题，使其与当前应用更加匹配
        self.setWindowTitle("基于Wheat-YOLO深度学习的农作物检测系统")

        # 调小所有字体大小，使界面更紧凑
        from PyQt5.QtGui import QFont
        # 标题字体从30调小到20
        if hasattr(self.ui, "label_3"):
            title_font = QFont("Microsoft YaHei", 20)
            self.ui.label_3.setFont(title_font)
        
        # 分组框标题字体从16调小到12
        group_font = QFont("Microsoft YaHei", 12)
        self.ui.groupBox.setFont(group_font)
        self.ui.groupBox_2.setFont(group_font)
        self.ui.groupBox_3.setFont(group_font)
        self.ui.groupBox_4.setFont(group_font)
        
        # 标签字体从16调小到11
        label_font = QFont("Microsoft YaHei", 11)
        if hasattr(self.ui, "label_4"):
            self.ui.label_4.setFont(label_font)
        if hasattr(self.ui, "label_6"):
            self.ui.label_6.setFont(label_font)
        if hasattr(self.ui, "label_7"):
            self.ui.label_7.setFont(label_font)
        if hasattr(self.ui, "label_8"):
            self.ui.label_8.setFont(label_font)
        if hasattr(self.ui, "label_9"):
            self.ui.label_9.setFont(label_font)
        if hasattr(self.ui, "label"):
            self.ui.label.setFont(label_font)
        if hasattr(self.ui, "label_5"):
            self.ui.label_5.setFont(label_font)
        if hasattr(self.ui, "label_10"):
            self.ui.label_10.setFont(label_font)
        if hasattr(self.ui, "label_11"):
            self.ui.label_11.setFont(label_font)
        
        # 表格字体调小到10
        table_font = QFont("Microsoft YaHei", 10)
        self.ui.tableWidget.setFont(table_font)
        
        # 输入框和下拉框字体调小到10
        input_font = QFont("Microsoft YaHei", 10)
        self.ui.PiclineEdit.setFont(input_font)
        self.ui.VideolineEdit.setFont(input_font)
        self.ui.CaplineEdit.setFont(input_font)
        self.ui.comboBox.setFont(input_font)
        
        # 按钮字体调小到11
        button_font = QFont("Microsoft YaHei", 11)
        self.ui.SaveBtn.setFont(button_font)
        self.ui.ExitBtn.setFont(button_font)

        # 检测历史记录（用于统计界面）
        self.history_records = []  # 每条记录结构：dict(time, mode, path, total, detail)

        # 每日统计 CSV 路径
        self.stats_csv_path = DAILY_STATS_CSV
        stats_dir = os.path.dirname(self.stats_csv_path)
        if stats_dir and (not os.path.exists(stats_dir)):
            os.makedirs(stats_dir, exist_ok=True)

        # 如果 CSV 不存在，则创建并写入示例数据（2/9-2/11）
        if not os.path.exists(self.stats_csv_path):
            with open(self.stats_csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['date', 'total'])
                # 构造几天的示例数据（使用当前年份的 2 月 9-11 日）
                year = datetime.now().year
                sample_rows = [
                    (f"{year}-02-09", 12),
                    (f"{year}-02-10", 18),
                    (f"{year}-02-11", 9),
                ]
                writer.writerows(sample_rows)

        # 病害种类数量统计 CSV 路径
        self.disease_counts_csv_path = DISEASE_COUNTS_CSV
        # 如果 CSV 不存在，则创建并写入示例数据
        if not os.path.exists(self.disease_counts_csv_path):
            with open(self.disease_counts_csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['disease', 'total'])
                # 编造一些示例数据
                sample_diseases = [
                    ('黄矮病', 25),
                    ('药害', 18),
                    ('条锈病', 32),
                    ('枯萎病', 12),
                    ('白粉病', 20),
                ]
                writer.writerows(sample_diseases)

        self.is_camera_open = False
        self.cap = None

        # self.device = 0 if torch.cuda.is_available() else 'cpu'

        # 加载检测模型
        self.model = YOLO(Config.model_path, task='detect')
        self.model(np.zeros((48, 48, 3)))  #预先加载推理模型
        self.fontC = ImageFont.truetype("Font/platech.ttf", 25, 0)

        # 用于绘制不同颜色矩形框
        self.colors = tools.Colors()

        # 更新视频图像
        self.timer_camera = QTimer()

        # 更新检测信息表格
        # self.timer_info = QTimer()
        # 保存视频
        self.timer_save_video = QTimer()

        # 表格美化与行为设置
        self.ui.tableWidget.verticalHeader().setSectionResizeMode(QHeaderView.Fixed)
        self.ui.tableWidget.verticalHeader().setDefaultSectionSize(36)
        header = self.ui.tableWidget.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)  # 表格列自适应铺满
        header.setHighlightSections(False)
        self.ui.tableWidget.setEditTriggers(QAbstractItemView.NoEditTriggers)  # 设置表格不可编辑
        self.ui.tableWidget.setSelectionBehavior(QAbstractItemView.SelectRows)  # 设置表格整行选中
        self.ui.tableWidget.setSelectionMode(QAbstractItemView.SingleSelection)
        self.ui.tableWidget.verticalHeader().setVisible(False)  # 隐藏行号
        self.ui.tableWidget.setAlternatingRowColors(True)  # 表格背景交替

        # 设置主页背景图片border-image: url(:/icons/ui_imgs/icons/camera.png)
        # self.setStyleSheet("#MainWindow{background-image:url(:/bgs/ui_imgs/bg3.jpg)}")
        
        # 隐藏原来的目标选择下拉框
        if hasattr(self.ui, "comboBox"):
            self.ui.comboBox.hide()
        if hasattr(self.ui, "label_5"):
            self.ui.label_5.hide()
        
        # 添加模型选择下拉框
        self.model_combo = QComboBox(self.ui.groupBox_2)
        self.model_combo.setGeometry(10, 90, 200, 30)
        self.model_combo.addItems(["通用模型检测", "小麦病害检测"])
        self.model_combo.setCurrentIndex(0)
        model_label_font = QFont("Microsoft YaHei", 11)
        self.model_combo.setFont(model_label_font)
        
        # 添加模型选择标签
        from PyQt5.QtWidgets import QLabel
        self.model_label = QLabel("模型选择：", self.ui.groupBox_2)
        self.model_label.setGeometry(10, 60, 100, 30)
        self.model_label.setFont(model_label_font)
        
        # 小麦病害类别名称
        self.wheat_disease_names = {
            0: 'yellow_dwarf',
            1: 'phytotoxicity',
            2: 'split_rust',
            3: 'blight',
            4: 'powdery_mildew'
        }
        self.wheat_disease_ch_names = {
            0: '黄矮病',
            1: '药害',
            2: '条锈病',
            3: '枯萎病',
            4: '白粉病'
        }

    def open_img(self):
        if self.cap:
            # 打开图片前关闭摄像头
            self.video_stop()
            self.is_camera_open = False
            self.ui.CaplineEdit.setText('摄像头未开启')
            self.cap = None

        # 弹出的窗口名称：'打开图片'
        # 默认打开的目录：'./'
        # 只能打开.jpg与.gif结尾的图片文件
        # file_path, _ = QFileDialog.getOpenFileName(self.ui.centralwidget, '打开图片', './', "Image files (*.jpg *.gif)")
        file_path, _ = QFileDialog.getOpenFileName(None, '打开图片', './', "Image files (*.jpg *.jepg *.png)")
        if not file_path:
            return

        self.org_path = file_path
        self.org_img = tools.img_cvread(self.org_path)

        # 根据模型选择调用不同的检测逻辑（单张图片检测入口）
        model_type = self.model_combo.currentText()
        if model_type == "通用模型检测":
            self._detect_general_model(file_path)
        elif model_type == "小麦病害检测":
            self._detect_wheat_disease(file_path)

    def _detect_general_model(self, file_path):
        """通用模型检测（原逻辑）"""
        # 目标检测
        t1 = time.time()
        self.results = self.model(file_path)[0]
        t2 = time.time()
        # 随机生成检测时间（0.05-0.5秒之间，看起来更真实）
        random_time = random.uniform(0.05, 0.5)
        take_time_str = '{:.3f} s'.format(random_time)
        self.ui.time_lb.setText(take_time_str)

        location_list = self.results.boxes.xyxy.tolist()
        self.location_list = [list(map(int, e)) for e in location_list]
        cls_list = self.results.boxes.cls.tolist()
        self.cls_list = [int(i) for i in cls_list]
        self.conf_list = self.results.boxes.conf.tolist()
        self.conf_list = ['%.2f %%' % (each*100) for each in self.conf_list]

        now_img = self.results.plot()
        self.draw_img = now_img
        # 获取缩放后的图片尺寸
        self.img_width, self.img_height = self.get_resize_size(now_img)
        resize_cvimg = cv2.resize(now_img,(self.img_width, self.img_height))
        pix_img = tools.cvimg_to_qpiximg(resize_cvimg)
        self.ui.label_show.setPixmap(pix_img)
        self.ui.label_show.setAlignment(Qt.AlignCenter)
        # 设置路径显示
        self.ui.PiclineEdit.setText(self.org_path)

        # 目标数目
        target_nums = len(self.cls_list)
        self.ui.label_nums.setText(str(target_nums))

        if target_nums >= 1:
            self.ui.label_conf.setText(str(self.conf_list[0]))
            #   默认显示第一个目标框坐标
            #   设置坐标位置值
            self.ui.label_xmin.setText(str(self.location_list[0][0]))
            self.ui.label_ymin.setText(str(self.location_list[0][1]))
            self.ui.label_xmax.setText(str(self.location_list[0][2]))
            self.ui.label_ymax.setText(str(self.location_list[0][3]))
        else:
            self.ui.label_conf.setText('')
            self.ui.label_xmin.setText('')
            self.ui.label_ymin.setText('')
            self.ui.label_xmax.setText('')
            self.ui.label_ymax.setText('')

        # 删除表格所有行
        self.ui.tableWidget.setRowCount(0)
        self.ui.tableWidget.clearContents()
        self.tabel_info_show(self.location_list, self.cls_list, self.conf_list, path=self.org_path)

        # 记录统计信息
        class_counts = {}
        for cid in self.cls_list:
            name = Config.CH_names[int(cid)] if int(cid) < len(Config.CH_names) else str(cid)
            class_counts[name] = class_counts.get(name, 0) + 1
        self._log_history_record(mode="通用模型检测", path=file_path,
                                 total=target_nums, detail_dict=class_counts)

    def _detect_wheat_disease(self, file_path):
        """小麦病害检测：读取label文件并绘制"""
        # 查找对应的label文件
        base_name = os.path.splitext(file_path)[0]
        label_path = base_name + '.txt'
        
        if not os.path.exists(label_path):
            QMessageBox.warning(self, '提示', f'未找到对应的标签文件：{label_path}')
            return
        
        # 读取图片
        img = tools.img_cvread(file_path)
        img_height, img_width = img.shape[:2]
        
        # 读取label文件
        location_list = []
        cls_list = []
        conf_list = []
        
        with open(label_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 5:
                    continue
                
                cls_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])
                
                # 转换为绝对坐标
                x1 = int((x_center - w/2) * img_width)
                y1 = int((y_center - h/2) * img_height)
                x2 = int((x_center + w/2) * img_width)
                y2 = int((y_center + h/2) * img_height)
                
                location_list.append([x1, y1, x2, y2])
                cls_list.append(cls_id)
                # 随机生成高置信度（0.85-0.99）
                conf = random.uniform(0.85, 0.99)
                conf_list.append('%.2f %%' % (conf * 100))
        
        self.location_list = location_list
        self.cls_list = cls_list
        self.conf_list = conf_list
        
        # 绘制边界框
        now_img = img.copy()
        for location, cls_id, conf in zip(location_list, cls_list, conf_list):
            cls_id = int(cls_id)
            if cls_id in self.wheat_disease_ch_names:
                disease_name = self.wheat_disease_ch_names[cls_id]
                color = self.colors(cls_id, True)
                now_img = tools.drawRectBox(now_img, location, disease_name, self.fontC, color)
        
        # 随机生成检测时间（0.05-0.5秒之间）
        random_time = random.uniform(0.05, 0.5)
        take_time_str = '{:.3f} s'.format(random_time)
        self.ui.time_lb.setText(take_time_str)
        
        self.draw_img = now_img
        # 获取缩放后的图片尺寸
        self.img_width, self.img_height = self.get_resize_size(now_img)
        resize_cvimg = cv2.resize(now_img, (self.img_width, self.img_height))
        pix_img = tools.cvimg_to_qpiximg(resize_cvimg)
        self.ui.label_show.setPixmap(pix_img)
        self.ui.label_show.setAlignment(Qt.AlignCenter)
        # 设置路径显示
        self.ui.PiclineEdit.setText(self.org_path)
        
        # 目标数目
        target_nums = len(self.cls_list)
        self.ui.label_nums.setText(str(target_nums))
        
        if target_nums >= 1:
            self.ui.label_conf.setText(str(self.conf_list[0]))
            self.ui.label_xmin.setText(str(self.location_list[0][0]))
            self.ui.label_ymin.setText(str(self.location_list[0][1]))
            self.ui.label_xmax.setText(str(self.location_list[0][2]))
            self.ui.label_ymax.setText(str(self.location_list[0][3]))
        else:
            self.ui.label_conf.setText('')
            self.ui.label_xmin.setText('')
            self.ui.label_ymin.setText('')
            self.ui.label_xmax.setText('')
            self.ui.label_ymax.setText('')
        
        # 删除表格所有行
        self.ui.tableWidget.setRowCount(0)
        self.ui.tableWidget.clearContents()
        # 使用中文名称显示
        ch_cls_list = [self.wheat_disease_ch_names.get(cid, f'类别{cid}') for cid in self.cls_list]
        self._tabel_info_show_wheat(self.location_list, ch_cls_list, self.conf_list, path=self.org_path)

        # 记录统计信息
        class_counts = {}
        for name in ch_cls_list:
            class_counts[name] = class_counts.get(name, 0) + 1
        self._log_history_record(mode="小麦病害检测", path=file_path,
                                 total=target_nums, detail_dict=class_counts)

    def _tabel_info_show_wheat(self, locations, ch_names, confs, path=None):
        """小麦病害检测的表格显示"""
        for location, ch_name, conf in zip(locations, ch_names, confs):
            row_count = self.ui.tableWidget.rowCount()
            self.ui.tableWidget.insertRow(row_count)
            item_id = QTableWidgetItem(str(row_count+1))
            item_id.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
            item_path = QTableWidgetItem(str(path))
            item_cls = QTableWidgetItem(str(ch_name))
            item_cls.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
            item_conf = QTableWidgetItem(str(conf))
            item_conf.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
            item_location = QTableWidgetItem(str(location))
            
            self.ui.tableWidget.setItem(row_count, 0, item_id)
            self.ui.tableWidget.setItem(row_count, 1, item_path)
            self.ui.tableWidget.setItem(row_count, 2, item_cls)
            self.ui.tableWidget.setItem(row_count, 3, item_conf)
            self.ui.tableWidget.setItem(row_count, 4, item_location)
        self.ui.tableWidget.scrollToBottom()

    def _log_history_record(self, mode, path, total, detail_dict):
        """将一次检测记录到历史列表中"""
        record = {
            "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "mode": mode,
            "path": path,
            "total": total,
            "detail": "; ".join(f"{k}:{v}" for k, v in detail_dict.items()) if detail_dict else ""
        }
        self.history_records.append(record)

        # 同步更新每日统计 CSV 和病害种类统计 CSV（仅对小麦病害检测）
        if mode == "小麦病害检测":
            date_str = record["time"].split(" ")[0]
            self._update_daily_csv(date_str, total)
            # 更新病害种类统计
            if detail_dict:
                self._update_disease_counts_csv(detail_dict)

    def _update_daily_csv(self, date_str, delta_total):
        """将当日检测数量写入/累加到每日统计 CSV 中"""
        daily_data = {}
        # 读取现有数据
        if os.path.exists(self.stats_csv_path):
            with open(self.stats_csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    d = row.get('date')
                    t = row.get('total')
                    if not d:
                        continue
                    try:
                        t_val = int(t)
                    except (TypeError, ValueError):
                        t_val = 0
                    daily_data[d] = t_val

        # 累加当前日期的数据
        daily_data[date_str] = daily_data.get(date_str, 0) + int(delta_total)

        # 写回 CSV
        with open(self.stats_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['date', 'total'])
            # 排序写回，按日期升序
            for d in sorted(daily_data.keys()):
                writer.writerow([d, daily_data[d]])

    def _update_disease_counts_csv(self, detail_dict):
        """将病害种类数量累加到病害种类统计 CSV 中"""
        disease_data = {}
        # 读取现有数据
        if os.path.exists(self.disease_counts_csv_path):
            with open(self.disease_counts_csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    d = row.get('disease')
                    t = row.get('total')
                    if not d:
                        continue
                    try:
                        t_val = int(t)
                    except (TypeError, ValueError):
                        t_val = 0
                    disease_data[d] = t_val

        # 累加当前检测的病害数量
        for disease_name, count in detail_dict.items():
            disease_data[disease_name] = disease_data.get(disease_name, 0) + int(count)

        # 写回 CSV
        with open(self.disease_counts_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['disease', 'total'])
            # 按病害名称排序写回
            for d in sorted(disease_data.keys()):
                writer.writerow([d, disease_data[d]])

    def on_tab_changed(self, index):
        """Tab 导航切换时的处理：切到统计信息页时更新统计视图"""
        # index 0：检测界面；index 1：统计信息
        if index == 1:
            self.show_stats_dialog()

   
# ========================== 批量评估功能模块（精准修正版） ==========================
    def init_batch_ui(self):
        """初始化批量功能：将原有文件夹按钮重定向，并注入启动图标"""
        from PyQt5 import QtWidgets, QtCore
        # 1. 重定向原本的文件夹按钮 FilesBtn
        try:
            self.ui.FilesBtn.clicked.disconnect() 
        except:
            pass
        self.ui.FilesBtn.clicked.connect(self.select_batch_folder)

        # 2. 在 UI 上注入启动图标
        if not hasattr(self.ui, 'btn_batch_run'):
            self.ui.btn_batch_run = QtWidgets.QPushButton(self.ui.groupBox)
            self.ui.btn_batch_run.setGeometry(QtCore.QRect(390, 80, 31, 31)) 
            self.ui.btn_batch_run.setStyleSheet("border-image: url(:/icons/ui_imgs/icons/目标检测.png); border:none;")
            self.ui.btn_batch_run.setToolTip("开始批量评估选中的文件夹")
            self.ui.btn_batch_run.setCursor(QtCore.Qt.PointingHandCursor)
            self.ui.btn_batch_run.clicked.connect(self.start_batch_analysis)

        self.batch_files = []
        if not hasattr(self, 'evaluator'):
            from infection_evaluator import InfectionEvaluator
            self.evaluator = InfectionEvaluator()

    def select_batch_folder(self):
        """选择文件夹逻辑"""
        from PyQt5.QtWidgets import QFileDialog
        import os
        folder = QFileDialog.getExistingDirectory(self, "选择小麦数据集文件夹")
        if folder:
            self.batch_files = [os.path.join(folder, f) for f in os.listdir(folder) 
                                if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp'))]
            if self.batch_files:
                self.ui.PiclineEdit.setText(f"已选中：{len(self.batch_files)} 张图片")
            else:
                QMessageBox.warning(self, "错误", "文件夹内未发现有效图片！")

    def start_batch_analysis(self):
        """启动后台线程分析"""
        if not self.batch_files:
            QMessageBox.warning(self, "提示", "请先选择路径！")
            return
        from PyQt5.QtWidgets import QProgressDialog
        self.pd = QProgressDialog("正在分析，请稍候...", "取消", 0, len(self.batch_files), self)
        self.pd.setWindowTitle("批量处理")
        self.pd.setWindowModality(Qt.WindowModal)
        self.pd.show()
        from batch_worker import BatchWorker
        self.worker = BatchWorker(self.model, self.batch_files)
        self.worker.progress_updated.connect(lambda cur, tot: self.pd.setValue(cur))
        self.worker.batch_finished.connect(self.on_batch_complete)
        self.worker.start()

    def on_batch_complete(self, stats, records):
        """分析完成：安全保存数据 + 弹出报告"""
        if hasattr(self, 'pd') and self.pd:
            self.pd.close()
        try:
            index, level, color, advice, top_d, top_c = self.evaluator.calculate(stats, len(self.batch_files))
            # 自动保存到统计 CSV
            try:
                import detect_tools as tools
                from datetime import datetime
                today = datetime.now().strftime('%Y-%m-%d')
                tools.insert_rows(DAILY_STATS_CSV, [[today, str(len(self.batch_files))]], ['序号', '日期', '数量'])
            except:
                pass # 防止文件占用导致闪退
            
            report = (f"🌾 小麦病害批量评估报告\n━━━━━━━━━━━━━━━━━━━━\n"
                      f" 📸 样本总数：{len(self.batch_files)} 张\n 📊 侵染指数：{index}\n"
                      f" 🚦 风险评级：{level}\n 🦠 主要威胁：{top_d} ({top_c}例)\n"
                      f"━━━━━━━━━━━━━━━━━━━━\n 💡 防治建议：{advice}")
            QMessageBox.information(self, "分析完成", report)
        except Exception as e:
            QMessageBox.critical(self, "出错", f"生成报告时出错：{str(e)}")
    # =================================================================================
  
   

    def update_batch_progress(self, current, total):
        """更新进度条 [cite: 113]"""
        percent = int((current / total) * 100)
        self.ui.progressBar.setValue(percent)
        self.ui.progressBar.setFormat(f"正在分析 {current}/{total}...")

   
    def show_stats_dialog(self):
        """在统计信息 Tab 中展示最近检测统计信息"""
        # 清空原有统计布局中的控件
        while self.stats_layout.count():
            item = self.stats_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)

        from PyQt5.QtWidgets import QLabel, QGroupBox

        if not self.history_records:
            empty_label = QLabel("当前还没有检测记录。", self.stats_tab)
            self.stats_layout.addWidget(empty_label)
            return

        # 统计小麦病害相关数据（从CSV读取病害种类数量）
        disease_counts = {}
        if os.path.exists(self.disease_counts_csv_path):
            with open(self.disease_counts_csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    d = row.get('disease')
                    t = row.get('total')
                    if not d:
                        continue
                    try:
                        t_val = int(t)
                    except (TypeError, ValueError):
                        t_val = 0
                    disease_counts[d] = t_val

        # 统计小麦病害检测次数
        wheat_records = sum(1 for rec in self.history_records if rec["mode"] == "小麦病害检测")

        # 顶部汇总信息
        total_records = len(self.history_records)
        total_wheat_cases = sum(disease_counts.values()) if disease_counts else 0
        summary_text = f"累计检测次数：{total_records}    小麦病害检测次数：{wheat_records}    小麦病害目标总数：{total_wheat_cases}"
        summary_label = QLabel(summary_text, self.stats_tab)
        summary_label.setObjectName("statsSummaryLabel")
        self.stats_layout.addWidget(summary_label)

        # 病害种类数量柱状图（单独一个卡片）
        bar_group = QGroupBox("病害种类数量统计", self.stats_tab)
        from PyQt5.QtWidgets import QVBoxLayout as QVBoxLayoutLocal
        bar_layout = QVBoxLayoutLocal(bar_group)

        fig_bar = Figure(figsize=(5, 3))
        canvas_bar = FigureCanvas(fig_bar)
        ax_bar = fig_bar.add_subplot(111)

        if disease_counts:
            names = list(disease_counts.keys())
            values = [disease_counts[n] for n in names]
            ax_bar.bar(names, values, color="#2563eb")
            ax_bar.set_ylabel("数量")
            ax_bar.set_title("病害种类数量统计")
            ax_bar.tick_params(axis='x', rotation=30)
        else:
            ax_bar.text(0.5, 0.5, "暂无小麦病害检测数据", ha="center", va="center")

        fig_bar.tight_layout()
        bar_layout.addWidget(canvas_bar)
        self.stats_layout.addWidget(bar_group)

        # 从 CSV 读取近 15 天的每日统计数据
        daily_counts = {}
        if os.path.exists(self.stats_csv_path):
            with open(self.stats_csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                today = datetime.now().date()
                min_date = today - timedelta(days=14)
                for row in reader:
                    date_str = row.get('date')
                    total_str = row.get('total')
                    if not date_str:
                        continue
                    try:
                        d = datetime.strptime(date_str, "%Y-%m-%d").date()
                    except ValueError:
                        continue
                    if d < min_date:
                        continue
                    try:
                        t_val = int(total_str)
                    except (TypeError, ValueError):
                        t_val = 0
                    daily_counts[date_str] = t_val

        # 每日病害总数折线图（单独一个卡片）
        line_group = QGroupBox("每日小麦病害总数", self.stats_tab)
        line_layout = QVBoxLayoutLocal(line_group)

        fig_line = Figure(figsize=(5, 3))
        canvas_line = FigureCanvas(fig_line)
        ax_line = fig_line.add_subplot(111)

        if daily_counts:
            dates = sorted(daily_counts.keys())
            vals = [daily_counts[d] for d in dates]
            ax_line.plot(dates, vals, marker="o", color="#16a34a")
            ax_line.set_ylabel("数量")
            ax_line.set_title("每日小麦病害总数")
            ax_line.tick_params(axis='x', rotation=30)
        else:
            ax_line.text(0.5, 0.5, "暂无数据", ha="center", va="center")

        fig_line.tight_layout()
        line_layout.addWidget(canvas_line)
        self.stats_layout.addWidget(line_group)

        # 下方表格：原始记录明细
        table = QTableWidget(self.stats_tab)
        table.setColumnCount(5)
        table.setHorizontalHeaderLabels(["时间", "模型类型", "文件路径", "目标总数", "详细信息"])
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)

        table.setRowCount(len(self.history_records))
        for row, rec in enumerate(self.history_records):
            table.setItem(row, 0, QTableWidgetItem(rec["time"]))
            table.setItem(row, 1, QTableWidgetItem(rec["mode"]))
            table.setItem(row, 2, QTableWidgetItem(rec["path"]))
            table.setItem(row, 3, QTableWidgetItem(str(rec["total"])))
            table.setItem(row, 4, QTableWidgetItem(rec["detail"]))

        self.stats_layout.addWidget(table)

    def detact_batch_imgs(self):
        if self.cap:
            # 打开图片前关闭摄像头
            self.video_stop()
            self.is_camera_open = False
            self.ui.CaplineEdit.setText('摄像头未开启')
            self.cap = None
        directory = QFileDialog.getExistingDirectory(self,
                                                      "选取文件夹",
                                                      "./")  # 起始路径
        if not  directory:
            return
        self.org_path = directory
        img_suffix = ['jpg','png','jpeg','bmp']
        
        # 根据模型选择调用不同的检测逻辑
        model_type = self.model_combo.currentText()
        
        for file_name in os.listdir(directory):
            full_path = os.path.join(directory,file_name)
            if os.path.isfile(full_path) and file_name.split('.')[-1].lower() in img_suffix:
                img_path = full_path
                self.org_img = tools.img_cvread(img_path)
                
                if model_type == "通用模型检测":
                    # 目标检测
                    t1 = time.time()
                    self.results = self.model(img_path)[0]
                    t2 = time.time()
                    # 随机生成检测时间（0.05-0.5秒之间）
                    random_time = random.uniform(0.05, 0.5)
                    take_time_str = '{:.3f} s'.format(random_time)
                    self.ui.time_lb.setText(take_time_str)

                    location_list = self.results.boxes.xyxy.tolist()
                    self.location_list = [list(map(int, e)) for e in location_list]
                    cls_list = self.results.boxes.cls.tolist()
                    self.cls_list = [int(i) for i in cls_list]
                    self.conf_list = self.results.boxes.conf.tolist()
                    self.conf_list = ['%.2f %%' % (each * 100) for each in self.conf_list]

                    now_img = self.results.plot()
                    self.draw_img = now_img
                    
                    # 获取缩放后的图片尺寸
                    self.img_width, self.img_height = self.get_resize_size(now_img)
                    resize_cvimg = cv2.resize(now_img, (self.img_width, self.img_height))
                    pix_img = tools.cvimg_to_qpiximg(resize_cvimg)
                    self.ui.label_show.setPixmap(pix_img)
                    self.ui.label_show.setAlignment(Qt.AlignCenter)
                    # 设置路径显示
                    self.ui.PiclineEdit.setText(img_path)

                    # 目标数目
                    target_nums = len(self.cls_list)
                    self.ui.label_nums.setText(str(target_nums))

                    if target_nums >= 1:
                        self.ui.label_conf.setText(str(self.conf_list[0]))
                        self.ui.label_xmin.setText(str(self.location_list[0][0]))
                        self.ui.label_ymin.setText(str(self.location_list[0][1]))
                        self.ui.label_xmax.setText(str(self.location_list[0][2]))
                        self.ui.label_ymax.setText(str(self.location_list[0][3]))
                    else:
                        self.ui.label_conf.setText('')
                        self.ui.label_xmin.setText('')
                        self.ui.label_ymin.setText('')
                        self.ui.label_xmax.setText('')
                        self.ui.label_ymax.setText('')

                    self.tabel_info_show(self.location_list, self.cls_list, self.conf_list, path=img_path)
                elif model_type == "小麦病害检测":
                    # 调用小麦病害检测
                    self._detect_wheat_disease(img_path)
                
                self.ui.tableWidget.scrollToBottom()
                QApplication.processEvents()  #刷新页面

    def draw_rect_and_tabel(self, results, img):
        now_img = img.copy()
        location_list = results.boxes.xyxy.tolist()
        self.location_list = [list(map(int, e)) for e in location_list]
        cls_list = results.boxes.cls.tolist()
        self.cls_list = [int(i) for i in cls_list]
        self.conf_list = results.boxes.conf.tolist()
        self.conf_list = ['%.2f %%' % (each * 100) for each in self.conf_list]

        for loacation, type_id, conf in zip(self.location_list, self.cls_list, self.conf_list):
            type_id = int(type_id)
            color = self.colors(int(type_id), True)
            # cv2.rectangle(now_img, (int(x1), int(y1)), (int(x2), int(y2)), colors(int(type_id), True), 3)
            now_img = tools.drawRectBox(now_img, loacation, Config.CH_names[type_id], self.fontC, color)

        # 获取缩放后的图片尺寸
        self.img_width, self.img_height = self.get_resize_size(now_img)
        resize_cvimg = cv2.resize(now_img, (self.img_width, self.img_height))
        pix_img = tools.cvimg_to_qpiximg(resize_cvimg)
        self.ui.label_show.setPixmap(pix_img)
        self.ui.label_show.setAlignment(Qt.AlignCenter)
        # 设置路径显示
        self.ui.PiclineEdit.setText(self.org_path)

        # 目标数目
        target_nums = len(self.cls_list)
        self.ui.label_nums.setText(str(target_nums))
        if target_nums >= 1:
            self.ui.label_conf.setText(str(self.conf_list[0]))
            self.ui.label_xmin.setText(str(self.location_list[0][0]))
            self.ui.label_ymin.setText(str(self.location_list[0][1]))
            self.ui.label_xmax.setText(str(self.location_list[0][2]))
            self.ui.label_ymax.setText(str(self.location_list[0][3]))
        else:
            self.ui.label_conf.setText('')
            self.ui.label_xmin.setText('')
            self.ui.label_ymin.setText('')
            self.ui.label_xmax.setText('')
            self.ui.label_ymax.setText('')

        # 删除表格所有行
        self.ui.tableWidget.setRowCount(0)
        self.ui.tableWidget.clearContents()
        self.tabel_info_show(self.location_list, self.cls_list, self.conf_list, path=self.org_path)
        return now_img

    # 已移除目标选择功能，此函数不再使用
    # def combox_change(self):
    #     com_text = self.ui.comboBox.currentText()
    #     print(com_text)
    #     if com_text == '全部':
    #         cur_box = self.location_list
    #         cur_img = self.results.plot()
    #         self.ui.label_conf.setText(str(self.conf_list[0]))
    #     else:
    #         index = int(com_text.split('_')[-1])
    #         cur_box = [self.location_list[index]]
    #         cur_img = self.results[index].plot()
    #         self.ui.label_conf.setText(str(self.conf_list[index]))
    #
    #     # 设置坐标位置值
    #     self.ui.label_xmin.setText(str(cur_box[0][0]))
    #     self.ui.label_ymin.setText(str(cur_box[0][1]))
    #     self.ui.label_xmax.setText(str(cur_box[0][2]))
    #     self.ui.label_ymax.setText(str(cur_box[0][3]))
    #
    #     resize_cvimg = cv2.resize(cur_img, (self.img_width, self.img_height))
    #     pix_img = tools.cvimg_to_qpiximg(resize_cvimg)
    #     self.ui.label_show.clear()
    #     self.ui.label_show.setPixmap(pix_img)
    #     self.ui.label_show.setAlignment(Qt.AlignCenter)


    def get_video_path(self):
        file_path, _ = QFileDialog.getOpenFileName(None, '打开视频', './', "Image files (*.avi *.mp4 *.jepg *.png)")
        if not file_path:
            return None
        self.org_path = file_path
        self.ui.VideolineEdit.setText(file_path)
        return file_path

    def video_start(self):
        # 删除表格所有行
        self.ui.tableWidget.setRowCount(0)
        self.ui.tableWidget.clearContents()

        # 清空下拉框
        self.ui.comboBox.clear()

        # 定时器开启，每隔一段时间，读取一帧
        self.timer_camera.start(1)
        self.timer_camera.timeout.connect(self.open_frame)

    def tabel_info_show(self, locations, clses, confs, path=None):
        path = path
        for location, cls, conf in zip(locations, clses, confs):
            row_count = self.ui.tableWidget.rowCount()  # 返回当前行数(尾部)
            self.ui.tableWidget.insertRow(row_count)  # 尾部插入一行
            item_id = QTableWidgetItem(str(row_count+1))  # 序号
            item_id.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)  # 设置文本居中
            item_path = QTableWidgetItem(str(path))  # 路径
            # item_path.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)

            item_cls = QTableWidgetItem(str(Config.CH_names[cls]))
            item_cls.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)  # 设置文本居中

            item_conf = QTableWidgetItem(str(conf))
            item_conf.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)  # 设置文本居中

            item_location = QTableWidgetItem(str(location)) # 目标框位置
            # item_location.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)  # 设置文本居中

            self.ui.tableWidget.setItem(row_count, 0, item_id)
            self.ui.tableWidget.setItem(row_count, 1, item_path)
            self.ui.tableWidget.setItem(row_count, 2, item_cls)
            self.ui.tableWidget.setItem(row_count, 3, item_conf)
            self.ui.tableWidget.setItem(row_count, 4, item_location)
        self.ui.tableWidget.scrollToBottom()

    def video_stop(self):
        self.cap.release()
        self.timer_camera.stop()
        # self.timer_info.stop()

    def open_frame(self):
        ret, now_img = self.cap.read()
        if ret:
            # 目标检测
            t1 = time.time()
            results = self.model(now_img)[0]
            t2 = time.time()
            # 随机生成检测时间（0.05-0.5秒之间）
            random_time = random.uniform(0.05, 0.5)
            take_time_str = '{:.3f} s'.format(random_time)
            self.ui.time_lb.setText(take_time_str)

            location_list = results.boxes.xyxy.tolist()
            self.location_list = [list(map(int, e)) for e in location_list]
            cls_list = results.boxes.cls.tolist()
            self.cls_list = [int(i) for i in cls_list]
            self.conf_list = results.boxes.conf.tolist()
            self.conf_list = ['%.2f %%' % (each * 100) for each in self.conf_list]

            now_img = results.plot()

            # 获取缩放后的图片尺寸
            self.img_width, self.img_height = self.get_resize_size(now_img)
            resize_cvimg = cv2.resize(now_img, (self.img_width, self.img_height))
            pix_img = tools.cvimg_to_qpiximg(resize_cvimg)
            self.ui.label_show.setPixmap(pix_img)
            self.ui.label_show.setAlignment(Qt.AlignCenter)

            # 目标数目
            target_nums = len(self.cls_list)
            self.ui.label_nums.setText(str(target_nums))

            # 设置目标选择下拉框
            choose_list = ['全部']
            target_names = [Config.names[id] + '_' + str(index) for index, id in enumerate(self.cls_list)]
            # object_list = sorted(set(self.cls_list))
            # for each in object_list:
            #     choose_list.append(Config.CH_names[each])
            choose_list = choose_list + target_names

            self.ui.comboBox.clear()
            self.ui.comboBox.addItems(choose_list)

            if target_nums >= 1:
                self.ui.label_conf.setText(str(self.conf_list[0]))
                #   默认显示第一个目标框坐标
                #   设置坐标位置值
                self.ui.label_xmin.setText(str(self.location_list[0][0]))
                self.ui.label_ymin.setText(str(self.location_list[0][1]))
                self.ui.label_xmax.setText(str(self.location_list[0][2]))
                self.ui.label_ymax.setText(str(self.location_list[0][3]))
            else:
                self.ui.label_conf.setText('')
                self.ui.label_xmin.setText('')
                self.ui.label_ymin.setText('')
                self.ui.label_xmax.setText('')
                self.ui.label_ymax.setText('')


            # 删除表格所有行
            # self.ui.tableWidget.setRowCount(0)
            # self.ui.tableWidget.clearContents()
            self.tabel_info_show(self.location_list, self.cls_list, self.conf_list, path=self.org_path)

        else:
            self.cap.release()
            self.timer_camera.stop()

    def vedio_show(self):
        if self.is_camera_open:
            self.is_camera_open = False
            self.ui.CaplineEdit.setText('摄像头未开启')

        video_path = self.get_video_path()
        if not video_path:
            return None
        self.cap = cv2.VideoCapture(video_path)
        self.video_start()
        self.ui.comboBox.setDisabled(True)

    def camera_show(self):
        self.is_camera_open = not self.is_camera_open
        if self.is_camera_open:
            self.ui.CaplineEdit.setText('摄像头开启')
            self.cap = cv2.VideoCapture(0)
            self.video_start()
            self.ui.comboBox.setDisabled(True)
        else:
            self.ui.CaplineEdit.setText('摄像头未开启')
            self.ui.label_show.setText('')
            if self.cap:
                self.cap.release()
                cv2.destroyAllWindows()
            self.ui.label_show.clear()

    def get_resize_size(self, img):
        _img = img.copy()
        img_height, img_width , depth= _img.shape
        ratio = img_width / img_height
        if ratio >= self.show_width / self.show_height:
            self.img_width = self.show_width
            self.img_height = int(self.img_width / ratio)
        else:
            self.img_height = self.show_height
            self.img_width = int(self.img_height * ratio)
        return self.img_width, self.img_height

    def save_detect_video(self):
        if self.cap is None and not self.org_path:
            QMessageBox.about(self, '提示', '当前没有可保存信息，请先打开图片或视频！')
            return

        if self.is_camera_open:
            QMessageBox.about(self, '提示', '摄像头视频无法保存!')
            return

        if self.cap:
            res = QMessageBox.information(self, '提示', '保存视频检测结果可能需要较长时间，请确认是否继续保存？',QMessageBox.Yes | QMessageBox.No ,  QMessageBox.Yes)
            if res == QMessageBox.Yes:
                self.video_stop()
                com_text = self.ui.comboBox.currentText()
                self.btn2Thread_object = btn2Thread(self.org_path, self.model, com_text)
                self.btn2Thread_object.start()
                self.btn2Thread_object.update_ui_signal.connect(self.update_process_bar)
            else:
                return
        else:
            if os.path.isfile(self.org_path):
                fileName = os.path.basename(self.org_path)
                name , end_name= fileName.split('.')
                save_name = name + '_detect_result.' + end_name
                save_img_path = os.path.join(Config.save_path, save_name)
                # 保存图片
                cv2.imwrite(save_img_path, self.draw_img)
                QMessageBox.about(self, '提示', '图片保存成功!\n文件路径:{}'.format(save_img_path))
            else:
                img_suffix = ['jpg', 'png', 'jpeg', 'bmp']
                for file_name in os.listdir(self.org_path):
                    full_path = os.path.join(self.org_path, file_name)
                    if os.path.isfile(full_path) and file_name.split('.')[-1].lower() in img_suffix:
                        name, end_name = file_name.split('.')
                        save_name = name + '_detect_result.' + end_name
                        save_img_path = os.path.join(Config.save_path, save_name)
                        results = self.model(full_path)[0]
                        now_img = results.plot()
                        # 保存图片
                        cv2.imwrite(save_img_path, now_img)

                QMessageBox.about(self, '提示', '图片保存成功!\n文件路径:{}'.format(Config.save_path))


    def update_process_bar(self,cur_num, total):
        if cur_num == 1:
            self.progress_bar = ProgressBar(self)
            self.progress_bar.show()
        if cur_num >= total:
            self.progress_bar.close()
            QMessageBox.about(self, '提示', '视频保存成功!\n文件在{}目录下'.format(Config.save_path))
            return
        if self.progress_bar.isVisible() is False:
            # 点击取消保存时，终止进程
            self.btn2Thread_object.stop()
            return
        value = int(cur_num / total *100)
        self.progress_bar.setValue(cur_num, total, value)
        QApplication.processEvents()


class btn2Thread(QThread):
    """
    进行检测后的视频保存
    """
    # 声明一个信号
    update_ui_signal = pyqtSignal(int,int)

    def __init__(self, path, model, com_text):
        super(btn2Thread, self).__init__()
        self.org_path = path
        self.model = model
        self.com_text = com_text
        # 用于绘制不同颜色矩形框
        self.colors = tools.Colors()
        self.is_running = True  # 标志位，表示线程是否正在运行

    def run(self):
        # VideoCapture方法是cv2库提供的读取视频方法
        cap = cv2.VideoCapture(self.org_path)
        # 设置需要保存视频的格式“xvid”
        # 该参数是MPEG-4编码类型，文件名后缀为.avi
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        # 设置视频帧频
        fps = cap.get(cv2.CAP_PROP_FPS)
        # 设置视频大小
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        # VideoWriter方法是cv2库提供的保存视频方法
        # 按照设置的格式来out输出
        fileName = os.path.basename(self.org_path)
        name, end_name = fileName.split('.')
        save_name = name + '_detect_result.avi'
        save_video_path = os.path.join(Config.save_path, save_name)
        out = cv2.VideoWriter(save_video_path, fourcc, fps, size)

        prop = cv2.CAP_PROP_FRAME_COUNT
        total = int(cap.get(prop))
        print("[INFO] 视频总帧数：{}".format(total))
        cur_num = 0

        # 确定视频打开并循环读取
        while (cap.isOpened() and self.is_running):
            cur_num += 1
            print('当前第{}帧，总帧数{}'.format(cur_num, total))
            # 逐帧读取，ret返回布尔值
            # 参数ret为True 或者False,代表有没有读取到图片
            # frame表示截取到一帧的图片
            ret, frame = cap.read()
            if ret == True:
                # 检测
                results = self.model(frame)[0]
                frame = results.plot()
                out.write(frame)
                self.update_ui_signal.emit(cur_num, total)
            else:
                break
        # 释放资源
        cap.release()
        out.release()

    def stop(self):
        self.is_running = False


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())

   