# -*- coding: utf-8 -*-
from PyQt5.QtCore import QThread, pyqtSignal
import cv2

class BatchWorker(QThread):
    # 定义信号：告诉主界面进度和最终结果 [cite: 113]
    progress_updated = pyqtSignal(int, int) # 当前张数, 总张数
    batch_finished = pyqtSignal(dict, list) # 统计结果, 详细记录

    def __init__(self, model, file_paths):
        super().__init__()
        self.model = model
        self.file_paths = file_paths

    def run(self):
        total = len(self.file_paths)
        disease_stats = {} # 记录每种病害的总数
        detailed_records = [] # 记录每一张图的结果，用于写CSV

        for i, path in enumerate(self.file_paths):
            # 执行推理 [cite: 41]
            results = self.model(path)
            
            # 提取检测到的类别名称
            current_image_diseases = []
            if results and len(results) > 0:
                boxes = results[0].boxes
                for box in boxes:
                    # 获取类别ID对应的名称 [cite: 46]
                    cls_id = int(box.cls[0])
                    label = self.model.names[cls_id]
                    
                    # 累计统计 [cite: 59]
                    disease_stats[label] = disease_stats.get(label, 0) + 1
                    current_image_diseases.append(label)

            # 进度回传 [cite: 113]
            self.progress_updated.emit(i + 1, total)

        # 完成后发送总结果 [cite: 113]
        self.batch_finished.emit(disease_stats, detailed_records)