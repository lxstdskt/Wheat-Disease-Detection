# -*- coding: utf-8 -*-
# 专门负责计算侵染指数和评级的模块

class InfectionEvaluator:
    def __init__(self):
        # 权重配置：根据规格书 v2.0 设定 [cite: 70]
        self.weights = {
            "条锈病": 1.2,
            "枯萎病": 1.3,
            "白粉病": 1.0,
            "药害": 0.6,
            "黄矮病": 1.1
        }

    def calculate(self, disease_counts, total_images):
        """
        disease_counts: 字典格式，如 {"条锈病": 5, "枯萎病": 2}
        total_images: 批量检测的图片总张数
        """
        if total_images == 0:
            return 0, "健康", "绿色", "暂无检测数据"

        # 1. 计算加权总分 [cite: 68]
        weighted_sum = 0
        max_disease = "无"
        max_count = 0

        for name, count in disease_counts.items():
            weight = self.weights.get(name, 1.0)
            weighted_sum += count * weight
            # 顺便找出数量最多的病害 [cite: 65]
            if count > max_count:
                max_count = count
                max_disease = name

        # 2. 计算侵染指数 [cite: 68]
        index = (weighted_sum / total_images) * 10
        
        # 3. 评级映射 [cite: 72]
        if index <= 10:
            level, color = "健康（无需防治）", "green"
        elif index <= 20:
            level, color = "轻度发生（注意监测）", "darkgreen"
        elif index <= 35:
            level, color = "中度发生（建议用药）", "orange"
        else:
            level, color = "重度爆发（立即防治）", "red"

        # 4. 生成防治建议 [cite: 74]
        advice = self.get_advice(max_disease, level)
        
        return round(index, 1), level, color, advice, max_disease, max_count

    def get_advice(self, disease, level):
        """根据病害和级别生成建议 [cite: 74]"""
        if "中度" in level or "重度" in level:
            advice_map = {
                "条锈病": "建议3-5日内喷洒三唑酮或戊唑醇",
                "枯萎病": "建议立即喷洒多菌灵或甲基硫菌灵",
                "白粉病": "建议喷洒嘧菌酯或乙嘧酚",
                "黄矮病": "建议防治蚜虫，喷洒吡虫啉",
                "药害": "检测到药害症状，建议暂停用药，喷施叶面肥缓解"
            }
            return advice_map.get(disease, "暂无需用药，建议加强田间巡查")
        return "暂无需用药，建议加强田间巡查"