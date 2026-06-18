"""
肥料计算器工具
根据面积和作物类型计算所需复合肥用量
"""

from typing import Any

from src.citrus_agent.tools.base import BaseTool, ToolRegistry


class FertilizerCalculator(BaseTool):
    """肥料计算器工具"""

    @property
    def name(self) -> str:
        return "fertilizer_calculator"

    @property
    def description(self) -> str:
        return (
            "根据种植面积（亩）和作物类型，计算所需的复合肥用量。"
            "适用于柑橘、蔬菜等常见作物的施肥量估算。"
        )

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "area": {
                    "type": "number",
                    "description": "种植面积，单位：亩",
                },
                "crop_type": {
                    "type": "string",
                    "description": "作物类型，如：柑橘、番茄、辣椒等",
                    "enum": ["柑橘", "番茄", "辣椒", "白菜", "黄瓜", "水稻"],
                },
                "fertilizer_type": {
                    "type": "string",
                    "description": "肥料类型，默认为复合肥",
                    "enum": ["复合肥", "尿素", "钾肥", "磷肥"],
                    "default": "复合肥",
                },
            },
            "required": ["area", "crop_type"],
        }

    async def execute(self, **kwargs) -> dict[str, Any]:
        """
        执行肥料用量计算

        基准用量参考（公斤/亩/年）：
        - 柑橘：复合肥 80-120kg，尿素 30-50kg
        - 番茄：复合肥 60-80kg，尿素 20-30kg
        - 辣椒：复合肥 50-70kg，尿素 15-25kg
        - 白菜：复合肥 40-60kg，尿素 15-20kg
        - 黄瓜：复合肥 55-75kg，尿素 18-28kg
        - 水稻：复合肥 35-50kg，尿素 20-30kg
        """
        area = kwargs.get("area", 0)
        crop_type = kwargs.get("crop_type", "")
        fertilizer_type = kwargs.get("fertilizer_type", "复合肥")

        # 基准用量表（公斤/亩）
        dosage_table = {
            "柑橘": {"复合肥": (80, 120), "尿素": (30, 50), "钾肥": (25, 40), "磷肥": (20, 35)},
            "番茄": {"复合肥": (60, 80), "尿素": (20, 30), "钾肥": (15, 25), "磷肥": (10, 20)},
            "辣椒": {"复合肥": (50, 70), "尿素": (15, 25), "钾肥": (12, 20), "磷肥": (8, 15)},
            "白菜": {"复合肥": (40, 60), "尿素": (15, 20), "钾肥": (10, 18), "磷肥": (8, 12)},
            "黄瓜": {"复合肥": (55, 75), "尿素": (18, 28), "钾肥": (14, 22), "磷肥": (10, 18)},
            "水稻": {"复合肥": (35, 50), "尿素": (20, 30), "钾肥": (8, 15), "磷肥": (10, 18)},
        }

        if crop_type not in dosage_table:
            return {
                "success": False,
                "message": f"暂不支持 {crop_type} 的肥料计算，当前支持：{list(dosage_table.keys())}",
            }

        crop_dosage = dosage_table[crop_type]
        if fertilizer_type not in crop_dosage:
            return {
                "success": False,
                "message": f"暂不支持 {fertilizer_type} 类型，当前支持：{list(crop_dosage.keys())}",
            }

        min_dosage, max_dosage = crop_dosage[fertilizer_type]
        min_total = round(area * min_dosage, 1)
        max_total = round(area * max_dosage, 1)

        return {
            "success": True,
            "result": {
                "crop_type": crop_type,
                "area": area,
                "area_unit": "亩",
                "fertilizer_type": fertilizer_type,
                "dosage_per_mu": f"{min_dosage}-{max_dosage} 公斤/亩",
                "total_dosage": f"{min_total}-{max_total} 公斤",
                "total_min_kg": min_total,
                "total_max_kg": max_total,
            },
            "message": f"{area}亩{crop_type}，建议{fertilizer_type}用量为 {min_total}-{max_total} 公斤（{min_dosage}-{max_dosage} 公斤/亩）",
        }


# 注册工具
ToolRegistry.register(FertilizerCalculator())
