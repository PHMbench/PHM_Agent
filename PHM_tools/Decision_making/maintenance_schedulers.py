from __future__ import annotations

"""用于生成最优维护策略的工具集。"""

from typing import Dict

from smolagents import tool
from utils.registry import register_tool


@register_tool("generate_maintenance_plan")
@tool
def generate_maintenance_plan(rul_prediction: Dict[str, float | list], cost_data: Dict[str, float]) -> Dict[str, object]:
    """综合RUL和成本信息，生成一份详细的维护建议。

    Args:
        rul_prediction: RUL预测结果。
        cost_data: 相关的成本数据。

    Returns:
        一份结构化的维护计划，包含建议、理由和预估成本。
    """
    predicted_rul = float(rul_prediction.get("predicted_rul", 0.0))
    spare_cost = cost_data.get("spare_part_cost", 0.0)
    labor_hours = cost_data.get("labor_hours", 0.0)
    hourly_rate = cost_data.get("hourly_rate", 0.0)
    downtime_rate = cost_data.get("downtime_cost_per_hour", 0.0)

    maintenance_cost = spare_cost + labor_hours * hourly_rate
    downtime_cost = downtime_rate * max(predicted_rul, 0.0)
    total_cost = maintenance_cost + downtime_cost

    if predicted_rul <= 50:
        recommendation = "建议立即安排维护，以避免失效。"
        urgency = "High"
    elif predicted_rul >= 200:
        recommendation = "建议按照计划周期进行维护。"
        urgency = "Low"
    else:
        recommendation = "建议在未来 100-120 小时内安排维护。"
        urgency = "Medium"

    return {
        "recommendation": recommendation,
        "reasoning": "该计划综合考虑了RUL预测和维护成本，在失效前进行维护。",
        "estimated_total_cost": float(total_cost),
        "urgency_level": urgency,
    }


if __name__ == "__main__":
    sample_rul = {"predicted_rul": 120.0, "confidence_interval": [100.0, 140.0]}
    sample_costs = {
        "spare_part_cost": 500,
        "labor_hours": 4,
        "hourly_rate": 80,
        "downtime_cost_per_hour": 900,
    }
    plan = generate_maintenance_plan(sample_rul, sample_costs)
    print("Sample maintenance plan:", plan)
