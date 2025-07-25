from __future__ import annotations

"""用于分析与维护相关的成本的工具集。"""

from typing import Dict

from smolagents import tool
from utils.registry import register_tool


@register_tool("get_maintenance_costs")
@tool
def get_maintenance_costs(component_id: str, knowledge_base: object) -> Dict[str, float]:
    """从知识库中检索特定组件的维护相关成本。

    Args:
        component_id: 需要查询的组件ID或名称。
        knowledge_base: 一个可以查询结构化数据的知识库接口，需支持 ``get_costs`` 方法或以字典形式提供数据。

    Returns:
        包含各项成本的字典，例如 ``{"spare_part_cost": 500, "labor_hours": 4}``。
    """
    if hasattr(knowledge_base, "get_costs"):
        return knowledge_base.get_costs(component_id)
    if isinstance(knowledge_base, dict):
        costs = knowledge_base.get(component_id)
        if costs is None:
            raise KeyError(f"No cost data for component '{component_id}'")
        return costs
    raise TypeError("knowledge_base must provide 'get_costs' or behave like a mapping")


if __name__ == "__main__":
    kb = {"bearing": {"spare_part_cost": 500, "labor_hours": 3, "hourly_rate": 80, "downtime_cost_per_hour": 1000}}
    result = get_maintenance_costs("bearing", kb)
    print("Sample cost data:", result)
