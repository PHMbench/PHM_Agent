"""
PHM_Strategy_Agent: 负责编排预测与规划工作流的高级Agent。
"""
from __future__ import annotations

from smolagents import ToolCallingAgent

from PHM_tools.Prognostics.health_indicators import (
    create_health_index,
    fit_degradation_model,
)
from PHM_tools.Prognostics.rul_predictors import predict_rul_from_model
from PHM_tools.Decision_making.cost_models import get_maintenance_costs
from PHM_tools.Decision_making.maintenance_schedulers import generate_maintenance_plan

SYSTEM_PROMPT = """
你的身份是'PHM战略规划师'，一个负责设备长期健康管理和维护决策的专家。你的工作是严格遵循一个清晰、逻辑严密的工作流来制定计划。**你唯一的任务就是按顺序调用已有的工具，将前一步的输出作为下一步的输入。**

**工作流指令:**
1.  **接收输入**: 你将从ManagerAgent处接收到`diagnostic_report`，其中包含了故障诊断结果和信号特征。
2.  **评估健康状态**:
    - 调用 `create_health_index` 将特征融合成健康指数(HI)。
    - 调用 `fit_degradation_model` 对HI序列进行建模，得到`degradation_model`。
3.  **预测未来趋势**:
    - 调用 `predict_rul_from_model`，并传入`degradation_model`和一个预设的`failure_threshold`，得到`rul_prediction`。
4.  **制定维护计划**:
    - 调用 `get_maintenance_costs` 获取相关成本信息。
    - 调用 `generate_maintenance_plan`，综合`rul_prediction`和成本信息，生成最终的`maintenance_plan`。
5.  **输出结果**: 将最终的`maintenance_plan`作为你的最终答案返回。
"""


def create_phm_strategy_agent(model) -> ToolCallingAgent:
    """创建并配置PHM战略规划师Agent。

    Args:
        model: 用于驱动Agent的大语言模型实例。

    Returns:
        一个配置好工具和指令的ToolCallingAgent实例。
    """
    prognostics_and_planning_tools = [
        create_health_index,
        fit_degradation_model,
        predict_rul_from_model,
        get_maintenance_costs,
        generate_maintenance_plan,
    ]

    strategy_agent = ToolCallingAgent(
        model=model,
        tools=prognostics_and_planning_tools,
        name ="PHM_Strategy_Agent",
        description="负责制定PHM维护计划的Agent",
        return_full_result=True,
        instructions=SYSTEM_PROMPT,
    )
    return strategy_agent


if __name__ == "__main__":
    class DummyModel:
        def __call__(self, messages):
            return type("Obj", (), {"content": "ok"})()

    agent = create_phm_strategy_agent(DummyModel())
    print("Strategy agent created with", len(agent.tools), "tools")
