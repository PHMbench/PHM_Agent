"""Dynamic analysis agent orchestrating PHM workflows."""
from __future__ import annotations

from smolagents.agents import CodeAgent
from smolagents.models import Model
from smolagents.tools import Tool

from utils.registry import register_agent

SYSTEM_PROMPT = """
你的身份是'PHM总工程师'，一个负责领导一个由“微服务化”虚拟专家组成的、包含研究与检索能力的完整团队来解决复杂诊断任务的专家。

## 核心资产与初始上下文
在每次运行时，你将接收到以下核心资产：
1. `data_manager`: 中央数据管理器实例。
2. `kb_manager`: 中央向量知识库管理器实例。
3. `reference_ids`: 参考信号ID列表。
4. `test_id`: 需要分析的目标信号ID。
5. 可用的专家团队: SP1D_SpecialistAgent, SP2D_SpecialistAgent, StatsFeatureAgent, PhysicalFeatureAgent, DeepResearchAgent, RetrievalAgent。

## 核心思考原则：知识驱动的、可溯源的专家委托
你的工作是规划并委托任务给专家团队，并将他们的发现通过 `kb_manager` 进行语义化存储和对齐。
你通过对知识库的语义查询来关联历史经验，并利用 `slicer_tools` 进行特征溯源，最终构建可追溯的证据链。

## 主要工作流程（高度动态的专家协作与特征溯源）
1. **初步分析与异常检测**：委托 StatsFeatureAgent 等专家提取特征并判断是否异常。
2. **深度诊断与特征溯源**：若发现异常，使用 SP1D/SP2D 专家与 slicer_tools 追踪异常来源，并让 PhysicalFeatureAgent 进行物理分析。
3. **知识关联与深度研究**：通过 kb_manager 查询相似案例，如有需要调用 DeepResearchAgent 进一步研究。
4. **最终决策与报告生成**：综合知识库信息，形成诊断结论并生成 Monitoring_Report.md。
"""


@register_agent("PHM_Analysis_Agent")
class PHMAnalysisAgent(CodeAgent):
    """Core agent capable of dynamic planning using PHM tools."""

    def __init__(self, tools: list[Tool], model: Model, **kwargs) -> None:
        instructions = SYSTEM_PROMPT
        super().__init__(
            tools=tools,
            model=model,
            name="phm_analysis_agent",
            description="Dynamic PHM analysis agent",
            instructions=instructions,
            **kwargs,
        )


if __name__ == "__main__":
    class DummyModel:
        def __call__(self, messages):
            return type("Obj", (), {"content": "ok"})()

    agent = PHMAnalysisAgent(tools=[], model=DummyModel())
    print("Created analysis agent:", agent.name)
