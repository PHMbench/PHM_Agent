"""Top-level manager delegating analysis to ``PHMAnalysisAgent``."""
from __future__ import annotations

from smolagents.agents import CodeAgent
from smolagents.models import Model

from utils.registry import register_agent, TOOL_REGISTRY, get_agent
# from agent_leader.phm_analysis_agent import create_phm_analysis_agent
# from agent_leader.phm_strategy_agent import create_phm_strategy_agent
from PHM_tools.data_management import DataManager
from PHM_tools.Retrieval.knowledge_base import VectorKnowledgeBaseManager


@register_agent("ManagerAgent")
class ManagerAgent(CodeAgent):
    """Top-level agent orchestrating analysis and planning stages."""

    def __init__(self, model: Model, **kwargs) -> None:

        # analysis_agent = create_phm_analysis_agent(model)

        deep_research_agent = get_agent("DeepResearchAgent")
        retrieval_agent = get_agent("RetrievalAgent")
        # phm_agent = get_agent("PHMAgent")
        Stats_Feature_Agent = get_agent("StatsFeatureAgent")
        sp1d_specialist_agent = get_agent("SP1DSpecialistAgent")
        sp2d_specialist_agent = get_agent("SP2DSpecialistAgent")
        agent_list = [
            deep_research_agent,
            retrieval_agent,
            Stats_Feature_Agent,
            sp1d_specialist_agent,
            sp2d_specialist_agent,
        ]

        instructions = (
            "先调用PHMAnalysisAgent进行诊断，如有异常再调用PHMStrategyAgent制定维护计划。"
        )
        super().__init__(
            tools=[],
            model=model,
            managed_agents=agent_list,
            name="manager_agent",
            description="Delegates PHM analysis and planning",
            instructions=instructions,
            return_full_result=True,
            add_base_tools=True,
            **kwargs,
        )

    # def run_workflow(self, task_input: dict) -> object:
    #     """Run the PHM analysis workflow.

    #     Args:
    #         task_input: Dictionary containing ``metadata_path``, ``signal_path``,
    #             ``reference_ids`` and ``test_id``.

    #     Returns:
    #         Analysis results or maintenance plan depending on detected status.
    #     """
    #     dm = DataManager(task_input["metadata_path"], task_input["signal_path"])
    #     kb = VectorKnowledgeBaseManager()
    #     payload = {
    #         "data_manager": dm,
    #         "kb_manager": kb,
    #         "reference_ids": task_input.get("reference_ids", []),
    #         "test_id": task_input.get("test_id"),
    #     }
    #     diagnostic_report = self.analysis_agent.run(payload)
    #     if isinstance(diagnostic_report, dict):
    #         status = diagnostic_report.get("status")
    #         if status and status != "NORMAL" or "fault_label" in diagnostic_report:
    #             return self.strategy_agent.run({"diagnostic_report": diagnostic_report})
    #     return diagnostic_report


if __name__ == "__main__":
    class DummyModel:
        def __call__(self, messages):
            return type("Obj", (), {"content": "ok"})()

    import numpy as np
    import pandas as pd
    import h5py
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as tmp_csv:
        pd.DataFrame({"id": ["demo"], "label": ["normal"]}).to_csv(tmp_csv.name, index=False)
        csv_path = tmp_csv.name
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_h5:
        with h5py.File(tmp_h5.name, "w") as f:
            f.create_dataset("demo", data=np.random.randn(50))
        h5_path = tmp_h5.name

    manager = ManagerAgent(DummyModel())
    payload = {
        "metadata_path": csv_path,
        "signal_path": h5_path,
        "reference_ids": ["demo"],
        "test_id": "demo",
    }
    result = manager.run_workflow(payload)
    print("Workflow result:", result)
