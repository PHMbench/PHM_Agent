"""Agent definitions for PHM_Agent."""

from .phm_agent import PHMAgent
from .report_agent import ReportAgent
from .deep_research_agent import create_deep_research_agent
from .sp1d_specialist_agent import SP1DSpecialistAgent
from .sp2d_specialist_agent import SP2DSpecialistAgent
from .stats_feature_agent import StatsFeatureAgent
from .physical_feature_agent import PhysicalFeatureAgent
from .retrieval_agent import RetrievalAgent

__all__ = [
    "PHMAgent",
    "ReportAgent",
    "create_deep_research_agent",
    "SP1DSpecialistAgent",
    "SP2DSpecialistAgent",
    "StatsFeatureAgent",
    "PhysicalFeatureAgent",
    "RetrievalAgent",
]


if __name__ == "__main__":
    print("Available agents:", __all__)
