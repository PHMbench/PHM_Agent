"""PHM toolkit package providing registered tools."""

from .Feature_extracting import (
    extract_time_features,
    extract_frequency_features,
)
from .Signal_processing import (
    normalize,
    detrend,
    bandpass,
    fft,
    cepstrum,
    envelope_spectrum,
    spectrogram,
    mel_spectrogram,
    scalogram,
    gramian_angular_field,
    markov_transition_field,
    recurrence_plot,
    cepstrogram,
    envelope_spectrogram,
)
from .Prognostics import (
    create_health_index,
    fit_degradation_model,
    predict_rul_from_model,
)
from .Decision_making import (
    isolation_forest_detector,
    svm_fault_classifier,
    get_maintenance_costs,
    generate_maintenance_plan,
)
from .Anomaly_Detection import calculate_statistical_divergence
from .Fault_Diagnosis import classify_fault_from_features
from .text_web_browser import (
    SimpleTextBrowser,
    VisitTool,
    PageUpTool,
    PageDownTool,
    FinderTool,
    FindNextTool,
    ArchiveSearchTool,
)
from .text_inspector_tool import TextInspectorTool
from .model_download import model_download_tool
from .Retrieval import (
    RetrieverTool,
    build_vector_store,
    create_retriever_tool,
    build_local_vector_store,
    create_local_retriever_tool,
)
from .data_management import DataManager
from .Retrieval.knowledge_base import VectorKnowledgeBaseManager
from .Signal_processing.slicer_tools import apply_transform, extract_slice

__all__ = [
    "extract_time_features",
    "extract_frequency_features",
    "normalize",
    "detrend",
    "bandpass",
    "fft",
    "cepstrum",
    "envelope_spectrum",
    "spectrogram",
    "mel_spectrogram",
    "scalogram",
    "gramian_angular_field",
    "markov_transition_field",
    "recurrence_plot",
    "cepstrogram",
    "envelope_spectrogram",
    "create_health_index",
    "fit_degradation_model",
    "predict_rul_from_model",
    "calculate_statistical_divergence",
    "classify_fault_from_features",
    "isolation_forest_detector",
    "svm_fault_classifier",
    "get_maintenance_costs",
    "generate_maintenance_plan",
    "RetrieverTool",
    "build_vector_store",
    "create_retriever_tool",
    "build_local_vector_store",
    "create_local_retriever_tool",
    "SimpleTextBrowser",
    "VisitTool",
    "PageUpTool",
    "PageDownTool",
    "FinderTool",
    "FindNextTool",
    "ArchiveSearchTool",
    "TextInspectorTool",
    "DataManager",
    "VectorKnowledgeBaseManager",
    "apply_transform",
    "extract_slice",
]


if __name__ == "__main__":
    import numpy as np

    sig = np.random.randn(500)
    norm_sig = normalize(sig)
    features = extract_time_features(norm_sig)
    print("Normalized signal shape:", norm_sig.shape)
    print("Extracted feature keys:", list(features)[:3])
