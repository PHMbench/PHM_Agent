"""Decision-making tools for predictive maintenance."""

from .decision_tools import isolation_forest_detector, svm_fault_classifier
from .cost_models import get_maintenance_costs
from .maintenance_schedulers import generate_maintenance_plan

__all__ = [
    "isolation_forest_detector",
    "svm_fault_classifier",
    "get_maintenance_costs",
    "generate_maintenance_plan",
]


if __name__ == "__main__":
    import numpy as np

    data = np.random.randn(20, 3)
    labels = np.random.randint(0, 2, 20)
    iso_out = isolation_forest_detector(data)
    svm_out = svm_fault_classifier(data, labels, data)
    cost = get_maintenance_costs("bearing", {"bearing": {"spare_part_cost": 500}})
    plan = generate_maintenance_plan({"predicted_rul": 100.0}, cost)
    print("IsolationForest sample:", iso_out[:5])
    print("SVM sample:", svm_out[:5])
    print("Cost lookup sample:", cost)
    print("Plan sample:", plan)
