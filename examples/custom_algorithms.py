"""Optional extension example: set DI_PLUGINS=examples.custom_algorithms before startup."""
from sklearn.ensemble import ExtraTreesClassifier

from di_validator.adapters import register_classifier


def extra_trees(seed):
    return ExtraTreesClassifier(n_estimators=150, max_depth=8, class_weight="balanced", random_state=seed, n_jobs=1)


register_classifier("extra-trees", "Extra trees (local extension)", extra_trees, parameters={"n_estimators", "max_depth"})

