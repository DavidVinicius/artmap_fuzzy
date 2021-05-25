import pytest

from sklearn.utils.estimator_checks import check_estimator

from artmap_neural_network import TemplateEstimator
from artmap_neural_network import TemplateClassifier
from artmap_neural_network import TemplateTransformer


@pytest.mark.parametrize(
    "Estimator", [TemplateEstimator, TemplateTransformer, TemplateClassifier]
)
def test_all_estimators(Estimator):
    return check_estimator(Estimator)
