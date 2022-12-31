from baseline import BaselineModel
from scn import SemanticCompositionNetworkModel
from adaptative_attention import AdaptativeAttentionModel


AVAILABLE_MODELS = ['Baseline', 'SCN', 'AA']


def get_class(classname):
    if classname == "Baseline":
        return BaselineModel
    elif classname == "SCN":
        return SemanticCompositionNetworkModel
    elif classname == "AA":
        return AdaptativeAttentionModel
