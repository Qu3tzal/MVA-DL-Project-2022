from baseline import BaselineModel
from scn import SemanticCompositionNetworkModel
from adaptative_attention import AdaptativeAttentionModel


AVAILABLE_MODELS = ['baseline', 'SCN', 'AA']


def get_class(classname):
    if classname == "baseline":
        return BaselineModel
    elif classname == "SCN":
        return SemanticCompositionNetworkModel
    elif classname == "AA":
        return AdaptativeAttentionModel
