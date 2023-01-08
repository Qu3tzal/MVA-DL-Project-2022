import pycocotools as cc

from nltk.translate.bleu_score import sentence_bleu

AVAILABLE_METRICS = ["BLEU", "CIDER", "METEOR"]


def get_fn(fn_name):
    if fn_name == "BLEU":
        return compute_bleu
    elif fn_name == "CIDER":
        return compute_cider
    elif fn_name == "METEOR":
        return compute_meteor


def compute_bleu(query: list[str], caption: list[str]):
    """
    Computes the BLEU score between the query and the caption.
    Args:
        query: the tokenized query
        caption: the tokenized caption
    Return:
        the BLEU score
    """
    return sentence_bleu([query], caption)


def compute_cider(query: list[str], caption: list[str]):
    """
    Computes the CIDEr score between the query and the caption.
    Args:
        query: the tokenized query
        caption: the tokenized caption
    Return:
        the CIDEr score
    """
    pass


def compute_meteor(query: list[str], caption: list[str]):
    """
    Computes the METEOR score between the query and the caption.
    Args:
        query: the tokenized query
        caption: the tokenized caption
    Return:
        the METEOR score
    """
    pass
