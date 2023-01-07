import pycocotools as cc


AVAILABLE_METRICS = ["BLEU", "CIDER", "METEOR"]


def get_fn(fn_name):
    if fn_name == "BLEU":
        return None
    elif fn_name == "CIDER":
        return None
    elif fn_name == "METEOR":
        return None
