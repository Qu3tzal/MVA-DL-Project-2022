import PIL.Image
import torch
from torchvision.transforms import Compose, PILToTensor, ToTensor, Resize
import numpy as np

import models
import metrics
from query_embedder import tokenizer_fn


def load_pytorch_model(model_name, filepath, query_embedder):
    model = models.get_class(model_name)(query_embedder)
    model.load_state_dict(torch.load(filepath))
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    return model, device


def load_keras_model(model_name, filepath, _):
    model = models.get_class(model_name)()
    model.load_weights(filepath)

    return model, None


def compute_query_scores(metric, query, captions):
    metric_fn = metrics.get_fn(metric)
    scores = np.zeros((len(captions),))

    for idx, caption in enumerate(captions):
        scores[idx] = metric_fn(
            tokenizer_fn(query),
            tokenizer_fn(caption)
        )

    return scores


def decode_softmax_pytorch(softmax_tensor, vocabulary):
    vocab_indices = torch.argmax(softmax_tensor, dim=1).squeeze().to("cpu").numpy()

    words = []
    for i in range(vocab_indices.shape[0]):
        word = vocabulary[vocab_indices[i]]
        words.append(word)

    caption = " ".join(words)
    return caption


def decode_softmax_numpy(softmax_tensor, vocabulary):
    vocab_indices = np.argmax(softmax_tensor, axis=1)

    words = []
    for i in range(vocab_indices.shape[0]):
        words.append(vocabulary[vocab_indices[i]])

    caption = "".join(words)
    return caption


def get_captions_pytorch_model(model, device, image_filepaths, vocabulary):
    image_transform = Compose([
        Resize((224, 224)),
        ToTensor()
    ])

    captions = []
    for image_fp in image_filepaths:
        image = PIL.Image.open(image_fp)
        image_tensor = image_transform(image).unsqueeze(0).to(device)

        model_output = model.predict(image_tensor)
        caption = decode_softmax_pytorch(model_output, vocabulary)

        captions.append(caption)

    return captions


def get_captions_keras_model(model, _, image_filepaths, vocabulary):
    image_transform = Compose([
        Resize((224, 224)),
        ToTensor()
    ])

    captions = []
    for image_fp in image_filepaths:
        image = PIL.Image.open(image_fp)
        image_tensor = image_transform(image).numpy()

        model_output = model(image_tensor)
        caption = decode_softmax_numpy(model_output.to_numpy(), vocabulary)

        captions.append(caption)

    return captions
