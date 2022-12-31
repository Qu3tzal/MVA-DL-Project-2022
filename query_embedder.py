import string
import numpy as np
import torch
from torch import nn


"""
How to use:
    import query_embedder as qe
    
    embedder = qe.QueryEmbedder()
    encoded = embedder.encode("The wolf is out. STOP. Be careful, send the rangers. STOP.")
    decoded = embedder.decode(encoded)
    
    " ".join(decoded.tolist())
    #> 'the wolf is out stop be careful send the rangers stop'
"""

# https://medium.com/mlearning-ai/load-pre-trained-glove-embeddings-in-torch-nn-embedding-layer-in-under-2-minutes-f5af8f57416a
def load_glove_weights(filepath):
    """ Loads the pre-trained GloVe weights. """
    vocab, embeddings = [], []
    with open(filepath, "rt", encoding="utf8") as file:
        full_content = file.read().strip().split('\n')

    for line in full_content:
        word = line.split(' ')[0]
        vec = [float(val) for val in line.split(' ')[1:]]
        vocab.append(word)
        embeddings.append(vec)

    # Add the pad and unknown tokens.
    vocab.insert(0, '<pad>')
    vocab.insert(2, '<start>')
    vocab.insert(3, '<eos>')
    vocab.insert(1, '<unk>')

    np_embeddings = np.asarray(embeddings)
    pad_emb = np.zeros((1, np_embeddings.shape[1]))
    unk_emb = np.mean(np_embeddings, axis=0, keepdims=True)

    start_emb = np.zeros((1, np_embeddings.shape[1]))
    for i in range(start_emb.shape[0] // 2):
        start_emb = 1

    eos_emb = np.ones((1, np_embeddings.shape[1]))

    np_embeddings = np.vstack((pad_emb, start_emb, eos_emb, unk_emb, np_embeddings))

    return np.array(vocab), np_embeddings


def tokenizer_fn(input_str):
    # Lowercase
    # Remove all punctuation
    # Remove before/after whitespaces
    # Split on whitespaces
    return input_str.lower().translate(str.maketrans('', '', string.punctuation)).strip().split(' ')


class QueryEmbedder:
    """ Computes the embeddings for a query. """
    def __init__(self):
        self.tokenizer = tokenizer_fn
        self.glove_vocab, self.glove_embeddings = load_glove_weights("pretrained/glove.6B.50d.txt")

        self.glove_embeddings = torch.from_numpy(self.glove_embeddings).float()
        # Normalize the embeddings.
        glove_emb_norms = self.glove_embeddings.pow(2).sum(dim=1).sqrt()[:, None]
        glove_emb_norms[0] = 1.0 # Prevent division by 0
        self.glove_embeddings = self.glove_embeddings / glove_emb_norms

        self.embedding_layer = nn.Embedding.from_pretrained(self.glove_embeddings)

        self.pad_token = self.glove_embeddings[0, :]
        self.start_token = self.glove_embeddings[1, :]
        self.eos_token = self.glove_embeddings[2, :]
        self.unk_token = self.glove_embeddings[3, :]

    def encode(self, query_string):
        query_tokens = self.tokenizer(query_string)
        tokens_indices = []
        for token in query_tokens:
            idx = np.where(self.glove_vocab == token)[0]
            if len(idx) > 0:
                tokens_indices.append(idx[0])
            else:
                # 1 is the index for the <unk> word/token.
                tokens_indices.append(1)

        tokens_indices = torch.from_numpy(np.asarray(tokens_indices)).long()
        return self.embedding_layer(tokens_indices)

    def decode(self, inputs):
        # Compute cosine similarity between word vectors and the glove weights.
        # Use the argmax of the similarity to get the index.
        # Use the index to look-up the string.
        # Normalize the input vectors.
        inputs = inputs / inputs.pow(2).sum(dim=1).sqrt()[:, None]
        cosine_sim = torch.matmul(inputs, self.glove_embeddings.T)
        words_indices = cosine_sim.argmax(dim=1)

        return self.glove_vocab[words_indices]
