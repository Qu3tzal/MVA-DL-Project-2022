import torch.nn as nn
import torch.nn.init
from torchvision.models import resnet

import numpy as np
import query_embedder


class BaselineModel(nn.Module):
    def __init__(self, embedder):
        super().__init__()
        self.qe = embedder
        vocabulary_size = len(self.qe.glove_vocab)

        resnet_model = resnet.resnet50(weights=resnet.ResNet50_Weights.IMAGENET1K_V2, progress=True)
        resnet_modules = list(resnet_model.children())[:-1]
        self.cnn = nn.Sequential(*resnet_modules, nn.Flatten())

        self.visual_feature_embedder = nn.Linear(2048, 100, bias=True)
        self.hidden_2_embedding = nn.Linear(100, 50, bias=True)
        self.output_projection = nn.Sequential(
            nn.Linear(50, vocabulary_size, bias=True),
            nn.LogSoftmax()
        )

        self.lstm_cell = nn.LSTMCell(
            input_size=50,
            hidden_size=100,
            bias=True
        )

        # Freeze the weights.
        for p in self.cnn.parameters():
            p.requires_grad = False

    def forward(self, inputs, padded_captions):
        # Embed the captions.
        embedded_padded_captions = self.qe.embed(padded_captions)

        # Run the cnn.
        visual_features = self.cnn(inputs)
        embedded_visual_features = self.visual_feature_embedder(visual_features)

        # Take the last CNN pool, flatten it, project it, and set it as the initial context.
        h = torch.zeros(inputs.shape[0], 100).to(inputs.device)
        c = embedded_visual_features

        # Run the LSTM with teacher forcing.
        concatenated_outputs = []
        for i in range(embedded_padded_captions.shape[1]):
            # Learn with teacher forcing.
            h, c = self.lstm_cell(embedded_padded_captions[:, i, :], (h, c))
            concatenated_outputs.append(h)

        concatenated_outputs = torch.stack(concatenated_outputs)
        # Brings hidden states into word embedding space.
        concatenated_outputs = self.hidden_2_embedding(concatenated_outputs)
        # Brings word embeddings into vocabulary space.
        outputs = self.output_projection(concatenated_outputs).permute(1, 0, 2)
        return outputs

    def predict(self, inputs):
        # Run the cnn.
        visual_features = self.cnn(inputs)
        embedded_visual_features = self.visual_feature_embedder(visual_features)

        # Take the last CNN pool, flatten it, project it, and set it as the initial context.
        h = torch.zeros((inputs.shape[0], 100)).to(inputs.device)
        c = embedded_visual_features

        # Run the LSTM until we get the <END> token, or we reached maximum length.
        lstm_outputs = []
        output = self.qe.start_token.unsqueeze(0).to(inputs.device)
        for i in range(50):
            h, c = self.lstm_cell(output, (h, c))

            output = self.hidden_2_embedding(h)
            lstm_outputs.append(output)

        word_embedding_space_outputs = torch.stack(lstm_outputs)
        vocabulary_space_outputs = self.output_projection(word_embedding_space_outputs).permute(1, 0, 2)
        return vocabulary_space_outputs
