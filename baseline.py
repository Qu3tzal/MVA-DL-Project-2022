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

        self.visual_feature_embedder = nn.Linear(2048, 50, bias=True)
        self.output_projection = nn.Sequential(
            nn.Linear(100, vocabulary_size, bias=True),
            nn.LogSoftmax()
        )

        self.lstm = nn.LSTM(
            input_size=50,
            hidden_size=2048,
            num_layers=2,
            bias=True,
            batch_first=True,
            dropout=0.5,
            bidirectional=True,
            proj_size=50, # Project into the word embedding space
        )

        # Freeze the weights.
        for p in self.cnn.parameters():
            p.requires_grad = False

        # Initialize custom layers.
        """
        for p in self.visual_feature_embedder.parameters():
            torch.nn.init.xavier_normal_(p)
        for p in self.output_projection.parameters():
            torch.nn.init.xavier_normal_(p)
        """

    def forward(self, inputs, padded_captions):
        # Embed the captions.
        embedded_padded_captions = self.qe.embed(padded_captions)

        # Run the cnn,
        visual_features = self.cnn(inputs)
        embedded_visual_features = self.visual_feature_embedder(visual_features)

        # Take the last CNN pool, flatten it, project it, and set it add it to the first token of each sequence.
        visual_padded_captions = embedded_padded_captions.clone()
        visual_padded_captions[:, 1] = embedded_padded_captions[:, 1] + embedded_visual_features

        # Run the LSTM until we get the <END> token or we reached maximum length.
        concatenated_outputs, (h_last, c_last) = self.lstm(embedded_padded_captions)

        outputs = self.output_projection(concatenated_outputs)
        return outputs
