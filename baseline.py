import torch.nn as nn
from torchvision.models import resnet

import query_embedder


class BaselineModel(nn.Module):
    def __init__(self, embedder):
        super().__init__()
        self.qe = embedder

        resnet_model = resnet.resnet50(weights=resnet.ResNet50_Weights.IMAGENET1K_V2, progress=True)
        resnet_modules = list(resnet_model.children())[:-1]
        self.cnn = nn.Sequential(*resnet_modules, nn.Flatten())

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
        #self.softmax = nn.Softmax()

        # Freeze the weights.
        for p in self.cnn.parameters():
            p.requires_grad = False
        # Replace the FC layer by identity.
        self.cnn.fc = None

    def forward(self, inputs, padded_captions):
        # Run the cnn,
        visual_features = self.cnn(inputs)

        # Take the last CNN pool, flatten it, and set it as the LSTM context/hidden value.
        # Run the LSTM until we get the <END> token or we reached maximum length.
        h0, c0 = visual_features, visual_features
        output, (h_last, c_last) = self.lstm(padded_captions, (h0, c0))

        #for x in output:
        #    x = self.softmax(x)


        return output
