import torch
import torch.nn as nn
from torchvision import models

class Network(nn.Module):

    def __init__(self, opts, n_out_features):
        super(Network, self).__init__()

        original_model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)

        self.features = nn.Sequential(
            *list(original_model.features.children())[:-1]
        )
        
        self.gap = nn.AvgPool2d(7)

        self.dense_layer = nn.Linear(1024, n_out_features, bias=True)


    def forward(self, in_img):
        
        features = self.features(in_img)
        
        features = self.gap(features).squeeze()
        
        outputs = self.dense_layer(features)

        return outputs
