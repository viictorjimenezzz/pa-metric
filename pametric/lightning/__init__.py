from pametric.lightning.module import PosteriorAgreementModule

import torch
import torch.nn as nn

class SplitClassifier(nn.Module):
    """
    Splits classifier to retrieve the feature extractor.
    """
    def __init__(self, net: nn.Module, net_name: str):
        super().__init__()
        """
        Add net names according to your requirements.
        """

        if "wideresnet" in net_name.lower():
            self.feature_extractor = nn.Sequential(
                *list(net.children())[:-1], 
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten()
            )

            self.classifier = net.fc

        elif "resnet" in net_name.lower():
            self.feature_extractor = nn.Sequential(
                *list(net.children())[:-1],
                nn.Flatten() 
            )
            self.classifier = net.fc
            
        elif "densenet" in net_name.lower() or "efficient" in net_name.lower():
            self.feature_extractor = nn.Sequential(
                *list(net.children())[:-1],
                nn.Flatten() 
            )
            self.classifier = net.classifier

    def forward(self, x: torch.Tensor, extract_features: bool = False) -> torch.Tensor:
        x = x.to(next(self.parameters()).device)
        x = self.feature_extractor(x)
        if extract_features == False:
            x = self.classifier(x)
        return x