from pametric.lightning.callbacks.model_checkpoint import PA_ModelCheckpoint
from pametric.lightning.callbacks.metric import PA_Callback
from pametric.lightning.callbacks.batch_size_finder import MultienvBatchSizeFinder
from pametric.lightning.callbacks.measure_dataset import *
from pametric.lightning.callbacks.measure_output import *


class SplitClassifier(nn.Module):
    """
    Splits classifier to retrieve the feature extractor.
    """
    def __init__(self, net: nn.Module, net_name: str):
        super().__init__()
        """
        Add net names according to your requirements.
        """
        if "resnet" in net_name:
            self.feature_extractor = nn.Sequential(
                *list(net.children())[:-1],
                nn.Flatten() 
            )
            self.classifier = net.fc
            
        if "densenet" in net_name or "efficient" in net_name:
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