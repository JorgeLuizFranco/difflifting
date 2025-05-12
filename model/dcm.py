import torch
from baselines.modules.dcm import DCM
from baselines.modules.cell_network import CellNetwork
from baselines.modules.layers import MLP

class DCMModule(torch.nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.pre = MLP(
            hparams["pre_layers"], dropout=hparams["dropout"], final_activation=True
        )
        self.dcm = DCM(hparams)
        self.cell_network = CellNetwork(hparams)
        post_layers = hparams["post_layers"]
        post_layers[0] *= 2
        self.post = MLP(post_layers)
        self.cell_network = CellNetwork(hparams)

    def forward(self, data):
        if self.dcm:
            data.x = self.pre(data.x)
            data = self.dcm(data)
            #   print("Data after DCM: ", data)
            x = self.cell_network(data)
            return self.post(x), data["ne_probs"], data["np_probs"]