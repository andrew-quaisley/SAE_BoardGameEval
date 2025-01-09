import torch
from circuits.dictionary_learning.dictionary import AutoEncoder
from .architectures import SAEAnthropic

device='cuda' if torch.cuda.is_available() else 'cpu'

class HubenSAE(AutoEncoder):
    def __init__(self, sae_path):
        super().__init__(activation_dim=512, dict_size=1024)
        with open(sae_path, 'rb') as f:
            state_dict = torch.load(f)
        self.sae = SAEAnthropic(gpt=None, num_features=1024, sparsity_coefficient=0)
        self.sae.load_state_dict(state_dict, strict=False)
    
    def encode(self, x):
        loss, residual_stream, hidden_layer, reconstructed_residual_stream = self.sae(x)
        return hidden_layer
    
    def decode(self, f):
        print("Decoding not implemented for Huben saes")
        raise NotImplementedError
    
    def forward(self, x, output_features=False, ghost_mask=None):
        loss, residual_stream, hidden_layer, reconstructed_residual_stream = self.sae(x)
        if output_features:
            return reconstructed_residual_stream, hidden_layer
        else:
            return reconstructed_residual_stream
