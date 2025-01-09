import torch 
import logging
from abc import ABC

logger = logging.getLogger(__name__)
device='cuda' if torch.cuda.is_available() else 'cpu'

CHESS_WINDOW_START_TRIM = 0
CHESS_WINDOW_END_TRIM = 0

class SAETemplate(torch.nn.Module, ABC):
    '''
    abstract base class that defines the SAE contract
    '''
    def __init__(self, gpt, num_features:int, window_start_trim:int=4, window_end_trim:int=8):
        super().__init__()
        self.gpt=gpt
        self.num_features=num_features
        """ for param in self.gpt.parameters():
            #freezes the gpt model  
            param.requires_grad=False  """
        self.num_data_trained_on=0
        self.classifier_aurocs=None
        self.classifier_smds=None
        self.classifier_f1_scores=None
        try:
            self.residual_stream_mean=torch.load(f"huben_saes/model_params/residual_stream_mean.pkl", map_location=device)
            self.average_residual_stream_norm=torch.load(f"huben_saes/model_params/average_residual_stream_norm.pkl", map_location=device)
        except:
            self.residual_stream_mean=torch.zeros((1))
            self.average_residual_stream_norm=torch.ones((1))
            logger.warning(f"Please ensure the correct files are in huben_saes/model_params/residual_stream_mean.pkl and huben_saes/model_params/average_residual_stream_norm.pkl!")

    def create_linear_encoder_decoder(self, decoder_initialization_scale):
        residual_stream_size=512 #self.gpt.output_size
        decoder_initial_value=torch.randn((self.num_features, residual_stream_size))
        decoder_initial_value=decoder_initial_value/decoder_initial_value.norm(dim=1).unsqueeze(-1) # columns of norm 1
        decoder_initial_value*=decoder_initialization_scale # columns of norm decoder_initial_value
        encoder=torch.nn.Parameter(torch.clone(decoder_initial_value).transpose(0,1).detach())
        encoder_bias=torch.nn.Parameter(torch.zeros((self.num_features)))
        decoder=torch.nn.Parameter(decoder_initial_value)
        decoder_bias=torch.nn.Parameter(torch.zeros((residual_stream_size)))
        return encoder, encoder_bias, decoder, decoder_bias