import torch 
from torch.nn import functional as F
import logging
import numpy as np
from .sae_template import SAETemplate


logger = logging.getLogger(__name__)
device='cuda' if torch.cuda.is_available() else 'cpu'

class SAEAnthropic(SAETemplate):

    def __init__(self, gpt, num_features:int, sparsity_coefficient:float, decoder_initialization_scale=0.1):
        super().__init__(gpt=gpt, num_features=num_features)
        self.sparsity_coefficient=sparsity_coefficient
        self.encoder, self.encoder_bias, self.decoder, self.decoder_bias=self.create_linear_encoder_decoder(decoder_initialization_scale)

    def forward(self, residual_stream):
        hidden_layer=self.activation_function(residual_stream @ self.encoder + self.encoder_bias)
        reconstructed_residual_stream=hidden_layer @ self.decoder + self.decoder_bias
        loss = None
        return loss, residual_stream, hidden_layer, reconstructed_residual_stream

    def activation_function(self, encoder_output):
        return F.relu(encoder_output)
    
    def report_model_specific_features(self):
        return [f"Sparsity loss coefficient: {self.sparsity_coefficient}"]

class SAEDummy(SAETemplate):
    '''
    "SAE" whose hidden layer and reconstruction is just the unchanged residual stream
    '''

    def __init__(self, gpt, num_features=1024):
        super().__init__(gpt=gpt, num_features=num_features)
        self.to(device)

    def forward(self, residual_stream, compute_loss=False):
        return None, residual_stream,residual_stream,residual_stream

class MultiHeadedTopKSAE(SAETemplate):

    def __init__(self, gpt, num_features:int, sparsity:int, num_heads:int, decoder_initialization_scale=0.1):
        super().__init__(gpt=gpt, num_features=num_features)
        self.sparsity=sparsity
        self.num_heads=num_heads
        self.encoder, self.encoder_bias, self.decoder, self.decoder_bias=self.create_linear_encoder_decoder(decoder_initialization_scale)

    def activation_function(self, encoder_output):
        activations = F.relu(encoder_output)
        attention_by_head=activations.reshape((activations.shape[0],activations.shape[1], self.num_heads, self.num_features//self.num_heads))
        kth_value = torch.topk(attention_by_head, k=self.sparsity//self.num_heads).values.min(dim=-1).values
        masked_activations=suppress_lower_activations(attention_by_head, kth_value, epsilon=0, mode='relative')
        return masked_activations.reshape(activations.shape)
    
    def forward(self, residual_stream, compute_loss=False):
        '''
        takes the trimmed residual stream of a language model (as produced by run_gpt_and_trim) and runs the SAE
        must return a tuple (loss, residual_stream, hidden_layer, reconstructed_residual_stream)
        residual_stream is shape (B, W, D), where B is batch size, W is (trimmed) window length, and D is the dimension of the model:
            - residual_stream is unchanged, of size (B, W, D)
            - hidden_layer is of shape (B, W, D') where D' is the size of the hidden layer
            - reconstructed_residual_stream is shape (B, W, D) 
        '''
        hidden_layer=self.activation_function(residual_stream @ self.encoder + self.encoder_bias)
        reconstructed_residual_stream=hidden_layer @ self.decoder + self.decoder_bias
        loss=None
        return loss, residual_stream, hidden_layer, reconstructed_residual_stream

    def report_model_specific_features(self):
        return [f"Number of heads: {self.num_heads}", f"Sparsity (total): {self.sparsity}"]

#supported variants: mag_in_aux_loss, relu_only
#setting no_aux_loss=True implements a gated sae in a different way from the paper that makes more sense to me
#currently uses tied weights only
#to try: untied weights original version, as well as using sigmoid instead of step function for training to avoid aux_loss
class Gated_SAE(SAEAnthropic):
    def __init__(self, gpt, num_features: int, sparsity_coefficient: float, no_aux_loss=False, decoder_initialization_scale=0.1):
        super().__init__(gpt, num_features, sparsity_coefficient, decoder_initialization_scale)
        self.b_gate = self.encoder_bias #just renaming to make this more clear
        self.r_mag = torch.nn.Parameter(torch.zeros((num_features)))
        self.b_mag = torch.nn.Parameter(torch.zeros((num_features)))
        self.no_aux_loss = no_aux_loss

    def forward(self, residual_stream, compute_loss=False):
        if self.no_aux_loss:
            encoder = F.normalize(self.encoder, p=2, dim=1)
        else:
            encoder = self.encoder
        encoding = (residual_stream - self.decoder_bias) @ encoder
        if self.no_aux_loss:
                features_to_use = F.relu(encoding + self.b_gate)
                hidden_layer = F.relu(features_to_use * torch.exp(self.r_mag) + self.b_mag) #is b_mag really necessary here?
        else:
            hidden_layer_before_gating = F.relu(encoding * torch.exp(self.r_mag) + self.b_mag)
            hidden_layer = ((encoding + self.b_gate) > 0) * hidden_layer_before_gating
        normalized_decoder = F.normalize(self.decoder, p=2, dim=1)
        reconstructed_residual_stream = hidden_layer @ normalized_decoder + self.decoder_bias

        loss = None
        return loss, residual_stream, hidden_layer, reconstructed_residual_stream

class ActivationQueue:
    def __init__(self, length):
        self.list = []
        self.length = length

    def add(self, activations):
        self.list.insert(0, activations)
        while len(self.list) > self.length:
            self.list.pop()
    
    def sparsity_coefficient_factor(self, last_p, next_p):
        list_as_tensor = torch.stack(self.list).to(device)
        return torch.sum(list_as_tensor**last_p) / torch.sum(list_as_tensor**next_p)

class P_Annealing_SAE(SAEAnthropic):
    def __init__(self, gpt, num_features: int, sparsity_coefficient: float, anneal_proportion: float, p_end=0.2, queue_length=10, decoder_initialization_scale=0.1):
        super().__init__(gpt, num_features, sparsity_coefficient, decoder_initialization_scale)
        self.p = 1
        self.anneal_proportion = anneal_proportion
        self.p_end = p_end
        self.queue = ActivationQueue(queue_length)
    
    def training_prep(self, train_dataset=None, eval_dataset=None, batch_size=None, num_epochs=None):
        num_steps = len(train_dataset) * num_epochs / batch_size
        self.anneal_start = round(num_steps*(1-self.anneal_proportion))
        self.p_step = (1 - self.p_end)/(num_steps - self.anneal_start)
        return
    
    def after_step_update(self, hidden_layer=None, step = None):
        if self.anneal_start - step <= self.queue.length:
            self.queue.add(hidden_layer.detach())
        if step >= self.anneal_start:
            next_p = self.p - self.p_step
            self.sparsity_coefficient *= self.queue.sparsity_coefficient_factor(self.p, next_p)
            self.p = next_p
        return
    
class Gated_P_Annealing_SAE(P_Annealing_SAE, Gated_SAE):
    def __init__(self, gpt, num_features: int, sparsity_coefficient: float, anneal_proportion: float, p_end=0.2, queue_length=10, no_aux_loss=False, decoder_initialization_scale=0.1):
        P_Annealing_SAE.__init__(self, gpt, num_features, sparsity_coefficient, anneal_proportion, p_end, queue_length, decoder_initialization_scale)
        Gated_SAE.__init__(self, gpt, num_features, sparsity_coefficient, no_aux_loss=no_aux_loss)

    def forward(self, residual_stream, compute_loss=False):
        return Gated_SAE.forward(self, residual_stream, compute_loss)

#suppression_mode can be "relative" or "absolute"
class Leaky_Topk_SAE(SAETemplate):
    def __init__(self, gpt, num_features: int, epsilon: float, k:int, suppression_mode="relative", decoder_initialization_scale=0.1):
        super().__init__(gpt=gpt, num_features=num_features)
        self.epsilon = epsilon
        self.k=k
        self.suppression_mode = suppression_mode
        self.encoder, self.encoder_bias, self.decoder, self.decoder_bias=self.create_linear_encoder_decoder(decoder_initialization_scale)

    def activation_function(self, encoder_output):
        activations = F.relu(encoder_output)
        kth_value = torch.topk(activations, k=self.k).values.min(dim=-1).values
        return suppress_lower_activations(activations, kth_value, epsilon=self.epsilon, mode=self.suppression_mode)
    
    def forward(self, residual_stream, compute_loss=False):
        '''
        takes the trimmed residual stream of a language model (as produced by run_gpt_and_trim) and runs the SAE
        must return a tuple (loss, residual_stream, hidden_layer, reconstructed_residual_stream)
        residual_stream is shape (B, W, D), where B is batch size, W is (trimmed) window length, and D is the dimension of the model:
            - residual_stream is unchanged, of size (B, W, D)
            - hidden_layer is of shape (B, W, D') where D' is the size of the hidden layer
            - reconstructed_residual_stream is shape (B, W, D) 
        '''
        normalized_encoder = F.normalize(self.encoder, p=2, dim=1) #normalize columns
        normalized_decoder = F.normalize(self.decoder, p=2, dim=1) #normalize columns
        hidden_layer=self.activation_function((residual_stream - self.decoder_bias) @ normalized_encoder + self.encoder_bias)
        reconstructed_residual_stream=hidden_layer @ normalized_decoder + self.decoder_bias
        loss= None
        return loss, residual_stream, hidden_layer, reconstructed_residual_stream

    def report_model_specific_features(self):
        return [f"k (sparsity): {self.k}", f"Epsilon (leakyness): {self.epsilon}"]

    def post_copying_update(self, original_sae, new_feature_indices):
        '''
        if there are fewer features than k, make k equal the number of features
        '''
        if len(new_feature_indices)<self.k:
            self.k=len(new_feature_indices)

def suppress_lower_activations(t, bound, epsilon, inclusive=True, mode="absolute"):
    if torch.is_tensor(bound) and bound.numel() != 1:
        while bound.dim() < t.dim():
            bound = torch.unsqueeze(bound, -1)
    above_mask = (torch.abs(t) >= bound) if inclusive else (torch.abs(t) > bound)
    above_only = t * above_mask
    below_only = t * (~above_mask)
    if mode == "absolute":
        bad_bound_mask = bound <= 0 #to make sure we don't divide by 0
        return above_only + (~bad_bound_mask)*epsilon/(bound+bad_bound_mask) * below_only
    elif mode == "relative":
        return above_only + epsilon * below_only

def smoothed_piecewise(input, functions, transitions):
    assert len(functions) == len(transitions) + 1, "Incorrect number of transitions for number of functions given."
    for i in range(len(transitions)-1):
        assert transitions[i]["x"] < transitions[i+1]["x"], "Transition list not sorted by x-value in ascending order."
    sig = torch.nn.Sigmoid()
    sum = functions[0](input) #first add in the initial function
    for i, t in enumerate(transitions): #then at each transition we will subtract off the previous function and add on the next function
        g = functions[i]
        h = functions[i+1]
        if "focus" in t:
            if t["focus"] == "right":
                t["x"] = t["x"] - t["delta"]
                n = torch.log(abs(g(t["x"]+t["delta"])-h(t["x"]+t["delta"]))/t["epsilon"] - 1)/t["delta"]
            else:
                assert t["focus"] == "left", "Unrecognized focus for a transition (must be either right or left)."
                t["x"] = t["x"] + t["delta"]
                n = torch.log(abs(g(t["x"]-t["delta"])-h(t["x"]-t["delta"]))/t["epsilon"] - 1)/t["delta"]
        else:
            left_and_right = torch.stack((abs(g(t["x"]+t["delta"])-h(t["x"]+t["delta"])), abs(g(t["x"]-t["delta"])-h(t["x"]-t["delta"]))))
            n = torch.log(torch.max(left_and_right, dim=0).values/t["epsilon"] - 1)/t["delta"]
        sum += sig(n*(input-t["x"])) * h(input) - sig(n*(input-t["x"])) * g(input)
    return sum
