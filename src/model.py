"""
Defines a model.
"""

import torch
import torch.nn as nn
from transformers import DistilBertForSequenceClassification
from utils import set_seed, RATINGS, get_device
from tqdm import tqdm
import numpy as np
from datetime import datetime

MODEL_CLASSES = {
  'distilbert-base-cased': DistilBertForSequenceClassification,
}

class Model(nn.Module):
    
    def __init__(self, model_type='distilbert-base-cased', checkpoint_path=None):
        super(Model, self).__init__()
        # See https://huggingface.co/transformers/pretrained_models.html
        # for description of model_type
        self.model_type = model_type
        self.MODEL_ID = datetime.today().strftime('%Y-%m-%d-%H:%M')
        set_seed(24)    # This is required bc from_pretrained initializes the classification head with random weights
        self.lm = MODEL_CLASSES[self.model_type].from_pretrained(
            self.model_type,
            num_labels=len(RATINGS),     # 5 different possible ratings
        )
        if checkpoint_path:
          self.lm.load_state_dict(torch.load(checkpoint_path, map_location=get_device()))


    def forward(self, batch_input, **model_args):
        """
        Executes a forward pass of the network.

        Params:
        - Tensor batch_input: input data for a batch
        - dict model_args: any additional arguments needed for the model (e.g. attention mask)
        """
        # Unpack model args
        attention_mask = model_args.get("attention_mask", None)
        labels = model_args.get("labels", None)

        # Run a forward pass of the model
        outputs = self.lm(batch_input, attention_mask=attention_mask, labels=labels, return_dict=True)
        return outputs

    # pytorch actually takes care of everything below so it's not necessary
    # but the already generated checkpoints depend on the code below
    
    def state_dict(self):
        """
        Returns a dictionary storing model parameters.
        Used for loading/saving model checkpoints.
        """
        return self.lm.state_dict()

    # Note: These state dict operations are necessary
    def load_state_dict(self, state_dict):
        """
        Loads a state dict (parameters) into the model.
        """
        self.lm.load_state_dict(state_dict)

    
    def parameters(self):
        """
        Returns an iterator over the model parameters.
        Usually passed into an optimizer.
        """
        return self.lm.parameters()


    def eval(self):
        """
        Sets the model to test mode.
        """
        self.lm.eval()


    def train(self, is_train=True):
        """
        Sets the model to train mode.
        """
        self.lm.train(is_train)

        