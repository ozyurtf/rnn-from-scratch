import pandas as pd 
from torch import nn
import torch
import numpy as np

np.random.seed(0)
torch.manual_seed(0)

sequence_length = 30
batch_size = 50 
num_layers = 3
input_size = 50
hidden_size = 25

input = torch.randn(sequence_length, batch_size, input_size) # seq_length x batch_size x input_size
h0 = torch.randn(num_layers, batch_size, hidden_size)        # num_layers x batch_size x hidden_size

rnn = nn.RNN(input_size = input_size, hidden_size = hidden_size, num_layers = num_layers, nonlinearity='relu')
rnn_output, hn = rnn(input, h0)

W_xh_l0 = rnn.state_dict()['weight_ih_l0'] # hidden_size x input_size 
W_hh_l0 = rnn.state_dict()['weight_hh_l0'] # hidden_size x hidden_size
b_xh_l0 = rnn.state_dict()['bias_ih_l0']   # hidden_size
b_hh_l0 = rnn.state_dict()['bias_hh_l0']   # hidden_size

W_xh_l1 = rnn.state_dict()['weight_ih_l1'] # hidden_size x hidden_size 
W_hh_l1 = rnn.state_dict()['weight_hh_l1'] # hidden_size x hidden_size 
b_xh_l1 = rnn.state_dict()['bias_ih_l1']   # hidden_size
b_hh_l1 = rnn.state_dict()['bias_hh_l1']   # hidden_size

W_xh_l2 = rnn.state_dict()['weight_ih_l2'] # hidden_size x hidden_size 
W_hh_l2 = rnn.state_dict()['weight_hh_l2'] # hidden_size x hidden_size 
b_xh_l2 = rnn.state_dict()['bias_ih_l2']   # hidden_size
b_hh_l2 = rnn.state_dict()['bias_hh_l2']   # hidden_size

W_xh = W_xh_l0                                             
b_xh = b_xh_l0                                             

W_hh_vertical = torch.cat((W_xh_l1.unsqueeze(0), 
                           W_xh_l2.unsqueeze(0)), dim=0)   

b_hh_vertical = torch.cat((b_xh_l1.unsqueeze(0), 
                           b_xh_l2.unsqueeze(0)), dim=0)   

W_hh_horizontal = torch.cat((W_hh_l0.unsqueeze(0), 
                             W_hh_l1.unsqueeze(0), 
                             W_hh_l2.unsqueeze(0)), dim=0) 

b_hh_horizontal = torch.cat((b_hh_l0.unsqueeze(0), 
                             b_hh_l1.unsqueeze(0), 
                             b_hh_l2.unsqueeze(0)), dim=0) 

def relu(x): 
    return torch.max(x, torch.tensor(0.0))

class CustomRNN: 
    def __init__(self, input_size, hidden_size, num_layers):
        self.W_xh = W_xh                        # hidden_size x input_size
        self.W_hh_vertical = W_hh_vertical      # num_layers - 1 x hidden_size x hidden_size
        self.W_hh_horizontal = W_hh_horizontal  # num_layers x hidden_size x hidden_size
        
        self.b_xh = b_xh                        # hidden_size
        self.b_hh_vertical =  b_hh_vertical     # num_layers - 1 x hidden_size
        self.b_hh_horizontal = b_hh_horizontal  # num_layers x hidden_size
        
    def forward(self, input, h0): 
        sequence_length = input.shape[0]
        h = h0
        results = []
        
        for i in range(sequence_length): 
            h1 = torch.zeros_like(h) # Why here ? 
            
            layer = 0
            xi = input[i]          
            h_tmp = (xi @ self.W_xh.T + self.b_xh) + (h[layer] @ self.W_hh_horizontal[layer].T + self.b_hh_horizontal[layer])
            h_tmp = relu(h_tmp)
            h1[layer] = h_tmp
            
            for layer in range(1, num_layers):
                h_tmp = (h_tmp @ self.W_hh_vertical[layer-1].T + self.b_hh_vertical[layer-1]) + (h[layer] @ self.W_hh_horizontal[layer].T + self.b_hh_horizontal[layer])
                h_tmp = relu(h_tmp)
                h1[layer] = h_tmp
            
            yi = h1[-1]
            results.append(yi)
            h = h1    
        return torch.stack(results), h
    
customRNN = CustomRNN(input_size, hidden_size, num_layers)    

custom_output, custom_h = customRNN.forward(input, h0)


# Note: The mechanism that allows parallelization should be broken completely in such a way that 
# there should be no way to parallelize the results. Then the model should be compared with regular RNN 
# with parallelized version of the same input.  
        

