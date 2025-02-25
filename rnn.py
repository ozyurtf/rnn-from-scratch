import torch

sequence_length = 3
batch_size = 32 
num_layers = 3
input_size = 50
hidden_size = 4

input = torch.randn(sequence_length, batch_size, input_size) # seq_length x batch_size x input_size
h0 = torch.randn(num_layers, batch_size, hidden_size)        # num_layers x batch_size x hidden_size

def relu(x): 
    return torch.max(x, torch.tensor(0.0))

class CustomRNN: 
    def __init__(self, input_size, hidden_size, num_layers):
        self.W_xh = torch.randn(size = (hidden_size, input_size,)) # hidden_size x input_size
        self.W_hh_vertical = torch.randn(size = (num_layers - 1, hidden_size, hidden_size,)) # num_layers - 1 x hidden_size x hidden_size
        self.W_hh_horizontal = torch.randn(size = (num_layers, hidden_size, hidden_size,))  # num_layers x hidden_size x hidden_size
        
        self.b_xh = torch.randn(size = (hidden_size,)) # hidden_size
        self.b_hh_vertical =  torch.randn(size = (num_layers - 1, hidden_size, ))  # num_layers - 1 x hidden_size
        self.b_hh_horizontal = torch.randn(size = (num_layers, hidden_size,))    # num_layers x hidden_size
        
    def forward(self, input, h): 
        sequence_length = input.shape[0]
        results = []
        
        for i in range(sequence_length): 
            h1 = torch.zeros_like(h) 
            
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
    
    def backward(self, output): 
        return None
        
customRNN = CustomRNN(input_size, hidden_size, num_layers)    
custom_output, custom_h = customRNN.forward(input, h0)
        

