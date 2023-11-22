import numpy as np
input_size=3
hidden_size=4
sequence_length=5 #tell how many time we have to set back

#random select weight for input layer to hidden layer
wxh=np.random.randn(hidden_size, input_size)
#random select weight for hiden layer to hidden
whh = np.random.randn(hidden_size, hidden_size)
#bias for hidden layer
bh=np.zeros((hidden_size,1))


#next hidden state or we can say previous hidden state because all previous data is going to be 
h_prev=np.zeros((hidden_size,1))



#forward pass
def rnn_forward(x, h_prev, Wxh, Whh, bh):
    # Calculate hidden state
    h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h_prev) + bh)
    
    return h
 #Now generate the input

x = np.random.randn(input_size, sequence_length)

#forward pass for each time stamp
hidden_states = []

for t in range(sequence_length):
    h_prev = rnn_forward(x[:, t:t+1], h_prev, wxh, whh, bh)
    hidden_states.append(h_prev)
#print hidden states
for t, h in enumerate(hidden_states):
    print(f"Time Step {t+1}:")
    print(h)
