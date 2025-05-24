from mlp import MLP

# MLP Sample Usage
n = MLP(3,[4,4,1])

# Sample dataset 
x_data = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]

labels = [1.0, -1.0, -1.0, 1.0] # Desired target value (suppose this is a binary task)

num_epochs = 50
step_size = 0.05

for k in range (num_epochs): # gradient descent
    pred = [n(x)[0] for x in x_data] #forward pass
    loss = sum((y_pred - y_true)**2 for y_true, y_pred in zip(labels, pred))
    for p in n.parameters(): # zero out gradietns
        p.grad = 0.0 

    loss.backward() # backward pass
    
    for p in n.parameters(): # updating parameters 
        p.data += step_size * (-p.grad) 

    print(f"Epoch {k+1}: {loss.data}")