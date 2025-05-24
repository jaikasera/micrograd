# Micrograd-based MLP Neural Network

This project is a minimal neural network implementation using automatic differentiation based on Andrej Karpathy's **micrograd**. It includes a simple multi-layer perceptron (MLP) built from scratch with support for different activation functions and gradient backpropagation.

---

## File Descriptions

### `autodiff.py` (automatic differentiation code)
Implements the core `Value` class that supports scalar values with automatic differentiation.  
- Overloads arithmetic operators for computation graph construction.  
- Implements nonlinear activation functions: `tanh`, `relu`, `sigmoid`.  
- Provides a recursive topological sorting utility to ensure correct backpropagation order.  
- Includes the `backward()` method for reverse-mode autodiff.

### `network.py` (MLP code)
Defines the structure of the Multi-Layer Perceptron (MLP) and individual layers.  
- Supports specifying activation functions **per layer** or globally.  
- Includes implementations of `tanh`, `relu`, and `sigmoid` activations.  
- Contains the forward pass logic that applies the chosen activation to each layer's output.  
- Uses `Value` objects to allow gradient computation through the network.

### `mlp_train.py` (a sample training script)
Provides an example training loop for the MLP.  
- Demonstrates how to create datasets, perform forward passes, compute losses, and call `backward()` for gradients.  
- Illustrates updating parameters with gradient descent.  

### `engine_and_nn.ipynb` (rough script)
- A rough script in which I wrote the original code, before I transferred this code to separate .py files and added improvements.
---

## Summary of Changes and Enhancements to Original Program

- **Activation flexibility:** Allowed specifying activation functions on a per-layer basis or globally in `mlp.py`.  
- **Added Sigmoid Activation:** Added sigmoid activation function support in both `value.py` and `mlp.py`.  
- **Refactoring:** Cleaned up and improved comments for clarity.  
- **Separate Topological Sort:** Extracted and implemented a topological sort utility inside the `Value` class for proper backpropagation order.   

---

## Credits

This implementation is based on the work of Andrej Karpathy and his work on micrograd.