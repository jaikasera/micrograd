import math 
class Value:
    def __init__(self, data, _children = (), _op = ""):
        self.data = data
        self.grad = 0.0 # default to no effect 
        self._backward = lambda: None # empty function by default 
        self._prev = set(_children) # take the set of the children tuple 
        self._op = _op # last operation 

    def __repr__(self):
        return f"Value(data={self.data})" # display Value object nicely 
    
    def __add__(self, other): # __add__ is equivalent to "+"
        other = other if isinstance(other, Value) else Value(other) # If we add a value object to an int, wrap the int into a Value object
        out = Value(self.data + other.data, (self, other), "+") #set other and self to children 
        def _backward():
            self.grad += 1.0 * out.grad # because derivative do/da and do/db for addition d = a + b is just 1 
            other.grad += 1.0 * out.grad # use += to accumulate gradients in the case of multivariable chain rule (if we use the same node more than once in the network)
            # the mathematical formula would look like dl/da = dl/db * db/da + dl/dc * dc/da
        out._backward = _backward
        return out 
    
    def __radd__(self, other):
        return self + other
    
    def __neg__ (self): # Self * -1
        return self * -1 
    
    def __sub__(self, other): # self - other = self + (-other) (__neg__ operation)
        return self + (-other)
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other) 
        out = Value(self.data * other.data, (self, other), "*")
        def _backward():
            self.grad += other.data * out.grad # because derivative of do/da for o = a*b is b, multiplied by global derivative o.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out 
        
    
    def __rmul__(self,other): # Check the reverse multiplication if forward multiplication is invalid 
        return self*other
    
    def exp(self):
        out = Value(math.exp(self.data), (self, ), "exp")

        def _backward():
            self.grad += out.data * out.grad # d/dx(e^x) = e^x, in this case out.data = e^x
        out._backward = _backward
        return out 

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only support int/float power"
        out = Value(self.data ** other, (self, ), "pow")

        def _backward():
            self.grad += other* (self.data ** (other-1)) * out.grad # power rule 
        out._backward = _backward
        return out 
    
    
    
    def __truediv__(self, other): #calculate self ÷ other 
        return self * (other**-1) 
    
    def __rtruediv__(self, other): # other / self
        return other * self**-1

    
    def tanh(self): #implementing tanh nonlinearity activation function, could do relu or sigmoid instead as well 
        x = self.data 
        t = (math.exp(2*x)-1)/(math.exp(2*x) + 1) # formula for tanh 
        out = Value(t, (self, ), "tanh")

        def _backward():
            self.grad += (1-t**2) * out.grad # using formula for derivative of tanh, only self here (no "other" node) 
        out._backward = _backward
        return out 
    
    def relu(self): 
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out
    

    def backward (self):

        topological_sort = [] 
        visited = set() 
        def build_topological(v): # recursive topological sort algorithm: returns a list, topologically ordered
            if v not in visited: 
                visited.add(v) 
                for child in v._prev:
                    build_topological(child) 
                topological_sort.append(v) 

        build_topological(self)
        self.grad = 1.0 # base case
        # Note: we need to traverse list in reverse topological order to ensure we have the global gradients before computing local ones 
        for node in reversed(topological_sort):
            node._backward() 