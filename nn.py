class LinearWithReLu():
    def __init__(self, dim_in, dim_out):
        self.W = np.random.randn(dim_in, dim_out)
        self.b = np.random.randn(dim_out)
        self.affine = Affine(self.W, self.b)
        self.relu = ReLu()
        self.x = None
        self.dW = None
        self.db = None
        # self.out = None
        # print(f"Initialized with W: {self.W}, b: {self.b}")

    def __call__(self, x):
        # 当实例被“调用”时，执行的代码
        print(f"Calling with {x}")
        return self.forward(x)
    
    def __str__(self):
        return f"LinearWithReLu with W: {self.W}, b: {self.b}"
    
    def __repr__(self):
        return f"LinearWithReLu with W: {self.W}, b: {self.b}"
    
    def state_dict(self):
        return {'W': self.W, 'b': self.b}
    
    def forward(self, x):
        self.x = x
        out = self.affine.forward(x)
        out = self.relu.forward(out)
        self.out = out
        return out
    
    def backward(self, dout):
        dout = self.relu.backward(dout)
        dout = self.affine.backward(dout)
        self.dW = self.affine.dW
        self.db = self.affine.db
        return dout
    
    def save_state_dict(self, path):
        state_dict = {'W': self.W, 'b': self.b}
        np.savez(path, **state_dict)

    def load_state_dict(self, path):
        state_dict = np.load(path)
        self.W = state_dict['W']
        self.b = state_dict['b']
        self.affine = Affine(self.W, self.b)
    
linear_with_relu = LinearWithReLu(3, 2)
linear_with_relu