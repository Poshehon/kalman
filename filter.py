class Gauss:
    def __init__(self, expect = 0, var = 1):
        self.var = var
        self.expect = expect
    
    def __repr__(self):
        return f'N({self.expect}, {self.var})'

x = Gauss()
print(x)