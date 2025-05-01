import math
import typing
from pydantic import validate_call, BaseModel
import numpy as np
import matplotlib.pyplot as plt

class Gauss(BaseModel):
    mu: float = 0
    var: float = 1
    #Initialize Gauss(mu = x, var = sigma ** 2)
    @property
    def std(self):
        return math.sqrt(self.var)
    
    def __repr__(self):
        return f'N({self.mu}, {self.var})'
    
    def __add__(self, other):
        return Gauss(mu=self.mu + other.mu, var=self.var + other.var)

    def __mul__(self, other):
        new_mu = (self.var * other.mu + other.var * self.mu) / (self.var + other.var)
        new_var = (self.var * other.var) / (self.var + other.var)
        return Gauss(mu=new_mu, var=new_var)


class Mult_gauss():
    def __init__(self, dim, mu, matrix):
        self.dim = dim
        self.mu = mu
        self.matrix = matrix

    def __repr__(self):
        return f'mu = {self.mu}\ncov = {self.matrix}\n'


class Target():
    def __init__(self, x, speed):
        self.x = x
        self.speed = speed
    
    def move(self):
        self.x = self.x + self.speed
    
    def motion_and_measurement(self, n, var):
        ans = []
        for i in range(n):
            self.move()
            z = Gauss(mu = self.x + np.random.randn() * math.sqrt(var), var = var)
            ans.append(z)
        return ans



@validate_call
def kalman1D(state, diff: Gauss, measurements: typing.List[Gauss]) -> typing.List[Gauss]:
    ans = []
    for z in measurements:
        curr = state + diff
        curr = curr * z
        ans.append(curr) 
    return ans


#Make simulation
state = Gauss()
diff = Gauss(mu = 2, var = 4)
car = Target(1, 1)
measurements = car.motion_and_measurement(100, 20)
res = [elem.mu for elem in kalman1D(state, diff, measurements)]
plt.plot(res)
plt.show()