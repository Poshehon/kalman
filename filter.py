import math
import typing
from pydantic import validate_arguments, BaseModel
class Gauss(BaseModel):
    mu: float = 0
    var: float = 1
    # Initialize x = Gauss(mu = a, var = b)
    @property
    def std(self):
        return math.sqrt(self.var)
    
    def __repr__(self):
        return f'N({self.mu}, {self.var})'
    
    def __add__(self, other):
        return Gauss(self.mu + other.mu, self.var + other.var)

    def __mul__(self, other):
        new_mu = (self.var * other.mu + other.var * self.mu ) / (self.var + other.var)
        new_var = (self.var * other.var) / (self.var + other.var)
        return Gauss(new_mu, new_var)

@validate_arguments
def kalman(state: Gauss, measurements : typing.List[Gauss]) -> typing.List[Gauss]:
    x = Gauss()
    return [x]
x = Gauss(mu = 1, var = 2)
y = Gauss()
print(x.std)