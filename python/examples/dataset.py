import numpy as np
import math
from sklearn.datasets import make_moons
from scipy.stats import norm


# Create a simple dataset
def create_twomoon_dataset(n, p):
    relevant, y = make_moons(n_samples=n, shuffle=True, noise=0.1, random_state=None)
    print(y.shape)
    noise_vector = norm.rvs(loc=0, scale=1, size=[n,p-2])
    data = np.concatenate([relevant, noise_vector], axis=1)
    print(data.shape)
    return data, y


def create_sin_dataset(n, p):
    '''This dataset was added to provide an example of L1 norm reg failure for presentation.
    '''
    assert p == 2
    x1 = np.random.uniform(-math.pi, math.pi, n).reshape(n ,1)
    x2 = np.random.uniform(-math.pi, math.pi, n).reshape(n, 1)
    y = np.sin(x1)
    data = np.concatenate([x1, x2], axis=1)
    print("data.shape: {}".format(data.shape))
    return data, y
    
    

