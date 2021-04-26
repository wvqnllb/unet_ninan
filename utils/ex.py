import numpy as np

a = np.arange(27).reshape(3, 3, 3)
print(a)
a = np.where(a > 13, 1, 0)
print(a)