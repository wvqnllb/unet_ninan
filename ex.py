import torch as t
import torch.nn.functional as F
from torch.autograd import Variable

target = t.rand(3, 3, 3).random_(5)
print(target)