import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng
seed = 1
rng = default_rng(seed)
import os, os.path
from pathlib import Path
import torch
import torch.nn as nn

import importlib.util
spec = importlib.util.spec_from_file_location("module.fcae", str(Path(os.getcwd()).parent.parent) + '/' + "fcae.py")
module_fcae = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module_fcae)

m = 100
phi_arr = rng.random(size=m) * 2*np.pi
n = 16
dx = 2*np.pi / n
x_arr = np.arange(dx/2, 2*np.pi, dx)
X = np.array([np.cos(x_arr + phi) for phi in phi_arr])

# divide into train/test set
m_train = int(0.8 * m)
m_test = m - m_train;
i_train = rng.choice(range(m), size=m_train, replace=False)
i_test = list(set(list(range(m))) - set(i_train))
Xtrainraw = X[i_train]
Xtestraw = X[i_test]

# Normalize appropriatly, i.e., w.r.t to full input matrix X
# not needed for this data set
#meanx = np.mean(Xtrainraw)
#stdx = np.std(Xtrainraw)
#Xtrain = (Xtrainraw - meanx) / stdx
#Xtest = (Xtestraw - meanx) / stdx
Xtrain = Xtrainraw
Xtest = Xtestraw

for i in range(5):
    plt.plot(Xtrain[i])
plt.show()

# Create torch.Tensor and torch.utils.data.dataset.TensorDataset for training
tensor_xtrain = torch.Tensor(Xtrain)
tensor_xtest = torch.Tensor(Xtest)
train_dataset = torch.utils.data.TensorDataset(tensor_xtrain, tensor_xtrain)
test_dataset = torch.utils.data.TensorDataset(tensor_xtest, tensor_xtest)

# Network design
layer_widths = [n, 4, 1]
activation_function = nn.Tanh()
loss = nn.MSELoss()

# Training design
# Size for batch gradient descent
batch_size = int(m_train / 2)
# Learning Rate
lr = 1e-3
# Number of learning epochs
num_epochs = 20000

fcae = module_fcae.FullyConnectedAutoencoder(layer_widths, activation_function, seed)

fcae.train_model(loss, lr, batch_size, num_epochs, train_dataset, test_dataset)

tensor_xptrain = fcae.forward(tensor_xtrain)
tensor_xptest = fcae.forward(tensor_xtest)

Xptrain = tensor_xptrain.detach().numpy()
Xptest = tensor_xptest.detach().numpy()

plt.plot(x_arr, Xtest[0], label="$f$")
plt.plot(x_arr, Xptest[0], label="$\\tilde{f}$")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.show()

tensor_ctrain = fcae.encode(tensor_xtrain)
tensor_ctest = fcae.encode(tensor_xtest)

Ctrain = tensor_ctrain.detach().numpy()
Ctest = tensor_ctest.detach().numpy()

plt.plot(Ctest[:], phi_arr[i_test], "o")
plt.xlabel("Code c")
plt.ylabel("Shift " + "$\phi$")
plt.show()
