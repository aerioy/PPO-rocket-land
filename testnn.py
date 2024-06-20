import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F



class deepnet(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(deepnet, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

inputs = torch.zeros(10,9)
outputs = torch.zeros(10,4)
# for x in range(10):
#   inputs[x] = torch.rand(9)
#   outputs[x] = torch.rand(4)
#   print(inputs[x])
#   print(outputs[x])




# print(inputs)
# print(outputs)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.is_available())
testnet = deepnet(9,4).to(device)
optimizer = optim.Adam(testnet.parameters(), lr=0.001)
lf = nn.MSELoss()
print("-------------------")
print(testnet.forward(torch.tensor([1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0])))
iterations = 0
while iterations < 1000:
  iterations += 1
  optimizer.zero_grad()
  out = testnet.forward(inputs)
  loss = lf(out, outputs)
  loss.backward()
  optimizer.step()
  if iterations % 100 == 0:
    print("loss on step " + str(iterations) + "=" + str(loss))


file_path = 'model.pth'

# Save the model parameters
torch.save(testnet.state_dict(), file_path)


for x in range(10):
  print(testnet.forward(inputs[x]))
  print(outputs[x])
  print("-------------------")