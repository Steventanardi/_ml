import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(10, 5)   
        self.fc2 = nn.Linear(5, 1)    

    def forward(self, x):
        x = F.relu(self.fc1(x))      
        x = self.fc2(x)              
        return x

model = SimpleModel()

input_data = torch.randn(1, 10)

output = model(input_data)

print("Model output:", output)
