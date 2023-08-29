import torch
input = torch.arange(1, 5, dtype=torch.float32).view(1, 1, 2, 2) #the tensor creates an array [1,2,3,4]. View cretes a three dimensional array 1 is batch, 1 dimension, then widht and height
print(input)