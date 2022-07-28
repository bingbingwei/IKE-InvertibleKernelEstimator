import torch
import torch.nn as nn

class Quant(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        input = torch.clamp(input, -1, 1)
        output = ((input+1)/2.0 * 255.).round()
        output = ((output/255.) * 2.0 - 1)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class Quantization(nn.Module):
    def __init__(self):
        super(Quantization, self).__init__()

    def forward(self, input):
        return Quant.apply(input)
