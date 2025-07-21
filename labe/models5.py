__author__ = 'Jiri Fajtl'
__email__ = 'ok1zjf@gmail.com'
__version__= '1.8'
__status__ = "Research"
__date__ = "2/1/2020"
__license__= "MIT License"

import torch
import torch.nn as nn

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        if m.bias is not None:
            m.bias.data.zero_()

def weight_init(model, mean=0, std=0.02):
    for m in model._modules:
        normal_init(model._modules[m], mean, std)
    return

#===========================================================================================
class QuantizerFunc(torch.autograd.Function):
    """
    Custom autograd function for quantization. Rounds input tensor to a specified number of decimal points or applies sign quantization.
    """
    @staticmethod
    def forward(self, input, npoints=4, dropout=0):
        """
        Forward pass for quantization.
        Args:
            input (Tensor): Input tensor to quantize.
            npoints (int): Number of decimal points to keep. If negative, applies sign quantization.
            dropout (float): Not used.
        Returns:
            Tensor: Quantized tensor.
        """
        # self.save_for_backward(input)
        # self.constant = npoints
        if npoints < 0:
            x = torch.sign(input)
            x[x==0] = 1
            return x

        scale = 10**npoints
        input = input * scale
        input = torch.round(input)
        input = input / scale
        return input

    @staticmethod
    def backward(self, grad_output):
        """
        Backward pass for quantization. Passes gradient unchanged.
        Args:
            grad_output (Tensor): Gradient from next layer.
        Returns:
            Tuple: Gradient for input, None for npoints.
        """
        # input, = self.saved_tensors
        grad_input = grad_output.clone()
        # grad_input[input < 0] = 0
        # grad_input[:] = 1
        return grad_input, None


class Quantizer(nn.Module):
    """
    Module wrapper for QuantizerFunc. Applies quantization to input tensor.
    """
    def __init__(self, npoints=3):
        super().__init__()
        self.npoints = npoints
        self.quant = QuantizerFunc.apply

    def forward(self,x):
        """
        Forward pass applying quantization.
        Args:
            x (Tensor): Input tensor.
        Returns:
            Tensor: Quantized tensor.
        """
        x = self.quant(x, self.npoints)
        return x
 

#===========================================================================================
class EncConvResBlock32(nn.Module):
    """
    Encoder convolutional residual block for 32x32 images.
    Args:
        channels (int): Number of input channels.
        zsize (int): Size of latent vector.
        zround (int): Quantization precision for latent vector.
    """
    def __init__(self, channels, zsize, zround):
        super().__init__()
        self.channels = channels
        self.vae = False
        self.zsize = zsize
        self.zround = zround
        bias = False 
        c = 64

        inc = c
        self.ec0 = nn.Conv2d(channels, inc, kernel_size=3, stride=1, padding=1, bias=bias)
        self.bn0 = nn.BatchNorm2d(inc)
        self.ec1 = nn.Conv2d(inc, c, kernel_size=4, stride=2, padding=1, bias=bias)

        self.bn1 = nn.BatchNorm2d(c)
        self.b11 = nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, bias=bias)
        self.bn11 = nn.BatchNorm2d(c)
        self.b12 = nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, bias=bias)
        self.bn12 = nn.BatchNorm2d(c)

        c = c*2
        self.ec2 = nn.Conv2d(c//2, c, kernel_size=4, stride=2, padding=1, bias=bias)
        self.bn2 = nn.BatchNorm2d(c)
        self.b21 = nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, bias=bias)
        self.bn21 = nn.BatchNorm2d(c)
        self.b22 = nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, bias=bias)
        self.bn22 = nn.BatchNorm2d(c)

        c_out = c*2
        
        self.ec4 = nn.Conv2d(c, c_out, kernel_size=4, stride=2, padding=1, bias=False)

        fmres = 4*4 
        in_size =self.ec4.out_channels*fmres

        # vae is always False
        self.l0l = nn.Linear(in_size, zsize)
        self.quant = QuantizerFunc.apply

        self.act = nn.LeakyReLU(0.02)
        self.drop = None

    def forward(self, x):
        """
        Forward pass for encoder block.
        Args:
            x (Tensor): Input image tensor.
        Returns:
            Tuple: (quantized latent, None, latent, diff, quantization error)
        """
        x = self.ec0(x)
        x = self.bn0(x)
        x = self.act(x)

        x = self.ec1(x)
        x = self.bn1(x)
        y = x
        x = self.act(x)
        x = self.b11(x)
        x = self.bn11(x)
        x = self.act(x)
        x = self.b12(x)
        x = self.bn12(x)
        x = self.act(x+y)
        
        x = self.ec2(x)
        x = self.bn2(x)
        y = x
        x = self.act(x)
        x = self.b21(x)
        x = self.bn21(x)
        x = self.act(x)
        x = self.b22(x)
        x = self.bn22(x)
        x = self.act(x+y)

        x = self.ec4(x)
        x = x.view(x.size(0), -1)

        # QAE output (vae is always False)
        xe = None

        x = self.l0l(x)
        x = torch.tanh(x)

        xq = self.quant(x, self.zround)
        err_quant = torch.abs(x - xq)
        x = xq

        xe = x if xe is None else xe
        diff = ((x+xe) == 0).sum(1) 
        return x, None, xe, diff, err_quant.sum()/(x.size(0) * x.size(1))

#===========================================================================================
class GenConvResBlock32(nn.Module):
    """
    Generator convolutional residual block for 32x32 images.
    Args:
        channels (int): Number of output channels.
        dataset (str): Dataset name (e.g., 'mnist').
        zsize (int): Size of latent vector.
    """
    def __init__(self, channels, dataset, zsize):
        super().__init__()
        self.channels = channels
        self.dataset = dataset
        self.zsize = zsize

        bias = False
        c = inch = 128

        if dataset == 'mnist':
           inch = 1

        self.in_channels = inch

        self.dc2 = nn.ConvTranspose2d(inch, c, kernel_size=4, stride=2, padding=1, output_padding=0, bias=bias)
        self.bn2 = nn.BatchNorm2d(c)
        self.b21 = nn.ConvTranspose2d(c, c,kernel_size=3, stride=1, padding=1, bias=bias)
        self.bn21 = nn.BatchNorm2d(c)
        self.b22 = nn.ConvTranspose2d(c, c, kernel_size=3, stride=1, padding=1, bias=bias)
        self.bn22 = nn.BatchNorm2d(c)

        c = c//2
        self.dc3 = nn.ConvTranspose2d(c*2, c, kernel_size=4, stride=2, padding=1, output_padding=0, bias=bias)
        self.bn3 = nn.BatchNorm2d(c)
        self.b31 = nn.ConvTranspose2d(c, c,kernel_size=3, stride=1, padding=1, bias=bias)
        self.bn31 = nn.BatchNorm2d(c)
        self.b32 = nn.ConvTranspose2d(c, c, kernel_size=3, stride=1, padding=1, bias=bias)
        self.bn32 = nn.BatchNorm2d(c)

        self.dc4 = nn.ConvTranspose2d(c, c, kernel_size=4, stride=2, padding=1, output_padding=0, bias=False)
        self.bn4 = nn.BatchNorm2d(c)
        self.dc5 = nn.ConvTranspose2d(c, channels, kernel_size=3, stride=1, padding=1, bias=False)

        self.act = nn.LeakyReLU(0.02)
        self.quant = QuantizerFunc.apply

        self.in_channels = self.dc2.in_channels

        self.fmres = 4 
        out_size = self.in_channels*self.fmres*self.fmres
        bias = True
        self.l1l=nn.Linear(zsize, out_size, bias=bias)
        

    def forward(self, x, sw=None):
        """
        Forward pass for generator block.
        Args:
            x (Tensor): Latent vector.
            sw: (Unused)
        Returns:
            Tensor: Generated image tensor.
        """
        x = x.view(x.size(0), -1)

        x = self.l1l(x)
        x = x.view(x.size(0), self.in_channels,self.fmres,self.fmres)

        x = self.dc2(x)
        x = self.bn2(x)
        y = x
        x = self.act(x)
        x = self.b21(x)
        x = self.bn21(x)
        x = self.act(x)
        x = self.b22(x)
        x = self.bn22(x)
        x = self.act(x+y)

        x = self.dc3(x)
        x = self.bn3(x)
        y = x
        x = self.act(x)
        x = self.b31(x)
        x = self.bn31(x)
        x = self.act(x)
        x = self.b32(x)
        x = self.bn32(x)
        x = self.act(x+y)

        x = self.dc4(x)
        x = self.bn4(x)
        x = self.act(x)
        x = self.dc5(x)

        x = torch.sigmoid(x)
        return x

#=================================================================================
if __name__ == "__main__":
    """
    This file is not intended to be run as a standalone script.
    """
    print("NOT AN EXECUTABLE!")




