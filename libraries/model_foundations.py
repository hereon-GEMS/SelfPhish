from gauss_conv import *
import torch.nn.init as init

def initializer(m):
        if type(m) == nn.Conv2d:
            init.xavier_uniform_(m.weight)
            if m.bias is not None:
                init.zeros_(m.bias)
                
def discriminator_loss(real_output, fake_output):
    real_loss = torch.mean(torch.nn.BCEWithLogitsLoss()(real_output, torch.ones_like(real_output)))
    fake_loss = torch.mean(torch.nn.BCEWithLogitsLoss()(fake_output, torch.zeros_like(fake_output)))
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output, img_output, pred, l1_ratio = 10):
    #with autograd
    return torch.mean(torch.nn.BCEWithLogitsLoss()(fake_output, torch.ones_like(fake_output))) + \
              l1_ratio * torch.mean(torch.abs(img_output - pred))

def dense_layer(in_features= 128, out_features = 128, dropout = 0.25, apply_batchnorm=True, transpose = False):
    initializer = torch.nn.init.xavier_uniform_(torch.empty(out_features, in_features))
    if apply_batchnorm:
        result = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.LeakyReLU(),
            nn.Dropout(dropout),    
        )
    else:
        result = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
        )

    if transpose:
        result = nn.Sequential(
            nn.Linear(in_features, out_features),
            Transpose(),
            nn.BatchNorm1d(out_features),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
        )
    return result

def conv2d_layer(in_channels, out_channels, kernel_size, stride, apply_batchnorm=True, normal_init = True):
    def initializer(m):
        if type(m) == nn.Conv2d:
            init.xavier_uniform_(m.weight)
            if m.bias is not None:
                init.zeros_(m.bias)
    if apply_batchnorm:
        result = nn.Sequential(
            nn.Conv2d(in_channels,
                out_channels, 
                kernel_size, 
                stride=stride, 
                padding_mode='zeros',
                bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.LeakyReLU())
    else:
        result = nn.Sequential(
            nn.Conv2d(in_channels,
                out_channels, 
                kernel_size, 
                stride=stride, 
                padding_mode='zeros',
                bias=False),
            nn.LeakyReLU())
    if normal_init:
        result.apply(initializer)
    return result

def deconv2d_layer(in_channels, out_channels, kernel_size, stride, apply_batchnorm=True, normal_init = True):
    def initializer(m):
        if type(m) == nn.Conv2d:
            init.xavier_uniform_(m.weight)
            if m.bias is not None:
                init.zeros_(m.bias)
    if apply_batchnorm:
        result = nn.Sequential(
            nn.ConvTranspose2d(in_channels,
                out_channels,
                kernel_size,
                stride=stride, 
                padding_mode='zeros',
                bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.LeakyReLU())
        
    else:
        result = nn.Sequential(
            nn.ConvTranspose2d(in_channels,
                out_channels,
                kernel_size,
                stride=stride, 
                padding_mode='zeros',
                bias=False),
            nn.LeakyReLU())
    if normal_init:
        result.apply(initializer)
    return result

class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(self.shape)
    
class Transpose(nn.Module):
    def __init__(self):
        super(Transpose, self).__init__()

    def forward(self, x):
        return x.view(-1, 1)
  
class Print_module(nn.Module):
    def __init__(self, word = ''):
        super(Print_module, self).__init__()
        self.word = word
    def forward(self, x):
        print(self.word, x.shape)
        return x