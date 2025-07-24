from unet import *
from torchvision.models.resnet import resnet18, ResNet18_Weights
# Assuming conv2d_layer and UNet are defined/imported elsewhere

def conv2d_layer(in_channels, out_channels, kernel_size, stride, padding=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)

class DiscriminatorFactory(nn.Module):
    def __init__(self, input_channels=1, dis_type='cnn', dis_depth=3, conv_num=3,
                 device='cuda:0', init_model = False, init_model_path = None, save_model_path = None):
        super().__init__()
        self.input_channels = input_channels
        self.dis_type = dis_type.lower()
        self.dis_depth = dis_depth
        self.conv_num = conv_num
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.init_model = init_model
        self.save_model_path = save_model_path
        self.init_model_path = init_model_path
        path = self.save_model_path if self.init_model_path is None else self.init_model_path
        # Build discriminator
        self.model = self._build_discriminator()
        self.to(self.device)

        # Initialize or load weights
        if self.init_model:
            if path:
                self._load_pretrained(path)
            else:
                self._init_weights()
        else:
            self._init_weights()

    def _build_discriminator(self):
        if self.dis_type == 'cnn':
            return self._make_cnn()
        elif self.dis_type == 'unet':
            return self._make_unet()
        elif self.dis_type == 'patchgan':
            return self._make_patchgan()
        elif self.dis_type == 'resnet':
            return self._make_resnet()
        elif self.dis_type == 'multiscale':
            return self._make_multiscale()
        else:
            raise ValueError(f"Unknown or unsupported dis_type: {self.dis_type}")

    def _make_cnn(self):
        # Classic DCGAN-style CNN discriminator
        in_ch = self.input_channels
        layers = []
        for i in range(self.dis_depth):
            out_ch = 2**(i + 4)
            layers.append(conv2d_layer(in_channels=in_ch, out_channels=out_ch,
                                       kernel_size=self.conv_num, stride=2))
            layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.LeakyReLU(0.2, inplace=False))
            in_ch = out_ch

        layers.append(nn.Conv2d(in_ch, 1, kernel_size=4, stride=1, padding=0))
        layers.append(nn.Flatten())
        layers.append(nn.Sigmoid())
        return nn.Sequential(*layers)

    def _make_unet(self):
        # U-Net discriminator for paired image tasks
        return nn.Sequential(
            UNet(n_channels=self.input_channels, n_classes=1, bilinear=True, base = 32),
            nn.Flatten(),
            nn.Sigmoid()
        )

    def _make_patchgan(self, patch_size=70):
        # PatchGAN discriminator as in pix2pix
        layers = []
        in_ch = self.input_channels
        nf = 64
        layers.append(nn.Conv2d(in_ch, nf, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        for i in range(1, self.dis_depth):
            mult = min(2**i, 8)
            layers.append(nn.Conv2d(nf * mult // 2, nf * mult,
                                     kernel_size=4, stride=2, padding=1))
            layers.append(nn.InstanceNorm2d(nf * mult))
            layers.append(nn.LeakyReLU(0.2, inplace=False))

        layers.append(nn.Conv2d(nf * mult, 1, kernel_size=patch_size, stride=1, padding=patch_size//2))
        layers.append(nn.Sigmoid())
        return nn.Sequential(*layers)

    def _make_resnet(self):
        # Projection ResNet discriminator with spectral norm
        # Use the new weights parameter to avoid deprecation warning
        base = resnet18(weights=None)
        base.conv1 = nn.Conv2d(self.input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        base.fc = nn.Linear(base.fc.in_features, 1)
        return base

    def _make_multiscale(self):
        # Multiple CNN discriminators at different scales
        return nn.ModuleList([
            self._make_cnn(),
            self._make_cnn(),
            self._make_cnn()
        ])

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.normal_(m.weight, 0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _load_pretrained(self, path):
        state = torch.load(path + 'discriminator.pth', map_location=self.device)
        self.model.load_state_dict(state.state_dict())

    def forward(self, x):
        if isinstance(self.model, nn.ModuleList):
            outputs = [m(x) for m in self.model]
            return sum(outputs) / len(outputs)
        return self.model(x)


if __name__ == "__main__":
    # Example run for all supported discriminator types
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")

    # Define input tensor
    x = torch.randn(1, 1, 512, 512, device=device)

    supported = ['cnn', 'unet', 'patchgan', 'resnet', 'multiscale']
    for dis_type in supported:
        print(f"Testing discriminator type: {dis_type}")
        model = DiscriminatorFactory(input_channels=1, dis_type=dis_type, device=device)
        out = model(x)
        print(f"  Output shape: {out.shape}\n")
