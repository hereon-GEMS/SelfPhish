""" Full assembly of the parts to form the complete network """
from torch_utils import *
from unet_parts import *
from transform2d import *
from torch.utils.checkpoint import checkpoint

def checkpoint_wrapper(func, *args):
    return checkpoint(func, *args, use_reentrant=False)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, use_checkpoint=True, base=64):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.use_checkpoint = use_checkpoint
        self.base = base

        # Encoder (down path)
        self.encoder = nn.ModuleList([
            DoubleConv(n_channels, base),
            Down(base, base * 2),
            Down(base * 2, base * 4),
            Down(base * 4, base * 8),
            Down(base * 8, (base * 16) // (2 if bilinear else 1)),
        ])
        # Decoder (up path)
        self.decoder = nn.ModuleList([
            Up(base * 16, (base * 8) // (2 if bilinear else 1), bilinear),
            Up(base * 8, (base * 4) // (2 if bilinear else 1), bilinear),
            Up(base * 4, (base * 2) // (2 if bilinear else 1), bilinear),
            Up(base * 2, base, bilinear),
        ])
        self.outc = OutConv(base, n_classes)

    def forward(self, x):
        f = checkpoint_wrapper if self.use_checkpoint else lambda f, *args: f(*args)
        # Encoder
        x1 = f(self.encoder[0], x)
        x2 = f(self.encoder[1], x1)
        x3 = f(self.encoder[2], x2)
        x4 = f(self.encoder[3], x3)
        x5 = f(self.encoder[4], x4)
        # Decoder
        x = f(self.decoder[0], x5, x4)
        x = f(self.decoder[1], x, x3)
        x = f(self.decoder[2], x, x2)
        x = f(self.decoder[3], x, x1)
        logits = f(self.outc, x)
        return logits
    

"""
Adapted from the following sources:
https://github.com/Ssshanto/wavelet-ensemble-unet/tree/main
"""

class DWTForward(nn.Module):
    """ Performs a 2d DWT Forward decomposition of an image

    Args:
        J (int): Number of levels of decomposition
        wave (str or pywt.Wavelet or tuple(ndarray)): Which wavelet to use.
            Can be:
            1) a string to pass to pywt.Wavelet constructor
            2) a pywt.Wavelet class
            3) a tuple of numpy arrays, either (h0, h1) or (h0_col, h1_col, h0_row, h1_row)
        mode (str): 'zero', 'symmetric', 'reflect' or 'periodization'. The
            padding scheme
        """
    def __init__(self, J=1, wave='db1', mode='zero', continous = False, bandwidth_frequency=1.5, center_frequency=1.0):
        #type of wave are 'db1', 'haar', 'coif1', 'bior1.3', 'rbio1.3', 'dmey', 'sym2', 'rbio3.1', 'rbio1.1', 'rbio1.5', 'rbio3.5', 'rbio3.7', 'rbio3.9', 'rbio1.7'
        super().__init__()
        if isinstance(wave, str):
            try:
                wave = pywt.Wavelet(wave)
            except:
                continous = True
                wave = pywt.ContinuousWavelet(wave)
        if continous is False:
            if isinstance(wave, pywt.Wavelet):
                h0_col, h1_col = wave.dec_lo, wave.dec_hi
                h0_row, h1_row = h0_col, h1_col
            else:
                    if len(wave) == 2:
                        h0_col, h1_col = wave[0], wave[1]
                        h0_row, h1_row = h0_col, h1_col
                    elif len(wave) == 4:
                        h0_col, h1_col = wave[0], wave[1]
                        h0_row, h1_row = wave[2], wave[3]
            # Prepare the filters
            filts = lowlevel.prep_filt_afb2d(h0_col, h1_col, h0_row, h1_row)
            self.register_buffer('h0_col', filts[0])
            self.register_buffer('h1_col', filts[1])
            self.register_buffer('h0_row', filts[2])
            self.register_buffer('h1_row', filts[3])
            self.J = J
            self.mode = mode
        else:
            wave.bandwidth_frequency = bandwidth_frequency
            wave.center_frequency = center_frequency
            self.J = J
        self.continous = continous
    def forward(self, x):
        """ Forward pass of the DWT.

        Args:
            x (tensor): Input of shape :math:`(N, C_{in}, H_{in}, W_{in})`

        Returns:
            (yl, yh)
                tuple of lowpass (yl) and bandpass (yh) coefficients.
                yh is a list of length J with the first entry
                being the finest scale coefficients. yl has shape
                :math:`(N, C_{in}, H_{in}', W_{in}')` and yh has shape
                :math:`list(N, C_{in}, 3, H_{in}'', W_{in}'')`. The new
                dimension in yh iterates over the LH, HL and HH coefficients.

        Note:
            :math:`H_{in}', W_{in}', H_{in}'', W_{in}''` denote the correctly
            downsampled shapes of the DWT pyramid.
        """
        yh = []
        ll = x
        if self.continous is False:
            mode = lowlevel.mode_to_int(self.mode)

            # Do a multilevel transform
            for j in range(self.J):
                # Do 1 level of the transform
                ll, high = lowlevel.AFB2D.apply(
                    ll, self.h0_col, self.h1_col, self.h0_row, self.h1_row, mode)
                yh.append(high)

            return ll, yh
        else:
            scales = range(1, self.J+1)
            coefficient, frequencies = pywt.cwt(x, scales=scales, wavelet='fbsp')
            return coefficient, frequencies

class DWTInverse(nn.Module):
    """ Performs a 2d DWT Inverse reconstruction of an image

    Args:
        wave (str or pywt.Wavelet or tuple(ndarray)): Which wavelet to use.
            Can be:
            1) a string to pass to pywt.Wavelet constructor
            2) a pywt.Wavelet class
            3) a tuple of numpy arrays, either (h0, h1) or (h0_col, h1_col, h0_row, h1_row)
        mode (str): 'zero', 'symmetric', 'reflect' or 'periodization'. The
            padding scheme
    """
    def __init__(self, wave='db1', mode='zero', continous = False, bandwidth_frequency=1.5, center_frequency=1.0):
        super().__init__()
        if isinstance(wave, str):
            try:
                wave = pywt.Wavelet(wave)
            except:
                wave = pywt.ContinuousWavelet(wave)
                continous = True
        if continous is False:
            if isinstance(wave, pywt.Wavelet):
                g0_col, g1_col = wave.rec_lo, wave.rec_hi
                g0_row, g1_row = g0_col, g1_col
            else:
                if len(wave) == 2:
                    g0_col, g1_col = wave[0], wave[1]
                    g0_row, g1_row = g0_col, g1_col
                elif len(wave) == 4:
                    g0_col, g1_col = wave[0], wave[1]
                    g0_row, g1_row = wave[2], wave[3]
            # Prepare the filters
            filts = lowlevel.prep_filt_sfb2d(g0_col, g1_col, g0_row, g1_row)
            self.register_buffer('g0_col', filts[0])
            self.register_buffer('g1_col', filts[1])
            self.register_buffer('g0_row', filts[2])
            self.register_buffer('g1_row', filts[3])
            self.mode = mode
        else:
            wave.bandwidth_frequency = bandwidth_frequency
            wave.center_frequency = center_frequency
        self.continous = continous
        

    def forward(self, coeffs):
        """
        Args:
            coeffs (yl, yh): tuple of lowpass and bandpass coefficients, where:
              yl is a lowpass tensor of shape :math:`(N, C_{in}, H_{in}',
              W_{in}')` and yh is a list of bandpass tensors of shape
              :math:`list(N, C_{in}, 3, H_{in}'', W_{in}'')`. I.e. should match
              the format returned by DWTForward

        Returns:
            Reconstructed input of shape :math:`(N, C_{in}, H_{in}, W_{in})`

        Note:
            :math:`H_{in}', W_{in}', H_{in}'', W_{in}''` denote the correctly
            downsampled shapes of the DWT pyramid.

        Note:
            Can have None for any of the highpass scales and will treat the
            values as zeros (not in an efficient way though).
        """
        yl, yh = coeffs
        ll = yl
        if self.continous is False:
            mode = lowlevel.mode_to_int(self.mode)
            # Do a multilevel inverse transform
            for h in yh[::-1]:
                if h is None:
                    h = torch.zeros(ll.shape[0], ll.shape[1], 3, ll.shape[-2],
                                    ll.shape[-1], device=ll.device)

                # 'Unpad' added dimensions
                if ll.shape[-2] > h.shape[-2]:
                    ll = ll[...,:-1,:]
                if ll.shape[-1] > h.shape[-1]:
                    ll = ll[...,:-1]
                ll = lowlevel.SFB2D.apply(
                    ll, h, self.g0_col, self.g1_col, self.g0_row, self.g1_row, mode)
            return ll
        else:
            return pywt.icwt(coeffs[0], coeffs[1], wavelet='fbsp')
        
class WaveletDownsampling(nn.Module):
    def __init__(self, wavelet='db1', level=1):
        super(WaveletDownsampling, self).__init__()
        self.wavelet = wavelet
        self.level = level
        self.dwt_forward = DWTForward(wave=self.wavelet, J=self.level)

    def forward(self, x):
        coeffs = self.dwt_forward(x)
        return coeffs[0]

class WaveletUpsampling(nn.Module):
    def __init__(self, wavelet='db1'):
        super(WaveletUpsampling, self).__init__()
        self.wavelet = wavelet
        self.dwt_inverse = DWTInverse(wave=self.wavelet)
    
    def forward(self, cA):
        coeffs = (cA, [None])
        x = self.dwt_inverse(coeffs)
        return x

class Wavelet_Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, wavelet):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            WaveletDownsampling(wavelet=wavelet, level=1),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Wavelet_Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, wavelet):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        self.up = WaveletUpsampling(wavelet=wavelet)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class Wavelet_UNet(nn.Module):
    def __init__(self, n_channels, n_classes, wavelet='db1', use_checkpoint = True):
        super(Wavelet_UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.use_checkpoint = use_checkpoint
        self.encoder = nn.ModuleList([
            DoubleConv(n_channels, 64),
            Wavelet_Down(64, 128, wavelet),
            Wavelet_Down(128, 256, wavelet),
            Wavelet_Down(256, 512, wavelet),
            Wavelet_Down(512, 1024 // 2, wavelet),
        ])
        self.decoder = nn.ModuleList([
            Wavelet_Up(1024, 512 // 2, wavelet),
            Wavelet_Up(512, 256 // 2, wavelet),
            Wavelet_Up(256, 128 // 2, wavelet),
            Wavelet_Up(128, 64, wavelet),
        ])
        self.outc = OutConv(64, n_classes)
        

    def forward(self, x):
        f = checkpoint_wrapper if self.use_checkpoint else lambda f, *args: f(*args)
        # Encoder
        x1 = f(self.encoder[0], x)
        x2 = f(self.encoder[1], x1)
        x3 = f(self.encoder[2], x2)
        x4 = f(self.encoder[3], x3)
        x5 = f(self.encoder[4], x4)
        # Decoder
        x = f(self.decoder[0], x5, x4)
        x = f(self.decoder[1], x, x3)
        x = f(self.decoder[2], x, x2)
        x = f(self.decoder[3], x, x1)
        logits = f(self.outc, x)
        return logits

class Wavelet_UNet(nn.Module):
    def __init__(self, n_channels, n_classes, wavelet='db1', use_checkpoint=False, base = 32):
        super(Wavelet_UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.use_checkpoint = use_checkpoint
        pow_cof = np.log2(base)
        powers = [int(2**(pow_cof + i)) for i in range(0, 4)]

        self.inc = (DoubleConv(n_channels, powers[0]))
        self.down1 = (Wavelet_Down(powers[0], powers[1], wavelet))
        self.down2 = (Wavelet_Down(powers[1], powers[2], wavelet))
        self.down3 = (Wavelet_Down(powers[2], powers[3], wavelet))
        self.down4 = (Wavelet_Down(powers[3], powers[3], wavelet))
        self.up1 = (Wavelet_Up(powers[3] * 2, powers[2], wavelet))
        self.up2 = (Wavelet_Up(powers[2] * 2, powers[1], wavelet))
        self.up3 = (Wavelet_Up(powers[1] * 2, powers[0], wavelet))
        self.up4 = (Wavelet_Up(powers[0] * 2, powers[0], wavelet))
        self.outc = (OutConv(powers[0], n_classes))
        
    def forward(self, x):
        f = checkpoint if self.use_checkpoint else lambda f, *args: f(*args)
        x1 = f(self.inc, x)
        x2 = f(self.down1, x1)
        x3 = f(self.down2, x2)
        x4 = f(self.down3, x3)
        x5 = f(self.down4, x4)
        x = f(self.up1, x5, x4)
        x = f(self.up2, x, x3)
        x = f(self.up3, x, x2)
        x = f(self.up4, x, x1)
        logits = f(self.outc, x)
        return logits    

class MRConcat(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MRConcat, self).__init__()
        self.conv1 = DoubleConv(in_channels, out_channels)
        self.conv2 = DoubleConv(out_channels * 2, out_channels)

    def forward(self, x1, x2):
        x1 = self.conv1(x1)
        x = torch.cat([x1, x2], dim=1)
        x = self.conv2(x)
        return x

class MR_UNet(nn.Module):
    def __init__(self, n_channels, n_classes, wavelet='db1'):
        super(MR_UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.wavelet_downsample = WaveletDownsampling(wavelet=wavelet, level=1)
        
        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.concat1 = MRConcat(n_channels, 128)
        self.down2 = (Down(128, 256))
        self.concat2 = MRConcat(n_channels, 256)
        self.down3 = (Down(256, 512))
        self.concat3 = MRConcat(n_channels, 512)
        self.down4 = (Down(512, 1024))
        self.concat4 = MRConcat(n_channels, 1024)
        
        self.up1 = (Up(1024, 512))
        self.up2 = (Up(512, 256))
        self.up3 = (Up(256, 128))
        self.up4 = (Up(128, 64))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        downsampled2 = self.wavelet_downsample(x)
        downsampled3 = self.wavelet_downsample(downsampled2)
        downsampled4 = self.wavelet_downsample(downsampled3)
        downsampled5 = self.wavelet_downsample(downsampled4)
        
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x2 = self.concat1(downsampled2, x2)
        x3 = self.down2(x2)
        x3 = self.concat2(downsampled3, x3)
        x4 = self.down3(x3)
        x4 = self.concat3(downsampled4, x4)
        x5 = self.down4(x4)
        x5 = self.concat4(downsampled5, x5)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

class UNET(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=1, features=[64, 128, 256, 512],
    ):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                # x = TF.resize(x, size=skip_connection.shape[2:])
                x = F.interpolate(x, size=skip_connection.shape[2:], mode='bilinear', align_corners=True)

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)

class Unet_skip(nn.Module):
    def __init__(
            self, 
            in_channels=3, 
            out_channels=1, 
            features=[16, 32, 64, 128], 
            bilinear=False
    ):
        super(Unet_skip, self).__init__()
        self.bilinear = bilinear

        # Downsampling path
        self.downs = nn.ModuleList()
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Upsampling path
        self.ups = nn.ModuleList()
        for feature in reversed(features):
            if bilinear:
                self.ups.append(nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                    nn.Conv2d(feature * 2, feature, kernel_size=1)
                ))
            else:
                self.ups.append(
                    nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
                )
            self.ups.append(DoubleConv(feature * 2, feature))

        # Final output layer
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Down path
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = F.max_pool2d(x, kernel_size=2)

        # Bottleneck
        x = self.bottleneck(x)

        # Reverse skip connections for upsampling
        skip_connections = skip_connections[::-1]

        # Up path
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            # Resize if shapes differ
            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:], mode='bilinear', align_corners=True)

            # Concatenate and apply DoubleConv
            x = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](x)

        return self.final_conv(x)

    def use_checkpointing(self):
        """Enable checkpointing to save memory."""
        for module in self.downs:
            module = torch.utils.checkpoint(module)
        self.bottleneck = torch.utils.checkpoint(self.bottleneck)
        for module in self.ups:
            module = torch.utils.checkpoint(module)
  
class ResidualUNet(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=1,
        features=[64, 128, 256, 512],
        bilinear=False
    ):
        super().__init__()
        self.bilinear = bilinear

        # Downsampling path
        self.downs = nn.ModuleList()
        for feature in features:
            self.downs.append(ResidualDoubleConv(in_channels, feature))
            in_channels = feature

        # Bottleneck
        self.bottleneck = ResidualDoubleConv(features[-1], features[-1] * 2)

        # Upsampling path
        self.ups = nn.ModuleList()
        for feature in reversed(features):
            if bilinear:
                self.ups.append(nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                    nn.Conv2d(feature * 2, feature, kernel_size=1)
                ))
            else:
                self.ups.append(
                    nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
                )
            self.ups.append(ResidualDoubleConv(feature * 2, feature))

        # Final output layer
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Down path
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = F.max_pool2d(x, kernel_size=2)

        # Bottleneck
        x = self.bottleneck(x)

        # Reverse skip connections for upsampling
        skip_connections = skip_connections[::-1]

        # Up path
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            # Resize if shapes differ
            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:], mode='bilinear', align_corners=True)

            # Concatenate and apply ResidualDoubleConv
            x = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](x)

        return self.final_conv(x)
  
def progressive_unfreeze(model, epoch: int, unfreeze_every: int = 10, weight_avg_factor: float = 0.5):
    """
    Progressive unfreezing for models with an encoder (e.g., UNet).
    Unfreezes encoder blocks gradually as training progresses.
    """
    if hasattr(model, 'encoder'):
        blocks = model.encoder
    else:
        print("No encoder attribute found – skipping unfreeze.")
        return

    num_blocks = len(blocks)
    num_unfrozen = min(epoch // unfreeze_every, num_blocks)

    for i, block in enumerate(blocks):
        trainable = i >= (num_blocks - num_unfrozen)
        for p in block.parameters():
            if trainable:
                # Smooth transition with weight averaging
                if hasattr(p, 'data') and hasattr(p, 'grad'):
                    p.data = weight_avg_factor * p.data + (1 - weight_avg_factor) * p.data.clone()
            p.requires_grad_(trainable)

    print(f"Unfroze last {num_unfrozen}/{num_blocks} encoder blocks at epoch {epoch}.")
    
def partial_load_unet(src: nn.Module, tgt: nn.Module):
    """
    Copy weights from src→tgt:
     - exact‐shape params are copied wholesale
     - mismatched params get the first min(src_dim, tgt_dim) channels copied
     - log any partial copies for later inspection
    """
    src_sd = src.state_dict()
    tgt_sd = tgt.state_dict()
    mismatches = {}

    for name, tgt_param in tgt_sd.items():
        if name not in src_sd:
            # entirely new tensor in target
            continue

        src_param = src_sd[name]
        if src_param.shape == tgt_param.shape:
            # direct copy
            tgt_param.copy_(src_param)
        else:
            # figure out how many channels/features overlap
            min_dims = tuple(min(s, t) for s, t in zip(src_param.shape, tgt_param.shape))
            slices = tuple(slice(0, d) for d in min_dims)

            # copy only the overlapping subtensor
            tgt_param[slices].copy_(src_param[slices])

            mismatches[name] = {
                "src_shape": src_param.shape,
                "tgt_shape": tgt_param.shape,
                "copied": min_dims
            }

    # write back into the model
    tgt.load_state_dict(tgt_sd)
    import logging
    if mismatches:
        logging.getLogger(__name__).info(
            f"Partial-loaded {len(mismatches)} tensors:\n" +
            "\n".join(f"  {n}: {v}" for n,v in mismatches.items())
        )
    return mismatches