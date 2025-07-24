from libraries.utils import *
from libraries.setup import *
from libraries.torch_utils import *

def ffactors_gpu(px, py, energy=None, zs=None, pv=None, fresnel_number=None, device='cpu',output = 'Tensor'):
    import torch
    from torch.fft import fftfreq as torch_fftfreq
    
    # Set up device
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # Compute Fresnel number if not provided
    if fresnel_number is None:
        fresnel_number = fresnel_calc(energy, zs, pv)
    
    # Convert fresnel_number to a PyTorch tensor
    # fresnel_number = torch.tensor(fresnel_number, dtype=torch.float32, device=device)
    
    # Precompute frequency grids
    freq_x = torch_fftfreq(px, dtype=torch.float32).to(device)
    freq_y = torch_fftfreq(py, dtype=torch.float32).to(device)
    xi, eta = torch.meshgrid(freq_x, freq_y, indexing='ij')
    xi_eta_sum = (xi ** 2 + eta ** 2).to(device)  # Precompute the shared term
    
    # Precompute frequency prefactors
    frequ_prefactors = (2 * torch.pi / fresnel_number).to(device)
    
    # Compute h using broadcasting
    h = torch.exp(-1j * frequ_prefactors[:, None, None] * xi_eta_sum[None, :, :] / 2).to(torch.complex64)
    # h = h.transpose(1,2)
    if output == 'numpy':
        h = tensor_to_np(h)
    # Return single array if only one Fresnel number, else full array
    return h[0] if h.shape[0] == 1 else h

def ffactors_jax(px, py, energy=None, zs=None, pv=None, fresnel_number=None):
    import jax
    import jax.numpy as jnp
    # Compute Fresnel number if not provided
    if fresnel_number is None:
        fresnel_number = fresnel_calc(energy, zs, pv)

    # Convert Fresnel number to JAX array
    fresnel_number = jnp.asarray(fresnel_number, dtype=jnp.float32)

    # Compute frequency grids
    freq_x = jnp.fft.fftfreq(px).astype(jnp.float32)
    freq_y = jnp.fft.fftfreq(py).astype(jnp.float32)
    xi, eta = jnp.meshgrid(freq_x, freq_y, indexing='ij')
    xi_eta_sum = (xi ** 2 + eta ** 2).astype(jnp.float32)  # Precompute the shared term

    # Precompute frequency prefactors
    frequ_prefactors = (2 * jnp.pi / fresnel_number).astype(jnp.float32)

    # Compute h using broadcasting
    h = jnp.exp(-1j * frequ_prefactors[:, None, None] * xi_eta_sum[None, :, :] / 2).astype(jnp.complex64)

    # Return single array if only one Fresnel number, else full array
    return h[0] if h.shape[0] == 1 else h

def get_wavefield(phase = None, attenuation = None, dsf = 1, pad = 2, mode = 'reflect', value = 'median', cut = None, horizontally = True, vertically = True):
        #resize the phase and attenuation images by the downsampling factor
        phase = torch_reshape(phase)
        phase = resize_with_diff_interpolation(phase, (int(phase.shape[2]/dsf), int(phase.shape[3]/dsf)), 'NEAREST_EXACT', 'tensor')
        attenuation = torch_reshape(attenuation)
        attenuation = resize_with_diff_interpolation(attenuation, (int(attenuation.shape[2]/dsf), int(attenuation.shape[3]/dsf)), 'NEAREST_EXACT', 'tensor')
        
        transmission_function = torch.complex(-attenuation, phase)
        wavefield = torch.exp(transmission_function)
        #now padding, 
        _, _, shape_x, shape_y = wavefield.shape
        if mode == 'constant':
            if value != '10':
                value_real = set_value(value, wavefield.real)
                value_imag = set_value(value, wavefield.imag)
            else:
                value_real = set_value(1, wavefield.real)
                value_imag = set_value(0, wavefield.imag)
            if cut == None:
                wavefield_real = F.pad(wavefield.real, (int((shape_y*pad - shape_y)/2), int((shape_y *pad- shape_y)/2), int((shape_x *pad- shape_x)/2), int((shape_x *pad- shape_x)/2)), mode = mode, value = value_real)
                wavefield_imag = F.pad(wavefield.imag, (int((shape_y*pad - shape_y)/2), int((shape_y *pad- shape_y)/2), int((shape_x *pad- shape_x)/2), int((shape_x *pad- shape_x)/2)), mode = mode, value = value_imag)
            else:
                wavefield_real = pad_on_all_sides_by_constant_torch(wavefield.real, cut = pad, value = value_real, horizonatally = horizontally, vertically = vertically)
                wavefield_imag = pad_on_all_sides_by_constant_torch(wavefield.imag, cut = pad, value = value_imag, horizonatally = horizontally, vertically = vertically)
            wavefield = torch.complex(wavefield_real, wavefield_imag)
        else:
            if pad is None:
                pad = 2
            if cut == None:
                wavefield = F.pad(wavefield, (int((shape_y*pad - shape_y)/2), int((shape_y *pad- shape_y)/2), int((shape_x *pad- shape_x)/2), int((shape_x *pad- shape_x)/2)), mode)
            else:
                wavefield = pad_on_all_sides_by_reflection_torch(wavefield, cut = pad, horizonatally = horizontally, vertically = vertically)
        
        return wavefield
    
class Fresnel_propagation(nn.Module):
    def __init__(self, **kwargs) -> None:
        kwargs = prepare_dict(**kwargs)
        super().__init__()     
        for key, value in kwargs.items():
            self.__setattr__(key, value)
        assert self.phase is not None or self.attenuation is not None, "phase or attenuation are not provided"
        self.wavefield = self.get_wavefield()
        _, _, new_shape_x, new_shape_y = self.wavefield.shape
        
        self.fresnel_number = fresnel_calc(self.energy, self.distance_sample_detector, self.detector_pixel_size) if self.fresnel_number is None else self.fresnel_number
        self.base_matrix = torch_reshape(get_base_coeff(self.wavefield)) if self.wavefield is not None else None
        self.fresnel_factor = torch_reshape(ffactors(px = new_shape_x, py = new_shape_y, energy = self.energy, zs = self.distance_sample_detector, pv = self.detector_pixel_size, fresnel_number= self.fresnel_number))
        # self.fresnel_factor = torch.fft.fftshift(self.fresnel_factor)
        # self.fresnel_factor = torch.transpose(self.fresnel_factor, 1, 2)
        self.image = self.forward() if self.image == None else self.image
        # self.transformed_images = transform(self.image, self.transform_type, self.transform_factor)
        
        self.cut = None if 'cut' not in kwargs.keys() else kwargs['cut']
        self.horizontally = True if 'horizontally' not in kwargs.keys() else kwargs['horizontally']
        self.vertically = True if 'vertically' not in kwargs.keys() else kwargs['vertically']
        
    def get_wavefield(self, phase = None, attenuation = None, dsf = 1, pad = 2, mode = None, value = None):
        pad = self.pad if pad is None else pad
        mode = self.mode if mode is None else mode
        value = self.value if value is None else value
        phase = self.phase if phase is None else phase
        attenuation = self.attenuation if attenuation is None else attenuation

        #resize the phase and attenuation images by the downsampling factor
        phase = torch_reshape(phase)
        phase = resize_with_diff_interpolation(phase, (int(phase.shape[2]/dsf), int(phase.shape[3]/dsf)), 'NEAREST_EXACT', 'tensor')
        attenuation = torch_reshape(attenuation)
        attenuation = resize_with_diff_interpolation(attenuation, (int(attenuation.shape[2]/dsf), int(attenuation.shape[3]/dsf)), 'NEAREST_EXACT', 'tensor')
        
        transmission_function = torch.complex(-attenuation, phase)
        wavefield = torch.exp(transmission_function)
        #now padding, 
        _, _, shape_x, shape_y = wavefield.shape
        if mode == 'constant':
            if value != '10':
                value_real = set_value(value, wavefield.real)
                value_imag = set_value(value, wavefield.imag)
            else:
                value_real = set_value(1, wavefield.real)
                value_imag = set_value(0, wavefield.imag)
            if self.cut == None:
                wavefield_real = F.pad(wavefield.real, (int((shape_y*pad - shape_y)/2), int((shape_y *pad- shape_y)/2), int((shape_x *pad- shape_x)/2), int((shape_x *pad- shape_x)/2)), mode = mode, value = value_real)
                wavefield_imag = F.pad(wavefield.imag, (int((shape_y*pad - shape_y)/2), int((shape_y *pad- shape_y)/2), int((shape_x *pad- shape_x)/2), int((shape_x *pad- shape_x)/2)), mode = mode, value = value_imag)
            else:
                wavefield_real = pad_on_all_sides_by_constant_torch(wavefield.real, cut = pad, value = value_real, horizonatally = self.horizontally, vertically = self.vertically)
                wavefield_imag = pad_on_all_sides_by_constant_torch(wavefield.imag, cut = pad, value = value_imag, horizonatally = self.horizontally, vertically = self.vertically)
            wavefield = torch.complex(wavefield_real, wavefield_imag)
        else:
            if pad is None:
                pad = 2
            if self.cut == None:
                wavefield = F.pad(wavefield, (int((shape_y*pad - shape_y)/2), int((shape_y *pad- shape_y)/2), int((shape_x *pad- shape_x)/2), int((shape_x *pad- shape_x)/2)), mode)
            else:
                wavefield = pad_on_all_sides_by_reflection_torch(wavefield, cut = pad, horizonatally = self.horizontally, vertically = self.vertically)
        
        return wavefield

    def forward(self, phase = None, attenuation = None, pad = None, mode = None, value = None, fresnel_factor = None, wavefield = None, **kwargs):
        """
        If the phase and attenuation are provided, it uses them to create the wavefield. Otherwise, it uses the wavefield provided.
        this function propagates the wavefield from the sample to the detector. If fresnel_factor is not provided, it calculates it.

        You can use the **dataloader.kwargs() as an input to this function:
        """
        phase = self.phase if phase is None else torch_reshape(phase)
        attenuation = self.attenuation if attenuation is None else torch_reshape(attenuation)
        _, _, shape_x, shape_y = phase.shape
        wavefield = self.get_wavefield(phase = phase, attenuation = attenuation, dsf = self.dsf, pad = pad, mode = mode, value = value) if wavefield is None else wavefield
        
        assert wavefield is not None and 'complex' in str(wavefield.dtype), "wavefield is not provided or is not complex"
        fresnel_factor = self.fresnel_factor if fresnel_factor is None else fresnel_factor
        
        self.propagated_wavefield = torch.fft.ifft2(fresnel_factor * torch.fft.fft2(wavefield))
        intenstiy = torch.abs(self.propagated_wavefield)**2
        
        if self.cut == None:
            self.intensity = intenstiy[:, :, int((intenstiy.shape[2] - shape_x)/2):int((intenstiy.shape[2] + shape_x)/2), int((intenstiy.shape[3] - shape_y)/2):int((intenstiy.shape[3] + shape_y)/2)]
        else:
            self.intensity = unpad_torch(intenstiy, (shape_x, shape_y))
            
        return self.intensity

def create_wavefield(phase, attenuation, dsf=1, pad=2, mode='reflect', value='median', cut=None, horizontally=True, vertically=True):
    assert phase is not None and attenuation is not None, "phase and attenuation must not be None"
    phase = torch_reshape(phase)
    phase = resize_with_diff_interpolation(phase, (int(phase.shape[2]/dsf), int(phase.shape[3]/dsf)), 'NEAREST_EXACT', 'tensor')
    attenuation = torch_reshape(attenuation)
    attenuation = resize_with_diff_interpolation(attenuation, (int(attenuation.shape[2]/dsf), int(attenuation.shape[3]/dsf)), 'NEAREST_EXACT', 'tensor')
    
    transmission_function = torch.complex(-attenuation, phase)
    wavefield = torch.exp(transmission_function)
    _, _, shape_x, shape_y = wavefield.shape
    if mode == 'constant':
        if value != '10':
            value_real = set_value(value, wavefield.real)
            value_imag = set_value(value, wavefield.imag)
        else:
            value_real = set_value(1, wavefield.real)
            value_imag = set_value(0, wavefield.imag)
        if cut is None:
            wavefield_real = F.pad(wavefield.real, (int((shape_y*pad - shape_y)/2), int((shape_y *pad- shape_y)/2), int((shape_x *pad- shape_x)/2), int((shape_x *pad- shape_x)/2)), mode=mode, value=value_real)
            wavefield_imag = F.pad(wavefield.imag, (int((shape_y*pad - shape_y)/2), int((shape_y *pad- shape_y)/2), int((shape_x *pad- shape_x)/2), int((shape_x *pad- shape_x)/2)), mode=mode, value=value_imag)
        else:
            wavefield_real = pad_on_all_sides_by_constant_torch(wavefield.real, cut=pad, value=value_real, horizonatally=horizontally, vertically=vertically)
            wavefield_imag = pad_on_all_sides_by_constant_torch(wavefield.imag, cut=pad, value=value_imag, horizonatally=horizontally, vertically=vertically)
        wavefield = torch.complex(wavefield_real, wavefield_imag)
    else:
        if pad is None:
            pad = 2
        if cut is None:
            wavefield = F.pad(wavefield, (int((shape_y*pad - shape_y)/2), int((shape_y *pad- shape_y)/2), int((shape_x *pad- shape_x)/2), int((shape_x *pad- shape_x)/2)), mode)
        else:
            wavefield = pad_on_all_sides_by_reflection_torch(wavefield, cut=pad, horizonatally=horizontally, vertically=vertically)
    return wavefield
