from visualize import *
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
        
    def get_wavefield(self, phase = None, attenuation = None, downsampling_factor = 1, pad = None, mode = None, value = None):
        pad = self.pad if pad is None else pad
        mode = self.mode if mode is None else mode
        value = self.value if value is None else value
        phase = self.phase if phase is None else phase
        attenuation = self.attenuation if attenuation is None else attenuation

        #resize the phase and attenuation images by the downsampling factor
        phase = torch_reshape(phase)
        phase = resize_with_diff_interpolation(phase, (int(phase.shape[2]/downsampling_factor), int(phase.shape[3]/downsampling_factor)), 'NEAREST_EXACT', 'tensor')
        attenuation = torch_reshape(attenuation)
        attenuation = resize_with_diff_interpolation(attenuation, (int(attenuation.shape[2]/downsampling_factor), int(attenuation.shape[3]/downsampling_factor)), 'NEAREST_EXACT', 'tensor')
        
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
        wavefield = self.get_wavefield(phase = phase, attenuation = attenuation, downsampling_factor = self.downsampling_factor, pad = pad, mode = mode, value = value) if wavefield is None else wavefield
        
        assert wavefield is not None and 'complex' in str(wavefield.dtype), "wavefield is not provided or is not complex"
        fresnel_factor = self.fresnel_factor if fresnel_factor is None else fresnel_factor
        
        self.propagated_wavefield = torch.fft.ifft2(fresnel_factor * torch.fft.fft2(wavefield))
        intenstiy = torch.abs(self.propagated_wavefield)**2
        
        if self.cut == None:
            self.intensity = intenstiy[:, :, int((intenstiy.shape[2] - shape_x)/2):int((intenstiy.shape[2] + shape_x)/2), int((intenstiy.shape[3] - shape_y)/2):int((intenstiy.shape[3] + shape_y)/2)]
        else:
            self.intensity = unpad_torch(intenstiy, (shape_x, shape_y))
            
        return self.intensity
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return super().__call__(*args, **kwds)
