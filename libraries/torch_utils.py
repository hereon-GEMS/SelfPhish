from libraries.setup import *
import torch
from torchvision.transforms.functional import adjust_contrast, adjust_brightness, InterpolationMode, resize
from torchvision.transforms import Resize
# import torchvision.transforms.functional as F
import torch.nn as nn
import torch.nn.functional as F

def filter_values(image, min_value = None, max_value = None, replace = None, replace_value = None):
    #replace nan values with the min value
    image = np.nan_to_num(image)
    if min_value is not None:
        image[image < min_value] = min_value 
    if max_value is not None:
        image[image > max_value] = max_value
    if replace is not None:
        image[image == replace] = replace_value
    return image

def torch_reshape(image, complex = False):
    #if it's tensor and of shape 4, return the image
    if 'tensor' in str(type(image)):
        if len(image.shape) == 4:
            #make sure it is B, C, H, W
            B, C, H, W = image.shape
            if H < B or H < C or W < B or W < C:
                #arrange increasing order and idx
                arranged, idx = torch.sort(torch.tensor([B, C, H, W]))
                #if the order is not B, C, H, W
                if arranged[1] == 1:
                    #rearrange the image
                    image = image.permute(idx)
                elif arranged[1] == 2:
                    image = image.permute([idx[1], idx[0], idx[2], idx[3]])
            elif C > B:
                if C != 3:
                    image = image.permute(1, 0, 2, 3)
            return image
                    
                

        if not torch.equal(arranged, torch.tensor([1, 0, 2, 3])):
            #rearrange the image
            image = image.permute(idx)
            
            return image
        elif len(image.shape) == 5:
            image = image.squeeze(1)
            return image
    
    if type(image) is list:
        #use joblib to parallelize the process
        if len(image) > 30:
            n_jobs = 30
        # else:
        #     n_jobs = 5
            image = Parallel(n_jobs=n_jobs)(delayed(torch.from_numpy)(image[i]) for i in range(len(image)))
        else:
            try:
                image = [torch.from_numpy(image[i]) for i in range(len(image))]
            except:
                image = [image[i] for i in range(len(image))]
        image = torch.stack(image)
    if 'numpy' in str(type(image)):
            image = torch.from_numpy(image)
    
    if len(image.shape) == 2:
        image = image.unsqueeze(0)
        image = image.unsqueeze(0)
    elif len(image.shape) == 3:
        image = image.unsqueeze(0)    
    elif len(image.shape) > 5:
        image = image.squeeze(1)
        return image
    
    #rearrange the dimension of the image [batch, channel, x, y]
    image = image.permute(1, 0, 2, 3)
    if complex or image.dtype == torch.complex64 or image.dtype == torch.complex128:
        image = image.type(torch.complex64)
    else:
        image = image.type(torch.float32)
    return image
    

def get_base_coeff(image):
    image = torch_reshape(image)
    _, _, shape_x, shape_y = image.shape
    freq_1 = torch.fft.fftfreq(shape_x)
    freq_2 = torch.fft.fftfreq(shape_y)
    xi, eta = torch.meshgrid(freq_1, freq_2)
    xi = xi.type(torch.float32)
    eta = eta.type(torch.float32)
    return torch.exp((xi**2 + eta**2)/2)

def torch_reshape(image, complex = False):
    #if it's tensor and of shape 4, return the image
    if 'tensor' in str(type(image)):
        if len(image.shape) == 4:
            return image
        elif len(image.shape) == 5:
            image = image.squeeze(1)
            return image
    
    if type(image) is list:
        try:
            image = [torch.from_numpy(image[i]) for i in range(len(image))]        
            image = torch.stack(image)
        except:
            image = [image[i] for i in range(len(image))]
            image = np.stack(image)
            image = image.astype(np.float32)
            image = torch.from_numpy(image)
    if 'numpy' in str(type(image)):
            image = torch.from_numpy(image)
    
    if len(image.shape) == 2:
        image = image.unsqueeze(0).unsqueeze(0)
    elif len(image.shape) == 3:
        image = image.unsqueeze(0)    
    elif len(image.shape) > 5:
        image = image.squeeze(1)
        return image
    
    
    #rearrange the dimension of the image [batch, channel, x, y]
    if len(image.shape) == 4 and image.shape[0] < image.shape[1] and image.shape[1]!=3:
        image = image.permute(1, 0, 2, 3)
    
    if len(image.shape) == 5:
        image = image.squeeze(1)

    if complex or image.dtype == torch.complex64 or image.dtype == torch.complex128:
        image = image.type(torch.complex64)
    else:
        image = image.type(torch.float32)
        
    if image.shape[0] == 1 and image.shape[1] > 1:
        #swap the first and second dimension
        image = image.permute(1, 0, 2, 3)
    return image 

def torchh_nortomo(img):
    img = torch_reshape(img)
    mean_tmp = torch.mean(img)
    std_tmp = torch.std(img)
    img = (img - mean_tmp) / std_tmp
    img = (img - torch.min(img)) / (torch.max(img) - torch.min(img))
    return img

def torch_norm(image):
    image = torch_reshape(image)
    image = (image - torch.mean(image))/torch.std(image)
    return image

def torchnor_phase(image):
    image = torch_reshape(image)
    image = (image - torch.mean(image))/torch.std(image)
    if torch.max(image) != 0:
        image = image / torch.max(image)
    return image

def torch_contrast(image, contrast_factor = 0.02):
    if type(image) is not torch.Tensor:
        image = torch_reshape(image)
    image = adjust_contrast(image, contrast_factor) 
    return image

def torch_brightness(image, brightness_factor = 0.02):
    image = torch_reshape(image)
    image = adjust_brightness(image, brightness_factor)
    return image

def torch_detector(image):
    image = torch_reshape(image, complex = True)
    image = torch.abs(image)**2
    return image



def transform(image, transform_type = None, factor = None):
    if transform_type is None:
        transform_type = 'reshape'
    if factor is None:
        factor = 0.5
    if transform_type == 'reshape':
        image = torch_reshape(image)
    elif transform_type == 'normalize':
        image = torchnor_phase(image)
    elif transform_type == 'normalize_1':
        image = 1 - torchnor_phase(image)
    elif transform_type == 'norm':
        image = torch_norm(image)
    elif transform_type == 'contrast':
        image = torch_reshape(image)
        image = torch_contrast(image, factor)
    elif transform_type == 'contrast_normalize':
        image = torch_reshape(image)
        image = torch_contrast(image, factor)
        image = torchnor_phase(image)
    elif transform_type == 'contrast_norm':
        image = torch_reshape(image)
        image = torch_contrast(image, factor)
        image = torch_norm(image)
    elif transform_type == 'brightness':
        image = torch_reshape(image)
        image = torch_brightness(image, factor)
    elif transform_type == 'brightness_normalize':
        image = torch_reshape(image)
        image = torch_brightness(image, factor)
        image = torchnor_phase(image)
    elif transform_type == 'brightness_norm':
        image = torch_reshape(image)
        image = torch_brightness(image, factor)
        image = torch_norm(image)
    elif transform_type == 'fourier':
        image = torch_reshape(image)
        image = torch.fft.fft2(image)
    elif transform_type =='minmax':
        image = torch_reshape(image)
        image = (image - torch.min(image))/(torch.max(image) - torch.min(image))
    elif transform_type == '0to1':
        image = torch_reshape(image)
        image = (image - torch.mean(image))/torch.std(image)
        image = (image - torch.min(image))/(torch.max(image) - torch.min(image))
    elif transform_type == 'log':
        image = torch_reshape(image)
        image = torch.log(image)
    else:
        image = torch_reshape(image)
    return image

def join_dict(dict2, base_dict, trans = False):
    res = base_dict.copy()
    res.update(dict2)
    if trans:
        res['image'] = torch_reshape(res['path'])
        res['transformed_images'] = transform(res['path'], res['transform_type'], res['transform_factor'])
    return res

def similar_terms_converter(**kwargs):
    similar_terms = [
        ['path', 'images', 'image', 'paths','image', 'i_inputs', 'i_input', 'hologram', 'intensity'],
        ['file_type', 'file_types', 'filetype', 'filetypes'],
        ['idx', 'indices', 'index'],
        ['energy', 'energy_kev'], 
        ['lam', 'lamda', 'wavelength', 'wave_length'],
        ['phase', 'phase_image'],
        ['attenuation', 'attenuation_image'],
        ['detector_pixel_size', 'pv'],
        ['distance_sample_detector', 'z'],
        ['fresnel_number', 'fresnel_number', 'fresnelnumbers', 'fresnelnumbers'],
        ['fresnel_factor', 'ffs', 'frensel_factors', 'fresnelfactor'],
        ['pad', 'pad_value', 'magnification_factor', 'upscale'],
        ['downsampling_factor'],
        ['mode', 'pad_mode'],
        ['experiment_name'],
        ['task', 'method'],
        ['alpha', 'alpha_value'],
        ['abs_ratio'],
        ['delta_beta', 'delta_beta_value'],
        ['shape_x', 'px'],
        ['shape_y', 'py'],
        ]
    for i, terms in enumerate(similar_terms):
        kwargs[terms[0]] = None if terms[0] not in kwargs.keys() else kwargs[terms[0]]
        for term in terms:
            if term in kwargs.keys()  and term != terms[0] and kwargs[term] is not None:
                kwargs[terms[0]] = kwargs[term]
                break
    return kwargs
    

def prepare_dict(**kwargs):
    similar_terms = [
        ['path', 'images', 'image', 'paths','image', 'i_inputs', 'i_input', 'hologram', 'intensity'],
        ['file_type', 'file_types', 'filetype', 'filetypes'],
        ['idx', 'indices', 'index'],
        ['energy', 'energy_kev'], 
        ['lam', 'lamda', 'wavelength', 'wave_length'],
        ['phase', 'phase_image'],
        ['attenuation', 'attenuation_image'],
        ['detector_pixel_size', 'pv'],
        ['distance_sample_detector', 'z'],
        ['fresnel_number', 'fresnel_number', 'fresnelnumbers', 'fresnelnumbers'],
        ['fresnel_factor', 'ffs', 'frensel_factors', 'fresnelfactor'],
        ['pad', 'pad_value', 'magnification_factor', 'upscale'],
        ['downsampling_factor'],
        ['mode', 'pad_mode'],
        ['experiment_name'],
        ['task', 'method'],
        ['alpha', 'alpha_value'],
        ['abs_ratio'],
        ['delta_beta', 'delta_beta_value'],
        ['shape_x', 'px'],
        ['shape_y', 'py'],
        ]

    dictionary = {'image':None, 'path':None, 'image_path': None, 'pad': 1, 'mode': 'reflect', 'task': 'learn_phase', 'alpha': 1e-8, 'delta_beta': 1e1, 'idx': None, 'file_type': 'tif', 'save_path': os.getcwd() + '/results/', 'idx': 0, 'save': False, 'transform_factor': 0.7, 'transform_type': 'reshape', 'save_format': 'tif', 'save_all': False, 'downsampling_factor': 1, 'fresnel_number': None, 'detector_pixel_size': None, 'distance_sample_detector': None, 'energy': None, 'phase': None, 'attenuation': None}
    optional_kwargs = {"fresnel_number": None, "distance_sample_detector": None, "detector_pixel_size": None, "energy": None, 'pad': 2, "transform_type": "reshape", "transform_factor": 1, "mode": "reflect", "value": "mean", "downsampling_factor": 1, "wave_field": None, 'fresnel_factor': None, 'device': 'cuda:0', 'cut':None, 'horizontally': True, 'vertically': True}
    dictionary.update(optional_kwargs)
    dictionary.update(kwargs)
    kwargs = dictionary

    for i, terms in enumerate(similar_terms):
        kwargs[terms[0]] = None if terms[0] not in kwargs.keys() else kwargs[terms[0]]
        for term in terms:
            if term in kwargs.keys()  and term != terms[0] and kwargs[term] is not None:
                kwargs[terms[0]] = kwargs[term]
                break
    
    kwargs['downsampling_factor'] = 1 if 'downsampling_factor' not in kwargs.keys() else kwargs['downsampling_factor']

    if kwargs['idx'] is not None:
        kwargs['idx'] = [kwargs['idx']] if type(kwargs['idx']) is not list else kwargs['idx']

    assert kwargs['path'] is not None or kwargs['phase'] is not None or kwargs['attenuation'] is not None, "path, phase or attenuation are not provided"
    kwargs['image'], kwargs['image_path'] = get_image(kwargs['path'], kwargs['idx'], kwargs['file_type']) if kwargs['path'] is not None else kwargs['image'], kwargs['image_path']
    
    try:
        kwargs['image'] = torch_reshape(kwargs['image']) if kwargs['image'] is not None else None
    except:
        try:
            kwargs['image'] = torch_reshape(kwargs['image'][0]) if kwargs['image'] is not None else None
        except:
            kwargs['image'] = torch_reshape(kwargs['image'][0][0]) if kwargs['image'] is not None else None
    if kwargs['image'] is not None and kwargs['phase'] is None and kwargs['attenuation'] is None:
        kwargs['phase'] = kwargs['image']
        kwargs['attenuation'] = kwargs['image']
    else:
        try:
            kwargs['phase'], kwargs['phase_path'] = get_image(kwargs['phase'], kwargs['idx'], kwargs['file_type']) if kwargs['phase'] is not None else kwargs['phase'], kwargs['phase_path']
        except:
            try:
                kwargs['phase'] = torch_reshape(kwargs['phase'])
                kwargs['attenuation'] = torch_reshape(kwargs['attenuation'])
            except:
                pass


    if kwargs['image'] is not None:
        _, _, shape_x, shape_y = torch_reshape(kwargs['image']).shape
        # kwargs['image'] = T.Resize((int(shape_x//kwargs['downsampling_factor']), int(shape_y//kwargs['downsampling_factor'])), antialias=True)(kwargs['image'])    
        kwargs['image'] = resize_with_diff_interpolation(kwargs['image'], (int(shape_x/kwargs['downsampling_factor']), int(shape_y/kwargs['downsampling_factor'])), 'NEAREST_EXACT', 'tensor')
        _, _, kwargs['shape_x'], kwargs['shape_y'] = kwargs['image'].shape if kwargs['image'] is not None else kwargs['phase'].shape
        # print('after resize image shape', kwargs['image'].shape)
    if kwargs['phase'] is not None:
        _, _, shape_x, shape_y = kwargs['phase'].shape
        # kwargs['phase'] = T.Resize((int(shape_x//kwargs['downsampling_factor']), int(shape_y//kwargs['downsampling_factor'])), antialias=True)(kwargs['phase'])
        kwargs['phase'] = resize_with_diff_interpolation(kwargs['phase'], (int(shape_x/kwargs['downsampling_factor']), int(shape_y/kwargs['downsampling_factor'])), 'NEAREST_EXACT', 'tensor')
    if kwargs['attenuation'] is not None:
        _, _, shape_x, shape_y = kwargs['attenuation'].shape
        kwargs['attenuation'] = resize_with_diff_interpolation(kwargs['attenuation'], (int(shape_x/kwargs['downsampling_factor']), int(shape_y/kwargs['downsampling_factor'])), 'NEAREST_EXACT', 'tensor')
        # kwargs['attenuation'] = T.Resize((int(shape_x//kwargs['downsampling_factor']), int(shape_y//kwargs['downsampling_factor'])), antialias=True)(kwargs['attenuation'])

    if kwargs['fresnel_number'] is not None:
        kwargs['fresnel_number'] = kwargs['fresnel_number'] * kwargs['downsampling_factor']**2
        # kwargs['fresnel_factor'] = ffactors(px = kwargs['shape_x'], py = kwargs['shape_y'], fresnel_number= kwargs['fresnel_number'])
    if kwargs['detector_pixel_size'] is not None:
        kwargs['detector_pixel_size'] = kwargs['detector_pixel_size'] * kwargs['downsampling_factor']
    kwargs['downsampling_factor'] = 1

    assert kwargs['phase'] is not None or kwargs['attenuation'] is not None, "phase or attenuation are not provided"
    _, _, kwargs['shape_x'], kwargs['shape_y'] = kwargs['phase'].shape if kwargs['phase'] is not None else kwargs['attenuation'].shape
    kwargs['shape'] = (kwargs['shape_x'], kwargs['shape_y'])
    kwargs['full_shape'] = kwargs['phase'].shape if kwargs['phase'] is not None else kwargs['attenuation'].shape
    return kwargs

def tensor_to_np(tensor):
    if type(tensor) is not list:
        if len(tensor.shape) <= 2:
            try:
                return tensor.detach().cpu().numpy()
            except:
                return tensor.numpy()
        elif len(tensor.shape) == 3:
            try:
                return tensor.detach().cpu().numpy()[0,:,:]
            except:
                return tensor.numpy()[0,:,:]
        elif len(tensor.shape) == 4:
            try:
                return np.array([t.detach().cpu().numpy()[0,:,:] for t in tensor]) if tensor.shape[0] > 1 else tensor[0,0,:,:].detach().cpu().numpy()
            except:
                return np.array([t.numpy()[0,:,:] for t in tensor]) if tensor.shape[0] > 1 else tensor[0,0,:,:].numpy()
    else:
        return [tensor_to_np(t) for t in tensor]
            
def to_device(x, device):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    try:
        return x.to(device)
    except:
        return torch_reshape(x).to(device)

def torch_zero_at_boundary(img, width=2):
    img[:, :, :width] = 0
    img[:, :, -width:] = 0
    img[:, :, :, :width] = 0
    img[:, :, :, -width:] = 0
    return img


def torch_noise_gaussian(image, std, mean = 0):
    image = torch_reshape(image)
    std = torch.tensor(std)
    mean = torch.tensor(mean)
    noise = torch.normal(mean, std, image.shape).to(image.device)
    return image + noise

def torch_noise_poisson(image, rescale_factor = 0.00001, generator = None):
    if rescale_factor == 0:
        rescale_factor += 1e-5
    image = torch_reshape(image)
    image = image * (1/rescale_factor)
    noise = torch.poisson(image, generator)
    noise = noise / (1/rescale_factor)
    return noise

def torch_noise_speckle(image, std, mean = 0):
    image = torch_reshape(image)
    std = torch.tensor(std)
    mean = torch.tensor(mean)
    noise = torch.normal(mean, std, image.shape).to(image.device)
    return image + image*noise

def inspect_img(*, img):
    plt.imshow(img, cmap='gray')
    print(f"Value of pixel with coordinates (14,9): {img[14, 9]}")
    
def resize_pytorch(*, img, output_size):
    img = resize(Image.fromarray(np.float64(img)), # Provide a PIL image rather than a Tensor.
                   size=output_size, 
                   interpolation=InterpolationMode.BICUBIC)
    img = np.asarray(img, dtype=np.float64) 
    # inspect_img(img=img)
    return img
    
def pos_neg_image(image, positive = None, reference_image = None, remove_extreme = False, retain_grad = False):
    """ 
    This function will transform the image into positive and negative image
    [False, True, 'linear_normalization', 'non_linear_normalization', 'match_histograms', 'to0', 'to1', 'relu', 'relu_inverted', 'sigmoid', 'tanh', 'softmax', 'log_softmax', 'exp', 'log', 'log1p', 'expm1', 'softplus', 'softsign', 'hardshrink', 'tanhshrink', 'softshrink', 'threshold', 'relu6', 'elu', 'celu', 'selu', 'gelu', 'rrelu', 'hardtanh', 'leakyrelu', 'leaky_relu_inverted', 'normalize', 'normalize_min_max', 'normalize_min_max_1']
    
    """
    
    image = torch_reshape(image)
    if positive == 'neg_abs'or positive == False:
        image = -1*( image + torch.abs(torch.min(image)))
    elif positive == 'neg':
        image = -1*(image - torch.min(image))
    
    elif positive is True or positive == 'pos':
        image = torch.max(image) - image
    
    elif positive == 'linear_normalization':
        if reference_image is not None:
            reference_image = torch_reshape(reference_image)
            min, max = torch.min(reference_image), torch.max(reference_image)
        else:
            min, max = 0, 1
        image = (image - torch.min(image))*(max-min)/(torch.max(image) - torch.min(image)) + min

    elif positive == 'non_linear_normalization':
        image = (image - torch.min(image))/(torch.max(image) - torch.min(image))
        # image = torch.tanh(image)
        # image = torch.nn.functional.softplus(image)
        image = torch.nn.functional.sigmoid(image)
    
    elif positive == 'match_histograms':
        assert reference_image is not None, "reference_image  must be provided"
        from skimage.exposure import match_histograms
        image = tensor_to_np(image)
        try:
            reference_image = tensor_to_np(reference_image)
        except:
            pass
        image = match_histograms(image, reference_image)
        image = torch_reshape(image)
    
    elif positive == 'to0':
        image = torch.max(image, torch.tensor(0))

    elif positive == 'to1':
        image = torch.max(image, torch.tensor(0))
        image = (image - torch.min(image))/(torch.max(image) - torch.min(image))
    elif positive == 'relu':
        image = torch.nn.functional.relu(image)
    elif positive == 'relu_inverted':
        image = -1*torch.nn.functional.relu(image)
    elif positive == 'sigmoid':
        image = torch.sigmoid(image)
    elif positive == 'tanh':
        image = torch.tanh(image)
    elif positive == 'softmax':
        image = torch.nn.functional.softmax(image)
    elif positive == 'log_softmax':
        image = torch.nn.functional.log_softmax(image)
    elif positive == 'exp':
        image = torch.exp(image)
    elif positive == 'log':
        image = torch.log(image)
    elif positive == 'log1p':
        image = torch.log1p(image)
    elif positive == 'expm1':
        image = torch.expm1(image)
    elif positive == 'softplus':
        image = torch.nn.functional.softplus(image)
    elif positive == 'softsign':
        image = torch.nn.functional.softsign(image)
    elif positive == 'hardshrink':
        image = torch.nn.functional.hardshrink(image)
    elif positive == 'tanhshrink':
        image = torch.nn.functional.tanhshrink(image)
    elif positive == 'softshrink':
        image = torch.nn.functional.softshrink(image)
    elif positive == 'threshold':
        image = torch.nn.functional.threshold(image, threshold=-10, value=0, inplace=True)
    elif positive == 'relu6':
        image = torch.nn.functional.relu6(image)
    elif positive == 'elu':
        image = torch.nn.functional.elu(image)
    elif positive == 'silu':
        image = torch.nn.Silu()(image)
    elif positive == 'celu':
        image = torch.nn.functional.celu(image)
    elif positive == 'selu':
        image = torch.nn.functional.selu(image)
    elif positive == 'gelu':
        image = torch.nn.functional.gelu(image, approximate='tanh')
    elif positive == 'gelu_inverted':
        image = -1 * torch.nn.functional.gelu(image)
    elif positive == 'rrelu':
        image = torch.nn.functional.relu(-image)
    elif positive == 'hardtanh':
        image = torch.nn.functional.hardtanh(image)
    elif positive == 'leakyrelu':
        image = torch.nn.functional.leaky_relu(image)
    elif positive == 'leaky_relu_inverted':
        image = torch.nn.functional.leaky_relu(-image)
    elif positive == 'normalize':
        image = (image - torch.mean(image))/torch.std(image)
    elif positive == 'normalize_1':
        image = (image - torch.mean(image))/torch.std(image)
        # image = image/torch.max(image)
        image = 1 - image
    elif positive == 'normalize_min_max':
        image = (image - torch.min(image))/(torch.max(image) - torch.min(image))
    elif positive == 'normalize_min_max_1':
        image = (image - torch.min(image))/(torch.max(image) - torch.min(image))
        image = 2*image - 1
    elif positive == 'pre_ssim':
        image = (image - torch.mean(image))/torch.std(image)
        image = (image - torch.min(image))/(torch.max(image) - torch.min(image))
    elif positive == 'gelu_inverted':
        image = -1*torch.nn.functional.gelu(image, approximate='tanh')    
        
    if remove_extreme and positive is not None and positive != 'relu':
        max = np.max(tensor_to_np(image)) - 10e-8
        min = np.min(tensor_to_np(image)) + 10e-8

        second_max = np.max(tensor_to_np(image[image!=max]))
        second_min = np.min(tensor_to_np(image[image!=min]))
        if np.abs(max -second_max) > np.abs(max-min):
            image = torch.nn.functional.threshold(image, threshold=max, value=second_max, inplace=True)
        if np.abs(min -second_min) > np.abs(max-min):
            image = torch.nn.functional.threshold(image, threshold=min, value=second_min, inplace=True)
    if retain_grad:
        image.retain_grad()
    return image

def set_value(value, image = None):
    image = torch_reshape(image)
    values = ['mean', 'median', 'min', 'max', 'zero', 'one', '10']
    if value in values:
        if value == 'mean':
            value = torch.mean(image).item()
        elif value == 'median':
            value = torch.median(image).item()
        elif value == 'min':
            value = torch.min(image).item()
        elif value == 'max':
            value = torch.max(image).item()
        elif value == 'zero':
            value = 0
        elif value == 'one':
            value = 1
        else:
            value = value
    
    return value
    
def pad_on_all_sides_by_constant_torch(image, cut = 2, value = 0, horizonatally = True, vertically = True):
    image = torch_reshape(image)
    value = set_value(value, image)
    # print('value', value)
    _, _, m,n = image.shape
    
    duplicate = image
    if horizonatally:
        left_side = to_device(torch_reshape(torch.ones((m, n//cut))*value), image.device)
        right_side = to_device(torch_reshape(torch.ones((m, n//cut))*value), image.device)
        # print(left_side.shape, image.shape)
        duplicate = torch.cat((left_side, image), dim = 3)
        # print(duplicate.shape)
        duplicate = torch.cat((duplicate, right_side), dim = 3)
    _, _, m,n = duplicate.shape
    if vertically:
        top_part = to_device(torch_reshape(torch.ones((m//cut, n))*value), image.device)
        bottom_part = to_device(torch_reshape(torch.ones((m//cut, n))*value), image.device)

        duplicate = torch.cat((top_part, duplicate), dim = 2)
        duplicate = torch.cat((duplicate, bottom_part), dim = 2)
    # print(duplicate.shape)
    #same device

    return duplicate

def pad_on_all_sides_by_reflection_torch(image, cut=2, value = None, horizonatally = True, vertically = True):
    image = torch_reshape(image)
    duplicate = image
    _, _, m,n = image.shape
    if horizonatally:
        left_side = to_device(torch_reshape(torch.flip(duplicate[:,:,:,:n//cut], [3])), image.device)
        right_side = to_device(torch_reshape(torch.flip(duplicate[:,:,:,n-n//cut:], [3])), image.device)
        duplicate = torch.cat((left_side, duplicate), dim = 3)
        duplicate = torch.cat((duplicate, right_side), dim = 3)
    _, _, m,n = duplicate.shape
    if vertically:
        top_part = to_device(torch_reshape(torch.flip(duplicate[:,:,:m//cut,:], [2])), image.device)
        bottom_part = to_device(torch_reshape(torch.flip(duplicate[:,:,m-m//cut:,:], [2])), image.device)
        duplicate = torch.cat((top_part, duplicate), dim = 2)
        duplicate = torch.cat((duplicate, bottom_part), dim = 2)
    # print(duplicate.shape)
    
    return duplicate

def unpad_torch(image, original_shape):
    image = torch_reshape(image)
    _, _, m,n = image.shape
    m_old,n_old = original_shape if len(original_shape) == 2 else original_shape[2:]
    range_n = n - n_old
    range_m = m - m_old
    return image[:, :, range_m//2:range_m//2+m_old, range_n//2:range_n//2+n_old]

def pad_multiple_times(im, N, pad_by = 2):
    shapes = []
    for i in range(N):
        im = torch_reshape(im)
        _, _, m, n = im.shape
        shapes.append(im.shape)
        # Fun.pad(wavefield.real, (int((shape_y*pad - shape_y)/2), int((shape_y *pad- shape_y)/2), int((shape_x *pad- shape_x)/2), int((shape_x *pad- shape_x)/2)), mode = mode, value = value_real)
        im = F.pad(im, (int((n*pad_by - n)/2), int((n *pad_by- n)/2), int((m *pad_by- m)/2), int((m *pad_by- m)/2),), mode = 'reflect')
    return im, shapes

def unpad_multiple_times(im, n, shapes):
    for i in range(n):
        im = unpad_torch(im, shapes[n-i-1])
    return im

def crop_on_all_sides(image, cut_x = 200, cut_y = 200, left = True, right = True, top = True, bottom = True, cut_shape = None, replace_value = None):
    """
    cut_shape: tuple of 4 values, (top, bottom, left, right)
    """
    image = torch_reshape(image)
    if replace_value is not None:
        replace_value = set_value(replace_value, image)
    # print('value', value)
    N, C, m,n = image.shape
    duplicate = image
    width = m - 2*cut_x
    height = n - 2*cut_y
    starting_place_x = cut_x
    starting_place_y = cut_y
    
    if cut_shape is None:
        subtract_bottom = starting_place_x if bottom else m
        subtract_top = starting_place_x + width if top else 0
        subtract_right = starting_place_y + height if left else n
        subtract_left = starting_place_y  if right else 0
        if replace_value is None:
            duplicate = duplicate[:, :, subtract_bottom:subtract_top, subtract_left:subtract_right]
        else:
            duplicate[:, :, :subtract_bottom, :] = replace_value
            duplicate[:, :, subtract_top:, :] = replace_value
            duplicate[:, :, :, :subtract_left] = replace_value
            duplicate[:, :, :, subtract_right:] = replace_value
            
        # print('cut shape:', subtract_top,':',subtract_bottom, subtract_left,':',subtract_right)
    else:
        duplicate = duplicate[:, :, cut_shape[0]:cut_shape[1], cut_shape[2]:cut_shape[3]]         
    return duplicate


def resize_with_diff_interpolation(img, size, interpolation = 'NEAREST_EXACT', out_type = 'np'):
    if type(img) is not torch.Tensor:
        img = torch_reshape(img)
    if interpolation == 'NEAREST' or 'nearest':
        new_image = (resize(img = (img), size = size, antialias=False, interpolation = InterpolationMode.NEAREST))
    elif interpolation == 'BOX' or 'box':
        new_image = (resize(img = (img), size = size, antialias=False, interpolation = InterpolationMode.BOX))
    elif interpolation == 'BILINEAR' or 'bilinear':
        new_image = (resize(img = (img), size = size, antialias=False, interpolation = InterpolationMode.BILINEAR))
    elif interpolation == 'HAMMING' or 'hamming':
        new_image = (resize(img = (img), size = size, antialias=False, interpolation = InterpolationMode.HAMMING))
    elif interpolation == 'BICUBIC' or 'bicubic':
        new_image = (resize(img = (img), size = size, antialias=False, interpolation = InterpolationMode.BICUBIC))
    elif interpolation == 'LANCZOS' or 'lanczos':
        new_image = (resize(img = (img), size = size, antialias=False, interpolation = InterpolationMode.LANCZOS))
    elif interpolation == 'NEAREST_EXACT' or 'nearest_exact':
        new_image = (resize(img = (img), size = size, antialias=False, interpolation = InterpolationMode.NEAREST_EXACT))
    if out_type == 'np':
        new_image = tensor_to_np(new_image)
    return new_image

#a function that does average pooling to resize an image to a new size
def binning(img, new_size):
    new_image = torch_reshape(img)
    # new_image = new_image.view(1, 1, new_image.shape[0], new_image.shape[1])
    new_image = torch.nn.functional.avg_pool2d(new_image, kernel_size = new_image.shape[2]//new_size[0], stride = new_image.shape[2]//new_size[0])
    new_image = new_image.view(new_image.shape[2], new_image.shape[3])
    new_image = tensor_to_np(new_image)
    return new_image

#A function that resizes using adaptive_max_pool2d
def resize_with_adaptive_max_pool2d(img, new_size):
    new_image = torch_reshape(img)
    new_image = torch.nn.functional.adaptive_max_pool2d(new_image, output_size = new_size)
    new_image = tensor_to_np(new_image)
    return new_image

def compare(img, new_size):
    from libraries.visualize import visualize
    import pandas as pd
    from IPython.display import display
    modes = ['NEAREST_EXACT', 'BILINEAR', 'Binning'] #'Max_pooling','BICUBIC', 'NEAREST', 
    dfs, new_images = [], []
    for mode in modes:
        if mode == 'Binning':
            new_image = binning(img, new_size)
            
        elif mode == 'Max_pooling':
            new_image = resize_with_adaptive_max_pool2d(img, new_size)
    
        else:
            new_image = resize_with_diff_interpolation(img, new_size, mode)
        df = pd.DataFrame({'Mode': mode, 'old max': [np.max(img)], 'new max': [np.max(new_image)], 'old min': [np.min(img)], 'new min': [np.min(new_image)], 'old mean': [np.mean(img)], 'new mean': [np.mean(new_image)], 'old std': [np.std(img)], 'new std': [np.std(new_image)], 'old median': [np.median(img)], 'new median': [np.median(new_image)],  'min_ratio': [np.min(new_image)/np.min(img)], 'max_ratio': [np.max(new_image)/np.max(img)], 'mean_ratio': [np.mean(new_image)/np.mean(img)], 'std_ratio': [np.std(new_image)/np.std(img)],  'min_ratio': [np.min(new_image)/np.min(img)], 'max_ratio': [np.max(new_image)/np.max(img)], 'mean_ratio': [np.mean(new_image)/np.mean(img)], 'std_ratio': [np.std(new_image)/np.std(img)], 'median_ratio': [np.median(new_image)/np.median(img)], 'old size':[img.shape], 'size': [new_image.shape]})

        dfs.append(df.T)
        new_images.append(new_image)
    df = pd.concat(dfs, axis = 1)
    visualize(new_images, cmap = 'gray', title = modes, show_or_plot = 'plot', images_per_row = len(modes), axis = 'off', plot_axis='half', zoomout_location='top left', colorbar = True, min_max=True, fontsize = 30)
    display(df)
    return df

def convert_to_polar_coordinate(img):
    img = torch_reshape(img)
    batch, in_channels, shape_x, shape_y = img.shape
    x = torch.linspace(-shape_x//2, shape_x//2, shape_x)
    y = torch.linspace(-shape_y//2, shape_y//2, shape_y)
    x, y = torch.meshgrid(x, y)
    r = torch.sqrt(x**2 + y**2)
    theta = torch.atan2(y, x)
    return r, theta

def convert_to_cartesian_coordinate(r, theta):
    x = r * torch.cos(theta)
    y = r * torch.sin(theta)
    return x, y

def radius_function_from_center(img, center = None):
    img = torch_reshape(img)
    if center is None:
        center = [img.shape[-2]//2, img.shape[-1]//2]
    x = torch.arange(img.shape[-2]) - center[0]
    y = torch.arange(img.shape[-1]) - center[1]
    X, Y = torch.meshgrid(x, y)
    r = torch.sqrt(X**2 + Y**2)
    #angle
    theta = torch.atan2(Y, X)
    return r, theta

def cartesian_from_polar_center(img, r, theta, center = None):
    img = torch_reshape(img)
    if center is None:
        center = [img.shape[-2]//2, img.shape[-1]//2]
    x = r * torch.cos(theta) + center[0]
    y = r * torch.sin(theta) + center[1]
    x = x.int()
    y = y.int()
    x = torch.clamp(x, 0, img.shape[-2]-1)
    y = torch.clamp(y, 0, img.shape[-1]-1)
    return img[:, :, x, y]

def rotate_from_polar_coordinate(img, r, theta, angle):
    x, y = convert_to_cartesian_coordinate(r, theta + angle)
    return cartesian_from_polar_center(img, x, y)


def compartment_image_torch(image, rows, cols, pad_by = 0, replace = False):
    """
    This function takes an image and splits it into compartments
    :param image: image to split
    :param rows: number of rows to split the image
    :param cols: number of columns to split the image
    :return: list of images
    """
    compartments = []
    image = torch_reshape(image)
    m,n,height, width = image.shape
    compartment_height = height // rows 
    compartment_width = width // cols
    for i in range(rows):    
        for j in range(cols):
            cell = torch.ones((m, n, compartment_height + 2 * pad_by, compartment_width+ 2 * pad_by))
            if not replace:
                compartment = image[:, :, i * compartment_height:(i + 1) * compartment_height,
                    j * compartment_width:(j + 1) * compartment_width]
                cell[:, :, pad_by:compartment_height + pad_by, pad_by:compartment_width + pad_by] = compartment
            else:
                "compartments will have cover bigger area that can be shared with other compartments"
                neighbors = positioning_comp(rows, cols, i * cols + j)
                if neighbors['left'] == 0:
                    left = 0
                    right = 2 * pad_by
                elif neighbors['right'] == 0:
                    left = 2 * pad_by
                    right = 0 
                else:
                    left = pad_by
                    right =  pad_by
                if neighbors['top'] == 0:
                    top = 0
                    bottom = 2 * pad_by
                elif neighbors['bottom'] == 0:
                    top =  2 * pad_by
                    bottom = 0
                else:
                    top = pad_by
                    bottom = pad_by
                compartment = image[:, :, i * compartment_height - top:(i + 1) * compartment_height + bottom,
                                    j * compartment_width - left:(j + 1) * compartment_width + right]
                cell = compartment
                
            compartments.append(cell)
    return compartments

def join_compartments_torch(compartments, rows, cols, padded_by = 0, replace = False):
    """
    This function takes a list of images and joins them into a single image
    :param compartments: list of images to join
    :param rows: number of rows to join the images
    :param cols: number of columns to join the images
    :param padded_by: padding that has been added to the compartments
    :return: single image
    """
    compartment_height, compartment_width = compartments[0].shape[-2:]
    
    if not replace:
                
        compartment_height -= 2 * padded_by
        compartment_width -= 2 * padded_by
        height = compartment_height * rows
        width = compartment_width * cols
        image = torch.zeros((1, 1, height, width))
        unpadded_compartments = [compartments[i][:, :, padded_by:compartment_height + padded_by, padded_by:compartment_width + padded_by] for i in range(rows * cols)]
        for i in range(rows):
            for j in range(cols):
                image[:, :, i * compartment_height:(i + 1) * compartment_height,
                    j * compartment_width:(j + 1) * compartment_width] = unpadded_compartments[i * cols + j]
    else:
        
        unpadded_compartments = []
        for i in range(rows * cols):
            neighbors = positioning_comp(rows, cols, i)
            cell_height, cell_width = compartments[i].shape[-2:]
            if neighbors['left'] == 0:
                left = 0
                right = 2 * padded_by
            elif neighbors['right'] == 0:
                left = 2 * padded_by
                right = 0
            else:
                left = padded_by
                right = padded_by
            if neighbors['top'] == 0:
                top = 0
                bottom = 2 * padded_by
            elif neighbors['bottom'] == 0:
                top = 2 * padded_by
                bottom = 0
            else:
                top = padded_by
                bottom = padded_by
            cell_height -= (left + right)
            cell_width -= (top + bottom)
            cell = torch.zeros((1, 1, cell_height, cell_width))
            cell = compartments[i][:, :, top:cell_height + top, left:cell_width + left]
            unpadded_compartments.append(cell)
        compartment_height = cell_height
        compartment_width = cell_width
        width = compartment_width * cols
        height = compartment_height * rows
        image = torch.zeros((1, 1, height, width))
        
        for i in range(rows):
            for j in range(cols):
                image[:, :, i * compartment_height:(i + 1) * compartment_height,
                    j * compartment_width:(j + 1) * compartment_width] = unpadded_compartments[i * cols + j]
                
    return image