from torch_utils import *
from typing import List, Optional, Tuple
from torch import Tensor
from torch.nn.functional import conv2d, pad as torch_pad
from torchvision.transforms.functional import rotate
# from torchvision.transforms.functional import  *
import torch.nn.functional as Fun

def _get_gaussian_kernel1d(kernel_size: int, sigma: float) -> Tensor:
    ksize_half = (kernel_size - 1) * 0.5

    x = torch.linspace(-ksize_half, ksize_half, steps=kernel_size)
    pdf = torch.exp(-0.5 * (x / sigma).pow(2))
    kernel1d = pdf / pdf.sum()

    return kernel1d

def _get_gaussian_kernel2d(
    kernel_size: List[int], sigma: List[float], dtype: torch.dtype, device: torch.device
) -> Tensor:
    kernel1d_x = _get_gaussian_kernel1d(kernel_size[0], sigma[0]).to(device, dtype=dtype)
    kernel1d_y = _get_gaussian_kernel1d(kernel_size[1], sigma[1]).to(device, dtype=dtype)
    kernel2d = torch.mm(kernel1d_y[:, None], kernel1d_x[None, :])
    return kernel2d

def _cast_squeeze_in(img: Tensor, req_dtypes: List[torch.dtype]) -> Tuple[Tensor, bool, bool, torch.dtype]:
    need_squeeze = False
    # make image NCHW
    if img.ndim < 4:
        img = img.unsqueeze(dim=0)
        need_squeeze = True

    out_dtype = img.dtype
    need_cast = False
    if out_dtype not in req_dtypes:
        need_cast = True
        req_dtype = req_dtypes[0]
        img = img.to(req_dtype)
    return img, need_cast, need_squeeze, out_dtype

def _cast_squeeze_out(img: Tensor, need_cast: bool, need_squeeze: bool, out_dtype: torch.dtype) -> Tensor:
    if need_squeeze:
        img = img.squeeze(dim=0)

    if need_cast:
        if out_dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            # it is better to round before cast
            img = torch.round(img)
        img = img.to(out_dtype)

    return img

def _is_tensor_a_torch_image(x: Tensor) -> bool:
    return x.ndim >= 2

def _assert_image_tensor(img: Tensor) -> None:
    if not _is_tensor_a_torch_image(img):
        raise TypeError("Tensor is not a torch image.")

def rotational_angular_blur(img, angle, rotation_num, center = None, pad_by = 5):
    img = torch_reshape(img)
    img = pad_on_all_sides_by_constant_torch(img, pad_by) if pad_by > 0 else img
    batch, in_channels, shape_x, shape_y = img.shape
    if center is None:
        center = [shape_x//2, shape_y//2]
    rotated_images = [rotate(img, angle*i, expand = False, center = center) for i in range(rotation_num)]
    final = torch.zeros_like(rotated_images[0])
    for i in range(rotation_num):
        final += rotated_images[i]
    return final/rotation_num, img

def gaussian_blur(img: Tensor, kernel_size: List[int], sigma: List[float]) -> Tensor:
    if not (isinstance(img, torch.Tensor)):
        raise TypeError(f"img should be Tensor. Got {type(img)}")

    _assert_image_tensor(img)

    dtype = img.dtype if torch.is_floating_point(img) else torch.float32
    kernel = _get_gaussian_kernel2d(kernel_size, sigma, dtype=dtype, device=img.device)
    kernel = kernel.expand(img.shape[-3], 1, kernel.shape[0], kernel.shape[1])
    img, need_cast, need_squeeze, out_dtype = _cast_squeeze_in(img, [kernel.dtype])

    # padding = (left, right, top, bottom)
    padding = [kernel_size[0] // 2, kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[1] // 2]
    img = torch_pad(img, padding, mode="reflect")
    img = conv2d(img, kernel, groups=img.shape[-3])

    img = _cast_squeeze_out(img, need_cast, need_squeeze, out_dtype)
    return img, kernel
   
def motion_blur_operator(shape_x, shape_y, a, b, Time_frame = 1, in_channels = 1):
    """
    this function creates the motion blur operator
    """

    u = torch.fft.fftfreq(shape_x)[None, :].unsqueeze(0).repeat(in_channels, 1, 1)
    v = torch.fft.fftfreq(shape_y)[:, None].unsqueeze(0).repeat(in_channels, 1, 1)
    
    omega = torch.pi*(u*a + v*b)
    H = (Time_frame/omega) * torch.sin(omega) * torch.exp(-(1.0j * omega))
    H[omega == 0] = 1

    H = H.reshape(in_channels, shape_x, shape_y)
    return H

def gaussian_kernel(kernel_size, sigma):
    if type(kernel_size) is not tuple and type(kernel_size) is not list:
        kernel_size = (kernel_size, kernel_size)
    if type(sigma) is not tuple and type(sigma) is not list:
        sigma = (sigma, sigma)
    sigma_x = sigma[0]
    sigma_y = sigma[1]
    kernel_size_x = kernel_size[0]
    kernel_size_y = kernel_size[1]
    x = torch.arange(kernel_size_x) - kernel_size_x // 2
    y = torch.arange(kernel_size_y) - kernel_size_y // 2
    X, Y = torch.meshgrid(x, y)
    kernel = torch.exp(-0.5*((X/sigma_x)**2 + (Y/sigma_y)**2))
    return kernel / kernel.sum()

def apply_kernel_conv(image_tensor, kernel, device = 'cuda:1'):
    image_tensor = to_device(torch_reshape(image_tensor), device)
    kernel = to_device(torch_reshape(kernel), device)
    kernel_size_x = kernel.shape[2]
    kernel_size_y = kernel.shape[3]
    padding = (kernel_size_x//2, kernel_size_y//2, kernel_size_x//2, kernel_size_y//2)
    image_tensor = torch_pad(image_tensor, padding, mode="reflect")
    blurred = conv2d(image_tensor, kernel, groups = image_tensor.shape[-3])
    return blurred
    
class Blur(nn.Module):
    def __init__(self, kernel_size_x: float, kernel_size_y: float, device = 'cuda', kernel = None, sigma = None) -> None:
        super().__init__()
        self.device = device
        if kernel is not None:
            self.kernel = to_device(torch_reshape(kernel), device)
            self.kernel_size_x = self.kernel.shape[2]
            self.kernel_size_y = self.kernel.shape[3]
            self.sigma = sigma if sigma is not None else (1,1)
        else:   
            assert kernel_size_x != None and kernel_size_y != None, 'Please provide kernel size'
            self.kernel_size_x = kernel_size_x
            self.kernel_size_y = kernel_size_y
            self.sigma = sigma
            self.kernel = to_device(torch_reshape(gaussian_kernel((self.kernel_size_x, self.kernel_size_y), sigma=self.sigma)), device)

    def forward(self, img: Tensor) -> Tensor:
        """
        image is a tensor of the form (batch, channel, shape_x, shape_y)
        """
        img = apply_kernel_conv(img, self.kernel, device = self.device)
        return img

class MotionBlur(nn.Module):
    def __init__(self, h: float, v: float, Time_frame: float = 1, shape_x: int = 512, shape_y: int = 512, in_channels = 1, device = 'cuda') -> None:
        super().__init__()
        self.h = h
        self.v = v
        self.Time_frame = Time_frame
        self.shape_x = shape_x
        self.shape_y = shape_y
        self.in_channels = in_channels
        self.H = to_device(motion_blur_operator(self.shape_x, self.shape_y, self.h, self.v, self.Time_frame, self.in_channels), device)

    def forward(self, img: Tensor) -> Tensor:
        """
        image is a tensor of the form (batch, channel, shape_x, shape_y)
        """
        batch, in_channels, shape_x, shape_y = img.shape
        if shape_x != self.shape_x or shape_y != self.shape_y:
            Warning('img shape is not the same as the initialized one')
            H = to_device(motion_blur_operator(shape_x, shape_y, self.h, self.v, self.T, in_channels), img.device)
            img = torch.fft.fftn(img)
            img = img * H
            img = torch.fft.ifftn(img)
            return  torch.abs(img)
        else:
            img = torch.fft.fftn(img)
            img = img * self.H
            img = torch.fft.ifftn(img)
            return  torch.abs(img)

class angular_blur(nn.Module):
    def __init__(self, img, angle: Optional[float] = 3, rotation_num: Optional[int] = 5,  center: Optional[List[int]] = None, pad_by: Optional[int] = 5, compress: Optional[bool] = False, **kwargs) -> None:
        super().__init__()
        print
        self.img = pad_on_all_sides_by_constant_torch(img, pad_by) if pad_by > 0 else torch_reshape(img)
        self.angle = angle
        self.rotation_num = rotation_num
        self.pad_by = 0
        self.center = center if center is not None else [self.img.shape[-2]//2, self.img.shape[-1]//2]
        self.device = self.img.device
        self.dtype = self.img.dtype
        self.shape = list(self.img.shape)
        self.compress = compress

    def forward(self, img: Optional[Tensor]=None) -> Tensor:
        """
        image is a tensor of the form (batch, channel, shape_x, shape_y)
        """
        img = torch_reshape(img) if img is not None else self.img
        blurred, img, rotated_images = rotational_angular_blur(img, self.angle, self.rotation_num, self.center, self.pad_by, self.compress)
        return blurred, img, rotated_images
   
class Gaussian_challenge_irreg_kernel(nn.Module):
    def __init__(self, img: torch.Tensor, kernel_size: List[int], sigma: List[float], device: Optional[torch.device] = None, kernel: Optional[torch.Tensor] = None, task = 'learn_gaussian', time_frame = 1) -> None:
        super().__init__()
        if not (isinstance(img, torch.Tensor)):
            img = torch_reshape(img)
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.device = device if device is not None else img.device
        self.dtype = img.dtype if torch.is_floating_point(img) else torch.float32
        self.shape = list(img.shape)

        if kernel is None:
            print("No Kernel")
            self.kernel = _get_gaussian_kernel2d(kernel_size, sigma, dtype=self.dtype, device=device)
            
            self.kernel = self.kernel.expand(self.shape[-3], 1, self.kernel.shape[0], self.kernel.shape[1])
            img, self.need_cast, self.need_squeeze, out_dtype = _cast_squeeze_in(img, [self.kernel.dtype])
            self.kernel = _cast_squeeze_out(self.kernel, self.need_cast, self.need_squeeze, out_dtype)
        else:
            self.shape = list(img.shape)
            self.kernel = kernel
            if isinstance(self.kernel, np.ndarray):
                self.kernel = torch.from_numpy(self.kernel)
                self.kernel = self.kernel.expand(self.shape[-3], 1, self.kernel.shape[0], self.kernel.shape[1])
            # else:
            #     self.kernel = self.kernel.expand(self.shape[-3], 1, self.kernel.shape[0], self.kernel.shape[1])
            img, self.need_cast, self.need_squeeze, out_dtype = _cast_squeeze_in(img, [self.kernel.dtype])
            self.kernel = _cast_squeeze_out(self.kernel, self.need_cast, self.need_squeeze, out_dtype)
            
        self.kernel = self.kernel.to(self.device)

    def forward(self, img: Tensor, kernel = None) -> Tensor:
        if not (isinstance(img, torch.Tensor)):
            img = torch_reshape(img)

        img = to_device(torch_reshape(img), self.device) 
        padding = [self.kernel_size[0] // 2, self.kernel_size[0] // 2, self.kernel_size[1] // 2, self.kernel_size[1] // 2]
        img = torch_pad(img, padding, mode="reflect")
        blurred = conv2d(img, self.kernel, groups=img.shape[-3])
        return _cast_squeeze_out(blurred, self.need_cast, self.need_squeeze, self.dtype), self.kernel

class Gaussian_challenge(nn.Module):
    def __init__(self, img: torch.Tensor, kernel_size: List[int], sigma: List[float], device: Optional[torch.device] = None, kernel: Optional[torch.Tensor] = None, task = 'learn_gaussian', time_frame = 1) -> None:
        super().__init__()
        if not (isinstance(img, torch.Tensor)):
            img = torch_reshape(img)
        
        self.device = device if device is not None else img.device
        self.dtype = img.dtype if torch.is_floating_point(img) else torch.float32
        self.shape = list(img.shape)
        self.task = task
        self.time_frame = time_frame
        if kernel is None:
            self.kernel = _get_gaussian_kernel2d(kernel_size, sigma, dtype=self.dtype, device=device) 
            if isinstance(self.kernel, np.ndarray):
                self.kernel = torch.from_numpy(self.kernel)
                self.kernel = self.kernel.expand(self.shape[-3], 1, self.kernel.shape[0], self.kernel.shape[1])
            img, self.need_cast, self.need_squeeze, out_dtype = _cast_squeeze_in(img, [self.kernel.dtype])
            self.kernel = _cast_squeeze_out(self.kernel, self.need_cast, self.need_squeeze, out_dtype)
        else:
            self.kernel = torch_reshape(kernel) if kernel is not None else None
        
        self.shape = list(img.shape)    
        # self.kernel = to_device(torch_reshape(self.kernel), device)
        self.kernel_size = list(self.kernel.shape)
        self.padding = [self.kernel_size[-2] // 2, self.kernel_size[-2] // 2, self.kernel_size[-1] // 2, self.kernel_size[-1] // 2]
        self.need_cast = False
        self.need_squeeze = False
    # def forward(self, img: Tensor, kernel = None) -> Tensor:
    #     if not (isinstance(img, torch.Tensor)):
    #         img = torch_reshape(img)
    #     if kernel is not None:
    #         self.kernel = to_device(torch_reshape(kernel), self.device)
    #         self.kernel_size = list(self.kernel.shape)
    #         self.padding = [self.kernel_size[2] // 2, self.kernel_size[2] // 2, self.kernel_size[3] // 2, self.kernel_size[3] // 2]
    #     img = to_device(torch_reshape(img), self.device)  
    #     if self.task == 'learn_gaussian':
    #         img = torch_pad(img, self.padding, mode="reflect")
    #         blurred = conv2d(img, self.kernel, groups=img.shape[-3])
    #         return _cast_squeeze_out(blurred, self.need_cast, self.need_squeeze, self.dtype), self.kernel
    #     elif self.task == 'learn_motion':
    #         img = torch.fft.fftn(img)
    #         img = img * self.kernel
    #         img = torch.fft.ifftn(img)
    #         return  torch.abs(img), self.kernel
        
        
    def forward(self, img: Tensor, kernel = None) -> Tensor:
        if not (isinstance(img, torch.Tensor)):
            img = torch_reshape(img)
        if kernel is None:
            kernel = self.kernel
        blurred = apply_kernel_conv(img, kernel, device = self.device)
        return blurred, kernel

#given a motion blur kernel, we can apply it to an image
class Blur(nn.Module):
    def __init__(self, kernel_size_x: float, kernel_size_y: float, device = 'cuda', kernel = None, sigma = None) -> None:
        super().__init__()
        self.device = device
        if kernel is not None:
            self.kernel = to_device(torch_reshape(kernel), device)
            self.kernel_size_x = self.kernel.shape[2]
            self.kernel_size_y = self.kernel.shape[3]
            self.sigma = sigma if sigma is not None else (1,1)
        else:   
            assert kernel_size_x != None and kernel_size_y != None, 'Please provide kernel size'
            self.kernel_size_x = kernel_size_x
            self.kernel_size_y = kernel_size_y
            self.sigma = sigma
            self.kernel = to_device(torch_reshape(gaussian_kernel((self.kernel_size_x, self.kernel_size_y), sigma=self.sigma)), device)

    def forward(self, img: Tensor) -> Tensor:
        """
        image is a tensor of the form (batch, channel, shape_x, shape_y)
        """
        img = apply_kernel_conv(img, self.kernel, device = self.device)
        return img

class Inpainiting(nn.Module):
    def __init__(self, img: torch.Tensor, mask: torch.Tensor, device: Optional[torch.device] = None, kernel: Optional[torch.Tensor] = None, task = 'learn_gaussian', time_frame = 1) -> None:
        super().__init__()
        if not (isinstance(img, torch.Tensor)):
            img = torch_reshape(img)
        self.device = device if device is not None else img.device
        self.dtype = img.dtype if torch.is_floating_point(img) else torch.float32
        self.shape = list(img.shape)
        self.task = task
        self.time_frame = time_frame
        self.mask = mask
        self.mask = to_device(torch_reshape(self.mask), self.device)
        self.mask = self.mask.expand(self.shape[-3], 1, self.mask.shape[2], self.mask.shape[3])
        self.mask = self.mask.to(self.device)
        self.img = img
        self.img = to_device(torch_reshape(self.img), self.device)
        self.img = self.img.expand(self.shape[-3], 1, self.img.shape[2], self.img.shape[3])
        self.img = self.img.to(self.device)
        self.need_cast = False
        self.need_squeeze = False
    def forward(self, img: Tensor, mask: Tensor) -> Tensor:
        if not (isinstance(img, torch.Tensor)):
            img = torch_reshape(img)
        img = to_device(torch_reshape(img), self.device)
        mask = to_device(torch_reshape(mask), self.device)
        img = img * mask
        return img

class angular_blur(nn.Module):
    def __init__(self, img, angle: Optional[float] = 3, rotation_num: Optional[int] = 5,  center: Optional[List[int]] = None, pad_by: Optional[int] = 5, compress: Optional[bool] = False, **kwargs) -> None:
        super().__init__()
        print
        self.img = pad_on_all_sides_by_constant_torch(img, pad_by) if pad_by > 0 else torch_reshape(img)
        self.angle = angle
        self.rotation_num = rotation_num
        self.pad_by = 0
        self.center = center if center is not None else [self.img.shape[-2]//2, self.img.shape[-1]//2]
        self.device = self.img.device
        self.dtype = self.img.dtype
        self.shape = list(self.img.shape)
        self.compress = compress

    def forward(self, img: Optional[Tensor]=None) -> Tensor:
        """
        image is a tensor of the form (batch, channel, shape_x, shape_y)
        """
        img = torch_reshape(img) if img is not None else self.img
        blurred, img, rotated_images = rotational_angular_blur(img, self.angle, self.rotation_num, self.center, self.pad_by, self.compress)
        return blurred, img, rotated_images
   
def get_angle(image, center = None):
    if center is None:
        center = (image.shape[0]//2, image.shape[1]//2)
    radius = np.sqrt(center[0]**2 + center[1]**2)
    width = np.sqrt(radius**2 - 1)
    # angle_1 = np.arctan(center[1]/center[0])
    # angle_2 = np.arctan((center[1] - 1) / center[0])
    # angle = angle_1 - angle_2
    angle = np.arctan(1/width)
    angle = np.rad2deg(angle)
    return angle

def calculate_rotation_angle(image_width, image_height):
    # Assume we rotate by one pixel
    one_pixel_rotation = 1
    # Calculate the diagonla length of the image
    diagonal_length = np.sqrt(image_width**2 + image_height**2)
    # Calculate the angle needed to rotate the image by one pixel
    angle = np.arctan(one_pixel_rotation / diagonal_length)
    # Convert the angle to degrees
    angle = np.rad2deg(angle)
    return angle

def rotate_batch(image, angles, center = None):
    """
    Rotate a batch of images by a batch of angles.
    
    Args:
        image (torch.Tensor): A 4D tensor representing the batch of images [N, 1, H, W].
        angles (torch.Tensor): A 1D tensor of angles in degrees.
        diagonal (int): The diagonal size of the padded image.
        center (tuple): The center of the rotation.

    Returns:
        torch.Tensor: A 4D tensor of rotated images [N, 1, H, W].
    """
    # Convert angles to radians and create rotation matrices
    angles = angles * (math.pi / 180)
    cos_vals = torch.cos(angles)
    sin_vals = torch.sin(angles)
    
    rotation_matrices = torch.stack([
        cos_vals, -sin_vals, torch.zeros_like(cos_vals),
        sin_vals, cos_vals, torch.zeros_like(sin_vals)
    ], dim=1).view(-1, 2, 3)
    
    grid = Fun.affine_grid(rotation_matrices, image.size(), align_corners=False)
    rotated_images = Fun.grid_sample(image, grid, align_corners=False)

    return rotated_images

def apply_circle_mask(sinogram, width, height):
    """
    Apply a circular mask to the sinogram to simulate a circular detection region.

    Args:
        sinogram (torch.Tensor): The sinogram to apply the mask to.
        width (int): The width of the padded image.
        height (int): The height of the padded image.

    Returns:
        torch.Tensor: The masked sinogram.
    """
    # Create a circular mask based on the dimensions of the padded image
    center_x, center_y = width // 2, height // 2
    y, x = torch.meshgrid(torch.arange(height), torch.arange(width))
    mask = ((x - center_x) ** 2 + (y - center_y) ** 2) <= (min(width, height) // 2) ** 2

    return sinogram * mask.sum(dim=1)

def radon_transform_parallel(image, angles, center = None, circle=False, angle_per_scan = None, reshape = True, multiply = 1):
    """
    Compute the Radon transform of the given image using parallel computation.
    
    Args:
        image (torch.Tensor): A 2D tensor representing the grayscale image (shape: [H, W]).
        angles (torch.Tensor): A 1D tensor representing the angles (in degrees) at which to compute the transform.
        circle (bool): If True, assumes the object is inscribed within a circle. 
                       If False, assumes the object is inscribed within a square.

    Returns:
        torch.Tensor: A 2D tensor containing the Radon transform (sinogram).
    """
    image = torch_reshape(image)
    image.cpu()
    _, _, H, W = image.shape
    if reshape:
        image = image[0, 0, :, :]
        diagonal = int(np.ceil(np.sqrt(2) * max(H, W)))

        # Pad the image to ensure it fits within the given dimensions
        padded_image = F.pad(image, [(diagonal - W) // 2] * 2 + [(diagonal - H) // 2] * 2, mode='constant', value=0)
        padded_image = padded_image.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions

        center = (diagonal // 2, diagonal // 2)
    else:
        diagonal = max(H, W)
        padded_image = image
        center = center or (W // 2, H // 2)
    # Rotate the image batch by all angles at once
    #make all in the same device
    device = image.device
    angles = angles.to(device)
    diagonal = torch.tensor(diagonal).to(device)
    padded_image = padded_image
    rotated_images = rotate_batch(padded_image.expand(len(angles), -1, -1, -1), angles, diagonal)

    if angle_per_scan is not None:
        #do the mean for the batch of rotated images with stepsize of angle_per_scan so the image is rotated angle_per_scan times and the mean is taken
        rotated_images_batches = rotated_images.chunk(len(angles)//angle_per_scan, dim=0)
        rotated_images = torch.stack([batch.mean(dim=0) for batch in rotated_images_batches])

    # Sum along the x-axis (dim=2) to get the sinogram
    sinogram = rotated_images.squeeze(1).sum(dim=2)
    
    if circle:
        sinogram = apply_circle_mask(sinogram, diagonal, diagonal)

    return to_device(sinogram, device), rotated_images

class RadonTransform(nn.Module):
    def __init__(self, total_angles = 180, stepsize = 180, multiply = 1, angle_per_scan = None, circle=False, center=None):
        super(RadonTransform, self).__init__()
        self.total_angles = total_angles
        self.stepsize = stepsize
        self.angle_per_scan = angle_per_scan
        self.multiply = multiply
        self.angles = torch.linspace(0, total_angles, steps=multiply * self.stepsize)
        self.circle = circle
        self.center = center
        if self.angle_per_scan == 0:
            self.angle_per_scan = None
    def forward(self, x, reshape = True):
        return radon_transform_parallel(x, self.angles, self.center, self.circle, self.angle_per_scan, reshape, self.multiply)
    