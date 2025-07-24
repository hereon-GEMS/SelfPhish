from torchmetrics.image.psnr import PeakSignalNoiseRatio as PSNR
from torchmetrics.image.ssim import MultiScaleStructuralSimilarityIndexMeasure as mSSIM
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure as SSIM
from torch.nn.modules.loss import _Loss
from scipy.stats import entropy as calculate_entropy
from torch_utils import *

def diff(real_image, comp_image, comparison = ''):
    real_image = torch_reshape(real_image)
    comp_image = torch_reshape(comp_image)
    if real_image.device != comp_image.device:
        comp_image = to_device(comp_image, real_image.device)
    if comparison == 'l2':
        return torch.norm(real_image - comp_image, p = 2)
    elif comparison == 'l1':
        return torch.norm(real_image - comp_image, p = 1)
    elif comparison == 'l0':
        return torch.norm(real_image - comp_image, p = 0)
    elif comparison == "l1_regularization":
        return diff(real_image, comp_image, comparison = 'l1') + torch.norm(real_image, p = 1)
    elif comparison == "l2_regularization":
        return diff(real_image, comp_image, comparison = 'l2') + torch.norm(real_image, p = 2)
    elif comparison == 'mse':
        return F.mse_loss(real_image, comp_image)
    elif comparison == 'mae':
        return F.l1_loss(real_image, comp_image)
    elif comparison == 'msle':
        return F.mse_loss(torch.log(real_image), torch.log(comp_image))
    elif comparison == 'psnr':
        psnr = to_device(PSNR(), real_image.device)
        return psnr(real_image, comp_image)
        # return 10 * torch.log10(1 / F.mse_loss(real_image, comp_image))
    elif comparison == 'ssim':
        ssim = to_device(SSIM(), real_image.device)
        return ssim(real_image, comp_image)
    elif comparison == 'cosine':
        return F.cosine_similarity(real_image, comp_image)
    elif comparison == 'correlation':
        return F.cosine_similarity(real_image - torch.mean(real_image), comp_image - torch.mean(comp_image))
    elif comparison == 'kl':
        return F.kl_div(real_image, comp_image)
    elif comparison == 'js':
        return F.kl_div(real_image, comp_image) + F.kl_div(comp_image, real_image)
    elif comparison == 'hellinger':
        return F.kl_div(real_image, comp_image) + F.kl_div(comp_image, real_image) - 2
    elif comparison == 'kid':
        from torchmetrics.image.kid import KernelInceptionDistance
        kid = KernelInceptionDistance(64, normalize=True)
        #convert to uint8
        real_image = real_image.type(torch.uint8)
        comp_image = comp_image.type(torch.uint8)
        kid.update(real_image, real=True)
        kid.update(comp_image, real=False)
        return kid.compute()

    elif comparison == 'FID':
        from torchmetrics.image.fid import FrechetInceptionDistance
        fid = FrechetInceptionDistance(64, normalize=True)
        #convert to uint8
        real_image = real_image.type(torch.uint8)
        comp_image = comp_image.type(torch.uint8)
        fid.update(real_image, real=True)
        fid.update(comp_image, real=False)
        return fid.compute()
    
    else:
        return real_image - comp_image
    
from torchvision.models import vgg16, VGG16_Weights
class VGGPerceptualLoss(nn.Module):
    def __init__(self, layer_names=None, weights=None):
        super().__init__()
        vgg = vgg16().features
        vgg.eval()  # freeze parameters
        for param in vgg.parameters():
            param.requires_grad = False
        
        # Default: compare at relu2_2, relu3_3, relu4_3
        self.layer_names = layer_names or ['3', '8', '15']  # these indices correspond to conv layers, or extract by name
        self.weights = weights or [1.0, 1.0, 1.0]
        
        # Split VGG into sequential blocks to extract intermediate activations
        self.blocks = nn.ModuleList()
        prev_idx = 0
        for idx in self.layer_names:
            idx = int(idx)
            block = nn.Sequential(*list(vgg.children())[prev_idx:idx+1])
            self.blocks.append(block)
            prev_idx = idx+1

        # Normalization layer matching ImageNet stats
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406])[None,:,None,None])
        self.register_buffer('std',  torch.tensor([0.229, 0.224, 0.225])[None,:,None,None])
        
    def forward(self, generated, target):
        # Assume generated, target in [0,1], shape (B,3,H,W)
        gen_norm = (generated - self.mean) / self.std
        tgt_norm = (target    - self.mean) / self.std
        
        loss = 0.0
        x_gen = gen_norm
        x_tgt = tgt_norm
        for i, block in enumerate(self.blocks):
            x_gen = block(x_gen)
            x_tgt = block(x_tgt).detach()  # detach target features
            diff = x_gen - x_tgt
            # Mean‐squared error per‐element
            loss += self.weights[i] * diff.pow(2).mean()
        return loss

# Assuming these are available from your libraries (e.g., comparison)
# For this example, we'll use placeholder functions or basic torch ops.
def compute_ssim(pred, target, data_range=1.0, device=None):
    pred = pos_neg_image(pred, 'pre_ssim')
    target = pos_neg_image(target, 'pre_ssim')
    if device is None:
        device = pred.device
    # Ensure pred and target have the same shape
    if pred.shape != target.shape:
        if pred.shape[1] != target.shape[1]:
            if pred.shape[1] == 1 and target.shape[1] > 1:
                pred = pred.repeat(1, target.shape[1], 1, 1)
            elif target.shape[1] == 1 and pred.shape[1] > 1:
                target = target.repeat(1, pred.shape[1], 1, 1)
    ssim_metric = SSIM(data_range=data_range).to(device)
    return 1.0 - ssim_metric(pred, target)

def compute_psnr(pred, target, data_range=1.0, device=None):
    pred = pos_neg_image(pred, 'pre_ssim')
    target = pos_neg_image(target, 'pre_ssim')    
    if device is None:
        device = pred.device
    # Ensure pred and target have the same shape (especially channels)
    if pred.shape != target.shape:
        if pred.shape[1] != target.shape[1]:
            if pred.shape[1] == 1 and target.shape[1] > 1:
                pred = pred.repeat(1, target.shape[1], 1, 1)
            elif target.shape[1] == 1 and pred.shape[1] > 1:
                target = target.repeat(1, pred.shape[1], 1, 1)
    psnr_metric = PSNR(data_range=data_range).to(device)
    return 1.0 - psnr_metric(pred, target)

def compute_total_variation_loss(img, weight, power=1):
    if weight == 0:
        return 0
    # Simplified TV loss for example
    bs_img, c_img, h_img, w_img = img.size()
    tv_h = torch.pow(torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]), power).sum()
    tv_w = torch.pow(torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]), power).sum()
    return weight * (tv_h + tv_w)

def regularize(x, reg_l1_ratio=0, reg_l2_ratio=0, gradient=False):
    l1_loss = reg_l1_ratio * torch.mean(torch.abs(x))
    l2_loss = reg_l2_ratio * torch.mean(torch.square(x))
    return l1_loss, l2_loss


def get_image_entropy(image_path):
    try:
        if isinstance(image_path, np.ndarray):
            #convert numpy array to PIL Image
            from PIL import Image
            img = Image.fromarray(image_path)
            hist = np.array(img.histogram())
            hist_sum = hist.sum()
            if hist_sum == 0:
                return np.nan
            prob_dist = hist / hist_sum
            # Filter out zero probabilities to avoid log(0)
            prob_dist = prob_dist[prob_dist > 0]
            return calculate_entropy(prob_dist, base=10)
        elif isinstance(image_path, list):
            # If a list of images is provided, calculate entropy for each image
            entropies = []
            for img_path in image_path:
                entropies.append(get_image_entropy(img_path))
                #change to 0 to 1
            entropies = np.array(entropies)
            entropies = (entropies - entropies.min()) / (entropies.max() - entropies.min() + 1e-8)
            return entropies
        else:
            raise ValueError("Unsupported image input type. Provide a file path, numpy array, or list of images.")
    except Exception as e:
        print(f"Error calculating entropy for {image_path}: {e}")
        return np.nan # Return NaN for errors

def calculate_shannon_entropy_loss(image_tensor, eps=1e-8):
    """
    Calculates a differentiable approximation of Shannon Entropy for a PyTorch tensor.
    This function treats the normalized pixel values as probabilities.
    Higher entropy is a higher loss (if you want to penalize it).

    Args:
        image_tensor (torch.Tensor): The input image tensor (e.g., model output).
                                     Expected shape: (N, C, H, W) or (N, H, W).
                                     Values should ideally be non-negative.
        eps (float): A small epsilon to avoid log(0).

    Returns:
        torch.Tensor: The scalar entropy loss.
    """
    # Ensure image_tensor is non-negative. If your image can have negative values,
    # you might need to map them to a positive range first (e.g., by adding min_val and normalizing).
    # For typical image data (0-1 or 0-255), this is fine.

    # Normalize pixel values to sum to 1 to represent a probability distribution.
    # We flatten all pixels for the distribution, assuming a global intensity distribution.
    # Sum over all pixels to get a distribution for the *entire* image batch.
    flattened_pixels = image_tensor.flatten()
    
    # If values are not normalized between 0-1 or similar, you might need to normalize them
    # For example, if pixel values are 0-255, divide by 255.
    # Or, if they are arbitrary values, normalize them to sum to 1 over the batch.
    
    # Simple normalization: treat values as 'counts' and normalize
    # This approach is a simplification for continuous values and may not be
    # true 'pixel intensity' histogram entropy, but it's differentiable.
    
    # A common approach for this is to use a softmax-like transformation if your
    # image values are not already a distribution.
    
    # Let's assume your image_tensor values are already normalized (e.g., 0 to 1) or
    # that their magnitudes are what you want to consider for 'probability'.
    # For a real "probability distribution" from pixel values,
    # you'd typically need to bin them and get counts, which is not easily differentiable.

    # Instead, let's use a common differentiable way to encourage / discourage distribution spread.
    # This is often done using a probability distribution derived from the values themselves.
    
    # A more common interpretation of "entropy loss" in deep learning for image generation
    # if you want to penalize simple, uniform outputs (i.e., encourage more varied outputs)
    # is to work with the pixel values directly.
    
    # Let's aim for a differentiable proxy that might encourage/discourage certain pixel value distributions.
    
    # If your pixels are in (0,1) range, consider them as probabilities directly.
    # This is an unusual application of Shannon entropy but differentiable.
    # A common way to make any tensor a "probability distribution" for entropy is softmax,
    # but that changes the values.

    # Let's consider a simpler form of entropy that encourages spread in pixel values.
    # A standard interpretation of entropy for continuous variables is more complex.
    # For images, we usually refer to discrete entropy of intensity levels.

    # Simpler, differentiable entropy proxy:
    # Treat image_tensor values (normalized to [0,1] or similar) as probabilities.
    # Example: if your model outputs are already in [0,1], then use them directly.
    # If not, you might want a sigmoid or softmax.
    
    # For entropy, p should be in [0,1] and sum to 1.
    # Normalizing the entire image batch to sum to 1:
    prob_dist = image_tensor / (image_tensor.sum() + eps)
    
    # Clamp values to avoid log(0) and log(1) issues if values become exactly 0 or 1.
    prob_dist = torch.clamp(prob_dist, eps, 1.0 - eps)

    # Calculate entropy
    # The negative sign is because we want to minimize this loss,
    # and standard entropy is usually positive.
    # If we want to penalize HIGH entropy, the loss is just H.
    # If we want to penalize LOW entropy, the loss is -H.
    
    entropy = - (prob_dist * torch.log(prob_dist)).sum()
    
    return entropy

def compute_image_metric(image, metric_type, device=None):
    """
    Compute a no-reference image quality metric using pyiqa.
    
    Args:
        image: Input image tensor (should be torch.Tensor, shape [N, C, H, W] or [C, H, W]).
        metric_type: String, one of the metrics supported by pyiqa (e.g., 'piqe', 'niqe').
        device: Torch device (optional). If None, use image.device if possible.
        
    Returns:
        Metric value as a numpy array.
    """
    import pyiqa
    import torch
    if metric_type == 'entropy':
        return get_image_entropy(image)
    
    if isinstance(image, list):
        image = torch_reshape(image)
    if not isinstance(image, torch.Tensor):
        raise ValueError("Input image must be a torch.Tensor")
    if device is None:
        device = image.device

    # Downsample or skip NIQE for large images
    if metric_type == 'niqe':
        h, w = image.shape[-2:]
        if h > 512 or w > 512:
            print(f"Skipping NIQE for large image ({h}x{w}).")
            return np.nan
        # Optionally, downsample:
        # image = torch.nn.functional.interpolate(image, size=(256, 256), mode='bilinear')

    metric = pyiqa.create_metric(metric_type, device=device)
    value = metric(image)
    if isinstance(value, torch.Tensor):
        value = tensor_to_np(value)
    if value.ndim > 0:
        value = (value - value.min()) / (value.max() - value.min() + 1e-8) 
    else:
        value = np.array(value)
    return value

def compute_image_multiple_metrics(image, metrics, device=None):
    """
    Compute multiple no-reference image quality metrics using pyiqa.
    Frees only the GPU memory used by this function.
    """
    import pyiqa
    import torch

    if isinstance(image, list):
        image = torch_reshape(image)
        
    if not isinstance(image, torch.Tensor):
        raise ValueError("Input image must be a torch.Tensor")
    if device is None:
        device = image.device

    image = pos_neg_image(pos_neg_image(image, 'pre_ssim'), 'pos')
    results = {}
    metric_objs = []
    try:
        for metric_name in metrics:
            # Create metric object inside the loop so it can be deleted after use
            if metric_name == 'entropy':
                value = [calculate_shannon_entropy_loss(im.unsqueeze(0)) for im in image]
            else:
                metric = pyiqa.create_metric(metric_name, device=device)
                metric_objs.append(metric)
                value = metric(image)
            if isinstance(value, torch.Tensor):
                value = tensor_to_np(value)
            results[metric_name] = value
            # Delete metric object and free memory after each metric
            if metric_name != 'entropy':
                del metric
            torch.cuda.empty_cache()
    finally:
        # Clean up any remaining local GPU objects
        del image
        for obj in metric_objs:
            del obj
        torch.cuda.empty_cache()
    return results
# Mock class for demonstration purposes, mimicking Forwardnet's context
class GeneratorLoss(nn.Module):
    def __init__(self, output_num=2, device='cuda', **kwargs):
        super().__init__()
        self.output_num = output_num
        self.device = device
        # Loss ratios/weights
        self.l1_ratio = kwargs.get('l1_ratio', 1.0)
        self.l2_ratio = kwargs.get('l2_ratio', 0.0)
        self.fourier_ratio = kwargs.get('fourier_ratio', 0.0)
        self.reg_l1_ratio = kwargs.get('reg_l1_ratio', 0.0)
        self.reg_l2_ratio = kwargs.get('reg_l2_ratio', 0.0)
        self.vgg_loss_ratio = kwargs.get('vgg_loss_ratio', 0.0)
        self.transform_factor = kwargs.get('transform_factor', 0.7)
        self.ssim_loss_weight = kwargs.get('ssim_loss_weight', 0.0)
        self.psnr_loss_weight = kwargs.get('psnr_loss_weight', 0.0)
        self.tv_weight = kwargs.get('tv_weight', 0.0)
        self.contrast_loss_weight = kwargs.get('contrast_loss_weight', 0.0)
        self.brightness_loss_weight = kwargs.get('brightness_loss_weight', 0.0)
        self.reg_entropy_loss=kwargs.get('reg_entropy_loss',0.0)
        self.final_loss = None
        if self.vgg_loss_ratio != 0:
            self._vgg_loss = VGGPerceptualLoss().to(self.device)
        self.softplus = to_device(torch.nn.Softplus(), self.device)
        
    def _calculate_fidelity_loss(self, pred, target):
        fidelity_loss = torch.tensor(0.0, device=pred.device)
        fidelity_loss += self.l1_ratio * torch.mean(torch.abs(target - pred))
        fidelity_loss += self.l2_ratio * torch.mean(torch.square(target - pred))
        if self.ssim_loss_weight > 0:
            fidelity_loss += self.ssim_loss_weight * (1 - compute_ssim(pred, target))
        if self.psnr_loss_weight > 0:
            fidelity_loss += self.psnr_loss_weight * (1.0 / (compute_psnr(pred, target) + 1e-6))
        if self.fourier_ratio > 0:
            fft_pred = torch.fft.fft2(pred)
            fft_target = torch.fft.fft2(target)
            fidelity_loss += self.fourier_ratio * torch.mean(torch.abs(fft_pred - fft_target))
        return fidelity_loss

    def _calculate_regularization_loss(self, image):
        # Regularization (TV) losses
        softplus = self.softplus
        if type(image) is list:
            reg_l1_loss = self.reg_l1_ratio * compute_total_variation_loss(softplus(-image[0])/ softplus(image[1]), self.reg_l1_ratio) if self.reg_l1_ratio != 0 else 0
            reg_l2_loss = self.reg_l2_ratio * compute_total_variation_loss(softplus(-image[0])/ softplus(image[1]), self.reg_l2_ratio, power = 2) if self.reg_l2_ratio != 0 else 0
            reg_entropy_loss = 0
            for img in image:
                reg_entropy_loss += self.reg_entropy_loss * calculate_shannon_entropy_loss(img)
        else:
            reg_l1_loss = self.reg_l1_ratio * compute_total_variation_loss(softplus(image), self.reg_l1_ratio) if self.reg_l1_ratio != 0 else 0
            reg_l2_loss = self.reg_l1_ratio * compute_total_variation_loss(softplus(image), self.reg_l2_ratio, power = 2) if self.reg_l2_ratio != 0 else 0
            reg_entropy_loss = self.reg_entropy_loss * calculate_shannon_entropy_loss(image)
        return reg_l1_loss + reg_l2_loss + reg_entropy_loss
        
    def _calculate_perceptual_loss(self, pred, target):
        perceptual_loss = torch.tensor(0.0, device=pred.device)
        pred_std = torch.std(pred)
        target_std = torch.std(target)
        perceptual_loss += torch.abs(pred_std - target_std)
        pred_mean = torch.mean(pred)
        target_mean = torch.mean(target)
        perceptual_loss += torch.abs(pred_mean - target_mean)
        return perceptual_loss

    def forward(self, phase, attenuation, propagated_intensity, transformed_images,
                target=None, pred=None, initial_pred=None):
        fidelity = self._calculate_fidelity_loss(propagated_intensity, transformed_images)
        if pred is not None and target is not None:
            fidelity += self._calculate_fidelity_loss(pred, target)
        # Perceptual loss
        # perceptual = self._calculate_perceptual_loss(propagated_intensity, transformed_images)
        # if pred is not None and target is not None:
        #     perceptual += self._calculate_perceptual_loss(pred, target)
        
        self.final_loss = fidelity #+ perceptual
            
        # VGG perceptual loss
        if self.vgg_loss_ratio != 0:
            vgg_loss = self._vgg_loss(propagated_intensity, transformed_images)
            self.final_loss = self.final_loss + self.vgg_loss_ratio * vgg_loss
        
        # Regularization
        if self.reg_entropy_loss != 0.0 or self.reg_l1_ratio != 0.0 or self.reg_l2_ratio != 0.0:
            self.final_loss = self.final_loss + self._calculate_regularization_loss([phase,attenuation])
        
        # Optionally add TV/regularization on initial_pred
        if initial_pred is not None:
            self.final_loss = self.final_loss + self._calculate_regularization_loss(initial_pred)
        return self.final_loss
    
# # --- Example Usage ---
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# model_instance = GeneratorLoss(output_num=2, device=device, l1_ratio=1.0, l2_ratio=0.5, ssim_loss_weight=0.8, psnr_loss_weight=0.05, tv_weight=0.001, vgg_loss_ratio=0.1)

# # Dummy inputs (replace with your actual data)
# dummy_target = torch.randn(1, 1, 256, 256).to(device)
# dummy_pred = torch.randn(1, 1, 256, 256).to(device)
# dummy_propagated_intensity = torch.randn(1, 1, 256, 256).to(device)
# dummy_initial_pred = torch.randn(1, 1, 256, 256).to(device)
# dummy_phase = torch.randn(1, 1, 256, 256).to(device)
# dummy_attenuation = torch.randn(1, 1, 256, 256).to(device)
# dummy_transformed_images = torch.randn(1, 1, 256, 256).to(device)

# # Calculate the total loss
# total_gen_loss = model_instance(
#     dummy_phase, dummy_attenuation, dummy_propagated_intensity, dummy_transformed_images,
#     target=dummy_target, pred=dummy_pred, initial_pred=dummy_initial_pred
# )
# print(f"Total Generator Loss: {total_gen_loss.item()}")
