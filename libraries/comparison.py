from torchmetrics.image.psnr import PeakSignalNoiseRatio as PSNR
from torchmetrics.image.ssim import MultiScaleStructuralSimilarityIndexMeasure as mSSIM
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure as SSIM
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
    
