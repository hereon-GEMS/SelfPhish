from libraries.available_experiments import propagate_others
from libraries.gan4inverse import make_ganrec_model
from libraries.utils import generate_qr_code
from libraries.visualize import visualize
import torch

# Example usage
image_path='data/qr/'
img = generate_qr_code("Phase (V20|BS10|B4): A Good Phase: Solving the Near-field Phase Retrieval Inverse Problem Quantitatively Using SelfPhish. Phase retrieval is a non-linear, ill-posed inverse problem that is essential for accurate wavefront reconstruction prior to tomographic analysis. SelfPhish enables quantitative phase re-trieval with high precision. ", image_path+'phase_qr_code.tif', 10)
img2 = generate_qr_code("Absorbance: Experimental results demonstrate SelfPhish's superior accuracy and robustness across diverse imaging scenarios. underscoring its potential to significantly enhance phase retrieval and image quality in X-ray imaging and beyond.", image_path+'abs_qr_code.tif', 10)

fresnel_prop = propagate_others(phase = img, attenuation = img2, fresnel_number =  10**(-2.5), task = 'learn_phase', dsf=2**0, positive_attenuation = 'relu', positive_phase = 'relu_inverted', transformation_type='ssim', abs_ratio = 1, model_type = 'wavelet', wavelet = 'db1', init_model = False, save_model_path = image_path, save_model = True, device = 'cuda:0', add_noise = False, noise_factor = 0.01, noise_type = 'gaussian', dis_depth = 2, dis_type = 'cnn')
visualize([fresnel_prop['transformed_images'], torch.abs(fresnel_prop['wavefield']), torch.abs(fresnel_prop['fresnel_factor'].imag)], title = ['input', 'wavefield', 'fresnel_factor'],  cmap = 'coolwarm', mode = 'show', axis = 'off', plot_axis = 'half', images_per_row=8, fig_size=(20,10), min_max = False)
fresnel_prop['device'] = 'cuda:1'
ssim_list, ground_ssim_list, ground_att_ssim_list = [], [], []
psnr_list, ground_psnr_list, ground_att_psnr_list = [], [], []
comparison = []
model = make_ganrec_model(**fresnel_prop)
model.train(iter_num = 400)
model.visualize(cmap = 'gray', vmode = 'zoom', axis = 'off', plot_axis = 'half', images_per_row=8, fig_size=(20,10), min_max = False)