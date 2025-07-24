import json
import os

all_config = {
    "ml": {
        'second_model': False,
        'model_type': 'unet',
        'use_diffusion': True,
        'diff_round': 5,
        'diff_round_counter': 0,
        'loss_box_size_factor': 1,
        'loss_box_number': 0,
        'update_rate': 1,
        'append_all': True,
        'input_channels': 1,
        'output_num': 2,
        'fc_depth': 0,
        'units': 64,
        'cnn_depth': 0,
        'dis_depth': 0,
        'stride': 1,
        'depth': 4,
        'base': 64,
        'conv_num':32,
        'conv_size': 3, 
        'dropout': 0.25,
        'apply_batchnorm': True,
        'normal_init': True,
        'init_model': False,
        'add_noise': False,
        'noise_factor': 0.0036,
        'no_of_noises': 1,
        'kernel': None,
        'noise_type': 'gaussian',
        'scheduler_type': 'cosine',
        'dis_type': 'cnn',
        'use_checkpoint':True,
        'seq_once':False,
    },
    "loss": {
        'l1_ratio': 10, 'l2_ratio': 0, 'reg_l1_ratio': 0, 'reg_l2_ratio': 0, 'contrast_normalize_ratio': 0, 
        'fourier_ratio': 0, 'norm_ratio': 0, 'entropy_ratio': 1, 'brightness_normalize_ratio': 0,
        'real_loss_ratio': 1, 'fake_loss_ratio': 1, 'vgg_loss_ratio': 0,
        'ssim_loss_weight': 0, 'psnr_loss_weight': 0,
    },
    "phase_retrieval": {
        'task': 'learn_phase',
        'ground_truth': None,
        'ground_attenuation': None,
        'dsf': 1, 'abs_ratio': 0.001,
        'cut': None,
        'horizontally': True,
        'vertically': True,
        'image': None, 'path': None, 'image_path': None, 'pad_times': 2,
        'alpha': 1e-8, 'delta_beta': 1e1, 'save': False, 'save_format': 'tif', 'save_all': False,
        'fresnel_number': None, 'detector_pixel_size': None, 'distance_sample_detector': None,
        'energy': None, 'phase': None, 'attenuation': None, 'wave_field': None, 'fresnel_factor': None,
        'idx': [30 * i for i in range(1)],
        'multidistance': False,
        'average_I': False,
        'average_P': False,
    },
    "transform": {
        'wavelet': 'db1',
        'transform_factor': 0.1,
        'transform_type': 'reshape',
        'pad': 2,
        'mode': 'reflect',
        'value': 'median',
        'tile_size': None,
        'overlap': None,
        'tile': False,
        'ground_transform_type': 'reshape',
        'ground_atten_transform_type': 'reshape',
        'positive_phase': 'relu_inverted',
        'positive_attenuation': 'relu',
    },
    "paths": {
        'save_model_path': '/beegfs/desy/user/hailudaw/new_data/',
        'save_path': '/beegfs/desy/user/hailudaw/new_data/',
        'file_type': 'tif',
        'save_model': False,
    },
    "visualization": {
        'cmap': '   ',
        'colorbar': True,
        'colorbar_normalize': False,
        'axis': 'off',
        'title': 'no_title',
        'colorbar_location': 'bottom',
        'vmode': 'all',
        'min_max': True,
        'zoomout_location': 'bottom 3',
        'zoomout_position': 'left',
        'move_vs': [-0.08, -0.08, 0.2],
        'move_hs': [-0.23, -0.05, 0.13],
        'sa_wspace': 0.01,
        'sa_hspace': 0.01,
        'shrink': 0.5,
        'fontsize': 33,
        'min_max': False, 'fontsize': 33, 'cmap': 'gray', 'label_size': 40, 
        'colorbar_size_factor': 100, 'legend_location': 'right', 
        'second_title': 'no_title', 'second_title_color': 'w', 
        'second_title_fontsize': 34, 'title_x': 0.5, 'title_y': 1, 
        'second_title_y': 0.05, 'title_horizontalalignment': 'center',
        'text_size':'small', 
    },
    "metric_column_map" : {
        'SSIM (I, I*)': 'ssim_list',
        'PSNR (I, I*)': 'psnr_list',
        'L1 (I,I*)': 'main_diff',
        'SSIM (Φ, φ)': 'ground_ssim_list',
        'PSNR (Φ, φ)': 'ground_psnr_list',
        'L1 (Φ, φ)': 'ground_main_diff_list',
        'SSIM (A, A*)': 'A_SSIM',
        'PSNR (A, A*)': 'A_PSNR',
        'L1 (A, A*)': 'A_L1'
    },
    "metric_final_column_map" : {
        'SSIM (I, I*)': 'final_ssim',
        'PSNR (I, I*)': 'final_psnr',
        'L1 (I,I*)': 'final_main_diff',
        'SSIM (Φ, φ)': 'final_ground_ssim',
        'PSNR (Φ, φ)': 'final_ground_psnr',
        'L1 (Φ, φ)': 'final_ground_main_diff',
        'SSIM (A, A*)': 'final_A_SSIM',
        'PSNR (A, A*)': 'final_A_PSNR',
        'L1 (A, A*)': 'final_A_L1'
    },
    "run": {
        'device': 'cuda:0',
        'iter_num': 100,
        'seed': 49,
        'print_result': False,
        'refine_param': False,
        'use_tqdm': True,
        'batch_size': 4,
        'ex_name': 'mg',
        'iter_nums': [10, 10, 10, 10, 10],
        'dsfs': [16, 8, 4, 2, 1],
        'iter_factor': 100,
        'strategy': 0,
        'conditions': ['mask', 'no_mask', 'direct'],
    },
}

class CONFIG:
    def __init__(self, config=None):
        self.config = config.copy() if config is not None else all_config.copy()

    def as_dict(self):
        return self.config

    def save(self, path='libraries/', filename='info.json'):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, filename), 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def forward(self):
        combined = {}
        for section in self.config.values():
            combined.update(section)
        return combined

json_kwargs = CONFIG()
json_kwargs.save()
json_file = json_kwargs.forward()

# def save_json(config, path='libraries/'):
#     os.makedirs(path, exist_ok=True)
#     with open(os.path.join(path, 'info.json'), 'w') as f:
#         json.dump(config, f, indent=2)

# if __name__ == "__main__":

