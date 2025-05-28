
import json
import os
def join_dictionaries(dict2, base_dict):
    res = base_dict.copy()
    res.update(dict2)
    return res

setup_info = {'target': None,
 'second_model': False,
 'model_type': 'wavelet',
 'wavelet': 'db1',
 'use_diffusion': True,
 'diff_round': 5,
 'diff_round_counter': 0,
 'loss_box_size_factor': 1,
 'loss_box_number': 0,
 'update_rate': 2,
 'append_all': False,
 'Time_frame': 1,
 'v': None,
 'h': None,
 'transform_factor': 0.1,
 'transform_type': 'reshape',
 'pad': 2,
 'mode': 'reflect',
 'value': 'median',
 'device': 'cuda:0',
 'iter_num': 100,
 'abs_ratio': 0.001,
 'input_channels': 1,
 'output_num': 2,
 'task': 'learn_phase',
 'seed': 49,
 'fc_depth': 0,
 'units': 128,
 'cnn_depth': 0,
 'dis_depth': 0,
 'stride': 1,
 'depth': 6,
 'conv_num': 32,
 'conv_size': 3,
 'dropout': 0.25,
 'apply_batchnorm': True,
 'normal_init': True,
 'ground_truth': None,
 'ground_attenuation': None,
 'init_model': False,
 'add_noise': False,
 'noise_factor': 0.0036,
 'no_of_noises': 1,
 'kernel': None,
 'save_model_path': '/beegfs/desy/user/hailudaw/new_data/',
 'save_path': '/beegfs/desy/user/hailudaw/new_data/',
 'file_type': 'tif',
 'save_model': False,
 'ground_transform_type': 'reshape',
 'ground_atten_transform_type': 'reshape',
 'positive_phase': 'relu_inverted',
 'positive_attenuation': 'gelu',
 'downsampling_factor': 2**0,
 'cut': None,
 'horizontally': True,
 'vertically': True,
 'noise_type': 'gaussian',
 'scheduler_type': 'cosine',
 'print_result':False,
 'refine_param':False,
 'use_tqdm':True,
 'append_all':True,
 'update_rate':1,
 
 }

ratio = {'l1_ratio': 10, 'contrast_ratio': 0, 'normalized_ratio': 0, 'brightness_ratio': 0, 'reg_l1_ratio': 0, 'reg_l2_ratio': 0, 'contrast_normalize_ratio': 0, 'brightness_normalize_ratio': 0, 'l2_ratio': 0, 'fourier_ratio': 0, 'norm_ratio': 0, 'entropy_ratio': 1, 'real_loss_ratio': 1, 'fake_loss_ratio': 1}
dictionary = {'image':None, 'path':None, 'image_path': None, 'pad': 1, 'mode': 'reflect', 'task': 'learn_phase', 'alpha': 1e-8, 'delta_beta': 1e1, 'idx': None, 'idx': 0, 'save': False, 'transform_factor': 0.7, 'transform_type': 'reshape', 'save_format': 'tif', 'save_all': False, 'downsampling_factor': 1, 'fresnel_number': None, 'detector_pixel_size': None, 'distance_sample_detector': None, 'energy': None, 'phase': None, 'attenuation': None}
optional_kwargs = {"fresnel_number": None, "distance_sample_detector": None, "detector_pixel_size": None, "energy": None, 'pad': 2, "transform_type": "reshape", "transform_factor": 1, "mode": "reflect", "value": "mean", "downsampling_factor": 1, "wave_field": None, 'fresnel_factor': None, 'device': 'cuda:0', 'cut':None, 'horizontally': True, 'vertically': True}
    

json_file = join_dictionaries(join_dictionaries(join_dictionaries(setup_info, ratio), dictionary), optional_kwargs)
#create a json file with the all infos
def save_json(info, path = 'libraries/'):
    with open(path + 'info.json', 'w') as f:
        json.dump(info, f)
    return path + 'info.json'
# save_json(json_file)
if __name__ == "__main__":
    save_json(json_file)