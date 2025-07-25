import sys
import os
import importlib
# Get the absolute path to the 'libraries' directory (one level up from 'examples')
current_dir = os.path.dirname(os.path.abspath(__file__))
libraries_dir = os.path.abspath(os.path.join(current_dir, '..', 'libraries'))
examples_dir = current_dir  # Absolute path to 'examples' directory
project_root = os.path.abspath(os.path.join(current_dir, '..'))

# Append the 'libraries' directory to sys.path if not already present
if libraries_dir not in sys.path:
    sys.path.append(libraries_dir)
# Append the 'examples' directory to sys.path if not already present
if examples_dir not in sys.path:
    sys.path.append(examples_dir)
# Optionally, append the project root if not already present
if project_root not in sys.path:
    sys.path.append(project_root)
from libraries.available_experiments import *
if __name__ == "__main__":
    print(f"Added {libraries_dir} to sys.path")
    available_experiments = importlib.import_module('available_experiments')
    gan4inverse = importlib.import_module('gan4inverse')
    visualize = importlib.import_module('visualize')

    jd = available_experiments.experiments.jd_mg(downsampling_factor = 2**1, positive_phase = 'relu_inverted', positive_attenuation = 'relu', transformation_type = 'leakyrelu', model_type = 'unet', abs_ratio = 5e-4, dis_depth = 2, dis_type = 'cnn', device = 'cuda:1', pad_times=2, pad = 1)
    jd['path'] = jd['path'].transpose()
    jd['path'].shape
    model = gan4inverse.solver(**jd)
    model.train(iter_num = 20)

    visualize.visualize([model.transformed_images, model.phase, -1*torch.log(model.attenuation)], title = ['Given hologram', 'Phase', 'Absorbance'], cmap = 'gray', images_per_row = 3, vmode = 'zoom', zoomout_location = 'bottom 3', axis = 'off', colorbar = True, colorbar_location = 'right', axin_axis = False, fontsize = 40, label_size = 30, min_max = False, move_hs = [-0.2,0.1,0.2], move_vs = [-0.05, 0.23, -0.05], pad = -0.02)