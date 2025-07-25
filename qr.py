import sys
"""
This script is a quick test for SelfPhish, which runs a simulation and visualizes the results.
You can run it directly after setting up the environment, or copy it to your working directory and adjust the paths accordingly.
Make sure to have the necessary libraries installed and the paths set correctly.
To run the script, simply execute:
```bash
python qr.py
```
You can modify the parameters in the `sim_bubbles_prop` function to suit your specific experiments.
Example:

- image size: 256x256
- number of bubbles: 10
- fresnel number: 1e-3
- positive phase: 'relu_inverted'
- positive attenuation: 'relu'
- absolute ratio: 1e-2
- downsampling factor: 1

This will generate a simulation with the specified parameters, train the model for 200 iterations, and visualize the results in a grid format.
"""
path_to_git = 'SelfPhish/libraries' if 'SelfPhish' in sys.path else '/data/hereon/wp/user/hailudaw/git_folders/paper_selfphish/SelfPhish/libraries'
if 'SelfPhish' not in sys.path:
    sys.path.append(path_to_git)
else:
    assert sys.path[0] == path_to_git, "Path to SelfPhish is not set correctly."

from libraries.available_experiments import sim_bubbles_prop
from libraries.gan4inverse import solver

sim = sim_bubbles_prop(256, 256, 10, 
    fresnel=5e-3, 
    positive_attenuation='relu',
    positive_phase='relu_inverted',
    transformation_type='reshape',
    abs_ratio=1,
    model_type='wavelet',
    init_model=False,
    device='cuda:0',
    add_noise=False,
    noise_factor=0.01,
    noise_type='gaussian',
    dis_type='cnn',
    dis_depth=2,
    dsf=1,
)
model = solver(**sim)
model.train(iter_num = 400)
model.visualize(images_per_row=3, vmode = 'show', title_fontsize = 5, colorbar = False, axis = 'off')
