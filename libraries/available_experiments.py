from visualize import *
from gauss_conv import *
from fresnel_propagator import *
from gan4inverse import *
from info import *

from skimage.data import shepp_logan_phantom, astronaut, camera, horse
from skimage.color import rgb2gray
import skimage.io as io
from skimage.draw import polygon
from PIL import Image, ImageDraw, ImageFont

setup_info = json_file
def one_matrix(n, m, i, j, r=0, shape='rectangle', letter = 'A'):
    A = np.zeros((n, m))
    A[i, j] = 1
    
    if r == 0:
        r = min(n, m) // 10
    
    if shape == 'rectangle':
        A[i-r//2:i+r//2, j-r//2:j+r//2] = 1
    elif shape == 'triangle':
        for x in range(n):
            for y in range(m):
                if np.abs(x-i) + np.abs(y-j) < r:
                    A[x, y] = 1
    elif shape == 'circle':
        y, x = np.ogrid[:n, :m]
        mask = (x - j)**2 + (y - i)**2 < r**2
        A[mask] = 1
    elif shape == 'gaussian circle':
        y, x = np.ogrid[:n, :m]
        A = np.exp(-((x - j)**2 + (y - i)**2) / (2 * r**2))    
    elif shape == 'letters':
        font_size = r * 5  # Make the font size proportional to r
        # Create a blank image with black background
        img = Image.new('L', (r*40, r*40), color=0)  # Grayscale 'L' mode, 0 is black
        draw = ImageDraw.Draw(img)

        # Load a default font or provide a .ttf font file
        try:
            font = ImageFont.truetype("/usr/share/fonts/gnu-free/FreeSansOblique.ttf", font_size)
        except IOError:
            font = ImageFont.load_default()

        # Get the bounding box of the letter and calculate its position
        bbox = draw.textbbox((0, 0), letter, font=font)
        text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]

        # Draw the letter centered in the image
        draw.text(((r*10) // 2, (r*10 - text_height) // 2), letter, fill=255, font=font)

        # Convert the image to a numpy array (scaled down to fit into matrix A)
        letter_array = np.array(img)

        # Normalize the array (make values between 0 and 1)
        letter_array = letter_array / 255

        # Determine the position where to place the letter array in A
        lx, ly = letter_array.shape
        i_start, i_end = max(0, i-lx//2), min(n, i+lx//2)
        j_start, j_end = max(0, j-ly//2), min(m, j+ly//2)

        # Place the letter_array into A
        A[i_start:i_end, j_start:j_end] = letter_array[:i_end-i_start, :j_end-j_start]
    
    return A

def multiple_one_matrices(p=10, n=128, m=128, r='any', shape='any', add_all = True, numbers = 1, letter = None, return_all = False):
    shapes = ['rectangle', 'circle', 'letters']
    if shape == 'letters' or shape == 'any':
        letters_available = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']#, 'phase', 'attenuation', 'inverse', 'future']
        letters = [np.random.choice(letters_available) for _ in range(p)] if letter is None else [letter for _ in range(p)]
    else:
        letters = [letter for _ in range(p)]

    if shape == 'any':
        random_shapes = [np.random.choice(shapes) for _ in range(p)]
    else:
        random_shapes = [shape for _ in range(p)]
    
    if numbers == 1:
        if r == 'any':
            A = [one_matrix(n, m, np.random.randint(0, n), np.random.randint(0, m), np.random.randint(0, min(n, m)//p + 20), shape=random_shapes[i], letter = str(letters[i])) for i in range(p)]
        else:
            A = [one_matrix(n, m, np.random.randint(0, n), np.random.randint(0, m), r, shape=shape, letter = str(letters[i])) for i in range(p)]
        if add_all:
            A_sum = np.sum(A, axis = 0)
            return A_sum if not return_all else (A_sum, p, n, m, r, shape)
        else:
            return A if not return_all else (A, p, n, m, r, shape)
    else:
        A = [multiple_one_matrices(p, n, m, r, shape, add_all, letter = letter) for i in range(numbers)]
        return A if not return_all else (A, p, n, m, r, shape)

def propagate_others(phase = None, attenuation = None, dsf = 1, abs_ratio = 1, fresnel_number = 0.0001, mode = 'reflect', positive_phase = 'relu_inverted', positive_attenuation = 'gelu', value = 'min', transform_type = 'reshape', add_noise =False, noise_factor = 0.0036, no_of_noises = 5, noise_type = 'gaussian', seed = 42, ground_transform_type = 'reshape', dict = setup_info, remove_extreme = False, **kwargs):
    assert phase is not None and attenuation is not None, "phase and attenuation must be in kwargs"
    if type(phase) is not torch.Tensor:
        if len(phase.shape) == 3:
            phase = rgb2gray(phase)
        if len(attenuation.shape) == 3:
            attenuation = rgb2gray(attenuation)
        dim = (int(phase.shape[0]//dsf), int(phase.shape[1]//dsf)) if 'dim' not in kwargs.keys() else kwargs['dim']
        if dim is None:
            dim = (kwargs['dim'], kwargs['dim']) if type(kwargs['dim']) is int else (kwargs['dim'][0], kwargs['dim'][1])
    else:
        dim = (int(phase.shape[2]//dsf), int(phase.shape[3]//dsf)) if 'dim' not in kwargs.keys() else kwargs['dim']
    
    fresnel_number = fresnel_number * dsf**2
    phase = pos_neg_image(phase, positive_phase, remove_extreme=remove_extreme)
    attenuation = pos_neg_image(attenuation, positive_attenuation, remove_extreme=remove_extreme)

    phase = Resize(size = dim, antialias=True)(phase)
    attenuation = Resize(size = dim, antialias=True)(attenuation) * abs_ratio

    simulation_sphere = {
        'experiment_name': 'simulation spheres',
        'phase': phase,
        'attenuation': attenuation,
        'fresnel_number': fresnel_number,
        'dsf': 1,
        'mode': mode,
        'pad' : 2,
        'abs_ratio': abs_ratio,
        'ground_truth': phase,
        'ground_attenuation': attenuation,
        'transform_type': transform_type,
        'ground_transform_type': ground_transform_type,
        'positive_phase': positive_phase,
        'positive_attenuation': positive_attenuation,
        'value': value,
        'add_noise': add_noise,
        'noise_factor': noise_factor,
        'no_of_noises': no_of_noises,
        'seed': seed,
        'noise_type': noise_type,
        'cut': None if 'cut' not in kwargs.keys() else kwargs['cut'],
        'horizontally': True if 'horizontally' not in kwargs.keys() else kwargs['horizontally'],
        'vertically': True if 'vertically' not in kwargs.keys() else kwargs['vertically'],
        'device': 'cuda:0' if 'device' not in kwargs.keys() else kwargs['device'],
        'iter_num': 100 if 'iter_num' not in kwargs.keys() else kwargs['iter_num'],
    }

    simulation_sphere = join_dict(join_dict(kwargs, simulation_sphere), json_file)
    prop = Fresnel_propagation(**simulation_sphere)
    prop.image = prop(phase = simulation_sphere['phase'], attenuation = simulation_sphere['attenuation'])
    simulation_sphere['image'] = prop.image
    if add_noise:
        if noise_type == 'random':
            noise_type = np.random.choice(['gaussian', 'poisson', 'speckle', None], 1)[0]
            noise_factor = np.random.uniform(0, 0.2, 1).item()
            
        if noise_type == 'gaussian':
            # print("Adding Gaussian noise with factor:", noise_factor)
            simulation_sphere['image'] = torch_noise_gaussian(prop.image, noise_factor)
        elif noise_type == 'poisson':
            simulation_sphere['image'] = torch_noise_poisson(prop.image, noise_factor, torch.Generator(device='cpu').manual_seed(seed))
        elif noise_type == 'speckle':
            simulation_sphere['image'] = torch_noise_speckle(prop.image, noise_factor)
        elif noise_type == 'all':
            simulation_sphere['image'] = torch_noise_gaussian(torch_noise_poisson(prop.image, noise_factor, torch.Generator(device='cpu').manual_seed(seed)), noise_factor)
            
            
        noise_type = 'gaussian' if noise_type is None else noise_type
        simulation_sphere['noise_factor'] = noise_factor
        simulation_sphere['noise_type'] = noise_type
        simulation_sphere['add_noise'] = True if noise_type is not None else False
    else:
        simulation_sphere['image'] = prop.image       
    simulation_sphere['path'] = simulation_sphere['image'] 
    simulation_sphere['image'] = pos_neg_image(simulation_sphere['image'], transform_type, remove_extreme=remove_extreme)
    simulation_sphere['transformed_images'] = simulation_sphere['image']
    simulation_sphere['wavefield'] = prop.wavefield
    simulation_sphere['fresnel_factor'] = prop.fresnel_factor
    return simulation_sphere

def sim_bubbles_prop(n = 512, m = 512, p= 30,  r = 'any', shape = 'any', numbers = 2, add_all = True, letter = None, fresnel = 1e-3, abs_ratio = 1, transform_type = 'reshape', positive_attenuation='pos', positive_phase='neg', mode = 'reflect', dsf = 1, value = 'constant', **kwargs):
    # n,m,p,r,shape, add_all, numbers, letter = 128, 128, 10, 2, 'any', True, 2, None
    A_sum, B_sum = multiple_one_matrices(p, n, m, r, shape, add_all, numbers, letter) 
    fresnel_numbers = [fresnel] if type(fresnel) is not list else fresnel
    prop = [propagate_others(A_sum, B_sum, fresnel_number=f, abs_ratio=abs_ratio, transform_type = transform_type, positive_phase = positive_phase, positive_attenuation=positive_attenuation, mode = mode, value = value, dsf = dsf, **kwargs) for f in fresnel_numbers]    
    return prop[0] if type(fresnel) is not list else prop

class available_experiments():
    def __init__(self, name = None, dictionary = None):
        self.experiments = ['spider_hair', 'spider_hair_best', 'sami_mouse_good', 'sami_best_distance', 'sami_farther', 'sami_closer', 'info_sami_spider', 'simulation_sphere', 'jh', 'melting', 'ivory', 'volcano', 'bubble']
        self.name = name
        if dictionary is not None:
            self.kwargs = dictionary
        else:
            self.kwargs = None
    
    def list_experiments(self):
        print(self.experiments)

    def __call__(self, **kwargs):
        self.kwargs = kwargs
        return self
    def cookie(self):
        image = plt.imread('/data/hereon/wp/user/hailudaw/git_folders/CUQIpy/cuqi/data/cookie.png')
        image = image[..., :3]
        image = rgb2gray(image)
        return image

    def spider_hair(self, idx = [120], setup_info=setup_info, **kwargs):
        image_path = '../data/spider/data_spider.tif'
        idx = [idx] if type(idx) is int else idx
        images = io.imread(image_path)
        if len(idx) >= 1:
            images = [images[i] for i in idx]
            images = np.stack(images)
        spider_hair = {
            'path': images,
            'experiment_name': 'spider_hair',
            'energy_kev': 11,
            'pv': 1.04735263e-7,
            'z': 7.8880960e-2,
            'dsf': 2**0,
            'abs_ratio': 0.05,
            'image_path': image_path,
        }
        return join_dict(join_dict(kwargs,  spider_hair), setup_info, False)
    
    def spider_hair_best(self, idx = [120], setup_info = setup_info, **kwargs):
        image_path = '../data/spider/data_spider.tif'
        idx = [idx] if type(idx) is int else idx
        images = io.imread(image_path)
        if len(idx) > 1:
            images = [images[i] for i in idx]
        spider_hair_best = {
            'path': images,
            'experiment_name': 'spider_hair_best',
            'energy_kev': 11.2999999,
            'pv': 1.04735263e-7,
            'z': 7.8880960e-2,
            'fresnel_number': 0.0012584630069243095,
            'abs_ratio':0.006866553519949974,
            
        }
        return join_dict(join_dict(kwargs, spider_hair_best), setup_info)

    def spider_jdora(self, **kwargs):
        image = io.imread('/data/hereon/wp/user/hailudaw/git_folders/Gans4Inverse/data/jdora/data/holograms/spider_hair.tiff')/1.1
        spider_jdora = {
            'image': image,
            'experiment_name': 'spider_jdora',
            'energy': 11.0,
            'pv': 6500.0,
            'z02': 19661.0*1e6,
            'z01': 79950000.0,
            'dsf': 2**0,
            'name': 'spider_jdora',
        }
        spider_jdora['fresnel_number'] = get_fresnel_from_cone(**spider_jdora)
        return join_dict(kwargs, spider_jdora)

    def jd_mg(self, setup_info=setup_info, **kwargs):
        # (energy=11.0, px_size=6500.0, z02=1.9661e10 )
        image_path = '../data/smart_phase/magnesium_wire.tiff'
        mg = {
            'image': np.transpose(io.imread(image_path))/1.1,
            'experiment_name': 'mg',
            'energy': 11.0,
            'pv': 6500.0,
            'z02': 1.9661e10,
            'z01': 470515625,   
            'dsf': 2**0,
            'name': 'mg_wire',
            'image_path':'../data/holograms/magnesium_wire.tiff',
        }
        mg['fresnel_number'] = get_fresnel_from_cone(**mg)
        return join_dict(join_dict(kwargs, mg), setup_info)
    
    def jd_spider_hair(self, setup_info=setup_info, **kwargs):
        # (energy=11.0,  px_size=6500.0, z02=19661.0*1e6 )
        spidy = {
            'image': np.flip(np.transpose(io.imread('../data/smart_phase/spider_hair.tiff')))/1.1,
            'experiment_name': 'spider_hair',
            'energy': 11.0,
            'pv': 6500.0,
            'z02': 19661.0*1e6,
            'z01': 79950000.0,
            'dsf': 2**0,
        }
        spidy['fresnel_number'] = get_fresnel_from_cone(**spidy)
        return join_dict(join_dict(kwargs, spidy), setup_info)
    
    def jd_tooth(self, setup_info=setup_info, **kwargs):
        # (energy=17.0,  px_size=6500.0, z02=19652000000.0 )
        tooth = {
            'image': np.flip(np.transpose(io.imread('../data/smart_phase/tooth.tiff'))/1.1),
            'experiment_name': 'tooth',
            'energy': 17.0,
            'pv': 6500.0,
            'z02': 19652000000.0,
            'z01': 81708500.0,
            'dsf': 2**0,
            'abs_ratio': 0.0067,
            'name': 'tooth',
        }
        tooth['fresnel_number'] = get_fresnel_from_cone(**tooth)
        return join_dict(join_dict(kwargs, tooth), setup_info)

    def mg_screw_sim(self):
        attenuation_orig = io.imread('../data/screw/ground_truth_projection_attenuation_50000_eV_00002_type_float32_4096x4096.tif')
        phase_orig = io.imread('../data/screw/ground_truth_projection_phase_50000_eV_00002_type_float32_4096x4096.tif')

        energy = 50
        pv = 0.00016 *1e-3
        z  = 1858301 * 1e-6
        fresnel_number = fresnel_calculator(energy_kev= energy, detector_pixel_size=pv, distance_sample_detector=z)
        print('fresnel_number', fresnel_number, 'distance_sample_detector', z, 'detector_pixel_size', pv)

        d = 4
        size = (int(attenuation_orig.shape[0]/d), int(attenuation_orig.shape[1]/d))
        attenuation = resize_with_diff_interpolation(attenuation_orig, size)    
        phase = resize_with_diff_interpolation(phase_orig, size)
        abs_ratio = np.mean(attenuation)
        print('shape', phase.shape, attenuation.shape)
        
        # noise_factors = [1, 10, 50, 100, 1000]
        save_path =  '../data/screw/reconstruction/'
        save_name = 'new_bone_screw'

        mg = propagate_others(tensor_to_np(phase), tensor_to_np(attenuation), fresnel_number=fresnel_number, abs_ratio= abs_ratio,  ground_transform_type='reshape', ground_atten_transform_type = 'reshape', transform_type = 'reshape', positive_phase = 'relu_inverted', positive_attenuation = 'gelu', mode = 'reflect',  value = 'median', dsf =2**0,  add_noise = False, noise_type = 'poisson', noise_factor = 0.2, cut = 3)
        mg['abs_ratio'] = abs_ratio*mg['abs_ratio']
        mg['add_noise'] = False
        mg['save_name'] = save_name
        mg['save_path'] = save_path
        return join_dict(mg, setup_info, False)
    
    def propagate_sphere(self, idx = 1,  dsf = 4, abs_ratio = 1, fresnel_number = 0.01, mode = 'reflect', value = 'min', transform_type = 'normalize', add_noise =False, noise_factor = 0.0036, no_of_noises = 5, folder = '/data/hereon/wp/user/jentscht/for_dawit/groundtruth/ratio_phase_attenuation_ratio_1e-2_double_sphere_30000_eV/ground_truth_projection/', file_type = 'tif', **kwargs):
        images = io.imread_collection(folder + '*.'+ file_type).files
        phase_word = '_projection_phase'
        phase = io.imread([element for element in images if phase_word in element][idx])

        attenuation_word = '_projection_attenuation'
        attenuation = io.imread([element for element in images if attenuation_word in element][idx])

        phase =  transform(phase, transform_type)
        attenuation = transform(attenuation, transform_type)
        return available_experiments.propagate_others(self, tensor_to_np(phase), tensor_to_np(attenuation), dsf, abs_ratio, fresnel_number, mode, value, 'reshape', add_noise, noise_factor, no_of_noises, **kwargs)
         
    def propagate(self, **kwargs):
        assert 'phase' in kwargs.keys() and 'attenuation' in kwargs.keys(), "phase and attenuation must be in kwargs"
        propagator = Fresnel_propagation(**kwargs)
        propagated_intensity = propagator.image
        return propagated_intensity

    def __repr__(self):
        return f"available experiments: {self.experiments}"
    
    def get_experiment(self, name = None, dictionary = None):
        if name is None:
            name = self.name
        else:
            assert name in self.experiments, "name not in experiments"

        exists = True
        for experiment in self.experiments:
            if experiment == name:
                kwargs = getattr(self, experiment)()
                if dictionary is not None:
                    kwargs.update(dictionary)
                return kwargs
            else:
                exists = False
        if exists is False:
            if dictionary is not None:
                kwargs = dictionary
                return kwargs            
            
    def get_data(self, name = None, dictionary= None, **kwargs):
        """"
        names = ['spider_hair', 'spider_hair_best', 'sami_mouse_good', 'sami_best_distance', 'sami_farther', 'sami_closer', 'info_sami_spider', 'simulation_sphere', 'jh']
        """
        kwargs = self.get_experiment(name, dictionary)
        kwargs = prepare_dict(**kwargs)
        if 'task' in kwargs.keys():
            if kwargs['task'] == 'learn_phase':
                kwargs['image'] = Fresnel_propagation(**kwargs).forward()
        return kwargs
  
experiments = available_experiments()