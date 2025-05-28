from libraries.visualize import *
from libraries.gauss_conv import *
from libraries.fresnel_propagator import *
from libraries.gan4inverse import *
from libraries.info import *

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
        letters_available = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'phase', 'attenuation', 'inverse', 'future']
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
            simulation_sphere['image'] = torch_noise_gaussian(prop.image, noise_factor)
        elif noise_type == 'poisson':
            simulation_sphere['image'] = torch_noise_poisson(prop.image, noise_factor, torch.Generator(device='cpu').manual_seed(seed))
        elif noise_type == 'speckle':
            simulation_sphere['image'] = torch_noise_speckle(prop.image, noise_factor)
            
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

def sim_bubbles_prop(n = 512, m = 512, p= 30,  r = 'any', shape = 'any', numbers = 2, add_all = True, letter = None, fresnel = 1e-3, abs_ratio = 1e-3,transform_type = 'reshape', positive_attenuation='relu', positive_phase='neg', mode = 'reflect', dsf = 1, value = 'constant', **kwargs):
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
    
    def lopez(self, setup_info=setup_info, **kwargs):
        # import sys
        # sys.path.append('/data/hereon/wp/user/hailudaw/pandas_for_petra/')
        # from create_csv import *
        # corrected_path =  '/gpfs/petra3/scratch/marinhoa/data_for_ai/new_images/flatfield_corrected/'
        # retrieved_path = '/gpfs/petra3/scratch/marinhoa/data_for_ai/new_images/phase_retrieved/'
        # corrected = create_csv_file(corrected_path, file_type='tiff', depth =4)['path']
        # phase_retrieved = create_csv_file(retrieved_path, file_type='tiff', depth =4)['path']
        corrected_path = pd.read_csv('/data/hereon/wp/user/hailudaw/git_folders/commits/Gans4Inverse/data/lopez/corrected_pandas.csv')['path']
        retrieved_path = pd.read_csv('/data/hereon/wp/user/hailudaw/git_folders/commits/Gans4Inverse/data/lopez/phase_retrieved_pandas.csv')['path']
        
        #we will now remove allready processes files from the list
        # path_of_processed_files = '/data/hereon/wp/user/hailudaw/new_data/new_folder/phase/'
        # processed_files = os.listdir(path_of_processed_files)
        # processed_files = [file.split('_')[0] for file in processed_files]
        # corrected_path = corrected_path[~corrected_path.isin(processed_files)]
        # retrieved_path = retrieved_path[~retrieved_path.isin(processed_files)]
        
        if 'idx' in kwargs:
            idx = kwargs['idx']
            if type(kwargs['idx']) is int:
                idx = [kwargs['idx']]
        else:
            idx = [300]
            
        
        try:
            image_path = list(corrected_path.iloc[idx].values) if idx is not None else list(corrected_path.values)
            ground_image_path = list(retrieved_path.iloc[idx].values) if idx is not None else list(retrieved_path.values)
            
            images = load_images_parallel(image_path, divide_by='mean')
            ground_images = load_images_parallel(ground_image_path)

        except:
            image_path = corrected_path.iloc[idx]
            images = [io.imread(image_path[i]) for i in range(len(image_path))]
            images = [im/np.median(im) for im in images]
            ground_image_path = retrieved_path.iloc[idx]
            ground_images = [io.imread(ground_image_path[i]) for i in range(len(ground_image_path))]
            ground_images = [im/np.median(im) for im in ground_images]
            
        # images = [im/np.median(im) for im in images]
        # ground_images = [im/np.median(im) for im in ground_images]
        lopez = {
            'experiment_name': 'lopez',
            'path': images,
            'energy': 11.0,
            'pv': 6500.0,
            'z02': 19930915635.601063,
            'z01': 290914035.60106295,
            'dsf': 2**0,
            'transform_type': 'reshape',
            'abs_ratio': 0.0007780000000000002,
            'ground_truth': ground_images,
            'ground_truth_path': ground_image_path,
            'image_path': image_path,
            'name': 'lopez',
        }
        lopez['fresnel_number'] =  get_fresnel_from_cone(**lopez)
        res = join_dict(join_dict(kwargs, lopez), setup_info)
        return res

    def jh(self, setup_info=setup_info, **kwargs):
        jh ={
            'experiment_name': 'nano3649_holotomo_Mg2Ag_corr_1',
            'path': io.imread('/data/hereon/wp/user/hailudaw/git_folders/ganrec/ganrectorch/data/JH/test_data_nano3649_holotomo_Mg2Ag_corr_1_130_angle.tiff'),
            'energy': energy_from_wavelength(0.1127).magnitude, 
            'fresnel_number': 5.3230e-04, #0.0005314791443087848 ,
            'dsf': 2**0,
            'abs_ratio' : 0.001,
            'name': 'jh_mg2ag',
        }
        res = join_dict(join_dict(kwargs, jh), setup_info)
        return res
        
    def cactus(self, setup_info=setup_info, **kwargs):
        cac = {
            'experiment_name': 'cactus',
            'path': io.imread('/data/hereon/wp/user/hailudaw/git_folders/Gans4Inverse/data/cactus/test_data_cactus_seed_large_dist_ln_flat.tiff'),
            'energy': energy_from_wavelength(0.0620).magnitude,
            'fresnel_number': 0.0022,
            'dsf': 1,
            'saving_path': '/data/hereon/wp/user/hailudaw/job_path/cactus/',
            'name': 'cactus',
        }
        return join_dict(join_dict(kwargs, cac), setup_info)

    def melting(self, idx = 0, comp_idx = None, setup_info=setup_info, compartment = False, crop = False, row = None, col = None, pad_by = None, cut_x = None, cut_y = None, min_value = 0, max_value = 5, replace = 0, replace_value = 1,med_ker_size = 3, **kwargs):
        if type(idx) != list:
            idx = [idx]
        row = 4 if row is None else row
        col = 4 if col is None else col
        pad_by = 10 if pad_by is None else pad_by
        cut_x = 450 if cut_x is None else cut_x
        cut_y = 350 if cut_y is None else cut_y
        
        # path = '/asap3/petra3/gpfs/p61/2023/data/11017784/processed/sample_bis_corrected/'
        # path = '/asap3/petra3/gpfs/p61/2023/data/11017784/processed/ss316_6066_1/sample_acq1/im_corr_acq1/'
        path = '/asap3/petra3/gpfs/p61/2023/data/11017784/processed/ss316_6066_1/sample_acq2/im_corr_acq2'
        ground_path = '/asap3/petra3/gpfs/p61/2023/data/11017784/processed/ss316_6066_1/sample_acq2/mba_acq2'
        images, image_path = get_image(path, idx, 'tif')
        ground_images, ground_image_path = get_image(ground_path, idx, 'tif')
        images = tensor_to_np(torch.transpose(pos_neg_image(median_filter_torch(images, med_ker_size,True),'normalize_min_max'), 2, 3))
        ground_images = tensor_to_np(torch.transpose(pos_neg_image(median_filter_torch(ground_images, med_ker_size,True), 'normalize_min_max'), 2, 3))
        if type(images) is not list:
            if len(images.shape)>2:
                images = [images[i] for i in range(images.shape[0])]
                ground_images = [ground_images[i] for i in range(ground_images.shape[0])]
            else:
                images = [images]
                ground_images = [ground_images]
        # images = [filter_values(images[i], min_value, max_value, replace, replace_value) for i in range(len(images))] if type(images) is list else filter_values(images, min_value, max_value, replace, replace_value)
        # melting['path'] = [filterout_artifacts(np.transpose(melting['path'][i]), use_median = True, use_gaussian = False, kernel_size = 3, min_threshold = 99.995, max_threshold = 0.005) for i in range(len(melting['path']))]
        # images = [filterout_artifacts(images[i], use_median = True, use_gaussian = False, kernel_size = 3, min_threshold = 99.9995, max_threshold = 1e-2) for i in range(len(images))]
        # ground_images = [filterout_artifacts(ground_images[i], use_median = True, use_gaussian = False, kernel_size = 3, min_threshold = 99.9995, max_threshold = 0.005) for i in range(len(ground_images))]
        if crop:
            left = kwargs['left'] if 'left' in kwargs else True
            right = kwargs['right'] if 'right' in kwargs else True
            top = kwargs['top'] if 'top' in kwargs else True
            bottom = kwargs['bottom'] if 'bottom' in kwargs else True
            images = tensor_to_np(crop_on_all_sides(images, cut_x, cut_y, replace_value = None,  left = left, right = right, top = top, bottom = bottom))
            ground_images = tensor_to_np(crop_on_all_sides(ground_images, cut_x, cut_y, replace_value = None,  left = left, right = right, top = top, bottom = bottom))
        if compartment:
            assert len(idx) == 1, 'only one image is allowed for compartment'
            print(images.shape) if type(images) is np.ndarray else [print(images[i].shape) for i in range(len(images))]
            images_com = [compartment_image(np.transpose(images[i]), row, col, pad_by) for i in range(len((images)))] if type(images) is list else compartment_image((images), row, col, pad_by)
            ground_images_com = [compartment_image(np.transpose(ground_images[i]), row, col, pad_by) for i in range(len((ground_images)))] if type(ground_images) is list else compartment_image((ground_images), row, col, pad_by)
            if comp_idx is not None:
                comp_idx = [comp_idx] if type(comp_idx) is int else comp_idx
                images_com = [images_com[i] for i in comp_idx]
            images = [torch_reshape(images_com[i]) for i in range(len(images_com))]
            ground_images = [torch_reshape(ground_images_com[i]) for i in range(len(ground_images_com))]
            # print(images.shape) if type(images) is np.ndarray else [print(images[i].shape) for i in range(len(images))]
            images = [images[i][j,0,:,:] for i in range(len(images)) for j in range(images[i].shape[0])]
            ground_images = [ground_images[i][j,0,:,:] for i in range(len(ground_images)) for j in range(ground_images[i].shape[0])]
        melting = {
            'experiment_name': 'radiography of melting material',
            'path': images,
            'image_path': image_path,
            'idx': None,
            'fresnel_number': 0.28,
            'dsf': 1,
            'abs_ratio': 1e-2,
            'name': 'melting',
            'row': row,
            'col': col,
            'positive_attenuation': True,
            'positive_phase': 'relu_inverted',
            'transform_type': 'leakyrelu',
            'model_type': 'wavelet',
            'wavelet': 'db4',
            'save_path': '/data/hereon/wp/user/hailudaw/new_data/',
            'ground_truth': ground_images,
            'ground_truth_path': ground_image_path,
        
        }
        return join_dict(join_dict(kwargs, melting), setup_info, True)

    def ivory(self, setup_info=setup_info, **kwargs):
        ivory = {
            'experiment_name': 'nano1588_inline_holotomo_A001_Ivory_continiued',
            'path': io.imread('/data/hereon/wp/user/hailudaw/git_folders/Gans4Inverse/data/ivory/test_data_nano1588_ivory_401_angle_dist_1.tif'),
            'energy': energy_from_wavelength(0.1127).magnitude,
            'fresnel_number': 1.9485e-04,
            'dsf': 2**0,
            'abs_ratio': 0.000073,
            'name': 'ivory',
        }
        ivory['path'] = ivory['path'][1800:3000, 1250:2600]
        ivory['path'] = tensor_to_np(rotate(torch_reshape(ivory['path']), 180))
        return join_dict(join_dict(kwargs, ivory), setup_info, True)

    def volcano(self, setup_info=setup_info, **kwargs):
        volcano = {
            'experiment_name': 'nano3603_holo_4a_vulcano_crystal_2',
            'path': io.imread('/data/hereon/wp/user/hailudaw/git_folders/Gans4Inverse/data/volcano/test_data_nano3603_holo_4a_vulcano_crystal_2_20_angle.tiff'),
            'energy': energy_from_wavelength(0.1127).magnitude,
            'fresnel_number': 1.2838e-04,
            'dsf': 2**0,
            'abs_ratio': 0.001,
            'name': 'volcano',
        }
        return join_dict(join_dict(kwargs, volcano), setup_info, True)

    def volcano_julian(self, setup_info=setup_info, **kwargs):
        volcano_julian = {
            'experiment_name': 'nano3603_holo_4a_vulcano_crystal_2',
            'path': io.imread('/data/hereon/wp/user/hailudaw/git_folders/Gans4Inverse/data/volcano/test_data_nano3603_holo_4a_vulcano_crystal_2_20_angle.tiff'),
            'energy': energy_from_wavelength(0.1127).magnitude,
            'fresnel_number': 1.2838e-04,
            'dsf': 2**0,
            'abs_ratio': 0.001,
            'name': 'volcano',
        }
        return join_dict(join_dict(kwargs, volcano_julian), setup_info, True)
    
    def bubble(self, setup_info=setup_info, **kwargs):
        bubble = {
            'experiment_name': 'nano2807_bubble',
            'path': io.imread('/data/hereon/wp/user/hailudaw/git_folders/Gans4Inverse/data/bubble/test_data_p2807_bubble_trId_1197456729.tif'),
            'energy': energy_from_wavelength(0.0688).magnitude,
            'fresnel_number': 7.2383e-04,
            'dsf': 1,
            'abs_ratio': 0.001,
            'name': 'bubble',
        }
        return join_dict(join_dict(kwargs, bubble), setup_info, True)

    def spider_hair(self, idx = [120], setup_info=setup_info, **kwargs):
        image_path = 'data/ganrec/data_spider.tif'
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
        return join_dict(join_dict(kwargs,  spider_hair), setup_info, True)
    
    def spider_hair_best(self, idx = [120], setup_info = setup_info, **kwargs):
        image_path = 'data/ganrec/data_spider.tif'
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

    def spider_big(self, setup_info = setup_info, **kwargs):
        image_path = '/data/hereon/wp/user/hailudaw/git_folders/Gans4Inverse/data/spider/test_data_nano3637_holotomo_step_R4B_highMag_100_angle.tiff'
        image = io.imread(image_path)
        spider_big = {
            'path': image,
            'experiment_name': 'spider_main',
            'fresnel_number': fresnel_calculator(energy_from_wavelength(0.1127*10**(-9)).magnitude, 0.1127*10**(-9), 26.1838*10**(-9), 7.8881e+07*1e-9),
            'energy_kev': energy_from_wavelength(0.1127*10**(-9)).magnitude,
            'lam': 0.1127*10**(-9),
            'z': 7.8881e+07*1e-9,
            'pv': 26.1838*10**(-9),
            'name': 'big_spider',
            'image_path': image_path,
        }
        return join_dict(join_dict(kwargs, spider_big), setup_info, True)
    
    def spider_jdora(self, setup_info = setup_info, **kwargs):
        spider_jdora = {
            'path': io.imread('/data/hereon/wp/user/hailudaw/git_folders/Gans4Inverse/data/jdora/data/holograms/spider_hair.tiff'),
            'experiment_name': 'spider_jdora',
            'energy': 11.0,
            'pv': 6500.0,
            'z02': 19661.0*1e6,
            'z01': 79950000.0,
            'dsf': 2**0,
            'name': 'spider_jdora',
        }
        spider_jdora['fresnel_number'] = get_fresnel_from_cone(**spider_jdora)
        return join_dict(join_dict(kwargs, spider_jdora), setup_info)

    def sami_mouse_good(self, idx = [680], setup_info=setup_info, **kwargs):
        # 'idx': [680],#[201, 234, 275, 326, 355, 388, 431, 482, 533, 584, 635, 680, 725, 776],
        sami_mouse_good = {
            'path': io.imread_collection('/asap3/petra3/gpfs/p05/2023/data/11017690/processed/nano111233_sf04_1p2p1_OV_JD/logs/nano111233_sf04_1p2p1_OV_reco_single/'+ '/*.tiff'),
            'idx': idx,
            'experiment_name': 'sami_mouse_good',
            'energy_kev': 11.0,
            'pv': 2.820144109415012e-06,
            'z': 1.9191507469051687,
            'name': 'sami_mouse_good',
        }
        return join_dict(join_dict(kwargs, sami_mouse_good), setup_info, False)
    
    def sami_all(self, idx = [680], setup_info=setup_info, compartment = False, crop = False, row = None, col = None, pad_by = None, cut_x = None, cut_y = None,  **kwargs):
        raw_path = '/asap3/petra3/gpfs/p05/2023/data/11017690/raw/nano111233_sf04_1p2p1_OV'
        images = io.imread_collection(raw_path + '/*.tiff')
        # images.files where 'ref' is in the files
        ref_images = [image for image in images.files if 'ref' in image]
        ref_images = io.imread_collection(ref_images)

        main_images = [image for image in images.files if 'ref' not in image]
        main_images = io.imread_collection(main_images)
        image = main_images[idx] if type(idx) is int else [main_images[i] for i in idx]
        image_path = main_images.files[idx] if type(idx) is int else [main_images.files[i] for i in idx]
        mean_ref = np.mean(ref_images, axis=0)
        image = image / mean_ref if type(image) is not list else [image[i] / mean_ref for i in range(len(image))]
        if crop:
            image = crop_on_all_sides(image, cut_x, cut_y, replace_value = None)
        if compartment:
            assert type(idx) is int, 'only one image is allowed for compartment'
            image = compartment_image(image, row, col, pad_by)
        sami_all = {
            'experiment_name': 'sami_mouse',
            'path': image,
            'image_path': image_path,   
            'idx': idx,
            'energy_kev': 11.0,
            'pv': 2.820144109415012e-06,
            'z': 1.90664,
            'iter_num': 100,
            'abs_ratio': 0.00078,
            'name': 'sami_all',
            'positive_attenuation': 'gelu',
            'positive_phase': 'relu_inverted',
            'transform_type': 'leakyrelu',
            'model_type': 'unet',
            'pad_times': 2,
            'wavelet': 'db4',
            'save_path': '/data/hereon/wp/user/hailudaw/new_data/',
        }
        return join_dict(join_dict(kwargs, sami_all), setup_info, True)


    def sami_best_distance(self, idx = [680], setup_info=setup_info, compartment = False, crop = False, row = None, col = None, pad_by = None, cut_x = None, cut_y = None,  **kwargs):
        if type(idx) != list:
            idx = [idx]
        row = 4 if row is None else row
        col = 4 if col is None else col
        pad_by = 10 if pad_by is None else pad_by
        cut_x = 450 if cut_x is None else cut_x
        cut_y = 350 if cut_y is None else cut_y
        
        path = '/asap3/petra3/gpfs/p05/2023/data/11017690/processed/nano111233_sf04_1p2p1_OV_JD/logs/nano111233_sf04_1p2p1_OV_reco_single/'
        images, image_path = get_image(path, idx, 'tiff')
        if crop:
            images = crop_on_all_sides(images, cut_x, cut_y, replace_value = None)
        if compartment:
            assert len(idx) == 1, 'only one image is allowed for compartment'
            images_com = [compartment_image(tensor_to_np(images[i]), row, col, pad_by) for i in range(len(tensor_to_np(images)))] if type(images) is list else compartment_image(tensor_to_np(images), row, col, pad_by)
            images = [torch_reshape(images_com[i]) for i in range(len(images_com))]
        sami_best_distance = {
            'experiment_name': 'sami_mouse',
            'path': images,
            'image_path': image_path,   
            'idx': idx,
            'energy_kev': 11.0,
            'pv': 2.820144109415012e-06,
            'z': 1.90664,
            'abs_ratio': 7.8e-3,
            'iter_num': 100,
            'abs_ratio': 1e-5, #0.00078,
            'name': 'sami_best_distance',
            'positive_attenuation': 'gelu',
            'positive_phase': 'relu_inverted',
            'transform_type': 'leakyrelu',
            'model_type': 'wavelet',
            'wavelet': 'db4',
            'save_path': '/data/hereon/wp/user/hailudaw/new_data/',
        }
        return join_dict(join_dict(kwargs, sami_best_distance), setup_info, True)

    def sami_farther(self, idx = [680], setup_info=setup_info, **kwargs):
        sami_farther = {
            'path': io.imread_collection('/asap3/petra3/gpfs/p05/2023/data/11017690/processed/nano111233_sf04_1p2p1_OV_JD/logs/nano111233_sf04_1p2p1_OV_reco_single/'+ '/*.tiff') , 
            'idx': idx,
            'experiment_name': 'sami_farther',
            'energy_kev': 11.0,
            'pv': 2.820144109415012e-06,
            'z': 1.95,
            'name': 'sami_farther',
        }
        return join_dict(join_dict(kwargs, sami_farther), setup_info, True)
       
    def info_sami_spider(self, idx = [680], setup_info=setup_info, **kwargs):
        info_sami_spider = {
            'path': io.imread_collection('/asap3/petra3/gpfs/p05/2023/data/11017690/processed/nano111233_sf04_1p2p1_OV_JD/logs/nano111233_sf04_1p2p1_OV_reco_single/'+ '/*.tiff') ,
            'idx': idx,
            'experiment_name': 'sami_spider',
            'energy': 11.0,
            'pv': 9.193699824924853e-08,
            'z': 0.2769425256194748, #0.0010809961997837324,
            'dsf': 1,
        }
        return join_dict(join_dict(kwargs, info_sami_spider), setup_info, True)
    
    def cactus_needles(self, setup_info=setup_info, **kwargs):
        cactus_needles = {
            'path': io.imread('/data/hereon/wp/user/hailudaw/git_folders/Gans4Inverse/data/jdora/data/holograms/cactus_needle.tiff'),
            'experiment_name': 'cactus_needles',
            'energy': 17.0,
            'pv': 6500,
            'z02': 19652000000.0,
            'z01': 285415625,
            'dsf': 2**0,
            'name': 'cactus_needles',
        }
        cactus_needles['fresnel_number'] = get_fresnel_from_cone(**cactus_needles)
        return join_dict(join_dict(kwargs, cactus_needles), setup_info)
    
    def jd_mg(self, setup_info=setup_info, **kwargs):
        # (energy=11.0, px_size=6500.0, z02=1.9661e10 )
        mg = {
            'path': io.imread('/data/hereon/wp/user/hailudaw/git_folders/Gans4Inverse/data/jdora/data/holograms/magnesium_wire.tiff'),
            'experiment_name': 'mg',
            'energy': 11.0,
            'pv': 6500.0,
            'z02': 1.9661e10,
            'z01': 470515625,
            'dsf': 2**0,
            'name': 'mg_wire',
        }
        mg['fresnel_number'] = get_fresnel_from_cone(**mg)
        return join_dict(join_dict(kwargs, mg), setup_info)
    
    def jd_spider_hair(self, setup_info=setup_info, **kwargs):
        # (energy=11.0,  px_size=6500.0, z02=19661.0*1e6 )
        spidy = {
            'path': io.imread('/data/hereon/wp/user/hailudaw/git_folders/Gans4Inverse/data/jdora/data/holograms/spider_hair.tiff'),
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
            'path': io.imread('/data/hereon/wp/user/hailudaw/git_folders/Gans4Inverse/data/jdora/data/holograms/tooth.tiff'),
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
        attenuation = io.imread('data/screw/ground_truth_projection_attenuation_60000_eV_00000_type_float32_8192x8192.tif') * 2 * np.pi / wavelength_from_energy(30).magnitude 
        phase = io.imread('data/screw/ground_truth_projection_phase_60000_eV_00000_type_float32_8192x8192.tif') * 2 * np.pi / wavelength_from_energy(30).magnitude 
        abs_ratio = 1 / (np.max(phase)/np.max(attenuation))
        d =  (2**(-1))
        print(512*d)
        
                
        attenuation = resize(img = torch_reshape(attenuation), size = (int(512*d), int(512*d)), antialias=False)
        phase = resize(size = (int(512*d), int(512*d)), img = torch_reshape(phase), antialias=True)
          
        energy = 30
        pv = 0.000711111*1e-3*((2048/(512*d))) # 0.7e-6 #
        z  = 24471460 * 1e-6
        # fresnel_number = 0.01
        fresnel_number = fresnel_calculator(energy_kev= energy, detector_pixel_size=pv, distance_sample_detector=z)
        print('fresnel_number', fresnel_number, 'distance_sample_detector', z, 'detector_pixel_size', pv)
        mg = propagate_others(tensor_to_np(phase), tensor_to_np(attenuation), fresnel_number=fresnel_number, abs_ratio= abs_ratio,  ground_transform_type='reshape', ground_atten_transform_type = 'reshape', transform_type = 'reshape', positive_phase = 'relu_inverted', positive_attenuation = 'gelu', mode = 'reflect',  value = 'median', dsf =2**0,  add_noise = False, noise_type = 'poisson', noise_factor = 0.2, cut = 3)
        return join_dict(mg, setup_info, True)
    

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