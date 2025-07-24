from libraries.utils import *
from libraries.setup import *
from libraries.torch_utils import *

import matplotlib
from matplotlib import pyplot as plt
import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ['GTK3Agg', 'GTK3Cairo', 'GTK4Agg', 'GTK4Cairo', 'MacOSX', 'nbAgg', 'QtAgg', 'QtCairo', 'Qt5Agg', 'Qt5Cairo', 'TkAgg', 'TkCairo', 'WebAgg', 'WX', 'WXAgg', 'WXCairo', 'agg', 'cairo', 'pdf', 'pgf', 'ps', 'svg', 'template']
# matplotlib.use('pdf')
import colorsys
named_colors = matplotlib.colors.get_named_colors_mapping()
def is_strong_color(color_name):
    """Returns True if the color is considered strong (not light or pastel)."""
    rgb = matplotlib.colors.to_rgb(color_name)  # Convert to RGB
    h, l, s = colorsys.rgb_to_hls(*rgb)         # Convert to HLS
    return l < 0.5  # Keep colors with low luminance (strong/dark colors)
strong_colors = sorted([name for name in named_colors if is_strong_color(named_colors[name])])


metrics_names_full = [
    'SSIM (I, I*)', 'PSNR (I, I*)', 'L1 (I,I*)',
    'SSIM (Φ, φ)', 'PSNR (Φ, φ)', 'L1 (Φ, φ)',
    'SSIM (A, A*)', 'PSNR (A, A*)', 'L1 (A, A*)'
]
metric_column_map = {
    'SSIM (I, I*)': 'ssim_list',
    'PSNR (I, I*)': 'psnr_list',
    'L1 (I,I*)': 'main_diff',
    'SSIM (Φ, φ)': 'ground_ssim_list',
    'PSNR (Φ, φ)': 'ground_psnr_list',
    'L1 (Φ, φ)': 'ground_main_diff_list',
    'SSIM (A, A*)': 'A_SSIM',
    'PSNR (A, A*)': 'A_PSNR',
    'L1 (A, A*)': 'A_L1'
}
metric_final_column_map = {
    'SSIM (I, I*)': 'final_ssim',
    'PSNR (I, I*)': 'final_psnr',
    'L1 (I,I*)': 'final_main_diff',
    'SSIM (Φ, φ)': 'final_ground_ssim',
    'PSNR (Φ, φ)': 'final_ground_psnr',
    'L1 (Φ, φ)': 'final_ground_main_diff',
    'SSIM (A, A*)': 'final_A_SSIM',
    'PSNR (A, A*)': 'final_A_PSNR',
    'L1 (A, A*)': 'final_A_L1'
}
       
bbox_to_anchors = {
    'lower center': (0.5,-0.17),
    'upper center': (0.5,1.1),
    'upper right': (1.3, 1.0),
    'upper left': (0.0, 1.0),
    'lower right': (1.3,-0.10),
    'lower left': (-0.3,-0.17),
    'center': (0.5,0.5),
    'upper': (0.5,1.1),
    'lower': (0.5,-0.17),
    'right': (1.35, 0.0),
    'center right': (0.9, 0.6),
    'center left': (0,0.5),
    'None': (None, None)
}

locs = {
    'lower center': 'lower center',
    'upper center': 'upper center',
    'upper right': 'upper right',
    'upper left': 'upper left',
    'lower right': 'lower right',
    'lower left': 'lower left',
    'center': 'center',
    'upper': 'upper',
    'lower': 'lower',
    'right': 'right',
    'center right': 'center right',
    'center left': 'center left',
    'None': None
}
def insert_axins(ax, loc = 'upper right', width = "50%", height = "5%"):
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    axins1 = inset_axes(
        ax,
        width=width,  # width: 50% of parent_bbox width
        height=height,  # height: 5%
        loc=loc,
    )
    axins1.xaxis.set_ticks_position("bottom")
    return axins1

def shorten(string):
    if 'e' in string:
        left = string.split('e')[0][:7]
        right = string.split('e')[1][:7]
        return left + 'e' + right
    else:
        if '.' in string:
            count = 0
            for i in range(len(string.split('.')[1])):
                if string[i] == '0':
                    count += 1
            return string[:count+5]
        else:
            return string[:7]
        
def give_title(image, title = '', idx = '', min_max = True):    
    if min_max:
        min_val_orig = np.min(image)
        max_val_orig = np.max(image)
        txt_min_val = '('+shorten(str(min_val_orig))+', '
        txt_max_val = shorten(str(max_val_orig))+')'
    else:
        txt_min_val = ''
        txt_max_val = ''    
    title = str(idx+1) if title == '' else title
    return title+'\n'+txt_min_val+txt_max_val if min_max else title

def give_titles(images, titles = [], min_max = True):
    titles = [titles] if type(titles) is not list else titles
    if len(titles) <= len(images):
        titles = [give_title(images[i], title = titles[i], idx=i, min_max = min_max) for i in range(len(titles))]
        n_for_rest = np.arange(len(titles), len(images))
        titles.extend([give_title(images[i], idx=i, min_max = min_max) for i in n_for_rest])
    else:
        titles = [give_title(images[i], title = titles[i], idx=i, min_max = min_max) for i in range(len(images))]
    return titles

def get_row_col(images, show_all = False, images_per_row = 5):
    if show_all:
        rows = int(np.sqrt(len(images)))
        cols = int(np.sqrt(len(images)))
        return rows, cols + (len(images) - rows*cols)//rows
    
    if len(images) == 1:
        rows = 1
        cols = 1
    elif len(images) <= images_per_row:
        rows = 1
        cols = len(images)
    else:
        rows = len(images)//images_per_row
        cols = images_per_row
        if rows*cols < len(images):
            rows += 1
    
    return rows, cols

def show_image_with_zoomed_in_part(image, left = None, right = None, buttom = None, top = None, vmode= 'show', plot_color = 'blue', color = 'red', text = None, second = None, third=None, save_name = None, save_path = None):
    #make a figure with size 20x20
    fig = plt.figure(figsize=(20,20))
    #add a subplot
    ax = fig.add_subplot(111)
    #show the image
    ax.imshow(image)
    if second == None:
        if [left, right, buttom, top] == [None, None, None, None]:
            #25% of the image at the center
            left = image.shape[1]//4
            right = 3*image.shape[1]//4
            buttom = image.shape[0]//4
            top = 3*image.shape[0]//4

        if vmode == 'show':
            ax.add_patch(plt.Rectangle((left, buttom), right - left, top - buttom, edgecolor=color, lw=2, facecolor='none'))
            axins = ax.inset_axes([0.7, 0.7, 0.3, 0.3])
            axins.imshow(image[buttom:top, left:right])
        elif vmode == 'plot':
            #add a line patch
            ax.add_patch(plt.Rectangle((left, buttom+(top-buttom)//2), right - left, 1, edgecolor=plot_color, lw=2, facecolor='none'))
            axins = ax.inset_axes([0.7, 0.7, 0.3, 0.3])
            axins.plot(np.arange(left, right), image[buttom+(top-buttom)//2, left:right], color=plot_color)
            axins.axis('off')
        else:
            """both"""
            #add a line patch
            ax.add_patch(plt.Rectangle((left, buttom+(top-buttom)//2), right - left, 1, edgecolor=plot_color, lw=2, facecolor='none'))
            axins = ax.inset_axes([0.7, 0.7, 0.3, 0.3])
            axins.plot(np.arange(left, right), image[buttom+(top-buttom)//2, left:right], color=plot_color)
            axins.axis('off')
            #add a rectangle patch
            ax.add_patch(plt.Rectangle((left, buttom), right - left, top - buttom, edgecolor=color, lw=2, facecolor='none')        )
            axins = ax.inset_axes([0, 0, 0.3, 0.3])
            axins.axis('off')
            ax.axis('off')
            axins.imshow(image[buttom:top, left:right])
    else:
        axins = ax.inset_axes(0.7, 0.7, 0.3, 0.3)
        axins.imshow(second)
        axins.axis('off')
    #write a text on the zoomed in part
    axins.text(0.8, 0.8, text, fontsize=8, ha='left', va='bottom', color='red')
    #put a third image that the user can add
    if third is not None:
        axins2 = ax.inset_axes([0.7, 0.7, 0.1, 0.1])
        # print(third)
        axins2.imshow(third)
        axins2.axis('off')
    #remove axis
    axins.axis('off')
    ax.axis('off')
    plt.show()

    if save_path is not None:
        save_path = os.getcwd() + '/plots/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if save_name is None:
            save_name = 'zoomed_in_image'
        save_path = save_path + save_name + '.png'
        plt.savefig(save_path)
                      
def val_from_images(image, type_of_image = None):
    if type_of_image is None:
        type_of_image = type(image)
    if 'ndarray' in str(type_of_image):
        if len(image.shape) == 2:
            val = image
        elif len(image.shape) == 3:
            val = [image[j,:,:] for j in range(len(image))]
        else:
            val = [image[j,0,:,:] for j in range(len(image))]
    elif 'Tensor' in str(type_of_image):
        image = tensor_to_np(image)
        if type(image) is not list:
            if len(image.shape) == 2:
                val = image
            elif len(image.shape) == 3:
                val = [image[j,:,:] for j in range(len(image))]
            elif len(image.shape) == 4:
                val = [image[j,0,:,:] for j in range(len(image))]
            elif len(image.shape) == 1:
                val = image
        else:
            val = image
    elif 'jax' in str(type_of_image):
        #jax to numpy
        image = np.array(image)
        if len(image.shape) == 2:
            val = image
        elif len(image.shape) == 3:
            val = [image[j,:,:] for j in range(len(image))]
        elif len(image.shape) == 4:
            val = [image[j,0,:,:] for j in range(len(image))]
        elif len(image.shape) == 1:
            val = image
        else:
            val = image
    elif type_of_image == 'str':
        val = io.imread_collection(image)
    elif 'collection' in str(type_of_image):
        val = image
    elif 'list' in str(type_of_image):
        val = [val_from_images(image, type_of_image = type(image)) for image in image]
    else:
        assert False, "type_of_image is not nd.array, list or torch.Tensor"
    return val
    
def convert_images(images, idx = None):
    if idx is not None:
        images = [images[i] for i in idx]
    if type(images) is list:
        vals = [val_from_images(image, type_of_image = type(image)) for image in images]
        # vals = [torch_reshape(image) for image in images]
        # vals = [tensor_to_np(image) for image in vals]
  
        for i, val in enumerate(vals):
            if type(val) is list:
                [vals.append(val[j]) for j in range(len(val))]
                vals.pop(i)
        images = vals
    else:
        images = val_from_images(images, type_of_image = type(images))
    for i, val in enumerate(images):
        if type(val) is list:
            [images.append(val[j]) for j in range(len(val))]
            images.pop(i)
    return images

def plot_func(ax, plot_axis, image, plot_color, add_patch, insert_axes = True, cmap = 'gray', axis = 'off', rectangle = [0.7, 0.7, 0.3, 0.3], axin_axis = 'off',  **kwargs):
    shape = image.shape
    if insert_axes:
        axin = ax.inset_axes(rectangle)
    else:
        axin = ax
    if add_patch:
        if plot_axis == 'half':
            ax.add_patch(plt.Rectangle((1, shape[0]//2), shape[1] - 1, 1, edgecolor=plot_color, lw=2, facecolor='none'))
            axin.plot(np.arange(1, shape[1]), image[shape[0]//2, 1:shape[1]], color=plot_color)
        elif plot_axis == 'vertical':
            ax.add_patch(plt.Rectangle((shape[1]//2, 1), 1, shape[0] - 1, edgecolor=plot_color, lw=2, facecolor='none'))
            axin.plot(np.arange(1, shape[0]), image[1:shape[0], shape[1]//2], color=plot_color)
        elif plot_axis == 'diagonal':
            initial_point = (1, 1)
            final_point = (shape[1], shape[0])
            ax.add_patch(plt.Arrow(initial_point[0], initial_point[1], final_point[0] - initial_point[0], final_point[1] - initial_point[1], edgecolor=plot_color, lw=2, facecolor='none'))
            axin.plot(np.arange(np.diag(image)[1:].shape[0]), np.diag(image)[1:], color=plot_color)
            
        elif plot_axis == 'diagonal_2':
            initial_point = (1, shape[0])
            final_point = (shape[1], 1)
            ax.add_patch(plt.Arrow(initial_point[0], initial_point[1], final_point[0] - initial_point[0], final_point[1] - initial_point[1], edgecolor=plot_color, lw=2, facecolor='none'))
            axin.plot(np.arange(np.diag(np.fliplr(image))[1:].shape[0]), np.diag(np.fliplr(image))[1:], color=plot_color)
        else:
            ax.add_patch(plt.Rectangle((1, plot_axis), shape[1] - 1, 1, edgecolor=plot_color, lw=2, facecolor='none'))
            axin.plot(np.arange(1, shape[1]), image[plot_axis, 1:shape[1]], color=plot_color)
    else:
        if plot_axis == 'half':
            axin.plot(np.arange(1, shape[1]), image[shape[0]//2, 1:shape[1]], color=plot_color)
        elif plot_axis == 'vertical':
            axin.plot(np.arange(1, shape[0]), image[1:shape[0], shape[1]//2], color=plot_color)
        elif plot_axis == 'diagonal':
            axin.plot(np.arange(1, shape[0]), np.diag(image)[1:], color=plot_color)
        elif plot_axis == 'diagonal_2':
            axin.plot(np.arange(1, shape[0]), np.diag(np.fliplr(image))[1:], color=plot_color)
        else:
            axin.plot(np.arange(1, shape[1]), image[plot_axis, 1:shape[1]], color=plot_color)
            
    if axis == 'off' or axis == False:
        ax.axis('off')
    else:
        if 'legend_size' in kwargs.keys():
            ax.legend(loc = 'upper center', fontsize = kwargs['legend_size'])
        if 'title' in kwargs.keys():
            if kwargs['title'] != 'no_title':
                ax.set_title(kwargs['title'])
        else:
            ax.set_title('Plot Profile')

    if axin_axis == 'off' or axin_axis == False:
        axin.axis('off')
        
    #adjust the axis size
    ax.axis('tight')
    axin.axis('tight')
    
    return ax

def chose_fig(images, idx = None, rows = None, cols = None, show_all = False, add_length = None, images_per_row = 5, fig_size = None):
    (rows, cols) = get_row_col(images, show_all, images_per_row) if rows is None or cols is None else (rows, cols)
    shape = images[0].shape
    if fig_size is not None:    
        fig_size = fig_size
    else:
        if shape[0] > 260:
            fig_size = (shape[1]*cols/100+1, shape[0]*rows/100)
        elif shape[0] > 100 and shape[0] <= 260:
            fig_size = (shape[1]*cols/50+1, shape[0]*rows/50)
        else:
            fig_size = (shape[1]*cols/25+1, shape[0]*rows/25)
        if add_length is None:
            add_length = 5
            fig_size = (fig_size[0]+add_length, fig_size[1])
    
    fig, ax = plt.subplots(rows, cols, figsize=fig_size, squeeze=False)
    ax.reshape(rows, cols)
    for i in range(len(images), rows*cols):
        ax[i//cols, i%cols].axis('off')
    
    if rows == 1 and cols == 1:
        return fig, ax, rows, cols, fig_size
    elif rows == 1:
        ax = ax.reshape(1, cols)
        return fig, ax, rows, cols, fig_size
    elif cols == 1:
        ax = ax.reshape(rows, 1)
        return fig, ax, rows, cols, fig_size
    else:
        return fig, ax, rows, cols, fig_size

def set_legend_location(legend_location, number_of_profiles = 1):
    """
    'upper center', 'lower center', 'upper right', 'upper left', 'lower right', 'lower left', 'center', 'upper', 'lower', 'right', 'left', 'center right', 'center left'
    """
    bbox_to_anchor = bbox_to_anchors[legend_location]
    loc = locs[legend_location]
    
    if legend_location not in ['lower right', 'upper right', 'right', 'center right', 'None']:
        print('Warning: The legend location not known. Please use one of the following: \nlower right, upper right, right, center right, None')
        ncol = number_of_profiles//2 + 2 if number_of_profiles%2 == 0 else number_of_profiles//2 + 3
    else:
        ncol = 1
    
    return bbox_to_anchor, loc, ncol

from matplotlib import patches
def rectangular_frame_for_zoom_boxes(zoom_box, ax, color = 'red', linewidth = 2):
    left, right, buttom, top = zoom_box
    ax.add_patch(patches.Rectangle((left, buttom), right-left, top-buttom, edgecolor = color, facecolor = 'none', linewidth = linewidth))
    return ax

def rectangle_shaper(image, position = 'middle', width = 0.1, height = 0.1, move_h = 0.1, move_v = 0.1):    
    if position == None:
        left = image.shape[1]//4
        buttom = image.shape[0]//4
    if width == None:
        width = 0.1
    if height == None:
        height = 0.1
        
    if position == 'middle':
        left = image.shape[1]//2
        buttom = image.shape[0]//2
    elif position == 'right':
        left = image.shape[1] - width*image.shape[1]
        buttom = image.shape[0]//2 - height*image.shape[0]
    elif position == 'left':
        left = 0
        buttom = image.shape[0]//2 - height*image.shape[0]
    elif position == 'bottom':
        left = image.shape[1]//2
        buttom = image.shape[0] - height*image.shape[0]
    elif position == 'top':
        left = image.shape[1]//2
        buttom = 0
    else:
        #use move_h and move_v to move the rectangle from the middle position
        left = image.shape[1]//2 + move_h*image.shape[1]
        buttom = image.shape[0]//2 + move_v*image.shape[0]
    
    right = width*image.shape[1] + left
    top = height*image.shape[0] + buttom
    return int(left), int(right), int(buttom), int(top)

def apply_kwargs(kwargs):    
    global legend_location, colorbar_normalize, colorbar_axins, colorbar_width, colorbar_height, sa_left, sa_right, sa_top, sa_bottom, sa_wspace, sa_hspace, colorbar_size_factor, colorbar_location, shrink, pad, spacing, lw, move_h, move_v, insert_axes, axin_axis, legend_size, use_line_style
    
    legend_location = kwargs.get('legend_location', 'None')
    colorbar_normalize = kwargs.get('colorbar_normalize', False)
    colorbar_axins = kwargs.get('colorbar_axins', None)
    colorbar_width = kwargs.get('colorbar_width', '50%')
    colorbar_height = kwargs.get('colorbar_height', '5%')
    sa_left = kwargs.get('sa_left', 0.1)
    sa_right = kwargs.get('sa_right', 0.9)
    sa_top = kwargs.get('sa_top', 0.9)
    sa_bottom = kwargs.get('sa_bottom', 0.1)
    sa_wspace = kwargs.get('sa_wspace', 0.01)
    sa_hspace = kwargs.get('sa_hspace', 0.2)
    colorbar_size_factor = kwargs.get('colorbar_size_factor', 100)
    colorbar_location = kwargs.get('colorbar_location', 'bottom')
    shrink = kwargs.get('shrink', 0.5)
    pad = kwargs.get('pad', -0.01)
    spacing = kwargs.get('spacing', 'proportional')
    lw = kwargs.get('lw', 4)
    move_h = kwargs.get('move_h', 0)
    move_v = kwargs.get('move_v', 0)
    insert_axes = kwargs.get('insert_axes', True)
    axin_axis = kwargs.get('axin_axis', True)
    legend_size = kwargs.get('legend_size', 20)
    use_line_style = kwargs.get('use_line_style', True)
    
def resample_profile(profile, target_length):
    """
    Resamples the input profile to a target length using linear interpolation.
    
    Parameters:
      profile       : 1D array representing the profile.
      target_length : Desired number of points.
      
    Returns:
      The resampled profile as a 1D array.
    """

    from scipy.interpolate import interp1d
    current_length = len(profile)
    if current_length == target_length:
        return profile
    # Create normalized x-axes for current and target lengths.
    x_current = np.linspace(0, 1, current_length)
    x_target = np.linspace(0, 1, target_length)
    interp_func = interp1d(x_current, profile, kind='linear')
    return interp_func(x_target)
  
def visualize(images, idx = None, rows = None, cols = None, vmode = 'show', cmap = 'coolwarm', title = '', axis = 'on', plot_axis = 'half', min_max = True, dict = None, save_path=None, save_name=None, show_all = False, images_per_row = 5, fig_size = None, coordinates = [None, None, None, None], plot_color = 'blue', color = 'red', colorbar = False, add_to = None, added_image = None, position = None, width = None, height = None, zoomout_location = 'top right', legend_location = 'upper center', overall_title = None, pyqt = False, use_sns = False, use_plotly = False, **kwargs):
    """
    cmaps: Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r',
    legend_locations = ['best', 'upper right', 'upper left', 'lower left', 'lower right', 'right', 'center left', 'center right', 'lower center', 'upper center', 'center']
    zoomout_location = ['top right', 'top left', 'bottom right', 'bottom left']
    positions = ['middle', 'random', 'top right', 'top left', 'bottom right', 'bottom left', 'custom', 'middle left', 'middle right', 'middle top', 'middle bottom']
    
    """
    #'Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r',
    if 'DataFrame' in str(type(images)):
        cond = kwargs['cond'] if 'cond' in kwargs.keys() else None
        cols = ['ssim_list', 'psnr_list', 'ground_ssim_list', 'ground_psnr_list', 'A_SSIM', 'A_PSNR', cond]
        keys = ['SSIM of holo and reco image', 'PSNR of holo and reco image', 'SSIM of ground truth and reco image', 'PSNR of ground truth and reco image', 'SSIM of ground truth and reco abs', 'PSNR of ground truth and reco abs']
        cols_new = ['SSIM(${I}$,$\\tilde{I}$)', 'PSNR(${I}$,$\\tilde{I}$)', 'SSIM($\\phi,\\tilde{\\phi}$)', 'PSNR($\\phi,\\tilde{\\phi}$)', 'SSIM($A,\\tilde{A}$)', 'PSNR($A,\\tilde{A}$)', cond]
        plot_pandas(images, column_range=cols_new, x_column=cond,  x_label=cond, y_label=cols)
        plot_pandas(images, column_range=cols_new, x_column=cond,  x_label=cond, y_label='Metric', use_sns =True, cond = cond)
        # regression_plot(images, cond, cols_new, cond, cols_new, order = 5)
        return None
    images = convert_images(images, idx)
    title = 'no_title' if title == 'None' or title == 'no' or title == False else title
    if title != 'no_title':
        titles = give_titles(images, title, min_max)
    images = [im[0] if type(im) is list else im for im in images]
    shape = images[0].shape
    dim1 = True
    for i in range(len(shape)-1):
        if shape[i] > 1: 
            dim1 = False
    if dim1:
        #if it's a list of lists, then it's a list of 1D arrays, change to one list using combination
        [plt.plot(images[i]) for i in range(len(images))]
        plt.legend(titles)
        plt.show()
        return None

    if pyqt:
        images = np.stack(images)
        import pyqtgraph as pg
        pg.show(images)
        return None
   
    if dict is not None:
        description_title, add_length = get_setup_info(dict)
    else:
        add_length = None

    insert_axes = kwargs['insert_axes'] if 'insert_axes' in kwargs.keys() else True
    axin_axis = kwargs['axin_axis'] if 'axin_axis' in kwargs.keys() else True
    legend_size = kwargs['legend_size'] if 'legend_size' in kwargs.keys() else 20
    colorbar_normalize = kwargs['colorbar_normalize'] if 'colorbar_normalize' in kwargs.keys() else False
    line_styles = ['solid', 'dashed', 'dashdot', 'dotted']
    # colors = ['blue', 'red', 'green', 'orange', 'purple', 'pink', 'brown', 'cyan', 'magenta', 'grey', 'lime', 'teal', 'black','maroon', 'navy', 'olive', 'silver', 'aqua', 'fuchsia', 'lime', 'teal', 'lavender', 'maroon', 'navy', 'olive', 'silver', 'aqua', 'fuchsia', 'blue', 'red', 'green', 'orange', 'purple', 'pink', 'brown', 'cyan', 'magenta', 'grey', 'lime', 'teal', 'black','maroon', 'navy', 'olive', 'silver', 'aqua', 'fuchsia', 'lime', 'teal', 'lavender', 'maroon', 'navy', 'olive', 'silver', 'aqua', 'fuchsia', 'blue', 'red', 'green', 'orange', 'purple', 'pink', 'brown', 'cyan', 'magenta', 'grey', 'lime', 'teal', 'black','maroon', 'navy', 'olive', 'silver', 'aqua', 'fuchsia', 'lime', 'teal', 'lavender', 'maroon', 'navy', 'olive', 'silver', 'aqua', 'fuchsia'] if 'colors' not in kwargs.keys() else kwargs['colors']
    colors = strong_colors

    if type(fig_size) is int:
        fig_size = (fig_size, fig_size)
    
    fig, ax, rows, cols, fig_size= chose_fig(images, idx, rows, cols, add_length, show_all, images_per_row, fig_size)
    upper_limit = rows*cols
    if rows*cols > len(images):
        upper_limit = len(images)
    
    legend_location = kwargs['legend_location'] if 'legend_location' in kwargs.keys() else 'None'
    bbox_to_anchor, loc, ncol = set_legend_location(legend_location, cols*rows)
    apply_kwargs(kwargs)
    cmap = [cmap] * len(images) if type(cmap) != list else cmap
    alpha = [1] * len(images) if 'alpha' not in kwargs.keys() else kwargs['alpha']

    positions = kwargs.get('positions', None)
    zoom_box = None
    zoombox_locations = {'top right': [0.7, 0.7, 0.3, 0.3], 'top left': [0.0, 0.7, 0.3, 0.3], 'bottom right': [0.7, 0.0, 0.3, 0.3], 'bottom left': [0.0, 0.0, 0.3, 0.3], 'below': [0.0, -0.3, 0.3, 0.3], 'bottom 3': [[0.0, -0.3, 0.3, 0.3], [0.35, -0.3, 0.3, 0.3], [0.7, -0.3, 0.3, 0.3]], 'bottom 2': [[0.0, -0.4, 0.4, 0.4], [0.45, -0.4, 0.4, 0.4]]}
    zoombox_locations.update({'obr': [1.1, 0.4, 0.6, 0.6], 'obl': [-0.5, -0.1, 0.4, 0.6], 'otr': [1.1, 0.7, 0.3, 0.3]}) if axin_axis == False else zoombox_locations.update({'obr': [0.7, -0.5, 0.3, 0.3], 'obl': [0.0, -0.5, 0.3, 0.3], 'otr': [0.0, 1.1, 0.3, 0.3]})
    zoom_box = zoombox_locations[zoomout_location]
    
    if zoomout_location == 'obr': #outside the box but to the right
        plot_location =[ 1.1, 0.0000, 0.6, 0.4] if axin_axis == False else [0.0, -0.5, 0.3, 0.3] 
    elif zoomout_location == 'obl':
        plot_location = [-0.5, 0.0000, 0.4, 0.4] if axin_axis == False else [0.7, -0.5, 0.3, 0.3]
    elif zoomout_location == 'otr':
        plot_location = [1.1, 0.7, 0.3, 0.3] if axin_axis == False else [0.0, 1.1, 0.3, 0.3]
    elif zoomout_location == 'bottom 3':
        # zoom_boxes = [[0.0, -0.4, 0.3, 0.3], [0.35, -0.4, 0.3, 0.3], [0.7, -0.4, 0.3, 0.3]]
        zoom_boxes = [[0.0, -0.5, 0.3, 0.3], [0.35, -0.5, 0.3, 0.3], [0.7, -0.5, 0.3, 0.3]]
        positions = ['custom', 'custom', 'custom']
        plot_location = [0.80, 0.0000, 0.2, 0.2] if axin_axis == False else [0.0, 0.0, 0.3, 0.3]
    elif zoomout_location == 'bottom 2':
        zoom_boxes = [[0.0, -0.4, 0.4, 0.4], [0.45, -0.4, 0.4, 0.4]]
        positions = ['bottom', 'left']
        plot_location =  [0.80, 0.0000, 0.2, 0.2] if axin_axis == False else [0.0, 0.0, 0.3, 0.3]
    else:
        plot_location = [0.00, 0.0000, 0.2, 0.2] if axin_axis == False else [0.0, 0.0, 0.3, 0.3]
    #get the rectangle coordinates
    if coordinates != [None, None, None, None]:
        left, right, buttom, top = coordinates
    else:
        if positions is None: 
            left, right, buttom, top = [], [], [], []
            for image in images:
                    l, r, b, t = rectangle_shaper(image, position = position, width = width, height = height, move_h = move_h, move_v = move_v)
                    left.append(l)
                    right.append(r)
                    buttom.append(b)
                    top.append(t)
        else:
            lefts, rights, buttoms, tops = [], [], [], []
            rects = []
            move_hs = [-0.2, 0.0, 0.2] if 'move_hs' not in kwargs.keys() else kwargs['move_hs']
            move_vs = [-0.2, 0.0, 0.2] if 'move_vs' not in kwargs.keys() else kwargs['move_vs']
            for i in range(len(positions)):
                rect_k = []
                left, right, buttom, top = [], [], [], []
                for j, image in enumerate(images):
                    l, r, b, t = rectangle_shaper(image, position = positions[i], width = width, height = height, move_h = move_hs[i], move_v = move_vs[i]) 
                    # rects.append([l, r, b, t])
                    #convert rects to the range of 0 to 1 with possibility to go beyond 1 and less that 0
                    rect_k.append([l/image.shape[1], t/image.shape[0], r/image.shape[1] - l/image.shape[1], t/image.shape[0] -  b/image.shape[0]])
                    left.append(l)
                    right.append(r)
                    buttom.append(b)
                    top.append(t)
                lefts.append(left)
                rights.append(right)
                buttoms.append(buttom)
                tops.append(top)
                rects.append(rect_k)
            #convert rects to the range of 0 to 1 with possibility to go beyond 1 and less that 0
            
    def show_with_zoom():
        
        if zoomout_location != 'bottom 3' and zoomout_location != 'bottom 2':                
                if use_sns:
                    import seaborn as sns
                    [sns.heatmap(images[i*cols + j], cmap=cmap[i*cols + j], ax=ax[i,j], cbar=False, cbar_kws={'shrink': shrink, 'pad': pad, 'spacing': spacing})for i in range(rows) for j in range(cols) if i*cols + j < upper_limit]
                elif use_plotly:
                    import plotly.express as px
                    [px.imshow(images[i*cols + j], color_continuous_scale=cmap[i*cols + j]) for i in range(rows) for j in range(cols) if i*cols + j < upper_limit]
                else:
                    [ax[i, j].add_patch(plt.Rectangle((left[i*cols + j], buttom[i*cols + j]), right[i*cols + j] - left[i*cols + j], top[i*cols + j] - buttom[i*cols + j], edgecolor=colors[(i*cols + j)%len(colors)], lw=lw, facecolor='none')) for i in range(rows) for j in range(cols) if i*cols + j < upper_limit]  
                
                [ax[i,j].add_patch(plt.Rectangle((left[i*cols + j], buttom[i*cols + j]), right[i*cols + j] - left[i*cols + j], top[i*cols + j] - buttom[i*cols + j], edgecolor=colors[(i*cols+j)%len(colors)], lw=lw, facecolor='none')) for i in range(rows) for j in range(cols) if i*cols + j < upper_limit]
                axins = [ax[i, j].inset_axes(zoom_box,) for i in range(rows) for j in range(cols) if i*cols + j < upper_limit]
                [axins[i*cols + j].imshow(images[i*cols + j][buttom[i*cols + j]:top[i*cols + j], left[i*cols + j]:right[i*cols + j]], cmap = cmap[i*cols + j], extent =  [left[i*cols + j], right[i*cols + j], buttom[i*cols+j],  top[i*cols + j]]) for i in range(rows) for j in range(cols) if i*cols + j < upper_limit]
                [axins[i*cols + j].axis('off') for i in range(rows) for j in range(cols) if i*cols + j < upper_limit]
                [axins[i*cols + j].add_patch(plt.Rectangle((left[i*cols + j], buttom[i*cols + j]), right[i*cols + j] - left[i*cols + j], top[i*cols + j] - buttom[i*cols + j], edgecolor=colors[(i*cols+j)%len(colors)], lw=lw, facecolor='none')) for i in range(rows) for j in range(cols) if i*cols + j < upper_limit]
                # [ax[i,j].indicate_inset_zoom(axins[i*cols + j], edgecolor=colors[(i*cols+j)%len(colors)], alpha = 0.9, lw=5) for i in range(rows) for j in range(cols) if i*cols + j < upper_limit]
                        
        else:
            #for each image, create three patches on the image
            for k in range(len(positions)):
                axins = []
                for i in range(rows):
                    for j in range(cols):
                        idx = i*cols + j
                        if idx >= upper_limit:
                            continue
                        ax[i, j].add_patch(plt.Rectangle((lefts[k][idx], buttoms[k][idx]), rights[k][idx] - lefts[k][idx], tops[k][idx] - buttoms[k][idx], edgecolor=colors[k], lw=lw, facecolor='none'))
                        axin = ax[i, j].inset_axes(zoom_boxes[k], transform=ax[i, j].transAxes) 
                        axins.append(axin)
                for i in range(rows):
                    for j in range(cols):
                        idx = i*cols + j
                        if idx >= upper_limit:
                            continue
                        axins[idx].add_patch(plt.Rectangle((lefts[k][idx], buttoms[k][idx]), rights[k][idx] - lefts[k][idx], tops[k][idx] - buttoms[k][idx], edgecolor=colors[k], lw=lw, facecolor='none'))
                        axins[idx].imshow(images[idx][buttoms[k][idx]:tops[k][idx], lefts[k][idx]:rights[k][idx]], cmap = cmap[idx], extent = [lefts[k][idx], rights[k][idx],  tops[k][idx], buttoms[k][idx]])
                        axins[idx].axis('off')
                
                # [ax[i,j].indicate_inset_zoom(axins[i*cols + j], edgecolor=colors[k]) for i in range(rows) for j in range(cols) if i*cols + j < upper_limit]   
        return ax, axins
                    
    if colorbar_normalize:
        from matplotlib.colors import Normalize
        from matplotlib import cm
        colorbar_size = 0.07 * (shape[0] / shape[1])
        colorbar_fontsize = colorbar_size * colorbar_size_factor

        # one colorbar for all the images
        min_images = [np.min(images[i*cols + j]) for i in range(rows) for j in range(cols) if i*cols + j < upper_limit]
        max_images = [np.max(images[i*cols + j]) for i in range(rows) for j in range(cols) if i*cols + j < upper_limit]
        min_images = np.min(min_images)
        max_images = np.max(max_images)
        norm = Normalize(vmin=min_images, vmax=max_images)
    
    else:
        norm = None

    if use_sns:
        import seaborn as sns
        [sns.heatmap(images[i*cols + j], cmap=cmap[i*cols + j], ax=ax[i,j], cbar=False, cbar_kws={'shrink': shrink, 'pad': pad, 'spacing': spacing})for i in range(rows) for j in range(cols) if i*cols + j < upper_limit]
        [ax[i, j].axis(axis) for i in range(rows) for j in range(cols) if i*cols + j < upper_limit]
    elif use_plotly:
        import plotly.graph_objects as go
        # Create a figure for each image using go.Image and set the colormodel property instead of colorscale.
        
        [go.Figure(go.Image(z=images[i*cols + j], colormodel='rgb'))
        for i in range(rows) for j in range(cols) if i*cols + j < upper_limit]

    else:
        [ax[i,j].imshow(images[i*cols + j], cmap = cmap[i*cols + j], alpha = alpha[i*cols+j], norm = norm) for i in range(rows) for j in range(cols) if i*cols + j < upper_limit]
        [ax[i, j].axis(axis) for i in range(rows) for j in range(cols) if i*cols + j < upper_limit]

    if vmode == 'plot':
        # Adjust subplots for image display
        fig.subplots_adjust(wspace=sa_wspace, hspace=sa_hspace,
                            left=sa_left, right=sa_right, top=sa_top, bottom=sa_bottom)
        
        # Add a rectangle patch on each image (using each image's own shape)
        for idx in range(upper_limit):
            i = idx // cols
            j = idx % cols
            img_shape = images[idx].shape
            ax[i, j].add_patch(plt.Rectangle((1, img_shape[0] // 2),
                                            img_shape[1] - 1, 1,
                                            edgecolor=colors[idx], lw=lw, facecolor='none'))
        
        # Extract profiles from each image
        profiles = []
        for idx in range(upper_limit):
            img = images[idx]
            img_shape = img.shape
            if plot_axis == 'diagonal':
                prof = np.diag(img)
            else:
                prof = np.array(img)[img_shape[0] // 2, 1:img_shape[1]]
            profiles.append(prof)
        
        unique_lengths = {len(p) for p in profiles}
        if len(unique_lengths) > 1:
            common_points = min(unique_lengths)
            normalized_profiles = [resample_profile(p, common_points) for p in profiles]
            common_x = np.linspace(0, 1, common_points)
        else:
            normalized_profiles = profiles
            common_x = np.linspace(0, 1, len(profiles[0]))
        
        # Determine line thickness (can be set via kwargs, defaults to 2)
        line_thickness = kwargs.get('line_thickness', 4)
        
        # Plot all the (possibly normalized) profiles in a single figure with different line styles.
        fig2, ax2 = plt.subplots(1, 1, figsize=(20, 10))
        if 'titles' not in locals():
            titles = ['im' + str(i + 1) for i in range(len(normalized_profiles))]
        legends = (['im' + str(i + 1) for i in range(len(normalized_profiles))]
                if title == '' or title == 'no_title' else titles)
        for i, prof in enumerate(normalized_profiles):
            
            dash_style = line_styles[i % len(line_styles)]
            ax2.plot(common_x, prof, color=colors[i], label=legends[i], linestyle=dash_style, linewidth=line_thickness) if use_line_style else ax2.plot(common_x, prof, color=colors[i], label=legends[i], linewidth=line_thickness)
              
        # Determine label size based on the smallest image height among all images
        min_img_height = min(img.shape[0] for img in images)
        label_size = kwargs.get('label_size', 25 / (512 / min_img_height))
        ax2.tick_params(axis='both', which='major', labelsize=label_size)
        
        fig2.subplots_adjust(wspace=sa_wspace, hspace=sa_hspace,
                            left=sa_left, right=sa_right, top=sa_top, bottom=sa_bottom)
        if legend_location != 'None':
            ax2.legend(fontsize=label_size, loc=loc, bbox_to_anchor=bbox_to_anchor, ncol=ncol)
        else:
            ax2.legend(fontsize=label_size, loc='right', bbox_to_anchor=(1.0, 0.2),
                    ncol=ncol, fancybox=True, shadow=True)
                    # Remove top and right spines
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)

        # Only show ticks on the left and bottom
        ax2.xaxis.set_ticks_position('bottom')
        ax2.yaxis.set_ticks_position('left')
        if overall_title is not None:
            ax2.set_title(overall_title, fontsize=label_size * 3 / 2)
        ax2.tick_params(axis='both', which='major', labelsize=label_size * 3 / 2)

                
    elif vmode == 'both':
        [plot_func(ax[i, j], plot_axis, images[i*cols + j], plot_color = colors[(i*cols + j)%len(colors)], add_patch = True, insert_axes = insert_axes, cmap = cmap, axis = axis, axin_axis=axin_axis, rectangle=plot_location) for i in range(rows) for j in range(cols) if i*cols + j < upper_limit]

    elif vmode == 'zoom':
        ax, axins = show_with_zoom()

    elif vmode == 'zoom_with_plot':
        ax, axins = show_with_zoom()

        plot_profiles = [np.diag(images[i*cols + j]) for i in range(rows) for j in range(cols) if i*cols + j < upper_limit] if plot_axis == 'diagonal' else [images[i*cols + j][shape[0]//2, 1:shape[1]] for i in range(rows) for j in range(cols) if i*cols + j < upper_limit]
        #plot all the profiles in the same image
        fig2, ax2 = plt.subplots(1, 1, figsize=(10, 10))
        legends = ['im'+str(i+1) for i in range(len(plot_profiles))] if title == '' else titles
        [ax2.plot(np.arange(1, shape[1]), plot_profiles[i], color=colors[i], label=legends[i]) for i in range(len(plot_profiles))]
        #increase the size of the x-axis and y-axis and formula for the lable size
        label_size = 30/(1024/shape[0])
        ax2.tick_params(axis='both', which='major', labelsize=label_size)
        #add overall title to the plot 
        ax2.legend(fontsize=label_size, loc=loc, bbox_to_anchor=bbox_to_anchor, ncol = ncol) if legend_location != 'None' else ax2.legend(fontsize=label_size)
        if overall_title is not None:
            ax2.set_title(overall_title, fontsize = label_size*2)
        else:
            ax2.set_title('Profiles', fontsize =  label_size*2)

    elif vmode == 'zoom_with_plot_line':
        ax, axins = show_with_zoom()
        [ax[i, j].add_patch(plt.Rectangle((1, shape[0]//2), shape[1] - 1, 1, edgecolor=colors[(i*cols + j)%len(colors)], lw=lw, facecolor='none')) for i in range(rows) for j in range(cols) if i*cols + j < upper_limit]

    elif vmode == 'all':
        ax, axins = show_with_zoom()
        [plot_func(ax[i, j], plot_axis, images[i*cols + j], plot_color = colors[(i*cols + j)%len(colors)], add_patch = True, insert_axes = insert_axes, cmap = cmap, axis = axis, axin_axis=axin_axis, rectangle=plot_location) for i in range(rows) for j in range(cols) if i*cols + j < upper_limit]
    
    elif vmode == 'add' or vmode == 'add_show' or vmode ==  'add_plot':
        assert added_image is not None, "added_image is None"
        if 'axes_given' in kwargs.keys():
            axes_given = kwargs['axes_given']
        else:
            axes_given = [0, 0.8, 0.2, 0.2]
        if type(added_image) is not list:
            added_image = [added_image] * len(images)
        if add_to is None:
            axins = [ax[i,j].inset_axes([0.1, 0.1, 0.3, 0.3]) for i in range(rows) for j in range(cols) if i*cols + j < upper_limit]
            [axins[i*cols + j].imshow(added_image[i*cols + j], cmap = cmap[i*cols + j], alpha = alpha[i*cols+j]) for i in range(rows) for j in range(cols) if i*cols + j < upper_limit]
            [axins[i*cols + j].axis('off') for i in range(rows) for j in range(cols) if i*cols + j < upper_limit]
        
        else:
            if type(add_to) == tuple:
                add_to_i = add_to[0]
                add_to_j = add_to[1]
            elif type(add_to) == int:
                add_to_i = add_to//cols
                add_to_j = add_to%cols
            #add a third image to the selected ones
            for i in range(rows):
                for j in range(cols):
                    if i == add_to_i and j == add_to_j:
                        ax[i,j].imshow(images[i*cols + j], cmap = cmap[i*cols + j])
                        axins = ax[i,j].inset_axes(axes_given)
                        axins.imshow(added_image[i*cols + j],cmap = cmap[i*cols + j])
                        axins.axis('off')
                    else:
                        ax[i,j].imshow(images[i*cols + j], cmap = cmap[i*cols + j])
                        axins.axis('off')
    

        if vmode == 'add_plot':
            if plot_axis == 'half':
                axins2 = [ax[i,j].inset_axes(plot_location) for i in range(rows) for j in range(cols) if i*cols + j < upper_limit]
                [axins2[i*cols + j].plot(np.arange(1, images[i*cols + j].shape[1]), images[i*cols + j][images[i*cols + j].shape[0]//2, :], color=plot_color) for i in range(rows) for j in range(cols) if i*cols + j < upper_limit]
                [axins2[i*cols + j].axis('off') for i in range(rows) for j in range(cols) if i*cols + j < upper_limit]

            else:
                assert type(plot_axis) == int or type(plot_axis) == float, "plot_axis should be an integer or a float"
                axins2 = [ax[i,j].inset_axes(plot_location) for i in range(rows) for j in range(cols) if i*cols + j < upper_limit]
                [axins2[i*cols + j].plot(np.arange(1, images[i*cols + j].shape[1]), images[i*cols + j][plot_axis, :], color=plot_color) for i in range(rows) for j in range(cols) if i*cols + j < upper_limit]
                [axins2[i*cols + j].axis('off') for i in range(rows) for j in range(cols) if i*cols + j < upper_limit]


        elif vmode == 'add_all':
            [ax[i,j].add_patch(plt.Rectangle((left[i*cols + j], buttom[i*cols + j]), right[i*cols + j] - left[i*cols + j], top[i*cols + j] - buttom[i*cols + j], edgecolor=colors[(i*cols + j)%len(colors)], lw=lw, facecolor='none')) for i in range(rows) for j in range(cols) if i*cols + j < upper_limit]
            [ax[i,j].add_patch(plt.Rectangle((1, buttom[i*cols + j]+(top[i*cols + j]-buttom[i*cols + j])//2), right[i*cols + j] - 1, 1, edgecolor=plot_color, lw=lw, facecolor='none')) for i in range(rows) for j in range(cols) if i*cols + j < upper_limit]
            axins = [ax[i,j].inset_axes([0.7, 0.7, 0.3, 0.3]) for i in range(rows) for j in range(cols) if i*cols + j < upper_limit]
            [axins[i*cols + j].imshow(images[i*cols + j][buttom[i*cols + j]:top[i*cols + j], left[i*cols + j]:right[i*cols + j]], cmap = cmap) for i in range(rows) for j in range(cols) if i*cols + j < upper_limit]
            [axins[i*cols + j].axis('off') for i in range(rows) for j in range(cols) if i*cols + j < upper_limit]
            axins2 = [ax[i,j].inset_axes([0.00, 0.0000, 0.3, 0.3]) for i in range(rows) for j in range(cols) if i*cols + j < upper_limit]
            [axins2[i*cols + j].plot(np.arange(1, right[i*cols + j]), images[i*cols + j][buttom[i*cols + j]+(top[i*cols + j]-buttom[i*cols + j])//2, 1:right[i*cols + j]], color=plot_color) for i in range(rows) for j in range(cols) if i*cols + j < upper_limit]
            [axins2[i*cols + j].axis('off') for i in range(rows) for j in range(cols) if i*cols + j < upper_limit]
        # inside your plotting function, before you start adding titles/text:
    def _annotate(
        ax, text, x, y,
        fontsize=24,
        fontweight='bold',
        color='black',
        ha='center',
        va='center',
        rotation=0,
        bbox=None,
        transform=None
    ):
        """
        Helper to add either a title (via set_title) or free text.
        If transform is None, assumes this is a title and uses ax.set_title();
        otherwise uses ax.text().
        """
        if bbox != 'ignore':
            if bbox is None:
                # default journal-style white box
                bbox = {
                    "facecolor": "white",
                    "edgecolor": "black",
                    "boxstyle": "round,pad=0.2",
                    "alpha": 0.8
                }
        else:
            bbox = None
        if transform is None:
            # use set_title for top-line titles
            ax.set_title(
                text, x=x, y=y,
                fontsize=fontsize,
                fontweight=fontweight,
                color=color,
                horizontalalignment=ha,
                bbox=bbox
            )
        else:
            ax.text(
                x, y, text,
                transform=transform,
                fontsize=fontsize,
                fontweight=fontweight,
                color=color,
                ha=ha,
                va=va,
                rotation=rotation,
                bbox=bbox
            )

    # now replace your three blocks with:

    if title != 'no_title':
        title_x  = kwargs.get('title_x', 0.5)
        title_y  = kwargs.get('title_y', 1.0)
        title_fs = kwargs.get('title_fontsize', 14)
        title_cl = kwargs.get('title_color', 'black')
        title_ha = kwargs.get('title_horizontalalignment', 'center')
        title_fw = kwargs.get('title_fontweight', 'bold')
        title_bbox=kwargs.get('title_bbox', 'ignore')
        # single-color or list
        title_colors = (
            [title_cl]*len(images)
            if not isinstance(title_cl, (list,tuple))
            else title_cl
        )
        for i in range(rows):
            for j in range(cols):
                idx = i*cols + j
                if idx >= upper_limit:
                    continue
                _annotate(
                    ax[i,j],
                    titles[idx],
                    title_x, title_y,
                    color=title_colors[idx % len(title_colors)],
                    fontsize=title_fs,
                    fontweight=title_fw,
                    ha=title_ha,
                    bbox=title_bbox,
                )
    second_title = kwargs.get('second_title', 'no_title')
    if second_title not in ['no_title', 'no', None]:
        st_x  = kwargs.get('stx', 0.3)
        st_y  = kwargs.get('sty', 0.05)
        st_fs = kwargs.get('stfontsize', 14)
        st_cl = kwargs.get('stcolor', 'white')
        st_ha = kwargs.get('sthorizontalalignment', 'center')
        st_va = kwargs.get('stverticalalignment', 'center')
        st_fw = kwargs.get('stfontweight', 'bold')
        st_bbox=kwargs.get('st_bbox', None)
        st_colors = (
            [st_cl]*len(images)
            if not isinstance(st_cl, (list,tuple))
            else st_cl
        )
        for i in range(rows):
            for j in range(cols):
                idx = i*cols + j
                if idx >= upper_limit:
                    continue
                _annotate(
                    ax[i,j],
                    second_title[idx],
                    st_x, st_y,
                    color=st_colors[idx % len(st_colors)],
                    fontsize=st_fs,
                    fontweight=st_fw,
                    ha=st_ha,
                    va=st_va,
                    transform=ax[i,j].transAxes,
                    bbox=st_bbox,
                )

    third_title = kwargs.get('third_title', 'no_title')
    if third_title not in ['no_title', 'no', None]:
        tt_x  = kwargs.get('ttx', -0.05)
        tt_y  = kwargs.get('tty', 0.5)
        tt_fs = kwargs.get('ttfontsize', 14)
        tt_cl = kwargs.get('ttcolor', 'white')
        tt_ha = kwargs.get('tthorizontalalignment', 'center')        
        tt_va = kwargs.get('ttverticalalignment', 'center')
        tt_fw = kwargs.get('ttfontweight', 'bold')
        tt_bbox=kwargs.get('tt_bbox', None)
        tt_colors = (
            [tt_cl]*len(images)
            if not isinstance(tt_cl, (list,tuple))
            else tt_cl
        )
        for i in range(rows):
            for j in range(cols):
                idx = i*cols + j
                if idx >= upper_limit:
                    continue
                _annotate(
                    ax[i,j],
                    third_title[idx],
                    tt_x, tt_y,
                    color=tt_colors[idx % len(tt_colors)],
                    fontsize=tt_fs,
                    fontweight=tt_fw,
                    ha=tt_ha,
                    va=tt_va,
                    rotation=90,
                    transform=ax[i,j].transAxes,
                    bbox=tt_bbox,
                )

    # plt.tight_layout()
    if vmode != 'plot':
        fig.patch.set_facecolor('xkcd:white')
    if colorbar_normalize:
        colorbar = False
    if colorbar:
        if type(colorbar) is not list:
            colorbar = [colorbar] * len(images)
        if type(cmap) is not list:
            cmap = [cmap] * len(images)
        colorbar_size = 0.07 * (shape[0] / shape[1])
        colorbar_fontsize = colorbar_size * colorbar_size_factor
        
        if colorbar_axins != None:
            cols = [fig.colorbar(ax[i, j].imshow(images[i*cols + j], cmap = cmap[i*cols + j]), ax=ax[i, j], cax = insert_axins(ax[i, j],colorbar_axins, colorbar_width, colorbar_height), location = colorbar_location, spacing = spacing, fraction = colorbar_size, shrink = shrink, pad = pad) for i in range(rows) for j in range(cols) if i*cols + j < upper_limit]
        else:
            cols = [fig.colorbar(ax[i, j].imshow(images[i*cols + j], cmap = cmap[i*cols + j]), ax=ax[i, j], location = colorbar_location, spacing = spacing, fraction = colorbar_size, shrink = shrink, pad = pad) for i in range(rows) for j in range(cols) if i*cols + j < upper_limit]
        #adjust the font
        [cols[i].ax.xaxis.set_tick_params('major', labelsize=colorbar_fontsize*5.5, pad = 0, rotation=0) for i in range(len(cols))]
        [cols[i].ax.yaxis.set_tick_params('major', labelsize=colorbar_fontsize*5.5) for i in range(len(cols))]
        #adjust the colorbar
        # [cols[r].set_label('$\it{(min, max):}$('+str(np.min(images[r]))+' , '+str(np.max(images[r]))+')', fontsize=fontsize) for r in range(len(cols))]
    
    if dict is not None:
        fig.subplots_adjust(left=add_length/150)
        fig.suptitle(description_title, fontsize=10, y=0.95, x=0.05, ha='left', va='center', wrap=True, color='blue')
    
    if save_path is not None:
        if save_name is None:
            save_name = get_file_nem(dict)
        save_path = save_path + save_name + '.png'
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1, dpi=300, facecolor='w', edgecolor='w', orientation='portrait', transparent=False, bbox_extra_artists=None, metadata=None)
    

    plt.subplots_adjust(left=sa_left, bottom=sa_bottom, right=sa_right, top=sa_top, wspace=sa_wspace, hspace=sa_hspace)
    
    if colorbar_normalize:            
        plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap[0]), ax=ax, shrink = shrink, pad = pad, location = colorbar_location, fraction = colorbar_size).ax.tick_params(labelsize=colorbar_fontsize*5.5)
        
    plt.show()        
    # return fig, ax, plt

def create_gif_with_progress_text(images, output_filename, duration=0.5, loop=0,
                                  normalize=True, texts = None, progress_bar=True,
                                  progress_bar_height=5, progress_bar_color=(0, 255, 0),
                                  add_text=True, text_color=(255, 255, 255),
                                  font_path=None, font_size=20, progress_bar_position='center', text_position=(10, 10)):
    """
    Create a GIF from a list of 2D numpy arrays (grayscale images) with a progress bar
    and optional text overlay on top of the progress bar.
    
    Parameters:
      images (list): List of 2D numpy arrays representing grayscale images.
      output_filename (str): Path/filename for the output GIF.
      duration (float): Seconds to display each frame.
      loop (int): Number of loops; 0 means infinite.
      normalize (bool): If True, each image is scaled to use the full 0–255 range.
      progress_bar (bool): If True, overlay a progress bar on each frame.
      progress_bar_height (int): Height in pixels of the progress bar.
      progress_bar_color (tuple): RGB color for the progress bar.
      add_text (bool): If True, overlay text on top of the progress bar.
      text_color (tuple): RGB color for the overlay text.
      font_path (str): Path to a TrueType font file. If None, the default font is used.
      font_size (int): Font size for the overlay text.
      text_position (str or tuple): 'center' to center text on the bar or a (x, y) tuple for a custom position.
      
    Returns:
      None (saves the looping GIF to output_filename)
    
    Example:
            
        create_gif_with_progress_text(model.phase_list, 'phase_rec.gif', duration=10.9, loop = 0, font_path='DejaVuSans-BoldOblique.ttf', font_size = 24, text_color = (0, 0, 255), texts = p_texts, progress_bar=True, add_text=False)
    """

    frames = []
    total_frames = len(images)
    
    # Load a TrueType font if provided; otherwise, use default.
    if font_path is not None:
        font = ImageFont.truetype(font_path, font_size)
    else:
        font = ImageFont.load_default()
        print("Warning: Using default font which may not support size adjustments.")
    
    for idx, img in enumerate(images):
        # Normalize and convert image if it's a 2D array.
        if img.ndim == 2:
            if normalize:
                min_val = img.min()
                max_val = img.max()
                if max_val > min_val:
                    img = (img - min_val) / (max_val - min_val) * 255
                else:
                    img = img * 0
            img = np.clip(img, 0, 255).astype(np.uint8)
            # Convert grayscale to RGB.
            img_rgb = np.stack((img,)*3, axis=-1)
        else:
            img_rgb = img
        
        # Convert to a PIL Image.
        pil_img = Image.fromarray(img_rgb)
        width, height = pil_img.size
        
        draw = ImageDraw.Draw(pil_img)
        if progress_bar:
            # Calculate progress fraction (0 to 1) for the current frame.
            progress_fraction = (idx + 1) / total_frames
            bar_width = int(width * progress_fraction)
            bar_top = height - progress_bar_height * 0.9
            bar_bottom = height - progress_bar_height * 0.1
            # Draw the progress bar.
            draw.rectangle([0, bar_top, bar_width, bar_bottom], fill=progress_bar_color)
            
            if add_text:
                # Create a text label showing the progress percentage.
                progress_percentage = int(progress_fraction * 100)
                text = f"{progress_percentage}%"                    
                # Use font.getbbox to measure text dimensions.
                bbox = font.getbbox(text)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                # Determine text position.
                if progress_bar_position == 'center':
                    x = (width - text_width) // 2
                    y = bar_top + (progress_bar_height - text_height) *2
                elif isinstance(progress_bar_position, tuple):
                    x, y = progress_bar_position
                else:
                    x, y = (10, bar_top)  # Default fallback position.
                # Draw the text on top of the progress bar.
                
                draw.text((x, y), text, fill=text_color, font=font)
        if texts is not None:
            if texts is not None:
                text_to_draw = texts[idx]
                draw.text(text_position, text_to_draw, fill=text_color, font=font)
            
            # Convert back to NumPy array
            frame_with_text = np.array(pil_img)
        # Append the modified frame.
        frames.append(np.array(pil_img))
    
    # Save all frames as a GIF.
    imageio.mimsave(output_filename, frames, duration=duration, loop=loop)

def pad_images_to_same_size(images, background_color=(0, 0, 0), normalize=True):
    """
    Pads a list of images (numpy arrays) to the same size based on the largest width and height.
    Optionally, each image is normalized (i.e. rescaled) to use the full 0-255 range before padding.
    
    Parameters:
      images (list): List of NumPy arrays. Each can be grayscale (2D) or RGB (3D).
      background_color (tuple): RGB color to use for padding (default is black).
      normalize (bool): If True, normalize each image to the full 0–255 range before padding.
    
    Returns:
      padded_images (list): List of images as NumPy arrays, all having the same dimensions.
    """
    widths, heights = [], []
    for img in images:
        if img.ndim == 2:
            h, w = img.shape
        else:
            h, w = img.shape[:2]
        widths.append(w)
        heights.append(h)
    max_width = max(widths)
    max_height = max(heights)
    
    padded_images = []
    for img in images:
        # Normalize if required.
        if img.ndim == 2:
            if normalize:
                min_val = img.min()
                max_val = img.max()
                if max_val > min_val:
                    # Scale values to 0-255.
                    img = (img - min_val) / (max_val - min_val) * 255
                else:
                    img = np.zeros_like(img)
            # Ensure the image is in 8-bit format.
            img = np.clip(img, 0, 255).astype(np.uint8)
            # Convert grayscale to RGB.
            img = np.stack((img,)*3, axis=-1)
        else:
            # Optionally, you could add normalization for RGB images here as well.
            pass
        
        pil_img = Image.fromarray(img)
        new_img = Image.new("RGB", (max_width, max_height), background_color)
        w, h = pil_img.size
        left = (max_width - w) // 2
        top = (max_height - h) // 2
        new_img.paste(pil_img, (left, top))
        padded_images.append(np.array(new_img).astype(np.uint8))
    
    return padded_images

import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from io import BytesIO

def get_plot_image(profile_data, mode='profile', figsize=(4, 3), dpi=100, fixed_ylim=None, 
                   plot_background_color='white', xlabel_fontsize=14, ylabel_fontsize=14,
                   tick_labelsize=12, title_fontsize=16, add_colorbar=False, cmap_name="viridis"):
    """
    Create a plot from profile_data and return it as a PIL Image.
    
    Parameters:
      profile_data (array-like): 
         - If mode=='profile': a 1D array to plot as a line.
         - If mode=='colormap': a 1D array (which will be expanded to 2D) to display via imshow.
      mode (str): Either 'profile' (line plot) or 'colormap' (image view).
      figsize (tuple): Figure size.
      dpi (int): Figure resolution.
      fixed_ylim (tuple or None): If provided, sets the y-axis limits.
      plot_background_color (str or tuple): Background color for the plot.
      xlabel_fontsize (int): Font size for the x-axis label.
      ylabel_fontsize (int): Font size for the y-axis label.
      tick_labelsize (int): Font size for tick labels.
      title_fontsize (int): Font size for the title.
      add_colorbar (bool): If True, adds a colorbar to the plot.
      cmap_name (str): Colormap to use.
      
    Returns:
      A PIL Image containing the plot.
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor(plot_background_color)
    ax.set_facecolor(plot_background_color)
    
    if mode == 'profile':
        ax.plot(profile_data, linestyle='-', marker=None)
        ax.set_title("Profile", fontsize=title_fontsize)
        ax.set_xlabel("X", fontsize=xlabel_fontsize)
        ax.set_ylabel("Intensity (mean pixel)", fontsize=ylabel_fontsize)
        if fixed_ylim is not None:
            ax.set_ylim(fixed_ylim)
            vmin, vmax = fixed_ylim
        else:
            vmin, vmax = min(profile_data), max(profile_data)
    elif mode == 'colormap':
        # Expand the 1D data into 2D (one row).
        data_img = np.atleast_2d(profile_data)
        im = ax.imshow(data_img, aspect='auto', cmap=cmap_name)
        # ax.set_title("Colormap", fontsize=title_fontsize)
        # ax.set_xlabel("X", fontsize=xlabel_fontsize)
        # ax.set_ylabel("Intensity", fontsize=ylabel_fontsize)
        if fixed_ylim is not None:
            # ax.set_clim(*fixed_ylim)
            vmin, vmax = fixed_ylim
        else:
            vmin, vmax = np.min(profile_data), np.max(profile_data)
    else:
        raise ValueError("mode must be either 'profile' or 'colormap'")
    
    ax.tick_params(axis='both', labelsize=tick_labelsize)
    
    if add_colorbar:
        vmin, vmax = np.min(profile_data), np.max(profile_data)
        
        if mode == 'profile':
            norm = plt.Normalize(vmin=vmin, vmax=vmax)
            cmap = plt.get_cmap(cmap_name)
            sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])
            plt.colorbar(sm, ax=ax)
        elif mode == 'colormap':
            plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png', facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)

def create_gif_with_progress_profile_text(images, output_filename, duration=0.5, loop=0,
                                          normalize=True, texts=None, progress_bar=True,
                                          progress_bar_height=5, progress_bar_color=(0, 255, 0),
                                          add_text=True, text_color=(255, 255, 255),
                                          font_path=None, font_size=20,
                                          progress_bar_position='center', text_position=(10, 10),
                                          add_profile=True, profile_mode='profile',
                                          profile_figsize=(4, 3), profile_dpi=100,
                                          background_color=(255, 255, 255),
                                          add_colorbar=False,
                                          add_main_colorbar=False,
                                          main_colorbar_width=30,
                                          main_colorbar_height=None,
                                          main_colorbar_cmap="viridis"):
    """
    Create a GIF from a list of 2D numpy arrays (grayscale images) with a progress bar
    (and optional text overlays) and an evolving profile plot added to each frame.
    Additionally, you can overlay a main colorbar on the base image (at the lower right corner).
    
    Parameters:
      images (list): List of 2D numpy arrays (grayscale images).
      output_filename (str): Path/filename for the output GIF.
      duration (float): Seconds to display each frame.
      loop (int): Loop count; 0 for infinite.
      normalize (bool): If True, scale each image to the full 0–255 range.
      texts (None, str, or list): Additional text overlay for each frame.
      progress_bar (bool): If True, overlay a progress bar.
      progress_bar_height (int): Height (in pixels) of the progress bar.
      progress_bar_color (tuple): RGB color of the progress bar.
      add_text (bool): If True, overlay progress percentage text on the progress bar.
      text_color (tuple): RGB color for overlay text.
      font_path (str): Path to a TrueType font file; if None, default font is used.
      font_size (int): Font size for overlay text.
      progress_bar_position (str or tuple): 'center' to center progress text or a (x,y) tuple.
      text_position (tuple): Position for any additional text overlay.
      add_profile (bool): If True, compute and add a profile plot (side by side) to each frame.
      profile_mode (str): Mode for profile plot ("profile" or "colormap").
      profile_figsize (tuple): Figure size for the profile plot.
      profile_dpi (int): DPI for the profile plot.
      background_color (tuple): Background color for compositing the final frame.
      add_colorbar (bool): If True, add a colorbar to the profile plot.
      add_main_colorbar (bool): If True, overlay a main colorbar on the base image.
      main_colorbar_width (int): Width of the main image colorbar.
      main_colorbar_height (int or None): Height of the main image colorbar. If None, defaults to 1/3 of base image height.
      main_colorbar_cmap (str): Colormap for the main image colorbar.
      
    Returns:
      None (saves a looping GIF to output_filename)
    """
    frames = []
    total_frames = len(images)
    
    # Load font for overlay text.
    if font_path is not None:
        font = ImageFont.truetype(font_path, font_size)
    else:
        font = ImageFont.load_default()
        print("Warning: Using default font which may not support size adjustments.")
    
    for idx, img in enumerate(images):
        original_img = img.copy()  # Save original for computing colorbar range.
        # Process the base image.
        if img.ndim == 2:
            if normalize:
                min_val = img.min()
                max_val = img.max()
                if max_val > min_val:
                    img = (img - min_val) / (max_val - min_val) * 255
                else:
                    img = np.zeros_like(img)
            img = np.clip(img, 0, 255).astype(np.uint8)
            img_rgb = np.stack((img,)*3, axis=-1)
        else:
            img_rgb = img  # Assume already displayable.
        
        base_img = Image.fromarray(img_rgb)
        width, height = base_img.size
        draw = ImageDraw.Draw(base_img)
        
        # Draw progress bar.
        if progress_bar:
            progress_fraction = (idx + 1) / total_frames
            bar_width = int(width * progress_fraction)
            bar_top = height - int(progress_bar_height * 0.9)
            bar_bottom = height - int(progress_bar_height * 0.1)
            draw.rectangle([0, bar_top, bar_width, bar_bottom], fill=progress_bar_color)
            if add_text:
                progress_percentage = int(progress_fraction * 100)
                progress_text = f"{progress_percentage}%"
                bbox = font.getbbox(progress_text)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                if progress_bar_position == 'center':
                    x = (width - text_width) // 2
                    y = bar_top + ((bar_bottom - bar_top) - text_height) // 2
                elif isinstance(progress_bar_position, tuple):
                    x, y = progress_bar_position
                else:
                    x, y = (10, bar_top)
                draw.text((x, y), progress_text, fill=text_color, font=font)
        
        # Add additional overlay text.
        if texts is not None:
            overlay_text = texts if isinstance(texts, str) else texts[idx]
            draw.text(text_position, overlay_text, fill=text_color, font=font)
        
        # Add main colorbar overlay if enabled.
        if add_main_colorbar:
            margin = 5
            if main_colorbar_height is None:
                main_colorbar_height = height // 3
            orig_min = float(original_img.min())
            orig_max = float(original_img.max())
            # Create a 1D gradient array spanning the original intensity range.
            gradient_data = np.linspace(orig_min, orig_max, num=256)
            # Use get_plot_image in 'colormap' mode to generate a colorbar image.
            colorbar_img = get_plot_image(gradient_data, mode=profile_mode, fixed_ylim=None,
                                           figsize=(4,10),#(main_colorbar_width/10, main_colorbar_height/100),
                                           dpi=100, add_colorbar=add_colorbar, cmap_name=main_colorbar_cmap)
            # # Resize the resulting image to the desired dimensions.
            colorbar_img = colorbar_img.resize((main_colorbar_width, main_colorbar_height))
            base_img.paste(colorbar_img, (width - main_colorbar_width - margin, height - main_colorbar_height - margin))
        
        # Optionally add a profile plot.
        if add_profile:
            # Compute a profile from the middle row of the original image.
            profile_data = original_img[len(original_img)//2]
            plot_img = get_plot_image(profile_data, mode=profile_mode, figsize=profile_figsize, dpi=profile_dpi,
                                      fixed_ylim=(0, 255), add_colorbar=add_colorbar, cmap_name="viridis")
            new_width = base_img.width + plot_img.width
            new_height = max(base_img.height, plot_img.height)
            combined_img = Image.new("RGB", (new_width, new_height), background_color)
            combined_img.paste(base_img, (0, 0))
            combined_img.paste(plot_img, (base_img.width, 0))
        else:
            combined_img = base_img
        
        frames.append(np.array(combined_img).astype(np.uint8))
    
    imageio.mimsave(output_filename, frames, duration=duration, loop=loop)

def visualize_interact(pure = []):
    import ipywidgets as widgets
    from ipywidgets import interact
    from IPython.display import display
    interact(visualize, pure = widgets.fixed(pure), mode = widgets.Dropdown(options=['show', 'plot'], value='show', description='Show or plot:'), rows = widgets.IntSlider(min=1, max=10, step=1, value=1, description='Rows:'), cols = widgets.IntSlider(min=1, max=10, step=1, value=3, description='Columns:'))
 
def plot_pandas(df, column_range = None, x_column = None, titles = None, min_max = False, x_label = '', y_label = '', images_per_row = 4, scatter = False, fig_size = None, use_sns = False, cond = None):
    """
    this function plots the metadata dataframe
    """
    
    if use_sns:
        import seaborn as sns
        sns.set_theme(style="whitegrid")
        
    cond = 'iteration' if cond is None else cond
    if column_range is None:
        column_range = df.columns[2:-1]
    elif column_range == 'all':
        column_range = df.columns
    elif type(column_range) is str:
        column_range = [column_range]
    elif type(column_range) is int:
        column_range = df.columns[column_range:-1]

    rows, cols = get_row_col(column_range, images_per_row = images_per_row)
    fig = plt.figure(figsize=fig_size) if fig_size is not None else plt.figure(figsize=(20,10))
    # print('rows: ', rows, 'cols: ', cols)
    # if rows*cols < 4:  
    #     fig_size = (10,15) #if fig_size == None else fig_size
    #     fig = plt.figure(figsize=fig_size)
    # else:
    #     fig_size = (10,20) #if fig_size == None else fig_size
    #     fig = plt.figure(figsize=(20,10))
    if min_max:
        min_vals = [df[column].min() for column in column_range], [df[column].idxmin() for column in column_range]
        max_vals = [df[column].max() for column in column_range], [df[column].idxmax() for column in column_range]
    else:
        min_vals, max_vals = '', ''
    if titles is None:
        # titles = [column + '\nmin = ' + str(min_per_column[i])+' at ' + str(df[column].idxmin()) +'\n max = ' + str(df[column].max())+' at ' + str(df[column].idxmax()) for i, column in enumerate(column_range)]
        if min_max: 
            titles = [column + '\nmin = ' + str(min_vals[0][i])+' at ' + str(min_vals[1][i]) +'\n max = ' + str(max_vals[0][i])+' at ' + str(max_vals[1][i]) for i, column in enumerate(column_range)]
        else:
            titles = [str(column) for column in column_range]


    for i, column in enumerate(column_range):
        ax = fig.add_subplot(rows, cols, i+1)
        if x_column is None:
            if use_sns:
                sns.regplot(x=cond, y=column, data=df, scatter_kws={'s': 10}, ci=None, order = 5) if not scatter else sns.scatterplot(x=cond, y=column, data=df)
            else:
                ax.plot(df[column]) if not scatter else ax.scatter(df.index, df[column])
            ax.set_xlabel(cond) if x_label == '' else ax.set_xlabel(x_label)
            ax.set_ylabel(column)  if y_label == '' else ax.set_ylabel(y_label)
            ax.set_title(titles[i])
        else: 
            if use_sns:
                sns.regplot(x=x_column, y=column, data=df, scatter_kws={'s': 10}, ci=None, order = 5) if not scatter else sns.scatterplot(x=x_column, y=column, data=df)
            else:            
                ax.plot(df[x_column], df[column]) if not scatter else ax.scatter(df[x_column], df[column])
            ax.set_xlabel(x_column)
            ax.set_ylabel(column)
            ax.set_title(titles[i])
    plt.show()
    plt.figure()        
    if use_sns:
        corr = df.corr()
        sns.heatmap(corr, cmap='coolwarm', center=0, annot=True, square=False, linewidths=.5, cbar_kws={"shrink": .8})
        plt.show()  
        plt.figure()
    plt.tight_layout()
    return min_vals, max_vals

def plot_image(plots, idx = None, title = '', fig = None, ax = None):
    if type(plots) is not list:
            plots = [plots]
    if idx is not None:
        plots = [plots[i] for i in idx]
    title = give_titles(plots, title)
    fig_size = (5,10) if len(plots) > 1 else (5,5)
    fig = plt.figure(figsize=fig_size) if fig is None else fig
    ax = fig.add_subplot(111)
    [ax.plot(plots[i]) for i in range(len(plots))]
    ax.set_title(title)
    ax.legend(title)
    plt.show()
    return fig, ax

def regression_plot(df, x_column, y_columns, x_label, y_label, order = 3):
    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.set_theme(style="whitegrid")
    y_columns = [y_columns] if not isinstance(y_columns, list) else y_columns
    y_col = df[y_columns[0]]
    x_col = df[x_column]
    # Plot the residuals after fitting a linear model
    # sns.residplot(x=x_col, y=y_col, lowess=False, color="g")
    # compute the correlation matrix and display it
    corr = df.corr()
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, cmap='coolwarm', center=0, annot=True, square=True, linewidths=.5)#, cbar_kws={"shrink": .5})
    plt.show()
    for i,y in enumerate(y_columns):
        # sns.lmplot(x=x_column, y=y, data=df, height=6)
        #non-linear regression
        sns.regplot(x=x_column, y=y, data=df, scatter_kws={'s': 10}, ci=None, order=order)
        plt.xlabel(x_label)
        plt.ylabel(y)
        plt.title('Regression plot between '+x_label+' and '+ y_label[i])#+'with order 3 showing the 95% confidence interval'+'\nPearson:'+str(round(corr[x_column][y], 4)))
        plt.show()
        
class IV:
    """ImageVisualizer class for plotting images."""
    def __init__(self, images, **kwargs):
        self.idx = kwargs.get("idx", None) 
        self.min_max = kwargs.get("min_max", True)
        self.title = kwargs.get("title", "")
        self.dict = kwargs.get("dict", None)
        self.rows = kwargs.get("rows", 1)
        self.cols = kwargs.get("cols", 5)
        self.fig_size = kwargs.get("fig_size", None)
        self.show_all = kwargs.get("show_all", False)
        self.images_per_row = kwargs.get("images_per_row", 5)
        self.cmap = kwargs.get("cmap", 'gray')
        
        self.images = convert_images(images, self.idx)
        self.shape = images[0].shape
        self.titles = give_titles(images, self.title, self.min_max) if self.title != None else None

        self.fig_size = (self.fig_size, self.fig_size) if type(self.fig_size) is int else self.fig_size
        self.disciption_title, self.add_length = get_setup_info(self.dict) if self.dict is not None else None, None

        self.fig, self.ax, self.rows, self.cols, self.fig_size = chose_fig(self.images, self.rows, self.cols, self.show_all, self.add_length, self.images_per_row, self.fig_size)
        self.upper_limit = self.rows * self.cols
        self.upper_limit = len(self.images) if self.upper_limit > len(self.images) else self.upper_limit
        
        self.zoomout_location = kwargs.get("zoomout_location", "top right")
        self.zoom_box = kwargs.get("zoom_box", [0.1, 0.1, 0.3, 0.3])
        self.zoom_boxes = kwargs.get("zoom_boxes", [[0.1, 0.1, 0.3, 0.3], [0.1, 0.1, 0.3, 0.3], [0.1, 0.1, 0.3, 0.3]])
        self.positions = kwargs.get("positions", ['top right', 'top right', 'top right'])
        self.left = kwargs.get("left", [0, 0, 0])
        
        self.colors = ['blue', 'red', 'green', 'orange', 'purple', 'pink', 'brown', 'cyan', 'magenta', 'grey', 'lime', 'teal', 'black','maroon', 'navy', 'olive', 'silver', 'aqua', 'fuchsia', 'lime', 'teal', 'lavender', 'maroon', 'navy', 'olive', 'silver', 'aqua', 'fuchsia', 'blue', 'red', 'green', 'orange', 'purple', 'pink', 'brown', 'cyan', 'magenta', 'grey', 'lime', 'teal', 'black','maroon', 'navy', 'olive', 'silver', 'aqua', 'fuchsia', 'lime', 'teal', 'lavender', 'maroon', 'navy', 'olive', 'silver', 'aqua', 'fuchsia', 'blue', 'red', 'green', 'orange', 'purple', 'pink', 'brown', 'cyan', 'magenta', 'grey', 'lime', 'teal', 'black','maroon', 'navy', 'olive', 'silver', 'aqua', 'fuchsia', 'lime', 'teal', 'lavender', 'maroon', 'navy', 'olive', 'silver', 'aqua', 'fuchsia'] if 'colors' not in kwargs.keys() else kwargs['colors']
        self.insert_axes = kwargs.get("insert_axes", True)
        self.axin_axis = kwargs.get("axin_axis", True)
        self.legend_size = kwargs.get("legend_size", 20)
        self.vmode = kwargs.get("vmode", "show")
        self.plot_axis = kwargs.get("plot_axis", "diagonal")
        self.plot_color = kwargs.get("plot_color", "blue")
        self.plot_location = kwargs.get("plot_location", [0.7, 0.7, 0.3, 0.3])
        self.colorbar = kwargs.get("colorbar", False)
        self.colorbar_normalize = kwargs.get("colorbar_normalize", False)
        self.colorbar_location = kwargs.get("colorbar_location", "right")
        self.colorbar_axins = kwargs.get("colorbar_axins", None)
        self.colorbar_width = kwargs.get("colorbar_width", 0.07)
        self.colorbar_height = kwargs.get("colorbar_height", 0.07)
        self.colorbar_size_factor = kwargs.get("colorbar_size_factor", 1)
        self.spacing = kwargs.get("spacing", 'uniform')
        self.shrink = kwargs.get("shrink", 0.5)
        self.pad = kwargs.get("pad", 0.05)
        self.alpha = kwargs.get("alpha", 1)
        
        self.save_path = kwargs.get("save_path", None)
        self.save_name = kwargs.get("save_name", None)
        self.loc = kwargs.get("loc", 'upper right')
        self.bbox_to_anchor = kwargs.get("bbox_to_anchor", (1.2, 1))
        self.ncol = kwargs.get("ncol", 1)
        self.overall_title = kwargs.get("overall_title", None)
        self.title_x = kwargs.get("title_x", 0.5)
        self.title_y = kwargs.get("title_y", 1.0)
        self.title_color = kwargs.get("title_color", 'black')
        self.title_horizontalalignment = kwargs.get("title_horizontalalignment", 'center')
        self.title_fontweight = kwargs.get("title_fontweight", 'bold')
        self.second_title = kwargs.get("second_title", 'no_title')
        self.stx = kwargs.get("stx", 0.3)
        self.sty = kwargs.get("sty", 0.05)
        self.stcolor = kwargs.get("stcolor", 'w')
        self.stfontsize = kwargs.get("stfontsize", 28)
        self.sthorizontalalignment = kwargs.get("sthorizontalalignment", 'center')
        self.stfontweight = kwargs.get("stfontweight", 'bold')
        self.third_title = kwargs.get("third_title", 'no_title')
        self.ttx = kwargs.get("ttx", 0.3)
        self.tty = kwargs.get("tty", 0.05)
        self.ttcolor = kwargs.get("ttcolor", 'w')
        self.ttfontsize = kwargs.get("ttfontsize", 28)
        self.tthorizontalalignment = kwargs.get("tthorizontalalignment", 'center')
        self.ttfontweight = kwargs.get("ttfontweight", 'bold')

    def _plot_images(self):
        """Plot the images."""
        if self.vmode == 'show':
            self._show_images()
        elif self.vmode == 'plot':
            self._plot_images()
        elif self.vmode == 'zoom':
            self._zoom_images()
        elif self.vmode == 'both':
            self._show_images()
            self._plot_images()
        elif self.vmode == 'zoom_with_plot':
            self._zoom_images()
            self._plot_images()
        elif self.vmode == 'zoom_with_patch':
            self._zoom_images()
            self._plot_images()
        elif self.vmode == 'all':
            self._zoom_images()
            self._plot_images()
            self._show_images()
        elif self.vmode == 'add':
            self._add_images()
        elif self.vmode == 'add_show':
            self._add_images()
            self._show_images()
        elif self.vmode == 'add_plot':
            self._add_images()
            self._plot_images()
        elif self.vmode == 'add_all':
            self._add_images()
            self._plot_images()
            self._show_images()
        else:
            raise ValueError("Invalid value for vmode")
    
    def _plot_zoom(self, ax, img, idx):
        """Add a zoom-in inset to a plot."""
        left, right, bottom, top = self._rectangle_shaper(img, idx)
        zoom_box = self._get_zoom_box_coords()
        
        ax.add_patch(
            patches.Rectangle((left, bottom), right - left, top - bottom, edgecolor=self.plot_color, lw=2, facecolor='none')
        )
        
        axins = ax.inset_axes(zoom_box)
        axins.imshow(img[bottom:top, left:right], cmap=self.cmap)
        axins.axis('off')
        ax.indicate_inset_zoom(axins, edgecolor=self.plot_color)
        
    def _rectangle_shaper(self, image, idx, width=0.1, height=0.1):
        """Create a rectangle for zooming."""
        h, w = image.shape
        left = int(w * (1 - width) / 2)
        right = int(w * (1 + width) / 2)
        bottom = int(h * (1 - height) / 2)
        top = int(h * (1 + height) / 2)
        return left, right, bottom, top
    
    def _get_zoom_box_coords(self):
        """Return the coordinates for the zoom box."""
        if self.zoomout_location == 'top right':
            return self.zoom_box
        if self.zoomout_location == 'top left':
            return self.zoom_boxes[0]
        if self.zoomout_location == 'bottom right':
            return self.zoom_boxes[1]
        if self.zoomout_location == 'bottom left':
            return self.zoom_boxes[2]
        raise ValueError("Invalid value for zoomout_location")
    
    def _prepare_images(self, images, idx):
        """Converts images to a standard format."""
        if not isinstance(images, list):
            images = [images]
        return [self._convert_image(im, idx) for im in images]
    
        
    def _show_images(self):
        """Show the images."""
        [ax[i,j].imshow(images[i*cols + j], cmap = cmap[i*cols + j], alpha = alpha[i*cols+j], norm = norm) for i in range(rows) for j in range(cols) if i*cols + j < upper_limit]
        [ax[i, j].axis(axis) for i in range(rows) for j in range(cols) if i*cols + j < upper_limit]
        
        for idx, (img, ax) in enumerate(zip(self.images, self.ax.flatten())):
            ax.imshow(img, cmap='gray')
            ax.set_title(self.titles[idx])
            ax.axis('off')
        plt.show()
        
    def _plot_images(self):
        """Plot the images."""
        for idx, (img, ax) in enumerate(zip(self.images, self.ax.flatten())):
            ax.imshow(img, cmap='gray')
            ax.set_title(self.titles[idx])
            ax.axis('on')
        plt.show()
        
    def _zoom_images(self):
        """Zoom in on the images."""
        for idx, (img, ax) in enumerate(zip(self.images, self.ax.flatten())):
            ax.imshow(img, cmap='gray')
            ax.set_title(self.titles[idx])
            ax.axis('on')
            self._plot_zoom(ax, img, idx)
        plt.show()
        
    def _add_images(self):
        """Add images to the plot."""
        for idx, (img, ax) in enumerate(zip(self.images, self.ax.flatten())):
            ax.imshow(img, cmap='gray')
            ax.set_title(self.titles[idx])
            ax.axis('on')
        plt.show()
        
        
    def _info(self):
        """Return the information about the class."""
        print(f"available colors:\n", plt.colormaps())
        print(f"positions for zoomout_location:\n", ['top right', 'top left', 'bottom right', 'bottom left'])
        print(f"available plot_axis:\n", ['diagonal', 'half', 'diagonal_1'])
        print(f"available axis:\n", ['on', 'off'])
        print(f"available vmode:\n", ['show', 'plot', 'zoom', 'both', 'zoom_with_plot', 'zoom_with_patch', 'all', 'add', 'add_show', 'add_plot', 'add_all'])
        print(f"available colorbar_location:\n", ['right', 'left', 'top', 'bottom'])
        print(f"available title_horizontalalignment:\n", ['center', 'left', 'right'])
        print(f"available sthorizontalalignment:\n", ['center', 'left', 'right'])
        
    def _prepare_images(self, images, idx):
        """Converts images to a standard format."""
        if not isinstance(images, list):
            images = [images]
        return [self._convert_image(im, idx) for im in images]

    def _convert_image(self, image, idx):
        """Convert image to a standard format."""
        if isinstance(image, str):
            return plt.imread(image)
        if isinstance(image, np.ndarray):
            return image
        if isinstance(image, list):
            return self._prepare_images(image, idx)
        if isinstance(image, dict):
            return self._prepare_images(image['images'], image.get('idx', idx))

    def _calculate_grid(self):
        """Determine the optimal number of rows and columns for the image grid."""
        if self.show_all:
            rows = int(np.sqrt(len(self.images)))
            cols = len(self.images) // rows + (len(self.images) % rows > 0)
        else:
            if len(self.images) <= self.images_per_row:
                return 1, len(self.images)
            rows = len(self.images) // self.images_per_row
            cols = self.images_per_row
        return rows, cols

    def _determine_fig_size(self):
        """Calculate an appropriate figure size."""
        if self.fig_size:
            return self.fig_size
        height, width = self.images[0].shape
        return (self.cols * width // 100 + 1, self.rows * height // 100 + 1)

    def set_titles(self, title='', min_max=True):
        """Set titles for the images."""
        self.titles = [
            self._format_title(idx, title, min_max) for idx in range(len(self.images))
        ]

    def _format_title(self, idx, title, min_max):
        """Format title based on provided index and min-max values."""
        min_val, max_val = np.min(self.images[idx]), np.max(self.images[idx])
        return f"{title}\n({min_val:.2f}, {max_val:.2f})" if min_max else title

    def _plot_zoom(self, ax, img, idx):
        """Add a zoom-in inset to a plot."""
        left, right, bottom, top = self._rectangle_shaper(img, idx)
        zoom_box = self._get_zoom_box_coords()
        
        ax.add_patch(
            patches.Rectangle((left, bottom), right - left, top - bottom, edgecolor=self.plot_color, lw=2, facecolor='none')
        )
        
        axins = ax.inset_axes(zoom_box)
        axins.imshow(img[bottom:top, left:right], cmap=self.cmap)
        axins.axis('off')
        ax.indicate_inset_zoom(axins, edgecolor=self.plot_color)

    def _rectangle_shaper(self, image, idx, width=0.1, height=0.1):
        """Create a rectangle for zooming."""
        h, w = image.shape
        left = int(w * (1 - width) / 2)
        right = int(w * (1 + width) / 2)
        bottom = int(h * (1 - height) / 2)
        top = int(h * (1 + height) / 2)
        return left, right, bottom, top

    def _get_zoom_box_coords(self):
        """Return zoom box coordinates based on the specified location."""
        return {
            'top right': [0.7, 0.7, 0.3, 0.3],
            'top left': [0.0, 0.7, 0.3, 0.3],
            'bottom right': [0.7, 0.0, 0.3, 0.3],
            'bottom left': [0.0, 0.0, 0.3, 0.3],
        }.get(self.zoomout_location, [0.7, 0.7, 0.3, 0.3])

    def plot_images(self, vmode='show', axis='on'):
        """Core function for plotting images."""
        fig, axes = plt.subplots(self.rows, self.cols, figsize=self._determine_fig_size())
        axes = axes.flatten() if self.rows * self.cols > 1 else [axes]
        
        for idx, (img, ax) in enumerate(zip(self.images, axes)):
            ax.imshow(img, cmap=self.cmap)
            ax.set_title(self.titles[idx] if self.titles else '')
            ax.axis(axis)

            if vmode == 'zoom':
                self._plot_zoom(ax, img, idx)

        # Final adjustments and saving
        self._finalize_plot(fig)

    def _finalize_plot(self, fig):
        """Finalize layout and handle saving."""
        plt.tight_layout()
        if self.save_path:
            fig.savefig(f"{self.save_path}/{self.save_name or 'plot'}.png", dpi=300)
        plt.show()


def plot_comparison(comps, labels=None, title=None, metrics_names=None, extreme_point_at_iteration=None):
    """
    Plot all comparison results stored in comps.
    Args:
        comps (list): List of comparison results.
        labels (list, optional): List of labels for each comparison.
        title (str, optional): Title for the whole plot.
        metrics_names (list, optional): List of metric display names to use (defaults to metrics_names_full).
        extreme_point_at_iteration (int or None): The iteration at which to compare the metrics.
            If None, use the highest iteration that is common in all comps.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    num_comparisons = len(comps)
    if labels is None:
        labels = [f'Comparison {i+1}' for i in range(num_comparisons)]

    if metrics_names is None:
        metrics_names = metrics_names_full

    num_metrics = len(metrics_names)
    cols = 3
    rows = int(np.ceil(num_metrics / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(6*cols, 4*rows))
    axs = axs.flatten()
    if title is not None:
        fig.suptitle(title, fontsize=16)

    # Determine the maximum common iteration length
    if extreme_point_at_iteration is None:
        min_len = min(len(comp[0]) if isinstance(comp[0], list) else len(comp[0]) for comp in comps)
        extreme_point_at_iteration = min_len - 1

    for i, metric_name in enumerate(metrics_names):
        for j, (comp, label) in enumerate(zip(comps, labels)):
            metric = comp[i]
            y = np.array(metric)
            x = np.arange(len(y))
            axs[i].plot(x, y, label=label)
        # Find the extreme value at the specified iteration
        values_at_iter = []
        for comp in comps:
            metric = comp[i]
            y = np.array(metric)
            if extreme_point_at_iteration < len(y):
                values_at_iter.append(y[extreme_point_at_iteration])
            else:
                values_at_iter.append(np.nan)
        if any(key in metric_name.lower() for key in ['ssim', 'psnr']):
            extreme_val = np.nanmax(values_at_iter)
            extreme_idx = np.nanargmax(values_at_iter)
        else:
            extreme_val = np.nanmin(values_at_iter)
            extreme_idx = np.nanargmin(values_at_iter)
        # Highlight the extreme point
        if not np.isnan(extreme_val):
            axs[i].scatter(extreme_point_at_iteration, extreme_val, color='red', zorder=5,
                           label=f'{"Max" if any(key in metric_name.lower() for key in ["ssim", "psnr"]) else "Min"}: {labels[extreme_idx]} ({extreme_val:.3f})')
            axs[i].annotate(f'{labels[extreme_idx]}\n{extreme_val:.3f}',
                            (extreme_point_at_iteration, extreme_val),
                            textcoords="offset points", xytext=(0,10), ha='center', color='red')
        axs[i].set_title(metric_name)
        axs[i].legend()
    plt.tight_layout(rect=[0, 0, 1, 0.95] if title else None)
    plt.show()

# plot_comparison_scientific already supports metrics_names as an argument, so just ensure it uses metrics_names_full by default

from plotly.subplots import make_subplots
import warnings # Ensure warnings is imported
import ipywidgets as widgets
import plotly.graph_objects as go

def plot_comparison_scientific(
    comps, labels=None, title=None, metrics_names=None,
    extreme_point_at_iteration=None, figsize=(18, 12),
    legend_ncol=4, text_size='large', rows = 3, cols = 3,
):
    num_comparisons = len(comps)
    if num_comparisons == 0:
        print("No comparison data provided.")
        return

    if labels is None:
        labels = [f'Comp {i+1}' for i in range(num_comparisons)]

    if metrics_names is None:
        metrics_names = [
            'SSIM (I, I*)', 'PSNR (I, I*)', 'L1 (I, I*)',
            'SSIM (Φ, φ)', 'PSNR (Φ, φ)', 'L1 (Φ, φ)',
            'SSIM (A, A*)', 'PSNR (A, A*)', 'L1 (A, A*)'
        ]
        metrics_names = [
            r'SSIM$(I, I^*)$', r'PSNR$(I, I^*)$', r'$L_{1}(I, I^*)$',
            r'SSIM$(\Phi, \varphi)$', r'PSNR$(\Phi, \varphi)$', r'$L_{1}(\Phi, \varphi)$',
            r'SSIM$(A, A^*)$', r'PSNR$(A, A^*)$', r'$L_{1}(A, A^*)$'
        ]


    num_metrics = len(metrics_names)
    
    # Increase the figure height slightly to make more room for the legend below
    legend_rows = int(np.ceil(num_comparisons / legend_ncol))

    # 1. Extract group keys (e.g., σ value) and subtypes (e.g., with/without discriminator)
    if labels is None:
        labels = [f'Comp {i+1}' for i in range(len(comps))]
    group_keys = []
    subtypes = []
    for label in labels:
        # Example: "σ = 0.0 without discriminator"
        parts = label.split(' ')
        group_keys.append(parts[2])  # e.g., "0.0"
        subtypes.append(' '.join(parts[3:]))  # e.g., "without discriminator"

    unique_groups = sorted(set(group_keys), key=lambda x: float(x))
    unique_subtypes = list(dict.fromkeys(subtypes))  # preserve order
    # 2. Assign colors to groups, line styles/markers to subtypes
    color_map = plt.get_cmap('tab10')
    group_color = {g: color_map(i % 10) for i, g in enumerate(unique_groups)}
    line_styles = ['-', '--', ':', '-.']
    markers = ['o', '_', '^', 'D', '*', '+', '|', 's']
    subtype_style = {st: line_styles[i % len(line_styles)] for i, st in enumerate(unique_subtypes)}
    subtype_marker = {st: markers[i % len(markers)] for i, st in enumerate(unique_subtypes)}
    


    num_metrics = len(metrics_names)
    rows = int(np.sqrt(num_metrics)) if rows is None else rows
    cols = int(np.ceil(num_metrics / rows))


    fig, axs = plt.subplots(rows, cols, figsize=(figsize[0], figsize[1]), constrained_layout=False)
    axs = axs.flatten()
    
    min_len = min(len(comp[0]) for comp in comps) if comps else 0
    if min_len == 0:
        print("No data points for comparison. Skipping plot.")
        plt.close(fig)
        return

    if extreme_point_at_iteration is None:
        extreme_point_at_iteration = min_len - 1
    elif not (0 <= extreme_point_at_iteration < min_len):
        import warnings
        warnings.warn(f"extreme_point_at_iteration ({extreme_point_at_iteration}) is out of bounds for data length {min_len}. Using {min_len-1} instead.")
        extreme_point_at_iteration = min_len - 1

    x_data = np.arange(min_len)
    lines = []
    for i in range(num_metrics):
        ax = axs[i]
        metric_name = metrics_names[i]
        y_data_all_comps = []
        for j in range(len(comps)):
            current_y_data = comps[j][i]
            if isinstance(current_y_data, np.ndarray):
                if current_y_data.ndim == 0:
                    y_data = np.array([current_y_data.item()])
                elif current_y_data.ndim == 1:
                    y_data = current_y_data.astype(float)
                else:
                    y_data = current_y_data.flatten().astype(float)
            elif isinstance(current_y_data, list):
                if current_y_data and isinstance(current_y_data[0], np.ndarray):
                    y_data = np.array([val.item() if isinstance(val, np.ndarray) else val for val in current_y_data], dtype=float)
                else:
                    y_data = np.array(current_y_data, dtype=float)
            else:
                y_data = np.array(current_y_data, dtype=float)

            plot_x_data = x_data[:len(y_data)]
            color = group_color[group_keys[j]]
            linestyle = subtype_style[subtypes[j]]
            marker = subtype_marker[subtypes[j]]
            msize     = 10
            mevery    = max(1, len(y_data)//10)
            line, = ax.plot(
                plot_x_data, y_data[:len(plot_x_data)],
                color      = color,
                linestyle  = linestyle,
                marker     = marker,
                markersize = msize,
                markevery  = mevery,
                linewidth  = 2.25,
                label      = labels[j] if i == 0 else "_nolegend_"
            )
            if i == 0:
                lines.append(line)
            y_data_all_comps.append(y_data[:min_len])

        # [extreme point logic unchanged...]

        ax.set_title(metric_name, fontsize=text_size)
        ax.set_xlabel('Iteration', fontsize=text_size)
        ax.tick_params(axis='both', which='major', labelsize=text_size)
        ax.grid(True, alpha=0.6)

    for i in range(num_metrics, rows * cols):
        fig.delaxes(axs[i])

    plt.subplots_adjust(top=0.9, bottom=0.1, hspace=0.3, wspace=0.2)
    fig.legend(lines, labels, loc='lower center', bbox_to_anchor=(0.5, 0.0),
               fancybox=True, shadow=True, ncol=legend_ncol, fontsize=text_size)

    if title:
        fig.suptitle(title, fontsize=text_size, y=0.98)
    plt.show()     

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ipywidgets as widgets
from IPython.display import display
import numpy as np
import warnings
# --- Utility to generate a consistent color map ---
def generate_consistent_color_map(labels):
    """
    Generates a consistent color map for a list of labels.
    Uses Plotly's default color cycle for distinct colors.
    """
    # These are Plotly's default discrete colors (Plotly 'plotly' template colors)
    # You can expand this list if you have more than 10 unique labels,
    # or use a different color palette (e.g., from plotly.express.colors)
    plotly_colors = [
        '#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A',
        '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52'
    ]
    plotly_colors = [
    'blue',      # 0
    'green',     # 1
    'orange',    # 2
    'purple',    # 3
    'brown',     # 4
    'red',       # 5
    'cyan',      # 6
    'magenta',   # 7
    'olive',     # 8
    'teal',      # 9
    'gold',      # 10
    'navy',      # 11
    'darkorange',# 12
    'lime',      # 13
    'pink',      # 14
    'gray',      # 15
    ]
    color_map = {}
    for i, label in enumerate(labels):
        color_map[label] = plotly_colors[i % len(plotly_colors)]
    return color_map

def interactive_plot_comparison(comps, labels=None, title=None):
    """
    Interactive plot for comparison results with a slider to select the x-value (iteration).
    At each x, highlights and annotates the max value among all comparisons for each metric.
    """
    metrics = ['SSIM', 'PSNR', 'Main Diff']
    num_metrics = 3
    num_comparisons = len(comps)
    if labels is None:
        labels = [f'Comparison {i+1}' for i in range(num_comparisons)]

    # Prepare y-data for each metric and comparison
    y_data = []
    min_len = min(len(comp[0]) if isinstance(comp[0], list) else len(comp[0]) for comp in comps)
    for comp in comps:
        comp_metrics = []
        for i in range(num_metrics):
            metric = comp[0][i] if isinstance(comp[0], list) else comp[i]
            y = metric.values[:min_len]
            comp_metrics.append(y)
        y_data.append(comp_metrics)

    x = np.arange(min_len)

    # Create figure with 3 subplots
    fig = make_subplots(
        rows=1, 
        cols=3, 
        subplot_titles=metrics, 
        horizontal_spacing=0.08,
    )
    fig.update_layout(width=1100, height=400)
    colors = ['blue', 'green', 'orange', 'purple', 'brown', 'red', 'cyan', 'magenta']

    # Add traces for each metric and comparison
    for i in range(num_metrics):
        for j in range(num_comparisons):
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y_data[j][i],
                    mode='lines',
                    name=labels[j],
                    legendgroup=labels[j],
                    showlegend=(i==0)
                ),
                row=1, col=i+1
            )

    # Add scatter for max points (to be updated)
    max_scatters = []
    for i in range(num_metrics):
        scatter = go.Scatter(
            x=[x[0]],
            y=[max(y_data[j][i][0] for j in range(num_comparisons))],
            mode='markers+text',
            marker=dict(color='red', size=12, symbol='star'),
            text=[''],
            textposition='top center',
            showlegend=False,
            name='Max'
        )
        fig.add_trace(scatter, row=1, col=i+1)
        max_scatters.append(scatter)

    fig.update_layout(title=title, height=400, width=1100)

    # Slider widget
    slider = widgets.IntSlider(value=0, min=0, max=min_len-1, step=1, description='Iteration')

    def update_plot(iter_idx):
        for i in range(num_metrics):
            # Find max value and which comparison at this x
            vals = [y_data[j][i][iter_idx] for j in range(num_comparisons)]
            max_val = np.max(vals)
            max_idx = np.argmax(vals)
            # Update scatter
            fig.data[num_metrics*num_comparisons + i].x = [x[iter_idx]]
            fig.data[num_metrics*num_comparisons + i].y = [max_val]
            fig.data[num_metrics*num_comparisons + i].text = [f'{labels[max_idx]}<br>{max_val:.3f}']
        fig.update_layout(title=f"{title or ''} (Iteration {iter_idx})")
        fig.show()

    def on_value_change(change):
        update_plot(change['new'])

    slider.observe(on_value_change, names='value')
    display(slider)
    update_plot(0)


# --- Updated interactive_plot_comparison_enhanced function ---
def interactive_plot_comparison_enhanced(comps, labels=None, title=None, metrics_names=None,
                                         extreme_point_at_iteration=None):
    """
    Enhanced interactive plot for comparison results using Plotly and ipywidgets,
    with consistent color mapping.
    
    Args:
        comps (list): List of comparison results. Each item is a list of metric lists.
                      e.g., [[ssim_list_1, psnr_list_1, ...], [ssim_list_2, psnr_list_2, ...], ...]
        labels (list, optional): List of labels for each comparison. If None, default labels are used.
        title (str, optional): Title for the whole plot.
        metrics_names (list, optional): Custom names for the metrics.
                                         Defaults to your original list.
        extreme_point_at_iteration (int or None): The iteration at which to highlight an extreme point.
            If None, uses the highest common iteration that is common in all comps.
    """
    num_comparisons = len(comps)
    if num_comparisons == 0:
        print("No comparison data provided.")
        return

    if labels is None:
        labels = [f'Comp {i+1}' for i in range(num_comparisons)]
    
    # Generate the consistent color map for all labels
    consistent_colors = generate_consistent_color_map(labels)

    if metrics_names is None:
        metrics_names = [
            'SSIM (I, I*)', 'PSNR (I, I*)', 'L1 (I,I*)',
            'SSIM (Φ, φ)', 'PSNR (Φ, φ)', 'L1 (Φ, φ)',
            'SSIM (A, A*)', 'PSNR (A, A*)', 'L1 (A, A*)'
        ]

    num_metrics = len(metrics_names)
    cols = 3
    rows = int(np.ceil(num_metrics / cols))

    min_len = min(len(comp[0]) for comp in comps) if comps else 0
    if min_len == 0:
        print("No data points for comparison. Skipping plot.")
        return
        
    x_data = np.arange(min_len)

    fig = go.FigureWidget(make_subplots(rows=rows, cols=cols,
                                        subplot_titles=metrics_names,
                                        shared_xaxes=True,
                                        vertical_spacing=0.1, horizontal_spacing=0.05))

    for i in range(num_metrics):
        for j in range(num_comparisons):
            current_y_data = comps[j][i]
            # Robust conversion for y_data: handle scalars and lists of arrays
            if isinstance(current_y_data, np.ndarray):
                if current_y_data.ndim == 0:
                    y_data = np.array([current_y_data.item()], dtype=float)
                elif current_y_data.ndim == 1:
                    y_data = current_y_data.astype(float)
                else:
                    y_data = current_y_data.flatten().astype(float)
            elif isinstance(current_y_data, list):
                if current_y_data and isinstance(current_y_data[0], np.ndarray):
                    y_data = np.array([val.item() if isinstance(val, np.ndarray) else val for val in current_y_data], dtype=float)
                else:
                    y_data = np.array(current_y_data, dtype=float)
            else:
                y_data = np.array(current_y_data, dtype=float)
            
            if len(y_data) < min_len:
                plot_x_data = x_data[:len(y_data)]
                warnings.warn(f"Data for {labels[j]}, metric {metrics_names[i]} is shorter than min_len. Plotting truncated data.")
            else:
                plot_x_data = x_data

            # Use the consistent color from the dictionary
            trace_color = consistent_colors.get(labels[j], '#000000') # Default to black if label not found
            
            fig.add_trace(go.Scatter(
                x=plot_x_data,
                y=y_data[:len(plot_x_data)],
                mode='lines',
                name=labels[j],
                line=dict(color=trace_color), # Set line color explicitly
                showlegend=(i == 0), # Only show legend for the first row of subplots
                hovertemplate=f"<b>{metrics_names[i]}</b><br>Iteration: %{{x}}<br>Value: %{{y:.4f}}<extra>{labels[j]}</extra>"
            ), row=int(i/cols) + 1, col=(i % cols) + 1)

        scatter_mode = 'markers'
        marker_symbol = 'star'
        marker_color = 'red' # Extreme point marker can be a consistent distinguishing color
        marker_size = 10
        
        vals_at_iter0 = []
        for k in range(num_comparisons):
            metric_data = comps[k][i]
            if isinstance(metric_data, (list, np.ndarray)) and len(metric_data) > 0:
                val = metric_data[0].item() if isinstance(metric_data[0], np.ndarray) else metric_data[0]
                vals_at_iter0.append(val)
            else:
                vals_at_iter0.append(np.nan)
        
        metric_lower = metrics_names[i].lower()
        if any(key in metric_lower for key in ['ssim', 'psnr']):
            extreme_val_iter0 = np.max(vals_at_iter0)
            extreme_idx_iter0 = np.argmax(vals_at_iter0)
            extreme_label_prefix = "Max"
        else:
            extreme_val_iter0 = np.min(vals_at_iter0)
            extreme_idx_iter0 = np.argmin(vals_at_iter0)
            extreme_label_prefix = "Min"

        fig.add_trace(go.Scatter(
            x=[x_data[0]],
            y=[extreme_val_iter0],
            mode=scatter_mode,
            marker=dict(symbol=marker_symbol, size=marker_size, color=marker_color),
            name=f'{extreme_label_prefix} ({labels[extreme_idx_iter0] if 0 <= extreme_idx_iter0 < len(labels) else "?"})',
            hovertemplate=f"<b>Extreme Point</b><br>Iteration: %{{x}}<br>Value: %{{y:.4f}}<extra></extra>",
            showlegend=False
        ), row=int(i/cols) + 1, col=(i % cols) + 1)

    # --- ONLY CHANGE IS HERE FOR LEGEND PLACEMENT ---
    fig.update_layout(
        title_text=title,
        height=rows * 300,
        showlegend=True,
        legend=dict(
            orientation="h",  # Horizontal legend
            yanchor="top",
            y=-0.1,           # Position below the plot area (negative value moves it down)
            xanchor="center",
            x=0.5,            # Center the legend horizontally
            traceorder="normal"
        )
    )
    # --- END OF ONLY CHANGE ---

    fig.update_xaxes(title_text="Iteration")

    initial_slider_value = extreme_point_at_iteration if extreme_point_at_iteration is not None and extreme_point_at_iteration < min_len else min_len - 1
    slider = widgets.IntSlider(value=initial_slider_value,
                               min=0, max=min_len-1, step=1, description='Iteration')

    def update_plot(change):
        iter_idx = change.new
        with fig.batch_update():
            for i in range(num_metrics):
                extreme_trace_index_in_subplot = num_comparisons
                trace_index_overall = i * (num_comparisons + 1) + extreme_trace_index_in_subplot

                vals_at_iter = [
                    (comps[j][i][iter_idx].item() if isinstance(comps[j][i][iter_idx], np.ndarray) else comps[j][i][iter_idx])
                    for j in range(num_comparisons)
                ]

                metric_lower = metrics_names[i].lower()
                if any(key in metric_lower for key in ['ssim', 'psnr']):
                    extreme_val = np.max(vals_at_iter)
                    extreme_idx_comp = np.argmax(vals_at_iter)
                    extreme_label_prefix = "Max"
                else:
                    extreme_val = np.min(vals_at_iter)
                    extreme_idx_comp = np.argmin(vals_at_iter)
                    extreme_label_prefix = "Min"

                fig.data[trace_index_overall].x = (x_data[iter_idx],)
                fig.data[trace_index_overall].y = (extreme_val,)
                label_name = labels[extreme_idx_comp] if 0 <= extreme_idx_comp < len(labels) else "?"
                fig.data[trace_index_overall].name = f'{extreme_label_prefix} ({label_name})'
                fig.data[trace_index_overall].hovertemplate = f"<b>{extreme_label_prefix} ({label_name})</b><br>Iteration: %{{x}}<br>Value: %{{y:.4f}}<extra></extra>"

    slider.observe(update_plot, names='value')

    display(widgets.VBox([fig, slider]))

def plot(results, title=None, plot=True):
    """
    Plot the results of computed metrics.
    
    Args:
        results: Dictionary with metric names as keys and their values as numpy arrays.
        title: Optional title for the plot.
        plot: If True, display the plot.
        
    Returns:
        None
    """
    import matplotlib.pyplot as plt

    if not isinstance(results, dict):
        raise ValueError("Results must be a dictionary")
    
    fig, ax = plt.subplots()
    for metric_name, values in results.items():
        ax.plot(values, label=metric_name)
    
    if title:
        ax.set_title(title)
    ax.legend()
    
    if plot:
        plt.show()

def plot_correlation(corr_matrix, title=''):
    """
    Plot a heatmap for the given correlation matrix using seaborn.

    Args:
        corr_matrix (pd.DataFrame): Correlation matrix to plot.
        title (str): Title for the heatmap.
    """
    import seaborn as sns
    plt.figure(figsize=(6, 5))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title(title)
    plt.show()

def plotly_plot_with_heatmap(results, title=None, type = 'scatter', save_path = None):
    """
    Plot the results of computed metrics using Plotly with heatmap.
    
    Args:
        results: Dictionary with metric names as keys and their values as numpy arrays.
        title: Optional title for the plot.
        
    Returns:
        None
    """
    import plotly.graph_objects as go
    import numpy as np
    
    fig = go.Figure()
    if type == 'heatmap':
        # Ensure both lists have the same length
        # Find the minimum length among all result arrays/series
        min_len = min(len(np.asarray(results[key])) for key in results.keys())
        # Truncate all arrays/series to min_len for alignment
        for key in results.keys():
            arr = np.asarray(results[key])
            if len(arr) > min_len:
                results[key] = arr[:min_len]
            
        # convert to pd.to_numeric if needed
        df = pd.DataFrame(results)
        # Convert all columns to numeric, coercing errors to NaN
        df = df.apply(pd.to_numeric, errors='coerce')
        # Create a heatmap
        plot_correlation(df.corr(), title=title)
        return
    elif type == 'scatter':
        for metric_name, values in results.items():
            fig.add_trace(go.Scatter(
                x=np.arange(len(values)),
                y=values,
                mode='lines+markers',
                name=metric_name
            ))
    
    else:
        #both scatter and heatmap side by side using subplots
        from plotly.subplots import make_subplots
        fig = make_subplots(rows=1, cols=2, subplot_titles=('Metrics', 'Heatmap'))
        for metric_name, values in results.items():
            fig.add_trace(go.Scatter(
                x=np.arange(len(values)),
                y=values,
                mode='lines',
                name=metric_name
            ), row=1, col=1)    
        # Create a heatmap
        df = pd.DataFrame(results)
       #use sns instead of plotly for heatmap
        corr_matrix = df.corr()
        #show the values in the heatmap
        corr_matrix = corr_matrix.round(2)  # Round to 2 decimal places for better readability
      
        fig.add_trace(go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale='Viridis',
            colorbar=dict(title='Correlation Coefficient'),
            zmin=-1, zmax=1
        ), row=1, col=2)
    # Update layout
    fig.update_layout(
        title_text=title if title else 'Image Quality Metrics',
        width=1000,
        height=500,
        showlegend=True,
        template='plotly_white'
    )  
    if title:
        fig.update_layout(title=title)
    #adjust the legend position
    fig.update_layout(legend=dict(x=0, y=1.2, orientation='h', xanchor='left', yanchor='bottom'))
    # Show the figure
    if type == 'scatter':
        fig.update_layout(xaxis_title='Image Index', yaxis_title='Metric Value')
    elif type == 'heatmap':
        fig.update_layout(xaxis_title='Metrics', yaxis_title='Metrics')
    else:
        fig.update_layout(xaxis_title='Image Index', yaxis_title='Metric Value')
    if save_path:
        #save the figure to a file
        fig.write_html(save_path)
    else:
        #show the figure
        if 'plotly.io' in sys.modules:
            import plotly.io as pio
            pio.renderers.default = 'browser'
        else:
            import plotly.io as pio
            pio.renderers.default = 'notebook'
        #show the figure in the notebook
        if 'plotly.offline' in sys.modules:
            import plotly.offline as pyo
            pyo.init_notebook_mode(connected=True)
        #show the figure in the notebook
        if 'plotly.offline' in sys.modules:
            import plotly.offline as pyo
            pyo.iplot(fig)
        else:
            #show the figure in the browser
            pio.show(fig)



