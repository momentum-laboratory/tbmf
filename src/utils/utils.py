### Includes all the function used throughout the repo
from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np
from models.build_models import torch
from design import winter, b_viridis, b_bwr, b_rdgy
import os

def ensure_directory_exists(directory_path):
    # Check if the directory exists
    if not os.path.exists(directory_path):
        # If it does not exist, create it
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.")



def pad_images(matrix, output_shape, size_idx):

    """
    matrix (np.array) - the matrix of images.
    output_size (tuple) - the desired output shape of each image.
    size_idx (tuple) - the indices of the height and width of the input image in order.
    """

    orig_height, orig_width = matrix.shape[size_idx[0]], matrix.shape[size_idx[1]]

    pad_height = output_shape[0] - orig_height
    pad_width = output_shape[1] - orig_width

    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left

    pad_width = [(0, 0), (0, 0), (0, 0)]


    pad_width[size_idx[0]] = (pad_top, pad_bottom)
    pad_width[size_idx[1]] = (pad_left, pad_right)

    padded_images = np.pad(matrix, tuple(pad_width), mode='constant', constant_values=np.nan)

    return padded_images



def convert_shape_to_original(image, original_shape):
    cur_shape = image.shape[1:3]
    x_space = int((cur_shape[0] - original_shape[0]) / 2)
    y_space = int((cur_shape[1] - original_shape[1]) / 2)

    original_image = image[:, x_space:x_space + original_shape[0], y_space:y_space + original_shape[1], :]

    return original_image



def preprocess_data_core_module(images_mat, parameters, normalization_value):
    
    images_mat = pad_images(images_mat, (normalization_value['IMAGE_SIZE'], normalization_value['IMAGE_SIZE']), (0,1))

    images_mat[np.isnan(images_mat)] = 0
    parameters = parameters.astype('float32')

    images_mat = torch.from_numpy(images_mat)
    parameters = torch.from_numpy(parameters)
    
    # Normalize data
    images_mat = images_mat / normalization_value['SCALE_IMAGES_VALUE']
    parameters = parameters / normalization_value['SCALE_PARAMETERS']
    
    # Seperate into input and labels
    input_images = images_mat[:,:,:6]
    ground_truth_images = images_mat[:,:,6:]

    return input_images.unsqueeze(0), parameters.unsqueeze(0), ground_truth_images.unsqueeze(0)



def preprocess_data_quantification_module(input_images_mat, parameters_mat, labels_mat, normalization_values): # vol4

    parameters_mat = parameters_mat.astype('float32')
   
    
    # normalize the data
    normalized_input_images_mat = input_images_mat / normalization_values['SCALE_IMAGES_VALUE']
    normalized_parameters_mat = parameters_mat / normalization_values['SCALE_PARAMETERS']

    labels_mat[0,:,:] = labels_mat[0, :, :] / normalization_values['SCALE_KSSW']
    labels_mat[1,:,:] = labels_mat[1, :, :] / normalization_values['SCALE_MT']
    labels_mat[2,:,:] = (labels_mat[2, :, :] + 1) / normalization_values['SCALE_B0']
    labels_mat[3,:,:] = labels_mat[3, :, :] / normalization_values['SCALE_B1']
    labels_mat[4,:,:] = labels_mat[4, :, :] / normalization_values['SCALE_T1']
    labels_mat[5,:,:] = labels_mat[5, :, :] / normalization_values['SCALE_T2']

    labels_mat[np.isnan(labels_mat)] = 0
    normalized_input_images_mat[np.isnan(normalized_input_images_mat)] = 0

    # Convert to torch
    labels_mat = torch.from_numpy(labels_mat)
    normalized_input_images_mat = torch.from_numpy(normalized_input_images_mat)
    normalized_parameters_mat = torch.from_numpy(normalized_parameters_mat)

    return normalized_input_images_mat.unsqueeze(0), normalized_parameters_mat.unsqueeze(0), labels_mat.unsqueeze(0)



def create_figure_core_module(prediction_mat, ground_truth_mat, savefig_dir):
    prediction_mat_original_shape = convert_shape_to_original(prediction_mat, original_shape=(116, 116))
    ground_truth_mat_original_shape = convert_shape_to_original(ground_truth_mat, original_shape=(116, 116))

    fig, axs = plt.subplots(nrows=2, ncols=6, figsize=(20, 10))
    for i in range(6):
        axs[0, i].imshow(np.flipud(np.transpose(ground_truth_mat_original_shape[0, :, :, i])), cmap='gray')
        axs[1, i].imshow(np.flipud(np.transpose(prediction_mat_original_shape[0, :, :, i])), cmap='gray')

        for ax in [axs[0, i], axs[1, i]]:
            # Hide the spines
            for spine in ax.spines.values():
                spine.set_visible(False)
            
            # Remove the ticks and tick labels
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        
        if i == 0:
            # Set the y-axis label for the first column and remove y-axis numbers
            axs[0, 0].set_ylabel(ylabel="Ground\nTruth", rotation=0, labelpad=50, fontsize=20)
            axs[0, 0].yaxis.set_ticks([])
            axs[1, 0].set_ylabel(ylabel="Predicted\nImages", rotation=0, labelpad=50, fontsize=20)
            axs[1, 0].yaxis.set_ticks([])

    cax = fig.add_axes([0.91, 0.3, 0.02, 0.4])
    sm = cm.ScalarMappable(cmap='gray', norm=plt.Normalize(vmin=0, vmax=1))
    fig.colorbar(sm, ax=axs[1, :], cax=cax, orientation='vertical')
    plt.subplots_adjust(top=0.99, bottom=0.01, hspace=-0.68, wspace=0)
    plt.savefig(os.path.join(savefig_dir, 'core_module_reults_fig.jpg'))    



def create_figure_quantification_module(prediction_mat, ground_truth_mat, savefig_dir): # (1, 144, 144, 6)
    prediction_mat_original_shape = convert_shape_to_original(prediction_mat, original_shape=(116, 88)) # (1, 116, 88, 6) 
    ground_truth_mat_original_shape = convert_shape_to_original(ground_truth_mat, original_shape=(116, 88)) # (1, 116, 88, 6) 
    
    fig, axs = plt.subplots(nrows=2, ncols=6, figsize=(20, 10))

    axs[0, 0].imshow(ground_truth_mat_original_shape[0,:,:,0] * 100,
                     vmin=0,
                     vmax=100,
                     cmap='magma')

    axs[0, 0].set_title(label="$K_{ssw} (s^{-1})$", fontsize=20)
    axs[0, 0].set_ylabel(ylabel="Ground\nTruth", rotation=0, labelpad=50, fontsize=20)
    axs[0, 0].set_yticks([])
    axs[0, 0].set_xticks([])

    axs[0, 1].imshow(ground_truth_mat_original_shape[0,:,:,1] * 27.27,
                     vmin=0,
                     vmax=27.27,
                     cmap=b_viridis)
    axs[0, 1].set_title(label="$f_{ss}$ (%)", fontsize=20)
    axs[0, 1].axis('off')

    axs[0, 2].imshow((ground_truth_mat_original_shape[0,:,:,2] * 2.7) - 1,
                     vmin=-0.6,
                     vmax=0.6,
                     cmap=b_bwr)
    axs[0, 2].set_title(label="$B_0\,\,(ppm)$", fontsize=20)
    axs[0, 2].axis('off')

    axs[0, 3].imshow(ground_truth_mat_original_shape[0,:,:,3] * 3.4944,
                     vmin=0.5,
                     vmax=1.5,
                     cmap=b_rdgy)
    axs[0, 3].set_title(label="$B_{1}\,\,(rel.)$", fontsize=20)
    axs[0, 3].axis('off')

    axs[0, 4].imshow(ground_truth_mat_original_shape[0,:,:,4] * 10000,
                     vmin=500,
                     vmax=2500,
                     cmap='hot')

    axs[0, 4].set_title(label="$T_{1}$ (ms)", fontsize=20)
    axs[0, 4].axis('off')

    axs[0, 5].imshow(ground_truth_mat_original_shape[0,:,:,5] * 1000,
                     vmin=30,
                     vmax=130,
                     cmap=winter,
                     )
    axs[0, 5].set_title(label="$T_{2}$ (ms)", fontsize=20)
    axs[0, 5].axis('off')

    axs[1, 0].imshow(prediction_mat_original_shape[0,:,:,0] * 100,
                     vmin=0,
                     vmax=100,
                     cmap='magma')
    axs[1, 0].set_ylabel(ylabel="Predicted\nImages", rotation=0, labelpad=50, fontsize=20)
    axs[1, 0].set_yticks([])
    axs[1, 0].set_xticks([])

    axs[1, 1].imshow(prediction_mat_original_shape[0,:,:,1] * 27.27,
                     vmin=0,
                     vmax=27.27,
                     cmap=b_viridis)
    axs[1, 1].axis('off')

    axs[1, 2].imshow((prediction_mat_original_shape[0,:,:,2] * 2.7) - 1,
                     vmin=-0.6,
                     vmax=0.6,
                     cmap=b_bwr,
                     )
    axs[1, 2].axis('off')

    axs[1, 3].imshow(prediction_mat_original_shape[0,:,:,3] * 3.4944, 
                     vmin=0.5,
                     vmax=1.5,
                     cmap=b_rdgy)
    axs[1, 3].axis('off')

    axs[1, 4].imshow(prediction_mat_original_shape[0,:,:,4] * 10000,
                     vmin=500,
                     vmax=2500,
                     cmap='hot')
    axs[1, 4].axis('off') 

    axs[1, 5].imshow(prediction_mat_original_shape[0,:,:,5] * 1000,
                     vmin=30,
                     vmax=130,
                     cmap=winter)
    axs[1, 5].axis('off')

    plt.subplots_adjust(top=0.99, bottom=0.01, hspace=-0.5, wspace=0)
    
    cbar_list = ['magma', b_viridis, b_bwr, b_rdgy, 'hot', winter]

    vmin = [0, 0, -0.6, 0.5, 500, 30]

    vmax = [100, 27.27, 0.6, 1.5, 2500, 130]
    
    for j in range(6):
        cax = fig.add_axes(
            [axs[0, j].get_position().x0 + 0.01, 0.1, axs[0, j].get_position().width - 0.02, 0.02])  # 0.2
        sm = cm.ScalarMappable(cmap=cbar_list[j], norm=plt.Normalize(vmin=vmin[j], vmax=vmax[j]))
        sm.set_array([])
        plt.colorbar(sm, cax=cax, orientation='horizontal', ticks=np.linspace(vmin[j], vmax[j], 5))

    plt.savefig(os.path.join(savefig_dir, 'quantification_module_results_fig.jpg'))
