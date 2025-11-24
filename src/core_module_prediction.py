from data.load_data import load_data_core_module, scipy
from utils.utils import preprocess_data_core_module, create_figure_core_module, ensure_directory_exists
from models.build_models import core_module, torch
from config.load_config import load_config


def predict_core_module():
    images_sequence = 'raw_data/core_module/vol9_mt_data.mat'
    parameters_sequence = 'raw_data/core_module/vol9_param_mat.mat'
    core_module_weights_path = 'saved_models/core_module_weights.pt'
    save_predictions_path = 'predictions/core_module_predictions.mat'

    # Load the config file
    config = load_config()

    # Load the data
    scan_images, scan_parameters = load_data_core_module(images_sequence, parameters_sequence)
    
    # Preprocess the data and split into input and output data
    input_images, scan_params, true_image_scans =  preprocess_data_core_module(scan_images, scan_parameters, config['normalization_values'])

    # Define the model
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = core_module(model_parameters=config['model_parameters']).to(device)
    model.load_state_dict(torch.load(core_module_weights_path))

    # Create the predictions
    model.eval()
    with torch.inference_mode():
        input_images, scan_params = input_images.to(device), scan_params.to(device)
        predictions = model(input_images, scan_params)

    # Save the predictions
    predictions = predictions.cpu().detach().numpy()
    ensure_directory_exists('predictions')
    scipy.io.savemat(save_predictions_path, {'mat':predictions.squeeze(0)})

    # Plot the predictions and the ground truth images
    create_figure_core_module(prediction_mat=predictions, ground_truth_mat=true_image_scans, savefig_dir='predictions')


if __name__ == '__main__':
    predict_core_module()
