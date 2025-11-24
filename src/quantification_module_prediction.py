from data.load_data import load_data_quantification_module, scipy
from utils.utils import preprocess_data_quantification_module, create_figure_quantification_module, ensure_directory_exists
from models.build_models import quantification_module, torch
from config.load_config import load_config

def predict_quantification_module():
    input_images_path = 'raw_data/quantification_module/dataset.mat'
    parameters_mat_path = 'raw_data/quantification_module/params.mat'
    output_labels_path = 'raw_data/quantification_module/labels.mat'
    quantification_module_weights_path = 'saved_models/quantification_module_weights.pt'
    save_predictions_path = 'predictions/quantification_module_predictions.mat'
    
    # Load the config file
    config = load_config()

    # Load the data
    scan_images, scan_parameters, labels = load_data_quantification_module(input_images_path, parameters_mat_path, output_labels_path)

    # Preprocess the data
    input_images, input_parameters, output_labels = preprocess_data_quantification_module(scan_images,
                                                                                          scan_parameters,
                                                                                          labels,
                                                                                          config['normalization_values'])

    # Define the model
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = quantification_module(model_parameters=config['model_parameters']).to(device)
    model.load_state_dict(torch.load(quantification_module_weights_path))

    # Create the predictions
    model.eval()
    with torch.inference_mode():
        input_images, input_parameters = input_images.to(device), input_parameters.to(device)
        predictions = model(input_images, input_parameters)

    # Save the predictions
    predictions = predictions.cpu().detach().numpy()
    ensure_directory_exists('predictions')
    scipy.io.savemat(save_predictions_path, {'mat':predictions.squeeze(0)})

    # Plot the predictions and the ground truth images
    create_figure_quantification_module(prediction_mat=predictions, ground_truth_mat=output_labels.permute(0,2,3,1), savefig_dir='predictions')


if __name__ == '__main__':
    predict_quantification_module()