##### Load the data given paths #####

import scipy.io


def load_data_core_module(data_path, params_path):
    brain_scans_mat = scipy.io.loadmat(data_path)['arr']
    scan_parameter_mat = scipy.io.loadmat(params_path)['params']

    return brain_scans_mat, scan_parameter_mat


def load_data_quantification_module(data_path, params_path, labels_path):
    input_images = scipy.io.loadmat(data_path)['res']
    scan_parameters = scipy.io.loadmat(params_path)['res']
    labels_images = scipy.io.loadmat(labels_path)['res']

    return input_images, scan_parameters, labels_images


