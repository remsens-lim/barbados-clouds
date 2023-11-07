from main import CloudProcessor

input_data  = 'PATH'
output_data = 'PATH'
plot_data   = 'PATH'

cloud_processor = CloudProcessor(file_path = input_data,
                                 dataset_path = output_data,
                                 plot_path =  plot_data)
cloud_processor.create_cloud_dataset(dataset_path = output_data,
                                     plot=True)

# adjust probability parameters if needed
# Default values for loc, scale, and invert
# ze_params = (-30, 5, True)
# vel_params = (-1, 0.2, False)

# Default values for shape, loc, scale, and invert
# beta_params = (6, 0.77e-5, 4.5e-06, False)
# probability_calculator = ProbabilityCalculator(path_to_classification_file, path_to_categorization_file, ze_params, vel_params, beta_params)
