from main import CloudProcessor

platform = 'BCO'

input_classification   = f'Path .. /target_classification/classification_new.nc'
input_categorization   = f'Path .. /target_categorization/categorization.nc'
output_data  = 'Path to output files'
plot_data    = 'Path to figures'

cloud_processor = CloudProcessor(classification_path = input_classification,
                                    categorization_path = input_categorization,
                                    dataset_path = output_data,
                                    plot_path    =  plot_data,
                                    platform     = platform)
cloud_processor.create_cloud_dataset(dataset_path = output_data,
                                        plot         = True,
                                        radar_data   = False)
