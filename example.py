from main import CloudProcessor

input_data  = 'PATH'
output_data = 'PATH'
plot_data   = 'PATH'

cloud_processor = CloudProcessor(file_path = input_data,
                                 dataset_path = output_data,
                                 plot_path =  plot_data)
cloud_processor.create_cloud_dataset(dataset_path = output_data,
                                     plot=True,
                                     radar_data = False)

