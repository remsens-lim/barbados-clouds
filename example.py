from main import CloudProcessor

dataset_path = '/Users/hanni/pythonProject6/trade_wind_clouds/data/'
data = '/Users/hanni/Documents/Masterarbeit/AMT-Paper/clouds/target_classification/20211202_new_cloudnet.nc'

cloud_processor = CloudProcessor(file_path = data,
                                 dataset_path = '/Users/hanni/trade_wind_clouds/data/',
                                 plot_path =  dataset_path)
cloud_processor.create_cloud_dataset(dataset_path = '/Users/hanni/trade_wind_clouds/data/',
                                     plot=True)


