from main import CloudProcessor

#input_data  = 'PATH'
#output_data = 'PATH'
#plot_data   = 'PATH'
#platform = 'RV-Meteor'
platform = 'BCO'
import datetime
start_date = datetime.date(2020, 1, 19)
end_date   = datetime.date(2020, 1, 20)
import glob
date_range = (end_date - start_date).days + 1


for i in range(date_range):
    sdate = start_date + datetime.timedelta(days=i)
    ymd = 10000*sdate.year + 100*sdate.month + sdate.day


    input_classification   = f'/Users/hanni/Documents/Masterarbeit/AMT-Paper/clouds/new_target_classification/20200119_barbados_classification_new.nc'
    input_categorization   = f'/Users/hanni/data_barbados/Cloudnet/EUREC4A/categorization/20200119_barbados_categorization.nc'

    #output_data  = 'Path to output files'
    #plot_data    = 'Path to figures'

    #input_data  = f'/Users/hanni/data_barbados/RV_meteor/new_cloudnet/{ymd}_new_cloudnet.nc'
    output_data = '/Users/hanni/Documents/Masterarbeit/AMT-Paper/clouds/new_target_classification/'
    plot_data   = '/Users/hanni/Documents/Masterarbeit/AMT-Paper/clouds/new_target_classification/'

    cloud_processor = CloudProcessor(classification_path = input_classification,
                                        categorization_path = input_categorization,
                                        dataset_path = output_data,
                                        plot_path    =  plot_data,
                                        platform     = platform)
    cloud_processor.create_cloud_dataset(dataset_path = output_data,
                                            plot         = True,
                                            radar_data   = False)
    #try:
    #   cloud_processor = CloudProcessor(classification_path = input_classification,
    #                                    categorization_path = input_categorization,
    #                                    dataset_path = output_data,
    #                                    plot_path    =  plot_data,
    #                                    platform     = platform)
    #   cloud_processor.create_cloud_dataset(dataset_path = output_data,
    #                                        plot         = True,
    #                                        radar_data   = False)
    #except FileNotFoundError:
    #    print(f"File not found for {ymd}. Skipping.")
    #except Exception as e:
    #    print(f"An error occurred for {ymd}: {e}")



