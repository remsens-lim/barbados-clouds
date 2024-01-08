from main import CloudProcessor

#input_data  = 'PATH'
#output_data = 'PATH'
#plot_data   = 'PATH'
#platform = 'RV-Meteor'
platform = 'BCO'
import datetime
start_date = datetime.date(2020, 1, 1)
end_date   = datetime.date(2020, 3, 1)
import glob
date_range = (end_date - start_date).days + 1


for i in range(date_range):
    sdate = start_date + datetime.timedelta(days=i)
    ymd = 10000*sdate.year + 100*sdate.month + sdate.day


    input_data   = f'/Users/hanni/Documents/Masterarbeit/AMT-Paper/clouds/target_classification/{ymd}_new_cloudnet.nc'

    output_data  = f'/Users/hanni/Documents/Masterarbeit/AMT-Paper/clouds/clouds_0/data/'
    plot_data    = f'/Users/hanni/Documents/Masterarbeit/AMT-Paper/clouds/clouds_0/figures/'

    #input_data  = f'/Users/hanni/data_barbados/RV_meteor/new_cloudnet/{ymd}_new_cloudnet.nc'
    #output_data = '/Users/hanni/data_barbados/RV_meteor/clouds_-10/data/'
    #plot_data   = '/Users/hanni/data_barbados/RV_meteor/clouds_-10/figures/'
    try:
       cloud_processor = CloudProcessor(file_path = input_data,
                                        dataset_path = output_data,
                                        plot_path =  plot_data, platform = platform)
       cloud_processor.create_cloud_dataset(dataset_path = output_data,
                                            plot=True,
                                            radar_data = False)
    except FileNotFoundError:
        print(f"File not found for {ymd}. Skipping.")
    except Exception as e:
        print(f"An error occurred for {ymd}: {e}")



