import datetime
import openpyxl
import numpy as np
import pandas as pd
import xarray as xr
from scipy import ndimage
from skimage import measure
import find_clouds as fc
import numpy.ma as ma
import utils
import os
import plot_clouds as pc
class CloudProcessor:
    def __init__(self, classification_path,categorization_path, dataset_path, platform , plot_path=None ):
        self.classification_path = classification_path
        self.categorization_path = categorization_path
        self.plot_path = plot_path
        self.dataset_path = dataset_path
        self.platform = platform

    def filter_out_haze_echos(self, input_cloudnet, input_radar = None):
        '''
        :param input_cloudnet: Cloudnet target classification
        :param input_radar: masked array from radar reflectivity data
        :return:
        '''
        radar  = ma.masked_where(input_cloudnet > 7,input_cloudnet).filled(0)#.mask
        radar  = ma.masked_where(radar==0, radar).mask

        if input_radar is not None:
            radar        = ma.masked_invalid(input_radar).mask


        haze_mask    = ma.masked_where(input_cloudnet > 10, input_cloudnet).mask#.astype(int)
        radar_masked = ma.masked_where(haze_mask == True,radar ).filled(True)
        return radar_masked

    def open_dataset(self, classification_path,categorization_path):
        cl_data =  xr.open_dataset(classification_path, engine='netcdf4')
        ca_data =  xr.open_dataset(categorization_path, engine='netcdf4')
        time = cl_data.time

        ####### Dimension needs to be reduced when height dimension can not be divided by 4 ##########
        if cl_data.target_classification_haze_echos.data.shape[1] % 4 == 0:
            target_classification_new = cl_data.target_classification_haze_echos.data[:,:]
            target_classification_old = cl_data.target_classification.data[:,:]
            cbh    = cl_data.cloud_base_height_agl.data
            Tw     = ca_data.Tw.data[:,:]
            height = ca_data.height.data[:]
            Ze     = ca_data.Z.data[:,:]
        else:
            number = cl_data.target_classification_haze_echos.data.shape[1]
            target_classification_new = cl_data.target_classification_haze_echos.data[:,:- (number % 4)]
            target_classification_old = cl_data.target_classification.data[:,:- (number % 4)]
            cbh    = cl_data.cloud_base_height_agl.data
            Tw     = ca_data.Tw.data[:,:- (number % 4)]
            height = ca_data.height.data[:- (number % 4)]
            Ze     = ca_data.Z.data[:,:- (number % 4)]
        ##### Filter out Haze echos #####
        return  cl_data, ca_data, target_classification_new,  target_classification_old, time, Tw, height, Ze, cbh


    def categorize_radar_pixels(self, input_radar, input_temperature, input_height, input_cloud_base_height, input_cloudnet):
        '''
        :param input_radar: radar signals with haze echos being filtered out (when present)
        :param input_temperature: wet bulb temperature [K]
        :param input_height: height [m]
        :param input_cloud_base_height: cloud base height [m]
        :param input_cloudnet: Cloudnet target classification
        :return:allclouds: clear sky (0);
                           clouds with cloud top above the height of the Tw-isotherm (1);
                           warm clouds (with tops below the height of the Tw-isotherm (2);
                           warm clouds with base below 1km (average height of the LCL over Barbados) (3)
        '''

        lab_image = measure.label(~input_radar.T, connectivity=1)



        Scc , Dcc , Str , Str_Cu, CNs ,Cir ,Ncc ,mix, dict_cloud_clear_sky = fc.detect_clouds(lab_image ,
                                                                                             input_temperature,
                                                                                             input_height,
                                                                                             input_cloudnet,
                                                                                             input_cloud_base_height)





        ## CNs clouds are often connected to Scc , Dcc
        CNs_labeled = fc.separate_connected_clouds(CNs)
        _2_Scc , _2_Dcc , _2_Str , _2_Str_Cu, _2_CNs ,_2_Cir ,_2_Ncc ,_2_mix,fall_streaks,_dict_cloud_clear_sky = fc.detect_NCs(CNs_labeled ,
                                                                                                                                input_temperature,
                                                                                                                                input_height,
                                                                                                                                input_cloud_base_height,
                                                                                                                               input_cloudnet)

        # Loop through the keys and append the lists
        for key in dict_cloud_clear_sky:
            dict_cloud_clear_sky[key].extend(_dict_cloud_clear_sky[key])

        # Calculate the sum for each list in the dictionary
        sums_dict = {}
        for key, value in dict_cloud_clear_sky.items():
            list_sum = sum(value)
            sums_dict[key] = list_sum

        # Now sums_dict contains the sum of values for each list


        CNs = _2_CNs+_2_Cir+_2_mix+ fall_streaks + mix
        CNs = np.where(CNs == 0, CNs, 5)
        ### some pixels are lost during the watershed separation, dilation will add pixels to cloud edges
        CNs_dilated = ndimage.binary_dilation(CNs, structure=np.ones((8, 8)))
        ### get original pixels
        CNs_o = ~(~CNs_dilated| input_radar.T )
        CNs = ma.masked_where(CNs_o == True, CNs_o).astype(int).filled(5)

        #### dilation sometimes involves that cloud parts from near by located clouds are included for CN-clouds #######
        CNs_mask    = ma.masked_where(CNs ==0, CNs).mask
        Str_Cu_mask = ma.masked_where(Str_Cu==0, Str_Cu).mask
        CNs = ma.masked_where( ( CNs_mask   | ~Str_Cu_mask)== True, CNs).filled(0)

        ### connect Dcc cloud-parts that were accidently separated
        Scc    = Scc +_2_Scc
        Dcc    = Dcc +_2_Dcc
        Ncc    = Ncc +_2_Ncc
        Str    = Str + _2_Str
        Str_Cu = Str_Cu + _2_Str_Cu

        Dcc    = np.where(Dcc    == 0, Dcc, 2)
        Scc    = np.where(Scc     == 0, Scc , 1)
        Str    = np.where(Str    == 0, Str, 3)
        Str_Cu = np.where(Str_Cu == 0, Str_Cu, 4)
        Ncc    = np.where(Ncc == 0, Ncc, 7)
        Scc_Dcc =  Scc + Dcc + Ncc + Str + Cir + CNs+  Str_Cu #+ mix
        Scc_Dcc = np.where(Scc_Dcc < 9, Scc_Dcc, 0)



        allclouds = fc.get_clouds_filtered(Scc_Dcc, input_radar, input_cloudnet , input_temperature)

        return allclouds, Scc_Dcc, sums_dict


    def create_cloud_dataset(self, dataset_path, radar_data = False, plot=False):
        cl_data, ca_data, target_classification_new, target_classification_old, time, Tw, height, Ze, cbh = self.open_dataset(self.classification_path,
                                                                                                                  self.categorization_path)
        radar_masked = self.filter_out_haze_echos(target_classification_new)

        if radar_data:
           radar_masked = self.filter_out_haze_echos(target_classification_new, Ze)

        allclouds, Scc_Dcc , dict_clouds = self.categorize_radar_pixels(radar_masked, Tw, height, cbh, target_classification_new)
        h0 = fc.get_T0(model_temp=Tw, height=height)
        start1, end1, start2, end2 = utils.get_time_interval(time)

        dates = pd.to_datetime(time[0].data)
        df_idx = datetime.date(dates.year, dates.month, dates.day)
        # Convert 'time steps' value to integer
        time_steps_value = int(dict_clouds['time steps'])
        dict_clouds['time steps'] = time_steps_value

        # Get the path of the current script
        current_script_path = os.path.dirname(os.path.abspath(__file__))
        # Construct the relative path to the data folder
        data_folder = os.path.join(current_script_path, '../..', 'data/cloud_objects.xlsx')
        print(dict_clouds)
        utils.append_to_excel(path_to_excel=data_folder,
                              date = df_idx,
                              warm_cloud_column = dict_clouds['Warm clouds'],
                              trade_cu_column = dict_clouds['Trade wind cumuli'],
                              cold_cloud_column = dict_clouds['Cold and mixed phase clouds'],
                              no_cloud_column = dict_clouds['No class clouds'],
                              time_steps_column= dict_clouds['time steps'])


        if plot:
            fig1, ax1 = pc.plot_clouds(start1, end1, allclouds, time, height, h0, cbh, self.plot_path)
            fig2, ax2 = pc.plot_clouds(start2, end2, allclouds, time, height, h0, cbh, self.plot_path)



        ds = xr.Dataset({
        'target_classification': xr.DataArray(
               data   = target_classification_old,   # enter data here
               dims   = ['time', 'height'],
               coords = {'time': time, 'height': height},
               attrs  = {
                   'long_name': 'Cloudnet target classification (old)',
                   'units'     : ''
                   }
               ),

        'target_classification_new': xr.DataArray(
                    data   = target_classification_new,   # enter data here
                    dims   = ['time', 'height'],
                    coords = {'time': time, 'height': height},
                    attrs  = {
                        'long_name': f'Cloudnet target classification (new)',
                        'units'     : f'Probability > 0.6 are Haze echos'
                        }
                    ),

        'Tw': xr.DataArray(
                    data   = Tw[:,:],   # enter data here
                    dims   = ['time', 'height'],
                    coords = {'time': time, 'height': height},
                    attrs  = {
                        'long_name': f'Wet bulb temperature (ECMWF)',
                        'units'     :f'K'
                        },
            ),


        'Scc_Dcc': xr.DataArray(
                    data   = Scc_Dcc.T[:,:],   # enter data here
                    dims   = ['time', 'height'],
                    coords = {'time': time, 'height': height[:]},
                    attrs  = {
                        'long_name': f'Barbados cloud types: Shallow Cumulus (Scc) : 1; Deeper cumulus (Dcc) : 2; Deeper cumulus with stratiform edges (Dcc_Str) : 3; Cumulu Nimbus (CNs) : 5; Stratus (Str) : 4; Cirrus (Cir) : 6; No category (Nc): 7 ',
                        'units'     :f''
                        },
            ),
         'clouds': xr.DataArray(
                    data   = allclouds.T[:,:],   # enter data here
                    dims   = ['time', 'height'],
                    coords = {'time': time, 'height': height[:]},
                    attrs  = {
                        'long_name': f'Cloud types; Clear Sky: 0, Cold Clouds: 1, Warm clouds:2, Warm clouds with cloud base below 1km: 3',
                        'units'     :f''
                        },
            ),


        },

        attrs={'description': 'Cloudnet target classification with the new classification inlcuding Haze echos',
               'longitude': '-59.50',
               'latitude' : '13.07',
               'reference': 'Johanna Roschke, jr55riqa@studserv.uni-leipzig.de'}
        )

        ymd = utils.get_date_str(time)
        ds.to_netcdf(self.dataset_path + f'{ymd}_'+self.platform+'_clouds.nc')

