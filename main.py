import numpy as np
import xarray as xr
from scipy import ndimage
from skimage import measure
import find_clouds as fc
import numpy.ma as ma
import utils
import plot_clouds as pc
class CloudProcessor:
    def __init__(self, file_path, dataset_path, plot_path=None):
        self.file_path = file_path
        self.plot_path = plot_path
        self.dataset_path = dataset_path

    def filter_out_haze_echos(self, input_cloudnet, input_radar):
        radar        = ma.masked_invalid(input_radar).mask
        haze_mask    = ma.masked_where(input_cloudnet==11, input_cloudnet).mask#.astype(int)
        radar_masked = ma.masked_where(haze_mask == True,radar ).filled(True)
        return radar_masked

    def open_dataset(self, input_path):
        data =  xr.open_dataset(input_path, engine='netcdf4')
        time = data.time
        ####### Dimension needs to be reduced when height dimension can not be divided by 4 ##########
        target_classification_new = data.target_classification_new.data[:,:-1]
        target_classification_old = data.target_classification.data[:,:-1]
        cbh    = data.cbh.data
        Tw     = data.Tw.data[:,:-1]
        height = data.height.data[:-1]
        Ze = data.Ze.data[:,:-1]
        ##### Filter out Haze echos #####
        return  data, target_classification_new,  target_classification_old, time, Tw, height, Ze, cbh


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
        Scc , Dcc , Str , Str_Cu, CNs ,Cir ,Ncc ,mix = fc.detect_clouds(lab_image ,
                                                                     input_temperature,
                                                                     input_height,
                                                                     input_cloud_base_height)

        ## CNs clouds are often connected to Scc , Dcc
        CNs_labeled = fc.separate_connected_clouds(CNs)
        _2_Scc , _2_Dcc , _2_Str , _2_Str_Cu, _2_CNs ,_2_Cir ,_2_Ncc ,_2_mix,fall_streaks = fc.detect_NCs(CNs_labeled ,
                                                                                                       input_temperature,
                                                                                                       input_height,
                                                                                                       input_cloud_base_height,
                                                                                                       input_cloudnet)

        CNs = _2_CNs+_2_Cir+_2_mix+ fall_streaks
        CNs = np.where(CNs == 0, CNs, 5)
        ### some pixels are lost during the watershed separation, dilation will add pixels to cloud edges
        CNs_dilated = ndimage.binary_dilation(CNs, structure=np.ones((8, 8)))
        ### get original pixels
        CNs_o = ~(~CNs_dilated| input_radar.T )
        CNs = ma.masked_where(CNs_o == True, CNs_o).astype(int).filled(5)

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
        Scc_Dcc =  Scc + Dcc + Ncc + Str + Cir + CNs+  Str_Cu + mix
        Scc_Dcc = np.where(Scc_Dcc < 9, Scc_Dcc, 0)

        allclouds = fc.get_clouds_filtered(Scc_Dcc, input_radar, input_cloudnet , input_temperature)

        return allclouds, Scc_Dcc

    def create_cloud_dataset(self, dataset_path, plot=False):
        data, target_classification_new, target_classification_old, time, Tw, height, Ze, cbh = self.open_dataset(self.file_path)
        radar_masked = self.filter_out_haze_echos(target_classification_new, Ze)

        allclouds, Scc_Dcc  = self.categorize_radar_pixels(radar_masked, Tw, height, cbh, target_classification_new)
        h0 = fc.get_T0(model_temp=Tw, height=height)
        start1, end1, start2, end2 = utils.get_time_interval(time)

        if plot:
            fig1, ax1 = pc.plot_clouds(start1, end1, allclouds, time, height, h0, cbh, self.plot_path)
            fig2, ax2 = pc.plot_clouds(start2, end2, allclouds, time, height, h0, cbh, self.plot_path)

        ds = xr.Dataset({
            'target_classification': xr.DataArray(
                data=target_classification_old,
                dims=['time', 'height'],
                coords={'time': time, 'height': height},
                attrs={
                    'long_name': 'Cloudnet target classification (old)',
                    'units': ''
                }
            ),
            # Add other data variables similarly
        }, attrs={
            'description': 'Cloudnet target classification with the new classification including Haze echos',
            'longitude': '-59.50',
            'latitude': '13.07',
            'reference': 'Johanna Roschke, jr55riqa@studserv.uni-leipzig.de'
        })

        ymd = utils.get_date_str(time)
        ds.to_netcdf(self.dataset_path + f'{ymd}_barbados_clouds.nc')

