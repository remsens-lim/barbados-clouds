import numpy as np
from scipy import ndimage
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage import measure, util
import cv2
import numpy.ma as ma

def separate_connected_clouds(cloud_type):
    # Load your binary image (binary_dilated) here
    # Downsample the image (needs to be dividable by 4)
    scale_factor = 4  # Adjust this value as needed
    downsampled_image = cv2.resize(cloud_type, None, fx=1/scale_factor, fy=1/scale_factor, interpolation=cv2.INTER_NEAREST)
    #downsampled_image shape (151, 720)
    # Calculate distance transform
    downsampled_image = ndimage.binary_dilation(downsampled_image, structure=np.ones((2,2)))
    distance          = ndimage.distance_transform_edt(downsampled_image)

    # Find local maxima
    coords = peak_local_max(distance, footprint=np.ones((1, 2)), labels=downsampled_image.astype(int))

    # Create the mask
    mask = np.zeros(downsampled_image.shape, dtype=bool)
    mask[tuple(coords.T)] = True

    # Label markers and perform watershed
    markers, _ = ndimage.label(mask)
    labels = watershed(-distance, markers, mask=downsampled_image)

    ##############################################################################################################################
    # Upsample the labeled image
    scale_factor = 4  # Adjust this value to match your previous downsampling factor
    upsampled_image = cv2.resize(labels, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_NEAREST)
    # Convert the upsampled image to the required data type (usually integer)
    upsampled_image = upsampled_image.astype(np.uint8)
    lab_image = upsampled_image

    return lab_image


def detect_clouds(lab_image , model_temp, height, target_classification_new, cloud_base_height):

    properties = ['label','coords', 'area','area_bbox', 'bbox','axis_major_length','axis_minor_length','extent','perimeter']
    table = measure.regionprops_table(lab_image, properties = properties)

    # Extract data from table
    base_h       = height[table['bbox-0']]
    top_h        = height[table['bbox-2']-1]
    length_t2    = table['bbox-3']
    length_t1    = table['bbox-1']
    input_labels = table['label']
    size         = table['area']
    height_t2 = table['bbox-2']
    height_t1 = table['bbox-0']
    cloud_time = length_t2-length_t1

    cloudnet = target_classification_new
    cbh = cloud_base_height
   # cbh_4 = ma.masked_where(cbh > 4000, cbh).filled(np.nan)
    cbh_1 = ma.masked_where(cbh > 1000, cbh).filled(0)
    cbh_2 = ma.masked_where(cbh > 3000, cbh).filled(0)
    cbh_2 = ma.masked_where(cbh_2 < 1000, cbh_2).filled(0)

    # Initialize empty lists for output labels
    output_Scc = []
    output_Dcc = []
    output_nc = []
    output_Str = []
    output_Ncc = []
    output_Str_Cu = []
    output_Cir = []
    output_CNs = []
    output_mixed = []
    output_warm_cloud_time_steps     = []
    output_trade_wind_cu_time_steps  = []
    output_mixed_and_cold_time_steps = []
    output_no_class_time_steps       = []
    # Loop through clouds and classify them
    for i in range(len(base_h)):
        #print(length_t2[i]-length_t1[i])
        t0_idx = np.nanmean(model_temp[length_t1[i]:length_t2[i]],0) - 273.15
        t0_idx = ma.masked_invalid(t0_idx).filled(-10)
        #print(t0_idx)
        t0_idx = height[ np.where(t0_idx < 0)[0][0] ] # relax or choose 0
        t0_idx = ma.masked_where(t0_idx == 0, t0_idx).filled(123)

        cbh1 = np.count_nonzero(ma.masked_invalid(cbh_1[length_t1[i]:length_t2[i]]))
        cbh2 = np.count_nonzero(ma.masked_invalid(cbh_2[length_t1[i]:length_t2[i]]))
        cbh_diff = cbh2- cbh1

        # Cloudnet target classification for the area of the bounding box
        target  = cloudnet[length_t1[i]:length_t2[i],height_t1[i]:height_t2[i]]
        target = ma.masked_where(target > 3, target).filled(0)
        # Drizzle percentage of target (no clear sky)
        drizzle = np.sum(target==2)*100/(ma.masked_where(target == 0, target ).compressed().shape[0])

        trade_inversion1 = 2500

        LCL = 1000

        condition_Scc      = (top_h[i] < t0_idx) & (400 < base_h[i] < LCL)  & ( top_h[i] < trade_inversion1)  & (cbh1 > 1)
        condition_Str      = (top_h[i] <= t0_idx) &(LCL <= base_h[i] < t0_idx)
        condition_Str_Cu   = (top_h[i] < t0_idx) &(base_h[i] <= LCL)  & (LCL<= top_h[i] < t0_idx) & (cbh_diff >1)
        no_class           = (top_h[i] < t0_idx) &(size[i] <= 2 )
        condition_Dcc      = (LCL < top_h[i] < t0_idx) & (base_h[i] < 300) & (cbh_diff <= 1)
        condition_Cir      = (base_h[i] >= t0_idx-1)  & ( top_h[i] > t0_idx)
        condition_CNs      = (base_h[i] < LCL)     & ( top_h[i] > t0_idx)
        condition_mixed    = (LCL < base_h[i] < t0_idx)     & ( top_h[i] >= t0_idx)

        if condition_Str:
           # fall streaks are usually cover more pixels in the height dimension compared to the number of pixels in the time dimension
           box = cloudnet[length_t1[i]:length_t2[i],height_t1[i]:height_t2[i]]
           ratio = box.shape[0]/box.shape[1]


           # Check if Stratus (Str) are fallstreaks
           if top_h[i] < 3000:
               if size[i] < 5:
                   if ratio > 1.5:
                       condition_Str == True
                   if ratio <= 1.5:
                       condition_Str = False
                       condition_mixed = True
           else:
               if drizzle == 100:
                   if ratio <= 1.5:
                       condition_Str = False
                       condition_mixed = True
               else:
                   condition_Str = True

        # Append output labels based on classification
        output_Scc.append(input_labels[i] * condition_Scc)
        output_Str.append(input_labels[i] * condition_Str)
        output_Str_Cu.append(input_labels[i] * condition_Str_Cu)
        output_Dcc.append(input_labels[i] * condition_Dcc)
        output_mixed.append(input_labels[i] * condition_mixed)
        output_Ncc.append(input_labels[i] * no_class)
        output_Cir.append(input_labels[i] * condition_Cir)
        output_nc.append(input_labels[i]  * no_class)
        output_CNs.append(input_labels[i] * condition_CNs)

        ###################################################################
        if condition_Str :
            output_warm_cloud_time_steps.append(cloud_time[i])

        elif condition_Str_Cu :
            output_warm_cloud_time_steps.append(cloud_time[i])

        elif condition_Dcc :
            output_warm_cloud_time_steps.append(cloud_time[i])
            output_trade_wind_cu_time_steps.append(cloud_time[i])
        elif condition_Scc:
            output_warm_cloud_time_steps.append(cloud_time[i])
            output_trade_wind_cu_time_steps.append(cloud_time[i])
        elif condition_Cir:
            output_mixed_and_cold_time_steps.append(cloud_time[i])
        elif condition_CNs:
            output_mixed_and_cold_time_steps.append(cloud_time[i])
        elif condition_mixed:
            output_mixed_and_cold_time_steps.append(cloud_time[i])
        elif no_class :
            output_no_class_time_steps.append(cloud_time[i])



    output_Scc = np.array(output_Scc)
    output_Dcc = np.array(output_Dcc)
    output_mixed = np.array(output_mixed)
    output_nc  = np.array(output_nc)
    output_Str = np.array(output_Str)
    output_Str_Cu = np.array(output_Str_Cu)
    output_Cir = np.array(output_Cir)
    output_CNs = np.array(output_CNs)

    Scc    = util.map_array(lab_image, input_labels, output_Scc)[:,:]
    Dcc    = util.map_array(lab_image, input_labels, output_Dcc)[:,:]
    Ncc    = util.map_array(lab_image, input_labels, output_nc) [:,:]
    Str    = util.map_array(lab_image, input_labels, output_Str)[:,:]
    Str_Cu = util.map_array(lab_image, input_labels, output_Str_Cu)[:,:]
    Cir    = util.map_array(lab_image, input_labels, output_Cir)[:,:]
    CNs    = util.map_array(lab_image, input_labels, output_CNs)[:,:]
    mix    = util.map_array(lab_image, input_labels, output_mixed)[:,:]

    #
    Scc    = np.where(Scc    == 0, Scc, 1)
    Dcc    = np.where(Dcc    == 0, Dcc, 2)
    Str    = np.where(Str    == 0, Str, 3)
    Str_Cu = np.where(Str_Cu == 0, Str_Cu, 4)

    ################## AVOID DOUBLE LABLED CLOUDS #################################
    Dcc_mask    = ma.masked_where(Dcc ==0, Dcc).mask
    Scc_mask    = ma.masked_where(Scc ==0, Scc).mask
    Str_Cu_mask = ma.masked_where(Str_Cu==0, Str_Cu).mask
    Str_Cu = ma.masked_where( ( ~Dcc_mask   | Str_Cu_mask)== True, Str_Cu).filled(0)
    Str_Cu = ma.masked_where( Str_Cu > 0, Str_Cu).filled(4)
    Str_Cu_mask = ma.masked_where(Str_Cu==0, Str_Cu).mask

    Scc = ma.masked_where( ( ~Str_Cu_mask  | Scc_mask)== True, Scc).filled(0)
    Scc = ma.masked_where(Scc > 0, Scc).filled(1)

    CNs = np.where(CNs == 0, CNs, 5)
    Cir = np.where(Cir == 0, Cir, 6)
    Ncc = np.where(Ncc == 0, Ncc, 7)
    mix = np.where(mix == 0, mix, 8)





    dict_cloud_clear_sky = {'Warm clouds': output_warm_cloud_time_steps    ,
                            'Trade wind cumuli': output_trade_wind_cu_time_steps ,
                            'Cold and mixed phase clouds': output_mixed_and_cold_time_steps,
                            'No class clouds': output_no_class_time_steps      ,
                            'time steps' :[cloudnet.shape[0]/2]
                            }
    return Scc , Dcc , Str , Str_Cu, CNs ,Cir ,Ncc ,mix, dict_cloud_clear_sky


def detect_NCs(lab_image , model_temp, height, cloud_base_height,target_classification_new):
    cbh = cloud_base_height
    properties = ['label','coords', 'area','area_bbox', 'bbox','axis_major_length','axis_minor_length','extent','perimeter']
    table = measure.regionprops_table(lab_image, properties = properties)

    # Extract data from table
    base_h       = height[table['bbox-0']]
    top_h        = height[table['bbox-2']-1]
    length_t2    = table['bbox-3']
    length_t1    = table['bbox-1']
    input_labels = table['label']
    temperature  = model_temp[table['bbox-0']] - 273.15
    size         = table['area']
    height_t2 = table['bbox-2']
    height_t1 = table['bbox-0']

    cloud_time = length_t2-length_t1
    # Initialize empty lists for output labels
    output_Scc = []

    output_Dcc = []
    output_nc = []
    output_Str = []
    output_Ncc = []
    output_Str_Cu = []
    output_Cir = []
    output_CNs = []
    output_mixed = []
    output_fall_streaks = []
    output_warm_cloud_time_steps     = []
    output_trade_wind_cu_time_steps  = []
    output_mixed_and_cold_time_steps = []
    output_no_class_time_steps       = []

    cloudnet = target_classification_new
    cdrizzle = ma.masked_where(target_classification_new > 4,target_classification_new).filled(0)
    cdrizzle = ma.masked_where(cdrizzle == 0, cdrizzle , np.nan)

    cbh_1 = ma.masked_where(cbh > 1000, cbh).filled(0)
    cbh_2 = ma.masked_where(cbh > 2000, cbh).filled(0)
    cbh_2 = ma.masked_where(cbh_2 < 1000, cbh_2).filled(0)
    cbh_3 = ma.masked_where(cbh  > 4000, cbh ).filled(0)
    cbh_3 = ma.masked_where(cbh_3 < 2000, cbh_3).filled(0)


    # Loop through clouds and classify them
    for i in range(len(base_h)):
        #print(length_t2, length_t1)
        t0_idx = np.nanmean(model_temp[length_t1[i]:length_t2[i]],0) - 273.15
        t0_idx = ma.masked_invalid(t0_idx).filled(-10)
        t0_idx = height[ np.where(t0_idx < 0)[0][0] ]# relax or choose 0
        t0_idx = ma.masked_where(t0_idx ==0, t0_idx).filled(123)


        target  = cloudnet[length_t1[i]:length_t2[i],height_t1[i]:height_t2[i]]
        target = ma.masked_where(target > 3, target).filled(0)
        drizzle = np.sum(target==2)*100/(ma.masked_where(target == 0, target ).compressed().shape[0])

        cbh1 = np.count_nonzero(ma.masked_invalid(cbh_1[length_t1[i]:length_t2[i]]))
        cbh2 = np.count_nonzero(ma.masked_invalid(cbh_2[length_t1[i]:length_t2[i]]))
        cbh3 = np.count_nonzero(ma.masked_invalid(cbh_3[length_t1[i]:length_t2[i]]))
        cbh_diff = cbh2- cbh1
        trade_inversion1 = 2500
        LCL = 1000

        condition_Scc      = (top_h[i] < t0_idx) & (400 < base_h[i] < LCL)  & ( top_h[i] < trade_inversion1)  & (cbh1 > 1)& ((np.nanmean(cbh1) - base_h[i]) < 100)
        condition_Str      = (top_h[i] <= t0_idx) &(LCL <= base_h[i] <= t0_idx)     & (drizzle < 60) #& (percentages[i]< 40)
        condition_fall_streaks = (top_h[i] <= t0_idx) &( base_h[i] <= t0_idx) & (drizzle > 60) #& (percentages[i]> 40)

        condition_Str_Cu   = (top_h[i] < t0_idx) &(base_h[i] < LCL)  & (LCL< top_h[i] < t0_idx) & (cbh_diff >1)
        no_class           = (top_h[i] < t0_idx) &(size[i] <= 2 )
        condition_Dcc      = (LCL < top_h[i] < t0_idx) & (base_h[i] < 300) & (cbh_diff < 1) #& (1000< ex[i]< t0_idx-base_h[i])

        condition_Cir          = (base_h[i] >= t0_idx-1)  & ( top_h[i] > t0_idx)
        condition_CNs          = (base_h[i] < LCL)     & ( top_h[i] > t0_idx)
        condition_mixed        =     (LCL < base_h[i] < t0_idx)     & ( top_h[i] >= t0_idx)

        if condition_Dcc:
            if drizzle > 60:
                condition_Dcc = False
                condition_fall_streaks = True
        elif condition_Str:

            cbh_2km = ma.masked_where(cbh_2[length_t1[i]:length_t2[i]]==0, cbh_2[length_t1[i]:length_t2[i]]).filled(np.nan)
            if base_h[i] < 2000:
                if  np.nanmean(cbh_2km)  < 1000:
                    if (cloudnet[length_t1[i]-2,height_t2[i]]) > 0:
                        condition_Str = False
                        condition_CNs = True
                    else:
                        condition_Str = False
                        condition_Dcc = True

        elif condition_Str_Cu:
            if base_h[i] < 2000:
                if np.nanmean(cbh2) < 2:
                    condition_Str_Cu = False
                    condition_CNs = True
            elif 2000 < base_h[i] < 4000:
                if np.nanmean(cbh3) < 2:
                    condition_Str_Cu = False
                    condition_CNs = True


        # Append output labels based on classification
        output_Scc.append(input_labels[i] * condition_Scc)
        output_Str.append(input_labels[i] * condition_Str)
        output_Str_Cu.append(input_labels[i] * condition_Str_Cu)
        output_Dcc.append(input_labels[i] * condition_Dcc)
        output_mixed.append(input_labels[i] * condition_mixed)
        output_Ncc.append(input_labels[i] * no_class)
        output_Cir.append(input_labels[i] * condition_Cir)
        output_nc.append(input_labels[i]  * no_class)
        output_CNs.append(input_labels[i] * condition_CNs)
        output_fall_streaks.append(input_labels[i]* condition_fall_streaks)
        ###################################################################
        if condition_Str :
            output_warm_cloud_time_steps.append(cloud_time[i])

        elif condition_Str_Cu :
            output_warm_cloud_time_steps.append(cloud_time[i])

        elif condition_Dcc :
            output_warm_cloud_time_steps.append(cloud_time[i])
            output_trade_wind_cu_time_steps.append(cloud_time[i])
        elif condition_Scc:
            output_warm_cloud_time_steps.append(cloud_time[i])
            output_trade_wind_cu_time_steps.append(cloud_time[i])
        elif condition_Cir:
            output_mixed_and_cold_time_steps.append(cloud_time[i])
        elif condition_CNs:
            output_mixed_and_cold_time_steps.append(cloud_time[i])
        elif condition_mixed:
            output_mixed_and_cold_time_steps.append(cloud_time[i])
        elif no_class :
            output_no_class_time_steps.append(cloud_time[i])



    output_Scc = np.array(output_Scc)
    output_Dcc = np.array(output_Dcc)
    output_mixed = np.array(output_mixed)
    output_nc  = np.array(output_nc)
    output_Str = np.array(output_Str)
    output_Str_Cu = np.array(output_Str_Cu)
    output_Cir = np.array(output_Cir)
    output_CNs = np.array(output_CNs)
    output_fall_streaks= np.array(output_fall_streaks)

    Scc    = util.map_array(lab_image, input_labels, output_Scc)[:,:]
    Dcc    = util.map_array(lab_image, input_labels, output_Dcc)[:,:]
    Ncc    = util.map_array(lab_image, input_labels, output_nc)[:,:]
    Str    = util.map_array(lab_image, input_labels, output_Str)[:,:]
    Str_Cu = util.map_array(lab_image, input_labels, output_Str_Cu)[:,:]
    Cir    = util.map_array(lab_image, input_labels, output_Cir)[:,:]
    CNs    = util.map_array(lab_image, input_labels, output_CNs)[:,:]
    mix    = util.map_array(lab_image, input_labels, output_mixed)[:,:]
    fall_streaks = util.map_array(lab_image, input_labels, output_fall_streaks)[:,:]

    Scc    = np.where(Scc    == 0, Scc, 1)
    Dcc    = np.where(Dcc    == 0, Dcc, 2)
    Str    = np.where(Str    == 0, Str, 3)
    Str_Cu = np.where(Str_Cu == 0, Str_Cu, 4)

    Dcc_mask    = ma.masked_where(Dcc   ==0, Dcc).mask
    Scc_mask    = ma.masked_where(Scc   ==0, Scc).mask
    Str_Cu_mask = ma.masked_where(Str_Cu==0, Str_Cu).mask
    Str_Cu      = ma.masked_where( ( ~Dcc_mask   | Str_Cu_mask)== True, Str_Cu).filled(0)
    Str_Cu      = ma.masked_where(Str_Cu > 0, Str_Cu).filled(4)
    Str_Cu_mask = ma.masked_where(Str_Cu ==0, Str_Cu).mask

    Scc = ma.masked_where( ( ~Str_Cu_mask  | Scc_mask)== True, Scc).filled(0)
    Scc = ma.masked_where(Scc > 0, Scc).filled(1)
    CNs = np.where(CNs == 0, CNs, 5)
    Cir = np.where(Cir == 0, Cir, 5)
    Ncc = np.where(Ncc == 0, Ncc, 7)
    mix = np.where(mix == 0, mix, 8)
    fall_streaks = np.where(fall_streaks == 0, fall_streaks, 5)

    dict_cloud_clear_sky = {'Warm clouds': output_warm_cloud_time_steps    ,
                            'Trade wind cumuli': output_trade_wind_cu_time_steps ,
                            'Cold and mixed phase clouds': output_mixed_and_cold_time_steps,
                            'No class clouds': output_no_class_time_steps      ,
                            'time steps' :[cloudnet.shape[0]/2]
                            }
    return Scc , Dcc , Str , Str_Cu, CNs ,Cir ,Ncc ,mix, fall_streaks, dict_cloud_clear_sky


def get_clouds_filtered(Scc_Dcc, radar_masked, target_classification_new , model_temp):

    maske_warm  =  ma.masked_where((Scc_Dcc  > 4), Scc_Dcc).mask
    maske_trade =  ma.masked_where(Scc_Dcc   > 2,  Scc_Dcc).mask
    maske_cold  =  ma.masked_where(Scc_Dcc   < 5,  Scc_Dcc).mask

    classi       = ma.masked_where(radar_masked == True,  target_classification_new).filled(0)
    classi       = ma.masked_where(classi >7 , classi).filled(0)

    classi_warm  = ma.masked_where(maske_warm  == True,   classi.T[:,:]).filled(0)
    classi_trade = ma.masked_where(maske_trade == True,   classi.T[:,:]).filled(0)
    classi_cold  = ma.masked_where(maske_cold  == True,   classi.T[:,:]).filled(0)

    warm_c  = ma.masked_where(classi_warm  > 0  ,classi_warm).filled(2)
    trade_c = ma.masked_where(classi_trade > 0  ,classi_trade).filled(3)
    cold_c  = ma.masked_where(classi_cold  > 0  ,classi_cold).filled(1)

    # check that all single pixels that have not been classified as cloud are filtered in the cold region
    Tw_     = model_temp
    Tw_mask = ma.masked_where(Tw_-273.15 < 0, Tw_-273.15).mask

    warm_c   = ma.masked_where(Tw_mask.T== True, warm_c  ).filled(0)
    trade_c  = ma.masked_where(Tw_mask.T== True, trade_c).filled(0)
    allclouds_ = warm_c +trade_c +cold_c

    return allclouds_

def get_T0(model_temp, height):
    h0 = []
    model_temp = ma.masked_invalid(model_temp).filled(1)
    for i in range(len(model_temp)):

        idx = np.where(model_temp[i]-273.15 <0)[0][0]
        if idx == 0:
            h0.append(height[123])
        else:
            h0.append(height[idx])
    return h0



