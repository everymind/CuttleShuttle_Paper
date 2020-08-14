# -*- coding: utf-8 -*-
"""
Paper: "An experimental method for evoking and characterizing dynamic color patterning of cuttlefish during prey capture" by Danbee Kim, Kendra Buresch, Roger Hanlon, and Adam R. Kampff
Analysis: Step 1 of process_cuttle_python Python Workflow

Processes cropped and aligned video of cuttlefish, measures contrast (aka granularity) in multiple spatial bands. 
Generate intermediate files with power at 7 spatial frequency bands for each frame.

Optional flags:
"--display": False (default) or True
"--saveVid": False (default) or True
"--ROI": 'backOnly' (default) or 'entireCuttlefish'

@author: Adam R Kampff and Danbee Kim
"""
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2
import datetime
import logging
import pdb
import argparse

###################################
# SET CURRENT WORKING DIRECTORY
###################################
cwd = os.getcwd()
###################################
# FUNCTIONS
###################################

##########################################################
#### MODIFY THIS FIRST FUNCTION BASED ON THE LOCATIONS OF:
# 1) video_dir (parent folder with all TGB videos)
# AND
# 2) plots_dir (parent folder for all plots output by this script)
### Current default uses a debugging source dataset
##########################################################
def load_data():
    video_dir = r'C:\Users\taunsquared\Dropbox\CuttleShuttle\CuttleShuttle-ManuallyAligned\CroppedAligned\MantleZoom\TentacleShots'
    plots_dir = r'C:\Users\taunsquared\Dropbox\CuttleShuttle\analysis\WoodsHoleAnalysis\draftPlots\intermediates'
    return video_dir, plots_dir
##########################################################

def genBandMasks(number_bands, crop_roi):
    # expected format for crop_roi = [roi_ul_x, roi_ul_y, roi_lr_x, roi_lr_y]
    # Measure ROI size
    roi_width = crop_roi[2] - crop_roi[0]
    roi_height = crop_roi[3] - crop_roi[1]
    # Generate band masks
    StartX = -np.round((roi_width+1)/2)
    EndX = StartX + roi_width
    StartY = -np.round((roi_height+1)/2)
    EndY = StartY + roi_height
    X,Y = np.meshgrid(np.arange(StartX, EndX), np.arange(StartY, EndY).T)
    bands = np.arange(number_bands,0, -1)
    radii = np.power(1/2, bands)
    band_masks = np.zeros((roi_height, roi_width, number_bands))
    for i in np.arange(number_bands):
        if (i == 0):
            band_screen = ((X/roi_width)**2 + (Y/roi_height)**2) <= radii[i]**2
        else:
            band_screen = (((X/roi_width)**2 + (Y/roi_height)**2) > radii[i-1]**2) & (((X/roi_width)**2 + (Y/roi_height)**2) <= radii[i]**2)
        #plt.imshow(band_screen)
        #plt.show()
        band_masks[:,:,i] = band_screen
    return band_masks

def computeFilteredVid(N_frames, N_bands, TS_video, TS_video_path, crop_roi, band_masks, display_bool, save_bool, save_folder):
    # expected format for crop_roi = [roi_ul_x, roi_ul_y, roi_lr_x, roi_lr_y]
    # Measure ROI size
    roi_width = crop_roi[2] - crop_roi[0]
    roi_height = crop_roi[3] - crop_roi[1]
    # Loop through all frames of TS_video and process
    band_energies = np.zeros((N_frames, N_bands))
    #N_frames = 50 # ...for debugging
    for frame in range(0, N_frames):
        # Read current frame
        success, image = TS_video.read()
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Crop
        crop = gray[crop_roi[1]:crop_roi[3], crop_roi[0]:crop_roi[2]]
        crop = np.float32(crop)
        # Transform to Weber contrasts
        mean_crop = np.mean(crop[:])
        weber = (crop-mean_crop)/mean_crop
        # 2D FFT and power spectrum
        fft = np.fft.fft2(weber)
        fft_centered = np.fft.fftshift(fft)
        spectrum = np.real(fft_centered * np.conj(fft_centered))
        # Apply band masks
        band_energy = np.zeros(N_bands)
        for i in np.arange(N_bands):
            band_energy[i] = np.sum(band_masks[:,:,i] * spectrum)
        band_energies[frame, :] = band_energy / sum(band_energy)
        # If displaying or saving, compute filtered images via inverse FFT
        if display_bool or save_bool:
            # setup output video
            output_video_name = os.path.basename(TS_video_path)[:-4] + "_filteredVid.avi"
            output_video_path = os.path.join(save_folder, output_video_name)
            output_video_size = (1280, 400)
            fourcc = cv2.VideoWriter_fourcc('F','M','P','4')
            output_video = cv2.VideoWriter(output_video_path, fourcc, 30, output_video_size, False)
            # load output frames
            filtered_images = np.zeros((roi_width, roi_height * (N_bands + 1)))
            filtered_images[:, 0:roi_height] = crop.T/255
            for i in np.arange(N_bands):
                filtered_fft = np.fft.ifftshift(band_masks[:,:,i] * fft_centered)
                filtered_image = np.real(np.fft.ifft2(filtered_fft))
                offset = (roi_height * (i + 1))
                filtered_images[:, offset:(offset+roi_height)] = filtered_image.T + 0.5
            filtered_images_small = cv2.resize(filtered_images, output_video_size)
        # If displaying, display
        if display_bool:
            #ret = cv2.imshow("Display", spectrum/100000)
            ret = cv2.imshow("Display", filtered_images_small)
            ret = cv2.waitKey(1)
        # If saving, setup and save output video
        if save_bool:
            # save output video
            filtered_images_small_u8 = (filtered_images_small * 200)
            filtered_images_small_u8[filtered_images_small_u8 > 255] = 255
            filtered_images_small_u8[filtered_images_small_u8 < 0] = 0
            filtered_images_small_u8 = np.uint8(filtered_images_small_u8)
            output_video.write(filtered_images_small_u8)
    # Save band energies as intermediate file
    data_name = os.path.basename(TS_video_path)[:-4] + "_bandEnergies.npy"
    data_path = os.path.join(save_folder, data_name)
    np.save(data_path, band_energies)
    # Plot band energies for entire TS_video
    # set fig path and title
    shot_type = os.path.basename(TS_video_path).split('_')[-2]
    animal_name = os.path.basename(TS_video_path).split('_')[1]
    figure_name = os.path.basename(TS_video_path)[:-4] + "_PowerFreqBandPlot.png"
    figure_path = os.path.join(save_folder, figure_name)
    figure_title = "Energy of each frequency band during tentacle shot (shot occurs at frame 180) \n Animal: {a}, Tentacle Shot type: {s}".format(a=animal_name, s=shot_type)
    # draw fig
    plt.figure(figsize=(16,8), dpi=200)
    plt.suptitle(figure_title, fontsize=12, y=0.99)
    plt.plot(band_energies)
    labels = ['band 0', 'band 1','band 2','band 3','band 4','band 5','band 6']
    plt.legend(labels)
    plt.ylabel('Power')
    plt.xlabel('Frame number')
    # save and show
    plt.savefig(figure_path)
    plt.show(block=False)
    plt.pause(1)
    plt.close()
    # Cleanup
    if save_bool:
        output_video.release()
    if display_bool:
        cv2.destroyAllWindows()

##########################################################
# BEGIN SCRIPT
##########################################################
if __name__=='__main__':
    parser = argparse.ArgumentParser(
        description='''Step 1 of process_cuttle_python Python Workflow.
        Processes cropped and aligned video of cuttlefish, measures contrast (aka granularity) in multiple spatial bands. 
        Generate intermediate files with power at 7 spatial frequency bands for each frame.''')
    parser.add_argument("--a", nargs='?', default="check_string_for_empty")
    parser.add_argument("--display", nargs=1, default=False, help="Set to 'True' to display frame by frame analysis as script is running.")
    parser.add_argument("--saveVid", nargs=1, default=False, help="Set to 'True' to save video of bandpass filtered images at each spatial frequency.")
    parser.add_argument("--ROI", nargs=1, default='backOnly', help="Change which part of the video frame to analyse. Options: 'backOnly' (default), 'entireCuttlefish'")
    args = parser.parse_args()
    ###################################
    # SCRIPT LOGGER
    ###################################
    # grab today's date
    now = datetime.datetime.now()
    today_dateTime = now.strftime("%Y-%m-%d_%H-%M-%S")
    logging.basicConfig(filename="process_cuttle_python_01_" + today_dateTime + ".log", filemode='w', level=logging.INFO)
    ###################################
    # SOURCE DATA AND OUTPUT FILE LOCATIONS 
    ###################################
    video_folder, plots_folder = load_data()
    logging.info('DATA FOLDER: %s \n PLOTS FOLDER: %s' % (video_folder, plots_folder))
    print('DATA FOLDER: %s \n PLOTS FOLDER: %s' % (video_folder, plots_folder))
    ###################################
    # DISPLAY AND SAVE TOGGLES
    ###################################
    display = args.display
    save = args.saveVid
    ###################################
    # SPECIFY CROP ROI (upper left pixel (x,y) and lower right pixel (x,y))
    ###################################
    # Only back of cuttlefish
    if args.ROI == 'backOnly':
        roi_ul_x = 600
        roi_ul_y = 350
        roi_lr_x = 1400
        roi_lr_y = 750
    # Entire cuttlefish
    if args.ROI == 'entireCuttlefish':
        roi_ul_x = 300
        roi_ul_y = 100
        roi_lr_x = 1700
        roi_lr_y = 1000
    CropRoi = [roi_ul_x, roi_ul_y, roi_lr_x, roi_lr_y]
    ###################################
    # COLLECT VIDEOS FROM VIDEO_FOLDER
    ###################################
    all_vids = glob.glob(video_folder + os.sep + "*.avi")
    # Loop through videos and process
    for video_path in all_vids: 
        # Open video
        video = cv2.VideoCapture(video_path)
        # Read video parameters
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        # Generate band masks
        NumBands = 7
        BandMasks = genBandMasks(NumBands, CropRoi)
        # If displaying, open display window
        if display:
            cv2.namedWindow("Display")
        computeFilteredVid(num_frames, NumBands, video, video_path, CropRoi, BandMasks, display, save, plots_folder)
        video.release()

#FIN