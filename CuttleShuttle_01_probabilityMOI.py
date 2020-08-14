# -*- coding: utf-8 -*-
"""
Paper: "An experimental method for evoking and characterizing dynamic color patterning of cuttlefish during prey capture" by Danbee Kim, Kendra Buresch, Roger Hanlon, and Adam R. Kampff
Analysis: Probability of MOI

Collects csv files of all moments of interest (MOI) from full, primary experimental dataset. 
Calculate the probability of an MOI happening after 1st, 2nd, or 3rd previous MOI (i.e. catch happening after tentacle shot, tentacle shot happening after orientation, etc).
Used in paper to calculate results section "Accuracy of Prey Capture". 

Optional flags:
"--mOI": Choose at least one from the following list of MOIs: catches (default), tentacle-shots, orients, homebase

@author: Adam R Kampff and Danbee Kim
"""
import os
import glob
import datetime
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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
# 1) dataset_dir (folder with full, primary experimental dataset)
# AND
# 2) plots_dir (parent folder for all plots output by this script)
### Current default uses a debugging source dataset
##########################################################
def load_data():
    dataset_dir = r"C:\Users\taunsquared\Dropbox\CuttleShuttle\CuttleShuttle-VideoDataset-Raw"
    plots_dir = r'C:\Users\taunsquared\Dropbox\CuttleShuttle\analysis\forPaper\plots'
    return dataset_dir, plots_dir
##########################################################

def convert_timestamps_to_secs_from_start(allA_timestamps_dict, list_of_MOIs):
    converted_dict = {}
    for animal in allA_timestamps_dict:
        converted_dict[animal] = {}
        all_session_dates = []
        all_session_lens = []
        all_food_offerings = []
        all_homebases = []
        all_orientations = []
        all_tentacle_shots = []
        all_catches = []
        all_mois = [all_food_offerings, all_homebases, all_orientations, all_tentacle_shots, all_catches]
        for session_date in sorted(allA_timestamps_dict[animal].keys()):
            all_session_dates.append(session_date)
            start_ts = allA_timestamps_dict[animal][session_date]['session-vids'][0]
            end_ts = allA_timestamps_dict[animal][session_date]['session-vids'][-1]
            session_len = (end_ts - start_ts).total_seconds()
            all_session_lens.append(session_len)
            if len(allA_timestamps_dict[animal][session_date].keys())==1 and 'session-vids' in allA_timestamps_dict[animal][session_date]:
                print("No moments of interest for animal {a} on {s}".format(a=animal,s=session_date))
                for moi in range(len(list_of_MOIs)):
                    all_mois[moi].append([])
            else:
                for moi in range(len(list_of_MOIs)):
                    if list_of_MOIs[moi] in allA_timestamps_dict[animal][session_date]:
                        this_session_mois = []
                        if allA_timestamps_dict[animal][session_date][list_of_MOIs[moi]].size <= 1:
                            timestamp = allA_timestamps_dict[animal][session_date][list_of_MOIs[moi]]
                            time_diff = (timestamp - start_ts).total_seconds()
                            this_session_mois.append(time_diff)
                        else:
                            for timestamp in allA_timestamps_dict[animal][session_date][list_of_MOIs[moi]]:
                                time_diff = (timestamp - start_ts).total_seconds()
                                this_session_mois.append(time_diff)
                        all_mois[moi].append(this_session_mois)
                    else:
                        all_mois[moi].append([])
        converted_dict[animal]['session dates'] = all_session_dates
        converted_dict[animal]['session durations'] = all_session_lens
        for moi in range(len(list_of_MOIs)):
            converted_dict[animal][list_of_MOIs[moi]] = all_mois[moi]
    return converted_dict

def calc_prob_MOI_sequence(secsFromStart_dict, MOI, prevMOI):
    MOIProb_dict = {}
    for animal in secsFromStart_dict:
        MOIProb_dict[animal] = {}
        all_MOI = secsFromStart_dict[animal][MOI]
        all_prevMOI = secsFromStart_dict[animal][prevMOI]
        for day in range(len(all_MOI)):
            if len(all_MOI[day])>0:
                for moi in all_MOI[day]:
                    if moi in all_prevMOI[day]:
                        attempt_number = all_prevMOI[day].index(moi)
                        MOIProb_dict[animal][attempt_number] = MOIProb_dict[animal].setdefault(attempt_number,0) + 1
                    else:
                        try:
                            attempt_number = [x-moi>0 for x in all_prevMOI[day]].index(True)
                            MOIProb_dict[animal][attempt_number] = MOIProb_dict[animal].setdefault(attempt_number,0) + 1
                        except Exception:
                            print('Error in {a}, day {d}'.format(a=animal, d=day))
    return MOIProb_dict

def plot_probMOIseq(probMOIseq_dict, MOI_str, prevMOI_str, plots_dir, todays_dt):
    allA_probMOIseq = []
    for animal in probMOIseq_dict:
        all_numPrevMOI = sorted(probMOIseq_dict[animal].keys())
        if len(all_numPrevMOI)>0:
            max_numPrevMOI = max(all_numPrevMOI)
            frequencies_to_plot = [0*x for x in range(max_numPrevMOI+1)]
            for numPrevMOI in all_numPrevMOI:
                frequencies_to_plot[numPrevMOI] = probMOIseq_dict[animal][numPrevMOI]
        else:
            print('No previous MOIs for {a}'.format(a=animal))
        allA_probMOIseq.append(frequencies_to_plot)
    mostTS_beforeCatch = max([len(x) for x in allA_probMOIseq])
    for animal in range(len(allA_probMOIseq)):
        pad_N = mostTS_beforeCatch-len(allA_probMOIseq[animal])
        padded_animal = allA_probMOIseq[animal]+[0]*pad_N
        allA_probMOIseq[animal] = padded_animal
    # set figure save path and title
    figure_name = 'Prob_MomentsOfInterestSeq_' + prevMOI_str + 'Before' + MOI_str + '_allAnimals_' + todays_dt + '.png'
    figure_path = os.path.join(plots_dir, figure_name)
    figure_title = 'Number of '+ prevMOI_str + ' before a ' + MOI_str + ' for all animals'
    # set axes and other figure properties
    ax = plt.figure(figsize=(16,9), dpi=200)
    plt.suptitle(figure_title, fontsize=12, y=0.98)
    x = np.arange(mostTS_beforeCatch)  # the label locations
    plt.xticks(x)
    plt.xlabel("{pM} before {M}".format(pM=prevMOI_str, M=MOI_str))
    plt.ylabel("Frequency")
    plt.grid(b=True, which='major', linestyle='-')
    width = 0.1  # the width of the bars
    # draw bars for 5 animals at each x tick
    plt.bar(x - 2*width, allA_probMOIseq[0], width, label='L1-H2013-01')
    plt.bar(x - width, allA_probMOIseq[1], width, label='L1-H2013-02')
    plt.bar(x, allA_probMOIseq[2], width, label='L1-H2013-03')
    plt.bar(x + width, allA_probMOIseq[3], width, label='L7-H2013-01')
    plt.bar(x + 2*width, allA_probMOIseq[4], width, label='L7-H2013-02')
    # Add legend
    ax.legend()
    # save and display fig
    plt.savefig(figure_path)
    plt.show(block=False)
    plt.pause(1)
    plt.close()

##########################################################
# BEGIN SCRIPT
##########################################################
if __name__=='__main__':
    parser = argparse.ArgumentParser(
        description='''Probability of MOI.
        Collects csv files of all moments of interest (MOI) from full, primary experimental dataset. 
        Calculate the probability of an MOI happening after 1st, 2nd, or 3rd previous MOI (i.e. catch happening after tentacle shot, tentacle shot happening after orientation, etc).
        Used in paper to calculate results section "Accuracy of Prey Capture". ''')
    parser.add_argument("--a", nargs='?', default="check_string_for_empty")
    parser.add_argument("--MOI", nargs='+', type=str, default='catches', help="Choose at least one from the following list of MOIs: catches (default), tentacle-shots, orients, homebase")
    args = parser.parse_args()
    ###################################
    # SCRIPT LOGGER
    ###################################
    # grab today's date
    now = datetime.datetime.now()
    today_dateTime = now.strftime("%Y-%m-%d_%H-%M-%S")
    logging.basicConfig(filename="probabilityMOIs_" + today_dateTime + ".log", filemode='w', level=logging.INFO)
    ###################################
    # SOURCE DATA AND OUTPUT FILE LOCATIONS 
    ###################################
    raw_dataset_folder, plots_folder = load_data()
    logging.info('DATA FOLDER: %s \n PLOTS FOLDER: %s' % (raw_dataset_folder, plots_folder))
    print('DATA FOLDER: %s \n PLOTS FOLDER: %s' % (raw_dataset_folder, plots_folder))
    ###################################
    # COLLECT CSV FILES FOR MOI
    ###################################
    animals = ['L1-H2013-01', 'L1-H2013-02', 'L1-H2013-03', 'L7-H2013-01', 'L7-H2013-02', 'L7-H2013-03']
    allMOI_allA = {}
    # extract data from csv and put into dictionary
    logging.info('Extracting raw data from csv...')
    print('Extracting raw data from csv...')
    for animal in animals:
        print('Working on animal {a}'.format(a=animal))
        allMOI_allA[animal] = {}
        MOI_homebase = glob.glob(raw_dataset_folder + os.sep + animal + os.sep + "*" + os.sep + "homebase*.csv")
        MOI_orients = glob.glob(raw_dataset_folder + os.sep + animal + os.sep + "*" + os.sep + "orients*.csv")
        MOI_TS = glob.glob(raw_dataset_folder + os.sep + animal + os.sep + "*" + os.sep + "tentacle_shots*.csv")
        MOI_catches = glob.glob(raw_dataset_folder + os.sep + animal + os.sep + "*" + os.sep + "catches*.csv")
        food_offerings = glob.glob(raw_dataset_folder + os.sep + animal + os.sep + "*" + os.sep + "food_available*.csv")
        session_vids = glob.glob(raw_dataset_folder + os.sep + animal + os.sep + "*" + os.sep + "session_video*.csv")
        all_MOI = [MOI_homebase, MOI_orients, MOI_TS, MOI_catches, food_offerings, session_vids]
        for MOI_type in range(len(all_MOI)):
            for csv_file in all_MOI[MOI_type]:
                if os.path.getsize(csv_file)>0:
                    csv_name = csv_file.split(os.sep)[-1]
                    csv_date = csv_file.split(os.sep)[-2]
                    csv_animal = csv_file.split(os.sep)[-3]
                    current_dict_level = allMOI_allA[animal].setdefault(csv_date,{})
                    # read csv file and convert timestamps into datetime objects
                    str2date = lambda x: datetime.datetime.strptime(x.decode("utf-8").split('+')[0][:-1], '%Y-%m-%dT%H:%M:%S.%f')
                    csv_MOI = np.genfromtxt(csv_file, dtype=None, delimiter=",", converters={0:str2date})
                    if MOI_type == 0:
                        allMOI_allA[animal][csv_date]['homebase'] = csv_MOI
                    if MOI_type == 1:
                        allMOI_allA[animal][csv_date]['orients'] = csv_MOI
                    if MOI_type == 2:
                        allMOI_allA[animal][csv_date]['tentacle-shots'] = csv_MOI
                    if MOI_type == 3:
                        allMOI_allA[animal][csv_date]['catches'] = csv_MOI
                    if MOI_type == 4:
                        allMOI_allA[animal][csv_date]['food-offerings'] = csv_MOI
                    if MOI_type == 5:
                        allMOI_allA[animal][csv_date]['session-vids'] = csv_MOI
    print('Finished extracting csv data!')
    logging.info('Finished extracting csv data!')
    ###################################
    # CONVERT TIMESTAMPS AND ANIMAL NUMBERS
    ###################################
    # convert timestamp obj's into ints (microseconds from start)
    MOIs = ['food-offerings','homebase','orients','tentacle-shots','catches']
    logging.info('Converting timestamps to microseconds from start...')
    print('Converting timestamps to microseconds from start...')
    allMOI_allA_converted = convert_timestamps_to_secs_from_start(allMOI_allA, MOIs)
    # convert animal numbers into names
    animal_names = {'L1-H2013-01':'Dora','L1-H2013-02':'Scar','L1-H2013-03':'Ender','L7-H2013-01':'Old Tom','L7-H2013-02':'Plato','L7-H2013-03':'Blaise'}
    ##################################################################################
    ### ---- PROBABILITY OF MOIS HAPPENING AFTER 1ST/2ND/3RD/4TH PREVIOUS MOI ---- ###
    ##################################################################################
    MOI_to_prevMOI = {'catches': 'tentacle-shots', 'tentacle-shots': 'orients', 'orients': 'food-offerings', 'homebase': 'catches'}
    MOIs = args.MOI
    for MOI in MOIs:
        logging.info('Calculating probability of {m} happening after {pm}...'.format(m=MOI, pm=MOI_to_prevMOI[MOI]))
        print('Calculating probability of {m} happening after {pm}...'.format(m=MOI, pm=MOI_to_prevMOI[MOI]))
        allA_probMOI = calc_prob_MOI_sequence(allMOI_allA_converted, MOI, MOI_to_prevMOI[MOI])
        # summary stats
        first_MOI_prob = {}
        second_MOI_prob = {}
        third_MOI_prob = {}
        for animal in allA_probMOI:
            if bool(allA_probMOI[animal]):
                total_prevMOIs = sum(allA_probMOI[animal].values())
                first_attempt = allA_probMOI[animal].get(0, 0)
                first_MOI_prob[animal] = first_attempt/total_prevMOIs
                two_attempts_or_less = allA_probMOI[animal].get(0, 0) + allA_probMOI[animal].get(1, 0)
                second_MOI_prob[animal] = two_attempts_or_less/total_prevMOIs
                three_attempts_or_less = allA_probMOI[animal].get(0, 0) + allA_probMOI[animal].get(1, 0) + allA_probMOI[animal].get(2, 0)
                third_MOI_prob[animal] = three_attempts_or_less/total_prevMOIs
        for animal in first_MOI_prob:
            logging.info('Probability of animal {a} making {m} after first {pm}: {prob:.4f}'.format(a=animal, m=MOI, pm=MOI_to_prevMOI[MOI], prob=first_MOI_prob[animal]))
            print('Probability of animal {a} making {m} after first {pm}: {prob:.4f}'.format(a=animal, m=MOI, pm=MOI_to_prevMOI[MOI], prob=first_MOI_prob[animal]))
        mean_first_attempt_success = np.mean([x for x in first_MOI_prob.values()])
        var_first_attempt_success = np.var([x for x in first_MOI_prob.values()])
        logging.info('Mean probability of all animals making {m} after first {pm}: {mean:.4f} +/- {var:.4f}'.format(m=MOI, pm=MOI_to_prevMOI[MOI], mean=mean_first_attempt_success, var=var_first_attempt_success))
        print('Mean probability of all animals making {m} after first {pm}: {mean:.4f} +/- {var:.4f}'.format(m=MOI, pm=MOI_to_prevMOI[MOI], mean=mean_first_attempt_success, var=var_first_attempt_success))
        for animal in second_MOI_prob:
            logging.info('Probability of animal {a} making {m} after second {pm}: {prob:.4f}'.format(a=animal, m=MOI, pm=MOI_to_prevMOI[MOI], prob=second_MOI_prob[animal]))
            print('Probability of animal {a} making {m} after second {pm}: {prob:.4f}'.format(a=animal, m=MOI, pm=MOI_to_prevMOI[MOI], prob=second_MOI_prob[animal]))
        mean_second_attempt_success = np.mean([x for x in second_MOI_prob.values()])
        var_second_attempt_success = np.var([x for x in second_MOI_prob.values()])
        logging.info('Mean probability of all animals making {m} after second {pm}: {mean:.4f} +/- {var:.4f}'.format(m=MOI, pm=MOI_to_prevMOI[MOI], mean=mean_second_attempt_success, var=var_second_attempt_success))
        print('Mean probability of all animals making {m} after second {pm}: {mean:.4f} +/- {var:.4f}'.format(m=MOI, pm=MOI_to_prevMOI[MOI], mean=mean_second_attempt_success, var=var_second_attempt_success))
        for animal in third_MOI_prob:
            logging.info('Probability of animal {a} making {m} after third {pm}: {prob:.4f}'.format(a=animal, m=MOI, pm=MOI_to_prevMOI[MOI], prob=third_MOI_prob[animal]))
            print('Probability of animal {a} making {m} after third {pm}: {prob:.4f}'.format(a=animal, m=MOI, pm=MOI_to_prevMOI[MOI], prob=third_MOI_prob[animal]))
        mean_third_attempt_success = np.mean([x for x in third_MOI_prob.values()])
        var_third_attempt_success = np.var([x for x in third_MOI_prob.values()])
        logging.info('Mean probability of all animals making {m} after third {pm}: {mean:.4f} +/- {var:.4f}'.format(m=MOI, pm=MOI_to_prevMOI[MOI], mean=mean_third_attempt_success, var=var_third_attempt_success))
        print('Mean probability of all animals making {m} after third {pm}: {mean:.4f} +/- {var:.4f}'.format(m=MOI, pm=MOI_to_prevMOI[MOI], mean=mean_third_attempt_success, var=var_third_attempt_success))
        # plot summary of catch accuracy
        plot_probMOIseq(allA_probMOI, MOI, MOI_to_prevMOI[MOI], plots_folder, today_dateTime)

# FIN