import argparse
import glob
import math
import ntpath
import os
import shutil
import pyedflib
import numpy as np
import pandas as pd
from scipy.signal import resample_poly ##Resampling

from sleepstage import stage_dict #Import sleep stage dictionary from sleepstage.py
from logger import get_logger #Import get_logger function from logger.py


# Have to manually define based on the dataset
##Changed keys according to Earable doc
ann2label = { ##Combined N1 & N2, changed all values accordingly
    "Sleep stage W": 0,
    "Sleep stage N1": 1, "Sleep stage N2": 1, #
    "Sleep stage N3": 2, "Sleep stage N4": 2, # Follow AASM Manual
    "Sleep stage R": 3, "Sleep stage REM": 3,
    "Sleep stage ?": 5,
    "Movement time": 4, "EEG arousal": 4,
    
    "Sleep stage Light Sleep": 1,
    "Sleep stage Deep Sleep": 2
}


def main():
    parser = argparse.ArgumentParser() #Create ArgumentParser object
    parser.add_argument("--data_dir", type=str, default="./data/sleepedf",
                        help="File path to the Sleep-EDF dataset.") ##Removed /sleep-cassette
    parser.add_argument("--output_dir", type=str, default="./data/sleepedf/eeg_fpz_cz",
                        help="Directory where to save outputs.") ##Removed /sleep-cassette
    parser.add_argument("--select_ch", type=str, default="ch5_OTE_R-FpZ",
                        help="Name of the channel in the dataset.") ## Changed channel name
    parser.add_argument("--log_file", type=str, default="info_ch_extract.log",
                        help="Log file.")
    parser.add_argument("--num_files", type=int, default=40,
                        help="Number of files to train with..") ##Added (SID fix)
    #^Add arguments; defines how a single command-line argument should be parsed
    args = parser.parse_args() #Parse arguments; convert argument strings to objects and assign them as attributes of the namespace
    
    # Output dir
    if not os.path.exists(args.output_dir): #Check if output directory exists, if not:
        os.makedirs(args.output_dir) #Create output directory
    else:
        shutil.rmtree(args.output_dir) #Delete entire directory tree
        os.makedirs(args.output_dir) #Create output directory

    args.log_file = os.path.join(args.output_dir, args.log_file) #Generate file path /data/sleepedf/sleep-cassette/eeg_fpz_cz/info_ch_extract.log

    # Create logger
    logger = get_logger(args.log_file, level="info")

    # Select channel
    select_ch = args.select_ch

    # Read raw and annotation from EDF files
    psg_fnames = glob.glob(os.path.join(args.data_dir, "*PSG.edf")) #Returns list of PSG.edf files
    ann_fnames = glob.glob(os.path.join(args.data_dir, "*Hypnogram.edf")) #Returns list Hypnogram.edf files
    psg_fnames.sort() #Sort the list
    ann_fnames.sort() #Sort the list
    psg_fnames = np.asarray(psg_fnames) #Conver to array
    ann_fnames = np.asarray(ann_fnames) #Convert to array
    
    ## SID fix: Set the number of files you want to train with as a command line argument
    n_files_to_train = args.num_files
    id_n = 0 ##SID fix
    
    for i in range(len(psg_fnames)): #Iterates through array of PSG files

        logger.info("Loading ...")  #Log message
        logger.info("Signal file: {}".format(psg_fnames[i]))
        logger.info("Annotation file: {}".format(ann_fnames[i]))

        psg_f = pyedflib.EdfReader(psg_fnames[i]) #Read EDF files
        ann_f = pyedflib.EdfReader(ann_fnames[i]) #Read EDF files
 
        assert psg_f.getStartdatetime() == ann_f.getStartdatetime() #Verify start date/time for PSG.edf and Hypnogram.edf files match
        start_datetime = psg_f.getStartdatetime()
        logger.info("Start datetime: {}".format(str(start_datetime)))

        file_duration = psg_f.getFileDuration()
        logger.info("File duration: {} sec".format(file_duration)) ##Debugging
        epoch_duration = 30.0 ##Set epoch duration manually, changed from epoch_duration = psg_f.datarecord_duration (datarecord_duration is 1 sec, is incorrect)
        
        logger.info("Epoch duration: {} sec".format(epoch_duration)) 

        # Extract signal from the selected channel
        ch_names = psg_f.getSignalLabels() #Returns all label names
        ch_samples = psg_f.getNSamples() 
        select_ch_idx = -1
        for s in range(psg_f.signals_in_file): #Iterate through signals in PSG file
            if ch_names[s] == select_ch: #Find select channel
                select_ch_idx = s #Index select channel
                break
        if select_ch_idx == -1:
            raise Exception("Channel not found.") #Select channel not found
        sampling_rate = psg_f.getSampleFrequency(select_ch_idx) #Returns frequencies of all signals
        
        logger.info("Select channel: {}".format(select_ch))
        logger.info("Select channel samples: {}".format(ch_samples[select_ch_idx]))
        logger.info("Sample rate: {}".format(sampling_rate))
        
        # Generate labels from onset and duration annotation
        labels = []
        total_duration = 0
        ann_onsets, ann_durations, ann_stages = ann_f.readAnnotations() #Get annotations from EDF file
        
        signals = psg_f.readSignal(select_ch_idx) #readSignal() returns physical data of select channel
        logger.info("Starting len(signals): {}".format(len(signals))) ##Debugging
        
        for a in range(len(ann_stages)):
            if "EEG" in ann_stages[a]: ##Ignore EEG arousal stages,
                onset_sec = int(ann_onsets[a])
                duration_sec = int(ann_durations[a])
                #arousal_dur = arousal_dur+duration_sec
                logger.info("EEG AROUSAL: Include onset:{}, duration:{}".format(onset_sec, duration_sec)) ##Debugging
                continue
        
            onset_sec = int(ann_onsets[a])
            duration_sec = int(ann_durations[a])
            ann_str = "".join(ann_stages[a])
            
            # Sanity check
            assert onset_sec == total_duration 
            
            # Get label value
            label = ann2label[ann_str]

            # Compute # of epoch for this stage
            if duration_sec % epoch_duration != 0: #duration_sec / epoch_duration = 30.0 sec should leave remainder of 0
                logger.info(f"Something wrong: {duration_sec} {epoch_duration}")
                raise Exception(f"Something wrong: {duration_sec} {epoch_duration}")
            duration_epoch = int(duration_sec / epoch_duration)
            
            total_duration += duration_sec
            
            ##Account for missing samples in signals array:
            if file_duration < total_duration: ##Missing samples in signals array
                print()
                logger.info("file_duration LESS than calculated total_duration") ##Debugging
                logger.info("Total duration: {} sec".format(total_duration)) ##Debugging
                if onset_sec < file_duration: ##Reduce signals array/labels array; discard signals/labels not within total_duration
                    total_duration = onset_sec ##Fix total duration
                    logger.info("Reduced total duration: {} sec".format(total_duration)) ##Debugging
                    file_duration = total_duration ##Fix file duration
                    signals_cutoff = total_duration * 125 ##Determine signals cutoff
                    logger.info("signals cutoff calculated: {}".format(signals_cutoff))
                    remove_rows = len(signals) - signals_cutoff
                    signals = signals[:-remove_rows]
                    logger.info("Removed {} rows from signals array".format(remove_rows))
                    logger.info("len(signals) after removal: {}".format(len(signals))) ##Debugging
                    print()
                    break ##Break for loop - can't have anymore labels if there are no signals
                else:
                    raise Exception(f"Something wrong: {file_duration} {onset_sec}") ##Should not hit ever
            
            # Generate sleep stage labels
            label_epoch = np.ones(duration_epoch, dtype=np.int) * label #Array of labels is generated
            labels.append(label_epoch) #Append to labels array

            logger.info("Include onset:{}, duration:{}, label:{} ({})".format(
                onset_sec, duration_sec, label, ann_str
            ))
        
        ##Account for extra samples in signals array:
        if file_duration > total_duration: ##Extra samples in signals array; cut off signals array at total_duration*125
            print()
            logger.info("file_duration GREATER than calculated total_duration") ##Debugging
            logger.info("Total duration: {} sec".format(total_duration)) ##Debugging
            file_duration = total_duration ##Fix file duration
            signals_cutoff = total_duration * 125 ##Determine signals cutoff
            logger.info("signals cutoff calculated: {}".format(signals_cutoff))
            remove_rows = len(signals) - signals_cutoff
            signals = signals[:-remove_rows]
            logger.info("Removed {} rows from signals array".format(remove_rows))
            logger.info("len(signals) after removal: {}".format(len(signals))) ##Debugging
            print()
            
        logger.info("len(signals) (BEFORE RESHAPE): {}".format(len(signals))) ##Debugging
        
        ##Resampling (occurs AFTER fixing signals array with missing/excess samples to ensure signals has an expected amount of samples before resampling):
        signals = resample_poly(signals, 4, 5)
        sampling_rate = 100.00 ##Update sampling rate manually
        logger.info("Sample rate updated: {}".format(sampling_rate))
        logger.info("len(signals) (AFTER RESAMPLE, BEFORE RESAHPE): {}".format(len(signals))) ##Debugging
    
        n_epoch_samples = int(epoch_duration * sampling_rate) #Calculates number of epoch samples
        signals = signals.reshape(-1, n_epoch_samples) #readSignal() returns physical data of select channel, reshape sets new array shape. -1 reduces the row dimension, n_epoch_samples is the column dimension. So this is an array of signals from the select channel with the fewest number of rows possible and n_epoch_samples columns. ##Changed
    
        # Sanity check
        n_epochs = total_duration / epoch_duration #n_epochs = data records in PSG file ##Equal to file duration rn ##Changed from n_epochs = psg_f.datarecords_in_file
        
        logger.info("len(signals) (AFTER RESHAPE): {}".format(len(signals))) ##Debugging
    
        assert len(signals) == n_epochs, f"signal: {signals.shape} != {n_epochs}" #Check that the number of elements in signals array is equal to the number of epochs..which should be the number of data records in the PSG file ##Wrong
        
        labels = np.hstack(labels) #Stack arrays column-wise to create single array

        # Remove annotations that are longer than the recorded signals
        labels = labels[:len(signals)]

        # Get epochs and their corresponding labels
        x = signals.astype(np.float32)
        y = labels.astype(np.int32)

        # Select only sleep periods
        w_edge_mins = 30
        nw_idx = np.where(y != stage_dict["W"])[0]
        start_idx = nw_idx[0] - (w_edge_mins * 2)
        end_idx = nw_idx[-1] + (w_edge_mins * 2)
        if start_idx < 0: start_idx = 0
        if end_idx >= len(y): end_idx = len(y) - 1
        select_idx = np.arange(start_idx, end_idx+1)
        logger.info("Data before selection: {}, {}".format(x.shape, y.shape))
        x = x[select_idx]
        y = y[select_idx]
        logger.info("Data after selection: {}, {}".format(x.shape, y.shape))

        # Remove movement and unknown
        move_idx = np.where(y == stage_dict["MOVE"])[0]
        unk_idx = np.where(y == stage_dict["UNK"])[0]
        if len(move_idx) > 0 or len(unk_idx) > 0:
            remove_idx = np.union1d(move_idx, unk_idx)
            logger.info("Remove irrelavant stages")
            logger.info("  Movement: ({}) {}".format(len(move_idx), move_idx))
            logger.info("  Unknown: ({}) {}".format(len(unk_idx), unk_idx))
            logger.info("  Remove: ({}) {}".format(len(remove_idx), remove_idx))
            logger.info("  Data before removal: {}, {}".format(x.shape, y.shape))
            select_idx = np.setdiff1d(np.arange(len(x)), remove_idx)
            x = x[select_idx]
            y = y[select_idx]
            #logger.info("  Data after removal: {}, {}".format(x.shape, y.shape))

        # Save
        ##SID fix:
        endfn = ".npz"
        if (i < n_files_to_train):
            endfn = "id{ind}.npz".format(ind = id_n) ##SID fix
            id_n += 1
            if (id_n == 20):
                id_n = 0
        
        filename = ntpath.basename(psg_fnames[i]).replace("-PSG.edf", endfn) #Conversion to .npz ##Changed for SIDs
        save_dict = {
            "x": x, 
            "y": y, 
            "fs": sampling_rate,
            "ch_label": select_ch,
            "start_datetime": start_datetime,
            "file_duration": file_duration,
            "epoch_duration": epoch_duration,
            "n_all_epochs": n_epochs,
            "n_epochs": len(x),
        }
        np.savez(os.path.join(args.output_dir, filename), **save_dict)

        logger.info("\n=======================================\n")


if __name__ == "__main__":
    main()