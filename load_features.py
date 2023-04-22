import os
import time

import numpy as np
import pandas as pd


import audiofile
import opensmile

#! usr/bin/env python3
# -*- coding: utf-8 -*-

'''
This features.py is used to extract audio features based on openSIMLE.
Require: openSMILE-2.2rc1
OpenSMILE only support audios in WAV format, 
so before using this script you could
transform MP3s into WAVs by transformat.sh.
'''

__author__ = 'huizhang'

import csv
import os
import shutil
from math import floor
import numpy as np


def extract_frame_feature(wavdir, distdir):
    '''Extract lld features in frame size: 60ms, step size: 10ms.

    Args:
        wavdir: Path to audios in WAV format.
        distdir: Path of distdir.
        opensmiledir: Path to opensimle project root.

    Returns:
        Distfiles containing lld features for each WAV.
    '''

    if os.path.exists(distdir):
        shutil.rmtree(distdir)
    os.mkdir(distdir)

    wav = [f for f in os.listdir(wavdir) if f[-4:] == ".wav"]
    for w in wav:
        wavpath = os.path.join(wavdir,w)
        distfile = os.path.join(distdir,w[:-4]+".csv")
        
        signal, sampling_rate = audiofile.read(
            wavpath,
            always_2d=True,
        )
        lld = opensmile.Smile(
            feature_set=opensmile.FeatureSet.ComParE_2016,
            feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
        )
        lld_d = opensmile.Smile(
            feature_set=opensmile.FeatureSet.ComParE_2016,
            feature_level=opensmile.FeatureLevel.LowLevelDescriptors_Deltas,
        )

        lld_data = lld.process_signal(
            signal,
            sampling_rate
        )

        lld_d_data = lld_d.process_signal(
            signal,
            sampling_rate
        )

        llds = pd.merge(lld_data, lld_d_data, on=["start", "end"])
        llds.to_csv(distfile,index=True)

    return lld.feature_names + lld_d.feature_names


def process_dynamic_feature(llddir, distdir, all_songs_distfile, featureNames):
    '''Obtain dynamic features in window size: 1s, shift size: 0.5s.

    Args:
        llddir: Path to lld feature files.
        distdir: Path of distdir.
        all_songs_distfile: Path of distfile.
        delimiter: csv delimiter in lld feature files, default=';'.

    Returns:
        Distfiles containing 260-dimension dynamic features all WAVs.
    '''

    if os.path.exists(distdir):
        shutil.rmtree(distdir)
    os.mkdir(distdir)

    # names of features
    headers = ['musicId', 'frameTime']
    headers += [str(f+'_mean') for f in featureNames]
    headers += [str(f +"_std") for f in featureNames]

    window = 1
    overlap = 0.5

    llds = [f for f in os.listdir(llddir) if f[-4:] == ".csv"]
    
    all_dynamic_features = []
    all_musicId = []

    for lld in llds:
        musicId = []
        lldpath = os.path.join(llddir,lld)
        single_song_distfile = os.path.join(distdir,lld)

        dynamic_features = _compute_feature_with_window_and_overlap(lldpath, window, overlap)
       
        for i in range(len(dynamic_features)):
            musicId.append(lld[:-4])
        _write_features_to_csv(headers, musicId, dynamic_features, single_song_distfile)

        all_musicId += musicId
        all_dynamic_features += dynamic_features

    _write_features_to_csv(headers, all_musicId, all_dynamic_features, all_songs_distfile)

def _compute_feature_with_window_and_overlap(lldpath, window, overlap):
    '''Compute the mean and std for frame-wise features in window size: 1s, shift size: 0.5s.'''

    fs = 0.01
    num_in_new_frame = floor(overlap/fs)
    num_in_window = floor(window/fs)

    # load the features from disk
    all_frame = []
    with open(lldpath) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            
            frame_feature = []
            for i in range(len(row)-2): #旧的frametime不用记录
                frame_feature.append(float(row[i+2]))
                
            all_frame.append(frame_feature)
    
    # compute new number of frames
    new_num_of_frame = floor(len(all_frame)/num_in_new_frame)
    all_new_frame = []

    # compute mean and std in each window as the feature corresponding to the frame. 
    for i in range(new_num_of_frame):
        start_index = num_in_new_frame * i
        new_frame_array = np.array(all_frame[start_index:start_index+num_in_window])

        mean_llds = np.mean(new_frame_array,axis=0)
        std_llds = np.std(new_frame_array,axis=0)
        new_frametime = i * overlap

        new_frame = [new_frametime] + mean_llds.tolist() + std_llds.tolist()
        all_new_frame.append(new_frame)

    

    return all_new_frame

def _write_features_to_csv(headers, musicIds, contents, distfile):
    '''Write all the features into one file, and add the last column as the annotation value'''
    
    with open(distfile,"w") as newfile:
        writer = csv.writer(newfile)
        writer.writerow(headers)

        for i in range(len(contents)):
            writer.writerow([musicIds[i]] + contents[i])


if __name__ == "__main__":
    wavdir ="dataset_deam/wav/"
    opensmiledir = "opensmile-3.0/"

    static_distfile = "static_features.arff"
    lld_distdir = "features_lld_260"
    dynamic_distdir = "features_out_260"
    all_dynamic_distfile = "dynamic_features260dim.csv"

    delimiter = ";"

    #extract_all_wav_feature(wavdir,static_distfile,opensmiledir)
    #print("EXTRACTING FRAME FEATURES")
    #names = extract_frame_feature(wavdir,lld_distdir)
    #print(len(names), "FEATURES EXTRACTED")
    lld = opensmile.Smile(
        feature_set=opensmile.FeatureSet.ComParE_2016,
        feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
        )
    lld_d = opensmile.Smile(
        feature_set=opensmile.FeatureSet.ComParE_2016,
        feature_level=opensmile.FeatureLevel.LowLevelDescriptors_Deltas,
    )
    names = lld.feature_names + lld_d.feature_names

    print("PROCESSING DYNAMIC FETAURES")
    process_dynamic_feature(lld_distdir,dynamic_distdir,all_dynamic_distfile, names)
    print("FEATURES EXTRACTED AND PROCESSED")

    


