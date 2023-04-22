import pandas as pd
import os
import numpy as np
from math import sqrt
import argparse
from sklearn.utils import shuffle
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, make_scorer
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate, KFold, train_test_split
from sklearn.pipeline import make_pipeline
from tqdm import tqdm_notebook
import IPython.display as ipd
import pickle
from load_features import *

def parse_args():
    desc = "Tool to create multiclass json labels file for stylegan2-ada-pytorch"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--verbose', action='store_true',
                        help='Print progress to console.')

    parser.add_argument('--input_folder', type=str,
                        default='./dataset/',
                        help='Directory path to the inputs folder. (default: %(default)s)')

    

    args = parser.parse_args()
    return args


def predict(reg_model, dataset, featureNames, labelName, filePrefix, outdir):
    X_data = dataset[featureNames]

 
    columns = ['musicId'] 
    results = pd.DataFrame(columns=columns)
    results['musicId'] = dataset['musicId']
    y_pred = reg_model.predict(X_data)

    # print(y_pred)
    results['prediction'] = y_pred
    results.to_csv(os.path.join(outdir,f'{filePrefix}_regression_results_{labelName}.csv'))
    return results
    


def main():
    args = parse_args()
    song_folder = args.input_folder


    DATASET_DIR = "new_songs"
    song_data = os.path.join(DATASET_DIR, song_folder)

    if not os.path.exists(DATASET_DIR):
        raise Exception("No dataset directory with new songs found. make sure to put new_songs/ folder in root directory")

    outdir = "output"
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    print("Collecting features for song...")

    wavdir = os.path.join(song_data, "wav")
    lld_distdir = os.path.join(song_data, "features_lld_260")
    dynamic_distdir = os.path.join(song_data, "features_out_260")
    all_dynamic_distfile = os.path.join(song_data, "dynamic_features260.csv")

    names = extract_frame_feature(wavdir,lld_distdir)
    process_dynamic_feature(lld_distdir,dynamic_distdir,all_dynamic_distfile, names)

    features = pd.read_csv(os.path.join(song_data, 'dynamic_features260.csv'))

  
    

    print("Loading regressor SVRs for Valence and Arousal...")
    model_arousal = pickle.load(open("model/SVR_Arousal_model_260.sav", 'rb'))
    model_valence = pickle.load(open("model/SVR_Valence_model_260.sav", 'rb'))


    featureNames = features.columns[2:]
    
    
    print('Predicting in Arousal dimension...')
    predict(model_arousal, features, featureNames, 'Arousal', song_folder.replace("/", ""), outdir)

    print('Predicting in Valence dimension...')
    predict(model_valence, features, featureNames, 'Valence', song_folder.replace("/", ""), outdir)


if __name__ == "__main__":
    main()