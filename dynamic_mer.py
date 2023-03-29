import pandas as pd
import os
import numpy as np
from math import sqrt
import argparse
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, make_scorer
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline
from tqdm import tqdm_notebook
import IPython.display as ipd
import pickle

def parse_args():
    desc = "Tool to create multiclass json labels file for stylegan2-ada-pytorch"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--verbose', action='store_true',
                        help='Print progress to console.')

    parser.add_argument('--input_folder', type=str,
                        default='./dataset/',
                        help='Directory path to the inputs folder. (default: %(default)s)')

    parser.add_argument('--output_folder', type=str,
                        default='./results/',
                        help='Directory path to the outputs folder. (default: %(default)s)')

    args = parser.parse_args()
    return args


def rmse(y, y_pred):
    return sqrt(mean_squared_error(y, y_pred))

def cross_val_regression(regressors, features, labels, preprocessfunc):
    columns = list(regressors.keys())
    scores = pd.DataFrame(columns=columns, index=['RMSE'])

    for reg_name, reg in tqdm_notebook(regressors.items(), desc='regressors'):
        scorer = {'rmse': make_scorer(rmse)}
        reg = make_pipeline(*preprocessfunc, reg)
        reg_score = cross_validate(reg, features, labels, scoring=scorer, cv=10, return_train_score=False) 
        scores.loc['RMSE', reg_name] = reg_score['test_rmse'].mean()
#         scores.loc['R', reg_name] = reg_score['test_r'].mean()
    return scores

def format_scores(scores):
    def highlight(s):
        is_min = s == min(s)
#         is_max = s == max(s)
#         is_max_or_min = (is_min | is_max)
        return ['background-color: yellow' if v else '' for v in is_min]
    scores = scores.style.apply(highlight, axis=1, subset=pd.IndexSlice[:, :scores.columns[-2]])
    return scores.format('{:.3f}')

def regression_results(reg, reg_name, trainset, testset, featureNames, labelName, filePrefix, preprocessfunc, outdir):
    X_train = trainset[featureNames]
    y_train = trainset[labelName]
    X_test = testset[featureNames]
    y_test = testset[labelName]

    # print("Test set", X_test)
    # print("Test labels", y_test)

    columns = ['musicId', 'y_test'] 
    results = pd.DataFrame(columns=columns)
    results['musicId'] = testset['musicId']
    results['y_test'] = y_test.values
    
    
    reg = make_pipeline(*preprocessfunc, reg)
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)

    filename = f'{reg_name}_{labelName}_model.sav'
    save_path = os.path.join(outdir, filename)
    pickle.dump(reg, open(save_path, 'wb'))

    # print(y_pred)
    results[reg_name] = y_pred
    results.to_csv(os.path.join(outdir,f'{filePrefix}_regression_results_{labelName}.csv'))
    
def compute_rmse_across_songs(resultsFile):
    results = pd.read_csv(resultsFile,index_col=0).dropna(axis=1, how='any')
    columns = results.columns[2:]
    scores = pd.DataFrame(columns=columns, index=['rmse_across_segments', 'rmse_across_songs'])
    rmse_across_songs = {}
    testsongs_num = len(results['musicId'].unique())

    for reg_name in columns:
        scores.loc['rmse_across_segments', reg_name] = rmse(results['y_test'], results[reg_name])
        rmse_across_songs[reg_name] = 0

    for i, g in results.groupby('musicId'):
        for reg_name in columns:
            rmse_across_songs[reg_name] += rmse(g['y_test'], g[reg_name])

    for reg_name in columns:
        scores.loc['rmse_across_songs', reg_name] = rmse_across_songs[reg_name]/testsongs_num
    
    mean_rmse = scores.mean(axis=1)
    std_rmse = scores.std(axis=1)
    
    scores['Mean'] = mean_rmse
    scores['std'] = std_rmse
    ipd.display(format_scores(scores))

def main():
    args = parse_args()

    DATASET_DIR = args.input_folder
    outdir = args.output_folder
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    print("Collecting features and annotations...")
    features = pd.read_csv(os.path.join(DATASET_DIR, 'dynamic_features.csv'))
    annotations = pd.read_csv(os.path.join(DATASET_DIR, 'annotations.csv'))
    
    dataset = pd.merge(features, annotations, on=['musicId', 'frameTime'])

    print("Setting up regressor SVR...")
    regressor = SVR(kernel='rbf', gamma='scale')

    dataset = dataset.sort_values(by=['musicId', 'frameTime'])
    print("Splitting dataset into train and test and eval")
    songs = dataset['musicId'].unique()
    
    train_split = 744
    test_split = 1000
    eval_split = 58
    train_songs = songs[:train_split]
    test_songs = songs[train_split:train_split + test_split]
    eval_songs = songs[train_split + test_split:]

    print("Testsongs:" , len(test_songs))
    print(list(test_songs))

    print("Train songs", len(train_songs))
    print(list(train_songs))

    print("Evaluation songs", len(eval_songs))
    print(list(eval_songs))

    iftestset = dataset['musicId'].apply(lambda x: x in test_songs)
    iftrainset = dataset['musicId'].apply(lambda x: x in train_songs)
    ifevalset = dataset['musicId'].apply(lambda x: x in eval_songs)

    testset = dataset[iftestset]
    trainset = dataset[iftrainset]
    evalset = dataset[ifevalset]

    featureNames = dataset.columns[2:-2]
    prefunc = [StandardScaler()]


    print('Predicting in Arousal dimension...')
    regression_results(regressor, "SVR", trainset, testset, featureNames, 'Arousal', 'audio', prefunc, outdir)

    print('Predicting in Valence dimension...')
    regression_results(regressor, "SVR", trainset, testset, featureNames, 'Valence', 'audio', prefunc, outdir)

    print('In Arousal dimension...')
    compute_rmse_across_songs(os.path.join(outdir,'audio_regression_results_Arousal.csv'))
    print('In Valence dimension...')
    compute_rmse_across_songs(os.path.join(outdir,'audio_regression_results_Valence.csv'))


if __name__ == "__main__":
    main()