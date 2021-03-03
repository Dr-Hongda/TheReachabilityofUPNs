import time

import sklearn
import numpy as np
import pandas as pd

import sys
sys.path.append("..")
from baggingPU import BaggingClassifierPU

import xgboost as xgb
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    columnwidth = max([len(x) for x in labels]) + 4
    empty_cell = " " * columnwidth
    print("    " + empty_cell, end=' ')
    for label in labels:
        print("%{0}s".format(columnwidth) % 'pred_' + label, end=" ")
    print()

    for i, label1 in enumerate(labels):
        print("    %{0}s".format(columnwidth) % 'true_' + label1, end=" ")
        for j in range(len(labels)):
            cell = "%{0}.1f".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            if cell:
                print(cell, end=" ")
        print()


def random_undersampling(tmp_df, TARGET_LABEL):
    df_majority = tmp_df[tmp_df[TARGET_LABEL] == 0]
    df_minority = tmp_df[tmp_df[TARGET_LABEL] == 1]

    df_majority_downsampled = resample(df_majority,
                                       replace=False,
                                       n_samples=len(df_minority),
                                       random_state=None)

    df_downsampled = pd.concat([df_majority_downsampled, df_minority])

    print("Undersampling complete!")
    print(df_downsampled[TARGET_LABEL].value_counts())
    return df_downsampled


if __name__ == '__main__':

    df_raw = pd.read_csv('../data/1place-independence.csv')

    df_raw['label'] = df_raw['label'].astype("int")
    print(df_raw.label.value_counts())
    print('Has null values', df_raw.isnull().values.any())

    df_downsampled = random_undersampling(df_raw, 'label')
    df_downsampled = df_downsampled.sample(frac=1)
    df_downsampled = df_downsampled.reset_index()
    df_downsampled = df_downsampled.drop(columns=['index'])

    x_data = df_raw.iloc[:, :-1]
    y_data = df_raw.iloc[:, -1]

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=7)

    x_train_input = pd.concat([x_train, y_train], axis=1)
    x_test_input = pd.concat([x_test, y_test], axis=1)

    print(len(x_train_input))
    print(len(x_test_input))

    print(x_train_input.label.value_counts())
    print('Has null values', x_train_input.isnull().values.any())

    print(x_test_input.label.value_counts())
    print('Has null values', x_test_input.isnull().values.any())

    df = x_train_input.copy()

    NON_LBL = [c for c in df.columns if c != 'label']
    X = df[NON_LBL]
    y = df['label']

    y_orig = y.copy()

    f1_orig = []
    f1_train = []
    f1_test = []

    for i in [77, 1077, 2077, 3077, 4077, 5077, 6077, 7077, 7177, 7277, 7377, 7477, 7577, 7677, 7777, 7877, 7977]:

        hidden_size = i

        y1 = y.copy(deep = True)


        y1.loc[
            np.random.choice(
                y1[y1 == 1].index,
                replace=False,
                size=hidden_size
            )
        ] = 0
        pd.Series(y1).value_counts()
        print('- %d samples and %d features' % (X.shape))
        print('- %d positive out of %d total before hiding labels' % (
        sum(df_downsampled.label), len(df_downsampled.label)))
        print('- %d positive out of %d total after hiding labels' % (sum(y1), len(y1)))

        print('Training XGboost model ...')



        model = xgb.XGBClassifier()

        model.fit(X, y1)

        print('Done')

        print('---- {} ----'.format('XGboost model'))
        print(print_cm(sklearn.metrics.confusion_matrix(y_orig, model.predict(X)), labels=['negative', 'positive']))
        print('')
        print('Precision: ', precision_score(y_orig, model.predict(X)))
        print('Recall: ', recall_score(y_orig, model.predict(X)))
        print('Accuracy: ', accuracy_score(y_orig, model.predict(X)))
        print('f1_score: ', f1_score(y_orig, model.predict(X)))

        f1_orig.append(f1_score(y_orig, model.predict(X)))


        print('Training bagging classifier...')
        pu_start = time.perf_counter()
        model = BaggingClassifierPU(xgb.XGBClassifier(),
                                    n_estimators=50,
                                    n_jobs=-1,
                                    max_samples=sum(y1)
                                    )
        model.fit(X, y1)
        pu_end = time.perf_counter()
        print('Done!')
        print('Time:', pu_end - pu_start)

        # train data
        print('---- {} ----'.format('PU Bagging'))
        print(print_cm(sklearn.metrics.confusion_matrix(y_orig, model.predict(X)), labels=['negative', 'positive']))
        print('')
        print('Precision: ', precision_score(y_orig, model.predict(X)))
        print('Recall: ', recall_score(y_orig, model.predict(X)))
        print('Accuracy: ', accuracy_score(y_orig, model.predict(X)))
        print('f1_score: ', f1_score(y_orig, model.predict(X)))

        f1_train.append(f1_score(y_orig, model.predict(X)))

        print('---- {} ----'.format('PU Bagging'))
        print(print_cm(sklearn.metrics.confusion_matrix(y_test, model.predict(x_test)), labels=['negative', 'positive']))
        print('')
        print('Precision: ', precision_score(y_test, model.predict(x_test)))
        print('Recall: ', recall_score(y_test, model.predict(x_test)))
        print('Accuracy: ', accuracy_score(y_test, model.predict(x_test)))
        print('f1_score: ', f1_score(y_test, model.predict(x_test)))

        f1_test.append(f1_score(y_test, model.predict(x_test)))

    x_coordinate = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000]
    name1 = ['F1_orig', 'F1_train', 'F1_test']
    test = pd.DataFrame(columns = x_coordinate, index = name1,
                        data = [list(reversed(f1_orig))[:],
                                list(reversed(f1_train))[:],
                                list(reversed(f1_test))[:]])
    test.to_csv('../result/1Place_diffHiddenNo_XGresult.csv')

    y_train_coordinate = list(reversed(f1_train))
    plt.plot(x_coordinate, y_train_coordinate)
    plt.show()

    y_test_coordinate = list(reversed(f1_test))
    plt.plot(x_coordinate, y_test_coordinate)
    plt.show()