import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
import scienceplots
import sklearn.metrics
from scipy.stats import wilcoxon

plt.style.use(['science', 'notebook'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if 'labels_df' not in locals() or 'features_df' not in locals():
    labels_file_path = Path(r"c:/data/sampleinfo_SCANB_t.csv")
    features_file_path = Path(r"c:\data\SCANB.csv")

    labels_df = pd.read_csv(labels_file_path)
    features_df = pd.read_csv(features_file_path)


def wilcoxon_p_value(x, y):
    w_test = wilcoxon(x, y)
    return w_test.pvalue


class FeatureSelectionTest:

    def __init__(self, features_file_path=None, labels_file_path=None, labels_df=None, features_df=None,
                 select_num_features=None, select_num_instances=None, selected_labels=None, test_size=0.2,
                 lin_regressor_label="Lympho", log_regressor_label="ER"):

        self.features_file_path = features_file_path
        self.labels_file_path = labels_file_path
        self.labels_df = labels_df
        self.features_df = features_df
        self.select_num_features = select_num_features
        self.select_num_instances = select_num_instances
        self.selected_labels = selected_labels
        self.features_and_labels_df = None
        self.shape = None
        self.test_size = test_size
        self.log_regressor_label = log_regressor_label
        self.lin_regressor_label = lin_regressor_label

        self.set_inputs()

    def set_inputs(self):

        self.set_dfs()
        self.set_wrapped_attributes()
        self.set_corr_matrices()
        self.set_feature_lists()

    def set_dfs(self):
        if self.labels_df is None:
            self.labels_df = pd.read_csv(self.labels_file_path)

        if self.features_df is None:
            self.features_df = pd.read_csv(self.features_file_path)

        self.labels_df.drop(["Unnamed: 0"], errors='ignore', inplace=True, axis=1)
        self.features_df.drop(["Unnamed: 0"], errors='ignore', inplace=True, axis=1)

        if self.select_num_instances is not None:
            instances_to_keep = range(self.select_num_instances)
            self.labels_df = self.labels_df.iloc[instances_to_keep]
            self.features_df = self.features_df.iloc[instances_to_keep]

        if self.select_num_features is not None:
            features_to_keep = range(self.select_num_features)
            # feature_names_to_keep = features_df.columns[features_to_keep]
            self.features_df = self.features_df.iloc[:, features_to_keep]

        if self.selected_labels is not None:
            self.labels_df = self.labels_df[self.selected_labels]

        self.features_and_labels_df = self.features_df.join(self.labels_df)
        self.num_features = self.features_df.shape[1]
        self.num_labels = self.features_df.shape[0]
        self.num_instances = self.features_df.shape[0]

    def remove_unnamed_cols():
        self.features_df.drop

    def set_wrapped_attributes(self):
        self.set_shape()
        self.set_columns()

    def set_shape(self):
        out_dict = {"features": self.features_df.shape, "labels": self.labels_df.shape}
        self.shape = pd.Series(out_dict)

    def set_columns(self):
        out_dict = {"features": self.features_df.columns, "labels": self.labels_df.columns}
        self.columns = pd.Series(out_dict)

    def head(self, n=5):
        return self.features_and_labels_df.head(n)

    def tail(self, n=5):
        return self.features_and_labels_df.tail(n)

    def run_linear_regression(self, selected_features=None):
        self.linear_regressor = LinearRegression()
        y = self.labels_df[self.lin_regressor_label]
        if self.features_already_selected_lin is None:
            x = self.features_df
        else:
            x = self.features_df[self.features_already_selected_lin]
        if selected_features is not None:
            x = x[selected_features]

        self.x_train_lin, self.x_test_lin, self.y_train_lin, self.y_test_lin = train_test_split(x, y,
                                                                                                test_size=self.test_size,
                                                                                                random_state=47)
        self.linear_regressor.fit(self.x_train_lin, self.y_train_lin)
        self.y_pred_lin = self.linear_regressor.predict(self.x_test_lin)
        self.lin_score = self.linear_regressor.score(self.x_test_lin, self.y_test_lin)

    def run_log_regression(self, selected_features=None):
        self.logistic_regressor = LogisticRegression()
        y = self.labels_df[self.log_regressor_label]
        if self.features_already_selected_log is None:
            x = self.features_df
        else:
            x = self.features_df[self.features_already_selected_log]

        if selected_features is not None:
            x = x[selected_features]

        self.x_train_log, self.x_test_log, self.y_train_log, self.y_test_log = train_test_split(x, y,
                                                                                                test_size=self.test_size,
                                                                                                random_state=47)
        self.logistic_regressor.fit(self.x_train_log, self.y_train_log)
        self.y_pred_log = self.logistic_regressor.predict(self.x_test_log)
        self.log_score = self.logistic_regressor.score(self.x_test_log, self.y_test_log)
        self.log_confusion_matrix = sklearn.metrics.confusion_matrix(y_true=self.y_test_log, y_pred=self.y_pred_log,
                                                                     normalize='all')

    def show_linear_regressor(self, figsize=(6, 6), ax=None):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        sns.scatterplot(x=self.y_pred_lin, y=self.y_test_lin, ax=ax)
        ax.set(
            title=f"linear regressor using dataset of total \n {self.x_train_lin.shape} train and {self.x_test_lin.shape} test. \nScore: {self.lin_score:.2f}",
            xlabel="y_test", ylabel="y_pred")
        ax.grid("minor")

    def show_log_regressor(self, figsize=(6, 6), ax=None, cmap="viridis"):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        sklearn.metrics.ConfusionMatrixDisplay.from_predictions(self.y_test_log, self.y_pred_log, ax=ax, cmap=cmap)
        ax.set(
            title=f"log regressor using dataset of total \n {self.x_train_log.shape} train and {self.x_test_log.shape} test. \nScore: {self.log_score:.2f}")

    def show(self, fig=None, figsize=(12, 6)):
        if fig is None:
            fig, axs = plt.subplots(1, 2, figsize=figsize)
        ax = axs.flatten()[0]
        ax.grid("minor")
        self.show_linear_regressor(ax=ax, figsize=(figsize[0] // 2, figsize[1]))
        ax = axs.flatten()[1]
        self.show_log_regressor(ax=ax, figsize=(figsize[0] // 2, figsize[1]))

    def show_classification_report(self):
        print(sklearn.metrics.classification_report(self.y_test_log, self.y_pred_log))

    def show_regression_report(self):
        print(sklearn.metrics.reg)

    def set_corr_matrices(self):
        self.correlation_matrix_lin = self.features_and_labels_df.drop(columns=self.log_regressor_label).corr().drop(
            self.lin_regressor_label)
        self.correlation_matrix_log = self.features_and_labels_df.drop(columns=self.lin_regressor_label).corr(
            method=wilcoxon_p_value).drop(self.log_regressor_label)
        self.correlation_ranking_lin = self.correlation_matrix_lin[self.lin_regressor_label].sort_values(
            ascending=False)
        self.correlation_ranking_log = self.correlation_matrix_log[self.log_regressor_label].sort_values(
            ascending=False)

    def set_feature_lists(self):
        self.features_already_selected_lin = []
        self.features_already_selected_log = []
        self.features_rejected_lin = []
        self.features_rejected_log = []
        self.features_not_yet_selected_lin = self.features_df.columns
        self.features_not_yet_selected_log = self.features_df.columns
        self.number_of_features_tested_lin = 0
        self.number_of_features_tested_log = 0

    def run_iterative_feature_selection(self, mode, num_iterations=None, correlation_threshold=0.1,
                                        min_num_of_features=None, debug_print=False, alert_selection=False):

        if num_iterations is None:
            num_iterations = self.num_features

        if mode == 'lin':
            self.features_to_test_list = self.correlation_matrix_lin[self.lin_regressor_label].abs().sort_values(
                ascending=False).index
        if mode == 'log':
            self.features_to_test_list = self.correlation_matrix_log[self.log_regressor_label].abs().sort_values(
                ascending=False).index
        if debug_print:
            print(f"features_to_test_list {self.features_to_test_list}")

        for iter in range(num_iterations):

            if debug_print:
                print(f"features to test list - {self.features_to_test_list}")
            if mode == 'lin':
                self.candidate_feature = str(self.features_to_test_list[self.number_of_features_tested_lin])
                self.features_already_selected = self.features_already_selected_lin
            if mode == 'log':
                self.candidate_feature = str(self.features_to_test_list[self.number_of_features_tested_log])
                self.features_already_selected = self.features_already_selected_log

            if debug_print:
                print(f"self.candidate_feature: {self.candidate_feature}")

            if len(self.features_already_selected) == 0:
                # self.features_already_selected.append(self.candidate_feature)

                if mode == 'lin':
                    self.features_already_selected_lin.append(self.candidate_feature)
                if mode == 'log':
                    self.features_already_selected_log.append(self.candidate_feature)
            ## Iterate
            else:
                if mode == 'lin':
                    self.selected_feature_data = self.features_df[self.features_already_selected_lin]
                if mode == 'log':
                    self.selected_feature_data = self.features_df[self.features_already_selected_log]

                self.candidate_feature_data = self.features_df[[self.candidate_feature]]

                if debug_print:
                    print(f"self.features_already_selected_lin: {self.features_already_selected_lin}")
                    print(f"candidate_feature_data: {self.candidate_feature_data}")
                    print(f"selected_feature_data: {self.selected_feature_data}")

                # self.df_candidate_vs_existing = self.candidate_feature_data.join(self.selected_feature_data)
                # # self.df_candidate_vs_existing = pd.concat([self.candidate_feature_data,self.selected_feature_data])
                self.candidate_and_selected_features = list(self.features_already_selected) + [self.candidate_feature]
                self.df_candidate_vs_existing = self.features_df[self.candidate_and_selected_features]

                if mode == 'lin':
                    self.test_corr_matrix = self.df_candidate_vs_existing.corr()
                if mode == 'log':
                    self.test_corr_matrix = self.df_candidate_vs_existing.corr(method=wilcoxon_p_value)

                self.correlation_with_prevs_features = self.test_corr_matrix[self.candidate_feature].drop(
                    self.candidate_feature)

                if debug_print:
                    print(f"correaltion with prevs: {self.correlation_with_prevs_features}")

                if self.correlation_with_prevs_features.abs().max() <= correlation_threshold:
                    if alert_selection:
                        print(f"found a new feature to use!")

            # advance index
            if mode == 'lin':
                self.features_already_selected_lin.append(self.candidate_feature)
                self.number_of_features_tested_lin = self.number_of_features_tested_lin + 1
            if mode == 'log':
                self.features_already_selected_log.append(self.candidate_feature)
                self.number_of_features_tested_log = self.number_of_features_tested_log + 1