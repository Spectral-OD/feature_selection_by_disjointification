import datetime
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from utils import utils
import scienceplots

plt.style.use(['science', 'notebook'])


def load_gene_expression_data(data_folder=r"c:\data", labels_file_name=r"sampleinfo_SCANB_t.csv",
                              features_file_name=r"SCANB.csv"):
    labels_file_path = Path(data_folder, labels_file_name)
    features_file_path = Path(data_folder, features_file_name)

    labels_df = pd.read_csv(labels_file_path)
    features_df = pd.read_csv(features_file_path)
    return labels_df, features_df


def create_test(*args, file=None, **kwargs):
    if file is None:
        return from_file(file)
    else:
        return Disjointification(*args, **kwargs)


def from_file(file):
    with open(file, 'rb') as f:
        loaded = pickle.load(f)
        loaded.set_inputs()
    return loaded


class Disjointification:
    def __init__(self, features_file_path=None, labels_file_path=None, labels_df=None, features_df=None,
                 select_num_features=None, select_num_instances=None, test_size=0.2,
                 lin_regressor_label="Lympho", log_regressor_label="ER", model_save_folder=None, do_autosave=True,
                 max_num_iterations=None):

        self.drop_label_temp = None
        self.focus_label_temp = None
        self.selected_labels_temp = None
        self.drop_labels_temp = None
        self.corr_matrix_temp = None
        self.max_num_iterations = max_num_iterations
        self.last_save_point_file = None
        self.do_autosave = do_autosave
        self.model_save_folder = model_save_folder
        self.correlation_with_previous_features_temp = None
        self.test_corr_matrix_temp = None
        self.df_candidate_vs_existing_temp = None
        self.candidate_feature_data_temp = None
        self.selected_feature_data_temp = None
        self.features_selected_in_disjointification_temp = None
        self.candidate_feature = None
        self.features_to_test_series_temp = None
        self.number_of_features_tested_log = None
        self.number_of_features_tested_lin = None
        self.features_not_yet_selected_log = None
        self.features_not_yet_selected_lin = None
        self.features_rejected_log = None
        self.features_rejected_lin = None
        self.features_selected_in_disjointification_log = None
        self.features_selected_in_disjointification_lin = None
        self.correlation_ranking_log = None
        self.correlation_ranking_lin = None
        self.correlation_matrix_log = None
        self.correlation_matrix_lin = None
        self.y_test_log = None
        self.y_train_log = None
        self.x_test_log = None
        self.x_train_log = None
        self.logistic_regressor = None
        self.lin_score = None
        self.y_pred_lin = None
        self.y_test_lin = None
        self.y_train_lin = None
        self.x_test_lin = None
        self.x_train_lin = None
        self.linear_regressor = None
        self.columns = None
        self.num_instances = None
        self.log_score = None
        self.log_confusion_matrix = None
        self.y_pred_log = None
        self.num_labels = None
        self.num_features = None
        self.candidate_and_selected_features_temp = None
        self.features_file_path = features_file_path
        self.labels_file_path = labels_file_path
        self.labels_df = labels_df
        self.features_df = features_df
        self.select_num_features = select_num_features
        self.select_num_instances = select_num_instances
        self.features_and_labels_df = None
        self.shape = None
        self.test_size = test_size
        self.log_regressor_label = log_regressor_label
        self.lin_regressor_label = lin_regressor_label
        self.set_inputs()

    def set_inputs(self):
        self.set_model_save_folder()
        self.set_dfs()
        self.set_wrapped_attributes()
        self.set_feature_lists()

    def run(self, persistent=True):
        self.run_correlation_matrix_calculation(persistent=persistent)
        self.run_disjointification()
        self.run_regressions()

    def run_correlation_matrix_calculation(self, mode=None, persistent=True):
        self.set_feature_corr_matrix(mode=mode, persistent=persistent)

    def set_model_save_folder(self, root="model", fmt="%m_%d_%Y__%H_%M_%S"):
        if self.model_save_folder is None:
            run_dt = utils.get_dt_in_fmt(fmt=fmt)
            if root is not None:
                folder_path = Path(root, run_dt)
            else:
                folder_path = Path(run_dt)
            self.model_save_folder = folder_path
            self.model_save_folder.mkdir(parents=True, exist_ok=True)

    def set_dfs(self):
        if self.labels_df is None:
            self.labels_df = pd.read_csv(Path(self.labels_file_path))

        if self.features_df is None:
            self.features_df = pd.read_csv(Path(self.features_file_path))

        self.labels_df.drop(["Unnamed: 0"], errors='ignore', inplace=True, axis=1)
        self.features_df.drop(["Unnamed: 0"], errors='ignore', inplace=True, axis=1)

        if self.select_num_instances is not None:
            # instances_to_keep = np.arange(self.select_num_instances)
            # self.labels_df = self.labels_df[instances_to_keep]
            # self.features_df = self.features_df[instances_to_keep]
            self.labels_df = self.labels_df[0:self.select_num_instances]
            self.features_df = self.features_df[0:self.select_num_instances]
        if self.select_num_features is not None:
            features_to_keep = range(self.select_num_features)
            # feature_names_to_keep = features_df.columns[features_to_keep]
            self.features_df = self.features_df.iloc[:, features_to_keep]

        self.selected_labels_temp = [self.lin_regressor_label, self.log_regressor_label]
        self.labels_df = self.labels_df[self.selected_labels_temp]

        self.features_and_labels_df = self.features_df.join(self.labels_df)
        self.num_features = self.features_df.shape[1]
        self.num_labels = self.features_df.shape[0]
        self.num_instances = self.features_df.shape[0]
        self.autosave()

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

    def run_regressions(self, mode=None):
        if mode is None:
            self.run_regressions(mode='lin')
            self.run_regressions(mode='log')
        else:
            if mode == 'lin':
                self.run_linear_regression(selected_features=self.features_selected_in_disjointification_lin)
            if mode == 'log':
                self.run_logistic_regression(selected_features=self.features_selected_in_disjointification_log)

    def run_linear_regression(self, selected_features=None):
        self.linear_regressor = LinearRegression()
        y = self.labels_df[self.lin_regressor_label].copy()
        x = self.features_df.copy()
        if selected_features is None:
            if self.features_selected_in_disjointification_lin is not None:
                x = x[self.features_selected_in_disjointification_lin]
        else:
            x = x[selected_features]

        self.x_train_lin, self.x_test_lin, self.y_train_lin, self.y_test_lin = \
            train_test_split(x, y, test_size=self.test_size, random_state=47)
        self.linear_regressor.fit(self.x_train_lin, self.y_train_lin)
        self.y_pred_lin = self.linear_regressor.predict(self.x_test_lin)
        self.lin_score = self.linear_regressor.score(self.x_test_lin, self.y_test_lin)

    def run_logistic_regression(self, selected_features=None):
        self.logistic_regressor = LogisticRegression()
        y = self.labels_df[self.log_regressor_label]
        if self.features_selected_in_disjointification_log is None:
            x = self.features_df
        else:
            x = self.features_df[self.features_selected_in_disjointification_log]

        if selected_features is not None:
            x = x[selected_features]

        self.x_train_log, self.x_test_log, self.y_train_log, self.y_test_log = \
            train_test_split(x, y, test_size=self.test_size, random_state=47)
        self.logistic_regressor.fit(self.x_train_log, self.y_train_log)
        self.y_pred_log = self.logistic_regressor.predict(self.x_test_log)
        self.log_score = self.logistic_regressor.score(self.x_test_log, self.y_test_log)
        self.log_confusion_matrix = confusion_matrix(y_true=self.y_test_log, y_pred=self.y_pred_log,
                                                     normalize='all')

    def autosave(self):
        if self.do_autosave:
            print(f"{utils.get_dt_in_fmt()} autosave function called by {__name__}")
            self.save_model()

    def save_model(self, printout=True):
        if printout:
            print(f"saving model...")
        save_folder = self.model_save_folder
        filename = f"{utils.get_dt_in_fmt()}_{self.features_df.shape}.pkl"
        file = Path(save_folder, filename)
        with open(file, 'wb') as f:
            pickle.dump(self, file=f)
        self.last_save_point_file = file
        if printout:
            print(f"saved model to {file.resolve()}")

    def show_linear_regressor(self, figsize=(6, 6), ax=None):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        sns.scatterplot(x=self.y_pred_lin, y=self.y_test_lin, ax=ax)

        title = f"linear regressor using dataset of total \n {self.x_train_lin.shape} " + \
                "train and {self.x_test_lin.shape} test. \nScore: {self.lin_score:.2f}"
        ax.set(
            title=title,
            xlabel="y_test", ylabel="y_pred")
        ax.grid("minor")

    def show_log_regressor(self, figsize=(6, 6), ax=None, cmap="viridis"):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        ConfusionMatrixDisplay.from_predictions(self.y_test_log, self.y_pred_log, ax=ax, cmap=cmap)
        title = f"log regressor using dataset of total \n {self.x_train_log.shape} " + \
                "train and {self.x_test_log.shape} test. \nScore: {self.log_score:.2f}"
        ax.set(
            title=title)

    def show(self, fig=None, axs=None, figsize=(12, 6)):
        if fig is None or axs is None:
            fig, axs = plt.subplots(1, 2, figsize=figsize)
        ax = axs.flatten()[0]
        ax.grid("minor")
        self.show_linear_regressor(ax=ax, figsize=(figsize[0] // 2, figsize[1]))
        ax = axs.flatten()[1]
        self.show_log_regressor(ax=ax, figsize=(figsize[0] // 2, figsize[1]))

    def show_classification_report(self):
        print(classification_report(self.y_test_log, self.y_pred_log))

    def set_feature_corr_matrix(self, mode=None, persistent=True):

        if mode is None:
            if not persistent or (self.correlation_matrix_lin is None):
                self.set_feature_corr_matrix(mode='lin', persistent=persistent)
            if not persistent or (self.correlation_matrix_log is None):
                self.set_feature_corr_matrix(mode='log', persistent=persistent)

        else:

            target = self.features_and_labels_df.copy()
            if mode == 'lin':
                drop_col = self.log_regressor_label

                method = 'pearson'
                print(f"calculating correlation matrix of linear regressor")
                # self.correlation_matrix_lin = target.copy().corr().drop(drop_col)
            if mode == 'log':
                drop_col = self.lin_regressor_label
                method = utils.wilcoxon_p_value
                print(f"calculating correlation matrix of logistic regressor")
                # self.correlation_matrix_log = self.correlation_matrix_log = \
                #     target.copy().corr(method=utils.wilcoxon_p_value).drop(drop_col)

            self.corr_matrix_temp = target.copy().drop(columns=drop_col).corr(method=method)
            self.autosave()

            if mode == 'lin':
                self.correlation_matrix_lin = self.corr_matrix_temp
                print(f"calculating correlation ranking of linear regressor")
                self.correlation_ranking_lin = self.correlation_matrix_lin[self.lin_regressor_label].sort_values(
                    ascending=False)
            if mode == 'log':
                self.correlation_matrix_log = self.corr_matrix_temp
                print(f"calculating correlation ranking of logistic regressor")
                self.correlation_ranking_log = self.correlation_matrix_log[self.log_regressor_label].sort_values(
                    ascending=False)
            self.autosave()

    def set_feature_lists(self):
        self.features_selected_in_disjointification_lin = []
        self.features_selected_in_disjointification_log = []
        self.features_rejected_lin = []
        self.features_rejected_log = []
        self.features_not_yet_selected_lin = self.features_df.columns
        self.features_not_yet_selected_log = self.features_df.columns
        self.number_of_features_tested_lin = 0
        self.number_of_features_tested_log = 0
        if self.max_num_iterations is None:
            self.max_num_iterations = np.inf

    def run_disjointification(self, mode=None, num_iterations=None, correlation_threshold=0.1,
                              min_num_of_features=np.inf, debug_print=False, alert_selection=False):
        if mode is None:
            self.run_disjointification(mode='lin', num_iterations=num_iterations,
                                       correlation_threshold=correlation_threshold,
                                       min_num_of_features=min_num_of_features,
                                       debug_print=debug_print, alert_selection=alert_selection)
            self.run_disjointification(mode='log', num_iterations=num_iterations,
                                       correlation_threshold=correlation_threshold,
                                       min_num_of_features=min_num_of_features,
                                       debug_print=debug_print, alert_selection=alert_selection)

        else:
            if num_iterations is None:
                num_iterations = self.max_num_iterations

            if mode == 'lin':
                self.focus_label_temp = self.lin_regressor_label
                self.drop_label_temp = self.log_regressor_label
                self.corr_matrix_temp = self.correlation_matrix_lin.copy()

            if mode == 'log':
                self.drop_label_temp = self.lin_regressor_label
                self.focus_label_temp = self.log_regressor_label
                self.corr_matrix_temp = self.correlation_matrix_log.copy()

            target = self.corr_matrix_temp[self.focus_label_temp].drop(self.drop_label_temp, errors='ignore')
            self.features_to_test_series_temp = target.abs().sort_values(ascending=False)

            if debug_print:
                print(f"features_to_test_series_temp {self.features_to_test_series_temp}")

            iter_num = 0
            for test_feature in self.features_to_test_series_temp.index:

                if iter_num >= num_iterations:
                    break

                if self.features_selected_in_disjointification_temp is None:
                    self.features_selected_in_disjointification_temp = [test_feature]

                if len(self.features_selected_in_disjointification_temp) == 0:
                    self.features_selected_in_disjointification_temp.append(test_feature)

                if len(self.features_selected_in_disjointification_temp) >= min_num_of_features:
                    break

                if target


                iter_num = iter_num + 1

            if mode == 'lin':
                self.features_selected_in_disjointification_lin = self.features_selected_in_disjointification_temp
            if mode == 'log':
                self.features_selected_in_disjointification_log = self.features_selected_in_disjointification_temp


            for iter_num in range(num_iterations):
                if mode == 'lin':
                    self.candidate_feature = str(self.features_to_test_series_temp[self.number_of_features_tested_lin])
                    self.features_selected_in_disjointification_temp = self.features_selected_in_disjointification_lin
                if mode == 'log':
                    self.candidate_feature = str(self.features_to_test_series_temp[self.number_of_features_tested_log])
                    self.features_selected_in_disjointification_temp = self.features_selected_in_disjointification_log

                if debug_print:
                    print(f"self.candidate_feature: {self.candidate_feature}")

                if (self.features_selected_in_disjointification_temp is None) or (len(self.features_selected_in_disjointification_temp) == 0):
                    # self.features_already_selected.append(self.candidate_feature)

                    if mode == 'lin':
                        self.features_selected_in_disjointification_lin.append(self.candidate_feature)
                    if mode == 'log':
                        self.features_selected_in_disjointification_log.append(self.candidate_feature)
                # Iterate
                else:

                    if mode == 'lin':
                        self.selected_feature_data_temp = self.features_df[self.features_selected_in_disjointification_lin]
                    if mode == 'log':
                        self.selected_feature_data_temp = self.features_df[self.features_selected_in_disjointification_log]

                    self.candidate_feature_data_temp = self.features_df[[self.candidate_feature]]

                    if debug_print:
                        print(f"self.features_already_selected_temp for mode {mode}: {self.selected_feature_data_temp}")
                        print(f"candidate_feature_data_temp : {self.candidate_feature_data_temp}")
                        print(f"selected_feature_data_temp: {self.selected_feature_data_temp}")

                    # self.df_candidate_vs_existing = self.candidate_feature_data.join(self.selected_feature_data)
                    # # self.df_candidate_vs_existing = pd.concat([self.candidate_feature_data,self.selected_feature_data])
                    self.candidate_and_selected_features_temp = list(self.features_selected_in_disjointification_temp) + [
                        self.candidate_feature]
                    self.df_candidate_vs_existing_temp = self.features_df[self.candidate_and_selected_features_temp]
                    self.test_corr_matrix_temp = self.df_candidate_vs_existing_temp.corr()
                    self.correlation_with_previous_features_temp = \
                        self.test_corr_matrix_temp[self.candidate_feature].drop(self.candidate_feature)

                    if debug_print:
                        print(f"correlation with previous: {self.correlation_with_previous_features_temp}")

                    if self.correlation_with_previous_features_temp.abs().max() <= correlation_threshold:
                        if alert_selection:
                            print(f"found a new feature to use!")
                        if mode == 'lin':
                            self.features_selected_in_disjointification_lin.append(self.candidate_feature)
                            self.features_selected_in_disjointification_lin = list(set(self.features_selected_in_disjointification_lin))
                            self.number_of_features_tested_lin = self.number_of_features_tested_lin + 1
                            if len(self.features_selected_in_disjointification_lin) >= min_num_of_features:
                                break
                        if mode == 'log':
                            self.features_selected_in_disjointification_log.append(self.candidate_feature)
                            self.features_selected_in_disjointification_log = list(set(self.features_selected_in_disjointification_log))
                            self.number_of_features_tested_log = self.number_of_features_tested_log + 1
                            if len(self.features_selected_in_disjointification_log) >= min_num_of_features:
                                break

                    self.autosave()
            self.save_model()


