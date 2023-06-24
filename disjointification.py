import pickle
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from utils import utils
from pathlib import Path
import pandas as pd
import scienceplots

plt.style.use(['science', 'notebook'])


def load_gene_expression_data(data_folder=r"c:\data", labels_file_name=r"sampleinfo_SCANB_t.csv",
                              features_file_name=r"SCANB_t.csv", labels_idx_column="samplename"):
    labels_file_path = Path(data_folder, labels_file_name)
    features_file_path = Path(data_folder, features_file_name)

    # features_df = pd.read_csv(features_file_path, index_col=labels_idx_column)
    # labels_df = pd.read_csv(labels_file_path, index_col=labels_idx_column)

    features_df = pd.read_csv(features_file_path)
    labels_df = pd.read_csv(labels_file_path)

    out_dict = {"labels": labels_df, "features": features_df}
    return out_dict


def create_test(*args, file=None, **kwargs):
    if file is None:
        return from_file(file)
    else:
        return Disjointification(*args, **kwargs)


def from_file(file):
    with open(file, 'rb') as f:
        loaded = pickle.load(f)
    return loaded


class Disjointification:
    def __init__(self, features_file_path=None, labels_file_path=None, labels_df=None, features_df=None,
                 select_num_features=None, select_num_instances=None, test_size=0.2,
                 lin_regressor_label="Lympho", log_regressor_label="ER", do_autosave=True,
                 max_num_iterations=None, root_save_folder=None, do_set=True, alert_selection=False):

        self.alert_selection = alert_selection
        self.last_save_time = None
        self.root_save_folder = root_save_folder
        self.description = None
        self.regression_sweep_y_lin = None
        self.regression_sweep_x_log = None
        self.regression_sweep_y_log = None
        self.regression_sweep_x_lin = None
        self.correlation_vals_temp = None
        self.features_list_temp = None
        self.drop_label_temp = None
        self.focus_label_temp = None
        self.selected_labels_temp = None
        self.drop_labels_temp = None
        self.corr_matrix_temp = None
        self.max_num_iterations = max_num_iterations
        self.last_save_point_file = None
        self.do_autosave = do_autosave
        self.model_save_folder = None
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
        if do_set:
            self.set()

    def set(self):
        self.set_model_save_folder()
        self.set_dfs()
        self.set_wrapped_attributes()
        self.set_feature_lists()
        self.set_label_correlation_lists()
        self.autosave()

    def describe(self):
        title = ("Disjointification Test Description")
        self.description = []
        self.description.append(title)
        self.description.append(f"features data: {self.features_df.shape}")
        self.description.append(f"labels data: {self.labels_df.shape}")
        self.description.append(f"last save point: {self.last_save_point_file}")
        p = [print(x) for x in self.description]
        # pprint(self.description)

    def run(self, show=False):
        self.run_disjointification(alert_selection=self.alert_selection)
        self.run_regressions()
        if show:
            self.show()

    def set_model_save_folder(self, root=None, fmt="%m_%d_%Y__%H_%M_%S"):
        this_run_dt = utils.get_dt_in_fmt(fmt=fmt)
        if self.model_save_folder is None:
            if root is None:
                root = self.root_save_folder
            folder_path = Path(root, this_run_dt)
            self.model_save_folder = folder_path
        self.model_save_folder.mkdir(parents=True, exist_ok=True)
        self.save_model_to_file(new_file=True)

    def set_dfs(self):
        if self.labels_df is None:
            raise NotImplementedError("can't initialize test without labels dataframe")

        if self.features_df is None:
            raise NotImplementedError("can't initialize test without features dataframe")

        for df in [self.labels_df, self.features_df]:
            df.drop(["Unnamed: 0"], errors='ignore', inplace=True, axis=1)
            df.dropna(axis=1, how='all', inplace=True)

        if self.select_num_instances is not None:
            ninst = self.select_num_instances
            nrows = self.features_df.shape[0]
            endp = utils.get_int_or_fraction(ninst, nrows)
            # keep only first lines of labels and features dfs
            self.labels_df = self.labels_df[0:endp]
            self.features_df = self.features_df[0:endp]

        if self.select_num_features is not None:
            ninst = self.select_num_features
            ncols = self.features_df.shape[1]
            endp = utils.get_int_or_fraction(ninst, ncols)
            # endp = ninst if ninst > 1 else int(ninst * ncols)
            features_to_keep = np.arange(endp)
            # feature_names_to_keep = features_df.columns[features_to_keep]
            self.features_df = self.features_df.iloc[:, features_to_keep]

        self.selected_labels_temp = [self.lin_regressor_label, self.log_regressor_label]
        self.labels_df = self.labels_df[self.selected_labels_temp]
        self.features_and_labels_df = pd.merge(self.features_df, self.labels_df, left_index=True, right_index=True,
                                               how='inner').dropna()

        label_col = [self.lin_regressor_label, self.log_regressor_label]
        self.features_df = self.features_and_labels_df.drop(columns=label_col)
        self.labels_df = self.features_and_labels_df[label_col]
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

    def run_regressions(self, mode=None, selected_features=None):
        if mode is None:
            self.run_regressions(mode='lin', selected_features=selected_features)
            self.run_regressions(mode='log', selected_features=selected_features)
        else:
            if mode == 'lin':
                self.run_linear_regression(selected_features=self.features_selected_in_disjointification_lin)
            if mode == 'log':
                self.run_logistic_regression(selected_features=self.features_selected_in_disjointification_log)

    def sweep_regression(self, mode=None, selected_feature_num=None):
        if mode is None:
            self.sweep_regression(mode='lin', selected_feature_num=selected_feature_num)
            self.sweep_regression(mode='log', selected_feature_num=selected_feature_num)
        else:
            total_features = None
            if selected_feature_num is None:
                if mode == 'lin':
                    total_features = len(self.features_selected_in_disjointification_lin)
                if mode == 'log':
                    total_features = len(self.features_selected_in_disjointification_log)
            step = np.sqrt(total_features).astype(int)
            num_selected_features_list = np.linspace(1, stop=total_features, num=step, endpoint=True).astype(int)
            score = None
            scores = []
            for selected_features in num_selected_features_list:
                self.run_regressions(mode=mode, selected_features=selected_features)
                if mode == 'lin':
                    score = self.lin_score
                if mode == 'log':
                    score = self.log_score
                scores.append(score)

            if mode == 'lin':
                self.regression_sweep_x_lin = num_selected_features_list
                self.regression_sweep_y_lin = scores
            if mode == 'log':
                self.regression_sweep_x_log = num_selected_features_list
                self.regression_sweep_y_log = scores

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
            train_test_split(x.dropna(), y.dropna(), test_size=self.test_size, random_state=47)
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

    def autosave(self, printout=False, new_file=False):
        if self.do_autosave:
            # print(f"{utils.get_dt_in_fmt()} autosave function called by {__name__}")
            self.save_model_to_file(printout=printout, new_file=new_file)

    def save_model_to_file(self, printout=True, new_file=False):
        call_time = utils.get_dt_in_fmt()
        if printout:
            print(f"saving model...")
        if new_file:
            save_folder = self.model_save_folder
            filename = f"{call_time}.pkl"
            file = Path(save_folder, filename)
            self.last_save_point_file = file
        else:
            file = self.last_save_point_file

        with open(file, 'wb') as f:
            pickle.dump(self, file=f)
        self.last_save_time = call_time
        if printout:
            print(f"saved model to {file.resolve()}")

    def show_regressor_sweep(self, figsize=(12, 6), axs=None):
        if axs is None:
            fig, axs = plt.subplots(1, 2, figsize=figsize)
        ax = axs[0]
        if np.all([self.regression_sweep_x_lin is not None, self.regression_sweep_y_lin is not None]):
            sns.scatterplot(x=self.regression_sweep_x_lin, y=self.regression_sweep_y_lin, ax=ax)
        title = f"linear regressor number of features vs. score"
        ax.set(title=title, xlabel="num features", ylabel="R2 Score")
        ax = axs[1]
        if np.all([self.regression_sweep_x_log is not None, self.regression_sweep_y_log is not None]):
            sns.scatterplot(x=self.regression_sweep_x_log, y=self.regression_sweep_y_log, ax=ax)
        title = f"logistic regressor number of features vs. score"
        ax.set(title=title, xlabel="num features", ylabel="R2 Score")

        for ax in axs.flatten():
            ax.grid("minor")

    def show_linear_regressor(self, figsize=(6, 6), ax=None):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        sns.scatterplot(x=self.y_pred_lin, y=self.y_test_lin, ax=ax)

        title = f"linear regressor using dataset of total \n {self.x_train_lin.shape} " + \
                f"train and {self.x_test_lin.shape} test. \nScore: {self.lin_score:.2f}"
        ax.set(title=title, xlabel="y_test", ylabel="y_pred")
        ax.grid("minor")

    def show_log_regressor(self, figsize=(6, 6), ax=None, cmap="viridis"):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        ConfusionMatrixDisplay.from_predictions(self.y_test_log, self.y_pred_log, ax=ax, cmap=cmap)
        title = f"log regressor using dataset of total \n {self.x_train_log.shape} " + \
                f"train and {self.x_test_log.shape} test. \nScore: {self.log_score:.2f}"
        ax.set(
            title=title)

    def show(self, fig=None, axs=None, figsize=(12, 6), report=False):
        if fig is None or axs is None:
            fig, axs = plt.subplots(1, 2, figsize=figsize)
        ax = axs.flatten()[0]
        ax.grid("minor")
        self.show_linear_regressor(ax=ax, figsize=(figsize[0] // 2, figsize[1]))
        ax = axs.flatten()[1]
        self.show_log_regressor(ax=ax, figsize=(figsize[0] // 2, figsize[1]))
        if report:
            self.show_classification_report()

    def show_classification_report(self):
        print(classification_report(self.y_test_log, self.y_pred_log))

    def set_label_correlation_lists(self):
        source = self.features_df

        method = 'pearson'
        label = self.lin_regressor_label
        target = self.labels_df[label]

        correlation_vals = source.corrwith(target, method=method)
        ranking = correlation_vals.abs().sort_values(ascending=False)
        self.correlation_ranking_lin = ranking

        method = utils.wilcoxon_p_value
        label = self.log_regressor_label
        target = self.labels_df[label]

        correlation_vals = source.corrwith(target, method=method)
        ranking = correlation_vals.abs().sort_values(ascending=False)
        self.correlation_ranking_log = ranking
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
                if debug_print:
                    print(f'self.correlation_ranking_lin: {self.correlation_ranking_lin}')
                self.features_list_temp = self.correlation_ranking_lin.copy()
            if mode == 'log':
                if debug_print:
                    print(f'self.correlation_ranking_log: {self.correlation_ranking_log}')
                self.features_list_temp = self.correlation_ranking_log.copy()

            for (feature_num, candidate_feature) in enumerate(self.features_list_temp.index):
                if debug_print:
                    print(f"candidate_feature: {candidate_feature}")
                if self.features_selected_in_disjointification_temp is None \
                        or len(self.features_selected_in_disjointification_temp) == 0:
                    self.features_selected_in_disjointification_temp = [self.features_list_temp.index[0]]
                    continue

                if len(self.features_selected_in_disjointification_temp) >= min_num_of_features:
                    break
                if feature_num >= num_iterations:
                    break

                self.autosave()
                self.corr_matrix_temp = self.features_df[self.features_selected_in_disjointification_temp]
                self.candidate_feature_data_temp = self.features_df[candidate_feature]
                self.correlation_vals_temp = self.corr_matrix_temp.corrwith(self.candidate_feature_data_temp)
                if self.correlation_vals_temp.abs().max() <= correlation_threshold:
                    if alert_selection:
                        print(f"candidate_feature: {candidate_feature} was selected!")
                    self.features_selected_in_disjointification_temp.append(candidate_feature)

            if mode == 'lin':
                self.features_selected_in_disjointification_lin = self.features_selected_in_disjointification_temp
            if mode == 'log':
                self.features_selected_in_disjointification_log = self.features_selected_in_disjointification_temp

            self.save_model_to_file()


if __name__ == "__main__":
    ge_data = load_gene_expression_data()
    features_df = ge_data["features"]
    labels_df = ge_data["labels"]
    select_num_features = 0.1
    select_num_instances = 0.1
    alert_selection = True
    debug_print = False

    test = Disjointification(features_file_path=None, labels_file_path=None, features_df=features_df,
                             labels_df=labels_df, select_num_features=select_num_features,
                             select_num_instances=select_num_instances)

    min_num_of_features = 50
    correlation_threshold = 0.99
    test.run_disjointification(min_num_of_features=min_num_of_features, correlation_threshold=correlation_threshold)
