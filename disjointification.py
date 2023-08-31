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
                              features_file_name=r"SCANB_t.csv"):
    labels_file_path = Path(data_folder, labels_file_name)
    features_file_path = Path(data_folder, features_file_name)

    ft_df = pd.read_csv(features_file_path)
    lbl_df = pd.read_csv(labels_file_path)

    out_dict = {"labels": lbl_df, "features": ft_df}
    return out_dict


def create_test(*args, file=None, **kwargs):
    if file is None:
        return from_file(file)
    else:
        return Disjointification(*args, **kwargs)


def from_file(file):
    with open(file, 'rb') as f:
        loaded = pickle.load(f)
    if not hasattr(loaded, "regression_correlation_method"):
        setattr(loaded, "regression_correlation_method", None)
    if not hasattr(loaded, "classification_correlation_method"):
        setattr(loaded, "classification_correlation_method", None)
    if not hasattr(loaded, "logistic_regression_max_iter"):
        setattr(loaded, "logistic_regression_max_iter", 1000)
    if not hasattr(loaded, "class_weight"):
        setattr(loaded, "class_weight", "balanced")
    try:
        loaded.features_df.set_index("samplename", inplace=True)
    except:
        pass
    try:
        loaded.labels_df.set_index("samplename", inplace=True)
    except:
        pass
    try:
        loaded.features_and_labels_df.set_index("samplename", inplace=True)
    except:
        pass

    return loaded


def validation_visualize_01(save_point, short_description="No Description", start_num_features=1, stop_num_features=300,
                            num_sweep=300, do_linear_regression_scatter=False, do_num_features_scatter=True, linear_features_list=None, log_feature_list=None):
    """

    :param do_num_features_scatter: validation scatter plot for regression
    :param do_linear_regression_scatter: validation scatter plot for both regression and classification
    :param save_point: .pkl file containing Disjointification object
    :param short_description: string describing save point
    :param start_num_features: minimum number of feats to sweep over
    :param stop_num_features: maximum number of feats to sweep over
    :param num_sweep: number of sweep points
    :return:
    """

    print(f"Validation Visualize: {short_description}")
    test = from_file(save_point)
    test.describe()

    if do_linear_regression_scatter:
        plt.figure()
        print("# Linear Prediction as a function of number of features kept (best and worst features)")
        test.sweep_regression_plot(mode='lin')
        test.sweep_regression_plot(mode='lin', order=-1)

    if do_num_features_scatter:
        print("Best features vs. Worst Features - Regression/Classification Score")
        test.init_scores_df()
        test.sweep_regression_scores(mode='lin', start_num_features=start_num_features,
                                     stop_num_features=stop_num_features,
                                     num_sweep=num_sweep, all_features_list=linear_features_list)
        test.sweep_regression_scores(mode='log', start_num_features=start_num_features,
                                     stop_num_features=stop_num_features,
                                     num_sweep=num_sweep, all_features_list=log_feature_list)

        plt.figure()
        fig, axs = plt.subplots(1, 2, figsize=(20, 10))
        data = test.scores_df

        ax = axs.flatten()[0]
        ax.set(title="Regression", xlabel="Number of Features", ylabel="Score")
        sns.scatterplot(data=data, y="scores_from_best_lin", x="num_features", ax=ax,
                        label="Using Best Features from Disjointification")
        sns.scatterplot(data=data, y="scores_from_worst_lin", x="num_features", ax=ax,
                        label="Using Worst Features from Disjointification")
        ax.grid('minor')

        ax = axs.flatten()[1]
        ax.set(title="Clssification", xlabel="Number of Features", ylabel="Score")
        sns.scatterplot(data=data, y="scores_from_best_log", x="num_features", ax=ax,
                        label="Using Best Features from Disjointification")
        sns.scatterplot(data=data, y="scores_from_worst_log", x="num_features", ax=ax,
                        label="Using Worst Features from Disjointification")
        ax.grid('minor')
        plt.show()


class Disjointification:
    def __init__(self, features_file_path=None, labels_file_path=None, labels_df=None, features_df=None,
                 select_num_features=None, select_num_instances=None, test_size=0.2,
                 lin_regressor_label="Lympho", log_regressor_label="ER", logistic_regression_max_iter=1000,
                 do_autosave=True,
                 regression_correlation_method="pearson", classification_correlation_method="kendall",
                 min_num_features=None, correlation_threshold=None,
                 max_num_iterations=None, root_save_folder=None, do_set=True, alert_selection=False):

        self.classification_correlation_method = classification_correlation_method
        self.regression_correlation_method = regression_correlation_method
        self.scores_df = None
        self.correlation_threshold = correlation_threshold
        self.min_num_features = min_num_features
        self.alert_selection = alert_selection
        self.last_save_time = None
        self.root_save_folder = root_save_folder
        self.description = None
        self.drop_label_temp = None
        self.focus_label_temp = None
        self.drop_labels_temp = None
        self.max_num_iterations = max_num_iterations
        self.last_save_point_file = None
        self.do_autosave = do_autosave
        self.model_save_folder = None
        self.candidate_feature = None
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
        self.logistic_regression_max_iter = logistic_regression_max_iter
        self.class_weight = "balanced"

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
        title = "Disjointification Test Description"
        self.description = []
        self.description.append(title)
        self.description.append(f"features data: {self.features_df.shape}")
        self.description.append(f"labels data: {self.labels_df.shape}")
        self.description.append(f"regression label: {self.lin_regressor_label}")
        self.description.append(f"classification label: {self.log_regressor_label}")
        self.description.append(f"correlation method regression: {self.regression_correlation_method}")
        self.description.append(f"correlation method regression: {self.classification_correlation_method}")
        self.description.append(f"min num of features to keep in disjointification: {self.min_num_features}")
        self.description.append(f"correlation threshold: {self.correlation_threshold}")
        self.description.append(f"last save point: {self.last_save_point_file}")
        self.description.append(f"number of features kept in disjointification: lin"
                                f" {len(self.features_selected_in_disjointification_lin)}, "
                                f"log {len(self.features_selected_in_disjointification_log)}")

        for x in self.description:
            print(x)

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

    def init_scores_df(self):
        column_names = ["num_features", "scores_from_best_lin", "scores_from_worst_lin", "scores_from_best_log",
                        "scores_from_worst_log"]
        self.scores_df = pd.DataFrame(columns=column_names)

    def set_dfs(self):
        self.init_scores_df()

        if self.labels_df is None:
            raise NotImplementedError("can't initialize test without labels dataframe")

        if self.features_df is None:
            raise NotImplementedError("can't initialize test without features dataframe")

        for df in [self.labels_df, self.features_df]:
            df.drop(["Unnamed: 0"], errors='ignore', inplace=True, axis=1)
            df.dropna(axis=1, how='all', inplace=True)
            try:
                df.set_index("samplename", inplace=True)
            except:
                pass

        if self.select_num_instances is not None:
            num_instances = self.select_num_instances
            num_rows = self.features_df.shape[0]
            end_point = utils.get_int_or_fraction(num_instances, num_rows)
            self.labels_df = self.labels_df[0:end_point]
            self.features_df = self.features_df[0:end_point]

        if self.select_num_features is not None:
            num_instances = self.select_num_features
            num_cols = self.features_df.shape[1]
            end_point = utils.get_int_or_fraction(num_instances, num_cols)
            features_to_keep = np.arange(end_point)
            self.features_df = self.features_df.iloc[:, features_to_keep]

        selected_labels_temp = [self.lin_regressor_label, self.log_regressor_label]
        self.labels_df = self.labels_df[selected_labels_temp]
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
        self.logistic_regressor = LogisticRegression(max_iter=self.logistic_regression_max_iter,
                                                     class_weight=self.class_weight)
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

    def sweep_regression_scores(self, mode=None, selected_feature_num=None, num_sweep=100,
                                start_num_features=3, stop_num_features=None, order=None, all_features_list=None):

        if mode is None:
            self.init_scores_df()
            self.sweep_regression_scores(mode='lin', selected_feature_num=selected_feature_num, num_sweep=num_sweep,
                                         start_num_features=start_num_features, stop_num_features=stop_num_features,
                                         order=order, all_features_list=all_features_list)
            self.sweep_regression_scores(mode='log', selected_feature_num=selected_feature_num, num_sweep=num_sweep,
                                         start_num_features=start_num_features, stop_num_features=stop_num_features,
                                         order=order, all_features_list=all_features_list)
        else:
            if all_features_list is None:
                if mode == 'lin':
                    all_features_list = self.features_selected_in_disjointification_lin
                if mode == 'log':
                    all_features_list = self.features_selected_in_disjointification_log
            if stop_num_features is None:
                num_of_features_disjointed = len(all_features_list)
            else:
                num_of_features_disjointed = stop_num_features
            nums_feats = np.geomspace(start=start_num_features,
                                      stop=num_of_features_disjointed,
                                      num=num_sweep, dtype=int)
            self.scores_df["num_features"] = nums_feats

            scores_from_best = []
            scores_from_worst = []
            for num_feats in nums_feats:
                if order is None or order >= 0:
                    best_features = all_features_list[0:num_feats]
                    # self.run_regressions(mode=mode, selected_features=best_features)
                    if mode == "lin":
                        self.run_linear_regression(selected_features=best_features)
                        scores_from_best.append(self.lin_score)
                    if mode == "log":
                        self.run_logistic_regression(selected_features=best_features)
                        scores_from_best.append(self.log_score)

                if order is None or order < 0:
                    worst_features = all_features_list[::-1][0:num_feats]
                    # self.run_regressions(mode=mode, selected_features=worst_features)

                    if mode == "lin":
                        self.run_linear_regression(selected_features=worst_features)
                        scores_from_worst.append(self.lin_score)

                    if mode == "log":
                        self.run_logistic_regression(selected_features=worst_features)
                        scores_from_worst.append(self.log_score)

            # store results in scores dataframe
            if mode == 'lin':
                if order is None or order >= 0:
                    # self.scores_from_best_features_xy_lin = (nums_feats, scores_from_best)
                    self.scores_df["scores_from_best_lin"] = scores_from_best
                if order is None or order < 0:
                    # self.scores_from_worst_features_xy_lin = (nums_feats, scores_from_worst)
                    self.scores_df["scores_from_worst_lin"] = scores_from_worst
            if mode == 'log':
                if order is None or order >= 0:
                    # self.scores_from_best_features_xy_log = (nums_feats, scores_from_best)
                    self.scores_df["scores_from_best_log"] = scores_from_best
                if order is None or order < 0:
                    # self.scores_from_worst_features_xy_log = (nums_feats, scores_from_worst)
                    self.scores_df["scores_from_worst_log"] = scores_from_worst

    def sweep_regression_plot(self, mode=None, selected_feature_num=None, num_sq=4,
                              start_num_features=3, stop_num_features=None, figsize=(20, 25),
                              order=1, all_features_list=None):
        if mode is None:
            self.sweep_regression_plot(mode='lin',
                                       selected_feature_num=selected_feature_num, num_sq=num_sq,
                                       start_num_features=3, stop_num_features=stop_num_features,
                                       figsize=figsize, order=order, all_features_list=all_features_list)
            self.sweep_regression_plot(mode='log',
                                       selected_feature_num=selected_feature_num,
                                       num_sq=num_sq, start_num_features=3, stop_num_features=stop_num_features,
                                       figsize=figsize, order=order, all_features_list=all_features_list)
        else:
            if all_features_list is None:
                if mode == 'lin':
                    all_features_list = self.features_selected_in_disjointification_lin
                if mode == 'log':
                    all_features_list = self.features_selected_in_disjointification_log
            if stop_num_features is None:
                num_of_features_disjointed = len(all_features_list)
            else:
                num_of_features_disjointed = stop_num_features
            nums_feats = np.geomspace(start=start_num_features,
                                      stop=num_of_features_disjointed,
                                      num=num_sq ** 2, dtype=int)
            plt.figure()
            fig, axs = plt.subplots(num_sq, num_sq, figsize=figsize)
            kept_features = all_features_list
            order_description = ''
            for ax, num_feats in zip(axs.flatten(), nums_feats):
                if order >= 0:
                    kept_features = all_features_list[0:num_feats]
                    order_description = 'best'
                if order < 0:
                    kept_features = all_features_list[::-1][0:num_feats]
                    order_description = 'worst'

                if mode == "lin":
                    self.run_linear_regression(selected_features=kept_features)
                    x_temp = self.y_pred_lin
                    y_temp = self.y_test_lin
                    score_temp = self.lin_score

                if mode == "log":
                    self.run_logistic_regression(selected_features=kept_features)
                    x_temp = self.y_pred_log
                    y_temp = self.y_test_log
                    score_temp = self.log_score

                ax.scatter(x_temp, y_temp, marker='.', label='model')
                title = fr"{num_feats} {order_description} fts. $R^2$: {score_temp: .3f}"
                ax.set(title=title, xlabel="Ground Truth", ylabel="Prediction")
                ax.plot(y_temp, y_temp, 'r--', label='ideal')
                ax.grid('minor')
                ax.legend()
            sup_title = f"Regressor ({mode}) Visualization as A Function of Number of Disjointed Features"
            fig.suptitle(sup_title)
            plt.show()

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

    def show_linear_regressor(self, figsize=(6, 6), ax=None, title=None):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        sns.scatterplot(x=self.y_pred_lin, y=self.y_test_lin, ax=ax)

        if title is None:
            title = f"Lin. Reg. data shape \n {self.x_train_lin.shape} " + \
                    f"train, {self.x_test_lin.shape} test. \nScore: {self.lin_score:.4f}"
        ax.set(title=title, xlabel="y_test", ylabel="y_pred")
        ax.grid("minor")

    def show_log_regressor(self, figsize=(6, 6), ax=None, cmap="viridis"):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        ConfusionMatrixDisplay.from_predictions(self.y_test_log, self.y_pred_log, ax=ax, cmap=cmap)
        title = f"log regressor using dataset of total \n {self.x_train_log.shape} " + \
                f"train and {self.x_test_log.shape} test. \nScore (Accuracy): {self.log_score:.4f}"
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

        method = self.regression_correlation_method
        label = self.lin_regressor_label
        target = self.labels_df[label]

        correlation_vals = source.corrwith(target, method=method)
        ranking = correlation_vals.abs().sort_values(ascending=False)
        self.correlation_ranking_lin = ranking

        method = self.classification_correlation_method
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
        if self.min_num_features is None:
            self.min_num_features = self.features_df.shape[1]

    def run_disjointification(self, mode=None, num_iterations=None, correlation_threshold=None,
                              min_num_features=None, debug_print=False, alert_selection=False):
        if correlation_threshold is None:
            correlation_threshold = self.correlation_threshold
        if min_num_features is None:
            min_num_features = self.min_num_features

        if mode is None:
            self.run_disjointification(mode='lin', num_iterations=num_iterations,
                                       correlation_threshold=correlation_threshold,
                                       min_num_features=min_num_features,
                                       debug_print=debug_print, alert_selection=alert_selection)
            self.run_disjointification(mode='log', num_iterations=num_iterations,
                                       correlation_threshold=correlation_threshold,
                                       min_num_features=min_num_features,
                                       debug_print=debug_print, alert_selection=alert_selection)

        else:
            features_selected_in_disjointification_temp = None
            if num_iterations is None:
                num_iterations = self.max_num_iterations

            if mode == 'lin':
                if debug_print:
                    print(f'self.correlation_ranking_lin: {self.correlation_ranking_lin}')
                features_list_temp = self.correlation_ranking_lin.copy()
            if mode == 'log':
                if debug_print:
                    print(f'self.correlation_ranking_log: {self.correlation_ranking_log}')
                features_list_temp = self.correlation_ranking_log.copy()

            for (feature_num, candidate_feature) in enumerate(features_list_temp.index):
                if debug_print:
                    print(f"candidate_feature: {candidate_feature}")
                if features_selected_in_disjointification_temp is None \
                        or len(features_selected_in_disjointification_temp) == 0:
                    features_selected_in_disjointification_temp = [features_list_temp.index[0]]
                    continue

                if len(features_selected_in_disjointification_temp) >= min_num_features:
                    break
                if feature_num >= num_iterations:
                    break

                self.autosave()
                corr_matrix_temp = self.features_df[features_selected_in_disjointification_temp]
                candidate_feature_data_temp = self.features_df[candidate_feature]
                correlation_vals_temp = corr_matrix_temp.corrwith(candidate_feature_data_temp)
                if correlation_vals_temp.abs().max() <= correlation_threshold:
                    if alert_selection:
                        print(f"candidate_feature: {candidate_feature} was selected!")
                    features_selected_in_disjointification_temp.append(candidate_feature)

            if mode == 'lin':
                self.features_selected_in_disjointification_lin = features_selected_in_disjointification_temp
            if mode == 'log':
                self.features_selected_in_disjointification_log = features_selected_in_disjointification_temp

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
    test.run_disjointification(min_num_features=min_num_of_features, correlation_threshold=correlation_threshold)
