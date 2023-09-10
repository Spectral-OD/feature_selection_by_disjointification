import pickle
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
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
    return loaded


class Disjointification2:
    def __init__(self, correlation_threshold, features_file_path=None, labels_file_path=None,
                 labels_df: pd.DataFrame = None, features_df: pd.DataFrame = None, features_and_labels_df=None,
                 select_num_features=None, select_num_instances=None, test_size: float = 0.2,
                 disjointification_label: str = "Lympho", do_autosave=True,
                 ranking_correlation_method="pearson", min_num_features=None,
                 max_num_iterations=np.inf, root_save_folder=None, do_set: bool = True):
        """

        :param correlation_threshold: maximum correlation allowed between any two features
        :param features_file_path: str or Path where features file exists, or None and provide features_df instead
        :param labels_file_path: str or Path where labels file exists, or None and provide labels_df instead
        :param labels_df: dataframe containing labels
        :param features_df: dataframe containing features
        :param select_num_features: for debug-purposes, use only a limited number of features out of the data
        :param select_num_instances: for debug-purposes, use only a limited number of instances out of the data
        :param test_size: after disjointification, use this to split train-test for Regression model
        :param disjointification_label: label to search in dataframes for regression
        :param do_autosave: saves model during disjointification in various points
        :param ranking_correlation_method: str or pair-wise function to correlate with regression label
        :param min_num_features: min. num. of features successfully disjointed before stopping, or all features if None.
        :param max_num_iterations: maximum number of iterations before disjointification stops
        :param root_save_folder: str or Path object. Model results will be saved in sub-folders under this root
        :param do_set: sets the model automatically after initializing
        """

        self.correlation_method = ranking_correlation_method
        self.scores_df = None
        self.correlation_threshold = correlation_threshold
        self.min_num_features = min_num_features
        self.last_save_time = None
        self.root_save_folder = root_save_folder
        self.description = None
        self.max_num_iterations = max_num_iterations
        self.last_save_point_file = None
        self.do_autosave = do_autosave
        self.model_save_folder = None
        self.features_selected_in_disjointification = None
        self.correlation_ranking = None
        self.columns = None
        self.num_instances = None
        self.num_labels = None
        self.num_features = None
        self.features_file_path = features_file_path
        self.labels_file_path = labels_file_path
        self.labels_df = labels_df
        self.features_df = features_df
        self.select_num_features = select_num_features
        self.select_num_instances = select_num_instances
        self.features_and_labels_df = features_and_labels_df
        self.shape = None
        self.test_size = test_size
        self.disjointification_label = disjointification_label

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
        self.description.append(f"regression label: {self.disjointification_label}")
        self.description.append(f"correlation method for disjointification ranking: {self.correlation_method}")
        self.description.append(f"min num of features to keep in disjointification: {self.min_num_features}")
        self.description.append(f"correlation threshold: {self.correlation_threshold}")
        self.description.append(f"last save point: {self.last_save_point_file}")
        self.description.append(f"number of features kept in disjointification:"
                                f" {len(self.features_selected_in_disjointification)}")

        for x in self.description:
            print(x)

    def run(self, show=False):
        self.run_disjointification()
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
        column_names = ["num_features", "scores_from_best", "scores_from_worst"]
        self.scores_df = pd.DataFrame(columns=column_names)

    def set_dfs(self):
        self.init_scores_df()

        if self.features_and_labels_df is not None:
            for df in [self.features_and_labels_df]:
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
                self.features_and_labels_df = self.features_and_labels_df[0:end_point]

            if self.select_num_features is not None:
                num_instances = self.select_num_features
                num_cols = self.features_df.shape[1]
                end_point = utils.get_int_or_fraction(num_instances, num_cols)
                features_to_keep = np.arange(end_point)
                selected_labels_temp = [self.disjointification_label]
                labels_df_temp = self.features_and_labels_df[selected_labels_temp]
                no_labels_df_temp = self.features_and_labels_df.drop(selected_labels_temp)
                no_labels_df_temp = no_labels_df_temp.iloc[:, features_to_keep]
                self.features_and_labels_df = pd.merge(no_labels_df_temp, labels_df_temp, left_index=True,
                                                       right_index=True,
                                                       how='inner').dropna()
        else:
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

            selected_labels_temp = [self.disjointification_label]
            self.labels_df = self.labels_df[selected_labels_temp]
            self.features_and_labels_df = pd.merge(self.features_df, self.labels_df, left_index=True, right_index=True,
                                                   how='inner').dropna()

        label_col = [self.disjointification_label]
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

    def head(self, *args, **kwargs):
        return self.features_and_labels_df.head(args, kwargs)

    def tail(self, *args, **kwargs):
        return self.features_and_labels_df.tail(*args, **kwargs)

    def set_label_correlation_lists(self):
        source = self.features_df
        method = self.correlation_method
        label = self.disjointification_label
        target = self.labels_df[label]

        correlation_names_and_vals = source.corrwith(target, method=method)
        ranking = correlation_names_and_vals.abs().sort_values(ascending=False)
        self.correlation_ranking = ranking
        self.autosave()

    def set_feature_lists(self):
        self.features_selected_in_disjointification = []

        if self.max_num_iterations is None:
            self.max_num_iterations = np.inf
        if self.min_num_features is None:
            self.min_num_features = self.features_df.shape[1]

    def run_disjointification(self, num_iterations: int = None, correlation_threshold: float = None,
                              min_num_features: int = None):
        """

        :param num_iterations: how many iterations to run for. Will take the initialized number if None
        :param correlation_threshold: abs. correlation allowed between two selected features. Use initialized if None
        :param min_num_features: number of features to find before stopping. Use initialized if None.
        :return:
        """

        if correlation_threshold is None:
            correlation_threshold = self.correlation_threshold
        if min_num_features is None:
            min_num_features = self.min_num_features

        current_iteration_num = 0
        num_found = 0

        features_selected_in_disjointification_temp = None
        if num_iterations is None:
            num_iterations = self.max_num_iterations

        features_list_temp = self.correlation_ranking.copy()

        for (feature_num, candidate_feature) in enumerate(features_list_temp.index):
            current_iteration_num = current_iteration_num + 1

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

            if correlation_vals_temp.abs().max() <= correlation_threshold:  # found a new feature
                features_selected_in_disjointification_temp.append(candidate_feature)
                num_found = num_found + 1

            if np.mod(current_iteration_num, 100) == 0:
                this_time = utils.get_dt_in_fmt()
                print(f"{this_time} - after {current_iteration_num} iterations, found {num_found} features!")

        self.features_selected_in_disjointification = features_selected_in_disjointification_temp

        self.save_model_to_file()

    def get_features_selected_for_regression(self):
        return self.features_selected_in_disjointification

    def get_num_features_selected_for_regression(self):
        return np.array(self.features_selected_in_disjointification).size


if __name__ == "__main__":
    ge_data = load_gene_expression_data()
    _features_df = ge_data["features"]
    _labels_df = ge_data["labels"]
    _select_num_features = 0.1
    _select_num_instances = 0.1

    _test = Disjointification(features_file_path=None, labels_file_path=None, features_df=_features_df,
                              labels_df=_labels_df, select_num_features=_select_num_features,
                              select_num_instances=_select_num_instances)

    _min_num_of_features = 50
    _correlation_threshold = 0.99
    _test.run_disjointification(min_num_features=_min_num_of_features, correlation_threshold=_correlation_threshold)
