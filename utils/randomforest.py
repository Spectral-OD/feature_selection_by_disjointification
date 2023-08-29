import pickle
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from pathlib import Path
import disjointification
from utils.utils import get_dt_in_fmt
import seaborn as sns
import matplotlib.pyplot as plt


def from_file(file):
    with open(file, 'rb') as f:
        loaded = pickle.load(f)
    return loaded


def show_feature_importance(rf, rf_cls, bins=20, figsize=(20, 10), log_scale=(False, True), grid='minor',
                            xlabels=("feature important", "feature important"),
                            ylabels=("counts", "counts"),
                            titles=("Random Forest Regressor", "Random Forest Classifier")):

    fig, axs = plt.subplots(1, 2, figsize=figsize)

    fig.suptitle("Feature Importance Distribution")
    xs = [rf.rf.feature_importances_, rf_cls.rf_cls.feature_importances_]
    for ax, xlabel, ylabel, title, x in zip(axs.flatten(), xlabels, ylabels, titles, xs):
        ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
        sns.histplot(x=x, bins=bins, log_scale=log_scale, ax=ax)
        if grid is not None:
            ax.grid(grid)

    plt.show()


class RandomForestRegressionComparison:
    def __init__(self, disjointification_model_pkl_file=None, disjointification_model=None, model_save_folder=None,
                 root_save_path=None):
        self.root_save_path = root_save_path
        self.disjointification_model_pkl_file = disjointification_model_pkl_file
        self.last_save_time = None
        self.last_save_point_file = None
        self.rf = None
        self.test_label = None
        self.train_label = None
        self.test_data = None
        self.train_data = None
        self.disjointification_model = disjointification_model
        self.model_save_folder = model_save_folder
        self.set_inputs()

    def set_inputs(self):
        if self.disjointification_model_pkl_file is not None:
            self.disjointification_model = disjointification.from_file(self.disjointification_model_pkl_file)
        self.set_model_save_folder()
        self.set_model()

    def set_model(self):
        feature_data_temp = self.disjointification_model.features_df.copy()
        y = self.disjointification_model.labels_df[self.disjointification_model.lin_regressor_label].copy()

        self.train_data, self.test_data, self.train_label, self.test_label = \
            train_test_split(feature_data_temp, y, test_size=self.disjointification_model.test_size, random_state=11)
        self.rf = RandomForestRegressor(random_state=0)
        self.save_model_to_file()

    def run(self):
        self.rf.fit(self.train_data, self.train_label)
        self.save_model_to_file()

    def set_model_save_folder(self, fmt="%m_%d_%Y__%H_%M_%S"):
        if self.model_save_folder is None:
            if self.root_save_path is not None:
                self.model_save_folder = Path(self.root_save_path)
            else:
                if self.disjointification_model_pkl_file is not None:
                    self.model_save_folder = Path(Path(self.disjointification_model_pkl_file).parents[0], "rf_model")
                else:
                    disjointification_save_path = Path(self.disjointification_model.model_save_folder)
                    folder_path = Path(disjointification_save_path, "rf_model")
                    self.model_save_folder = folder_path
        self.model_save_folder.mkdir(parents=True, exist_ok=True)
        self.save_model_to_file(new_file=True)

    def save_model_to_file(self, printout=True, new_file=False):
        call_time = get_dt_in_fmt()
        if printout:
            print(f"saving model...")
        if new_file:
            save_folder = self.model_save_folder
            filename = f"rf_{call_time}.pkl"
            file = Path(save_folder, filename)
            self.last_save_point_file = file
        else:
            file = self.last_save_point_file

        with open(file, 'wb') as f:
            pickle.dump(self, file=f)
        self.last_save_time = call_time
        if printout:
            print(f"saved model to {file.resolve()}")


class RandomForestClassificationComparison:
    def __init__(self, disjointification_model_pkl_file=None, disjointification_model=None, model_save_folder=None,
                 root_save_path=None):
        self.root_save_path = root_save_path
        self.disjointification_model_pkl_file = disjointification_model_pkl_file
        self.last_save_time = None
        self.last_save_point_file = None
        self.rf_cls = None
        self.test_label = None
        self.train_label = None
        self.test_data = None
        self.train_data = None
        self.disjointification_model = disjointification_model
        self.model_save_folder = model_save_folder
        self.set_inputs()

    def set_inputs(self):
        if self.disjointification_model_pkl_file is not None:
            self.disjointification_model = disjointification.from_file(self.disjointification_model_pkl_file)
        self.set_model_save_folder()
        self.set_model()

    def set_model(self):
        feature_data_temp = self.disjointification_model.features_df.copy()
        y = self.disjointification_model.labels_df[self.disjointification_model.log_regressor_label].copy()

        self.train_data, self.test_data, self.train_label, self.test_label = \
            train_test_split(feature_data_temp, y, test_size=self.disjointification_model.test_size, random_state=11)
        self.rf_cls = RandomForestClassifier(random_state=0)
        self.save_model_to_file()

    def run(self):
        self.rf_cls.fit(self.train_data, self.train_label)
        self.save_model_to_file()

    def set_model_save_folder(self, fmt="%m_%d_%Y__%H_%M_%S"):
        if self.model_save_folder is None:
            if self.root_save_path is not None:
                self.model_save_folder = Path(self.root_save_path)
            else:
                if self.disjointification_model_pkl_file is not None:
                    self.model_save_folder = Path(Path(self.disjointification_model_pkl_file).parents[0], "rf_model")
                else:
                    disjointification_save_path = Path(self.disjointification_model.model_save_folder)
                    folder_path = Path(disjointification_save_path, "rf_model")
                    self.model_save_folder = folder_path
        self.model_save_folder.mkdir(parents=True, exist_ok=True)
        self.save_model_to_file(new_file=True)

    def save_model_to_file(self, printout=True, new_file=False):
        call_time = get_dt_in_fmt()
        if printout:
            print(f"saving model...")
        if new_file:
            save_folder = self.model_save_folder
            filename = f"rf_cls_{call_time}.pkl"
            file = Path(save_folder, filename)
            self.last_save_point_file = file
        else:
            file = self.last_save_point_file

        with open(file, 'wb') as f:
            pickle.dump(self, file=f)
        self.last_save_time = call_time
        if printout:
            print(f"saved model to {file.resolve()}")
