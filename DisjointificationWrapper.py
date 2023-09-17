import disjointification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


class DisjointificationWrapper:
    def __init__(self, disjointification_model=None, disjointification_model_save_path=None,
                 validation_size_split=None, test_size_split=None, train_size_split=None,
                 initial_num_features=5, max_num_features=500, max_iter=10000, do_set=True):
        self.classification_label = None
        self.regression_label = None
        self.disjointification_model = disjointification_model
        self.disjointification_model_save_path = disjointification_model_save_path
        self.validation_size_split = validation_size_split
        self.test_size_split = test_size_split
        self.train_size_split = train_size_split
        self.initial_num_features = initial_num_features
        self.max_num_features = max_num_features
        self.max_iter = max_iter
        self.results = None
        self.class_weight = "balanced"
        if do_set:
            self.set()

    def set(self):
        if self.disjointification_model_save_path is not None:
            self.disjointification_model = disjointification.from_file(self.disjointification_model_save_path)
        self.regression_label = self.disjointification_model.lin_regressor_label
        self.classification_label = self.disjointification_model.log_regressor_label
        self.labels_df = self.disjointification_model.labels_df
        self.features_df = self.disjointification_model.features_df
        self.classification_features = self.disjointification_model.features_selected_in_disjointification_log
        self.regression_features = self.disjointification_model.features_selected_in_disjointification_lin
        self.set_dataset_splits()
        self.results = {"regression": None, "classification": None}

        # maybe take max num features from disjointification model

    def set_dataset_splits(self):
        if self.validation_size_split is None:
            self.validation_size_split = 1 - self.train_size_split - self.test_size_split
        if self.test_size_split is None:
            self.test_size_split = 1 - self.train_size_split - self.validation_size_split
        if self.train_size_split is None:
            self.train_size_split = 1 - self.validation_size_split - self.test_size_split

    def run_train(self):
        modes = ['classification', 'regression']

        intermediary_validation_size_split = self.validation_size_split / (
                self.validation_size_split + self.train_size_split)
        initial_num_features = self.initial_num_features
        max_num_features = self.max_num_features
        max_iter = self.max_iter

        for mode_num, mode in enumerate(modes):
            if mode_num == 0:
                features_list = self.classification_features
                y = self.labels_df[self.classification_label].copy().dropna()
                initial_model = LogisticRegression(max_iter=max_iter, class_weight=self.class_weight)
            if mode_num == 1:
                features_list = self.regression_features
                y = self.labels_df[self.regression_label].copy().dropna()
                initial_model = LinearRegression()

            initial_features = features_list[:initial_num_features]
            x = self.features_df.copy().dropna()
            x_train_validation, x_test, y_train_validation, y_test = train_test_split(x, y,
                                                                                      test_size=self.test_size_split,
                                                                                      random_state=97)
            x_train, x_validation, y_train, y_validation = train_test_split(x_train_validation, y_train_validation,
                                                                            test_size=intermediary_validation_size_split,
                                                                            random_state=34)

            x_train_initial_features, x_validation_initial_features = \
                x_train[initial_features], x_validation[initial_features]

            initial_model.fit(x_train_initial_features, y_train)
            initial_score = initial_model.score(x_validation_initial_features, y_validation)

            current_model = initial_model
            current_score = initial_score
            current_features_train = x_train_initial_features
            current_features = initial_features
            scores = [current_score]
            tested_feature_nums = np.arange(initial_num_features, max_num_features)
            number_of_features = np.append(tested_feature_nums, max_num_features)
            num_features_used = [initial_num_features]

            for new_feature_index in tested_feature_nums:
                new_feature = features_list[new_feature_index]
                # print(new_feature)
                new_features_train = current_features_train.join(x_train[new_feature])
                # print(new_features_train)
                new_dataset_validation = x_validation[new_features_train.columns]

                if mode_num == 0:
                    new_model = LogisticRegression(max_iter=max_iter, class_weight=self.class_weight)
                if mode_num == 1:
                    new_model = LinearRegression()

                new_model.fit(new_features_train, y_train)
                new_score = new_model.score(new_dataset_validation, y_validation)

                if new_score > current_score:
                    current_features_train = new_features_train
                    current_score = new_score
                    current_model = new_model
                    current_features.append(new_feature)

                scores.append(current_score)
                num_features_used.append(len(current_features_train.columns))

            result_dict = {"model": current_model, "features": current_features, "scores": scores,
                           "num_features_used": num_features_used,
                           "number_of_features_tested": number_of_features,
                           "y_train": y_train, "y_validation":y_validation, "y_test":y_test,
                           "x_train": x_train, "x_validation": x_validation, "x_test": x_test}
            self.results[mode] = result_dict

    def run(self):
        self.run_train()
        self.run_test()

    def run_test(self):
        mode_str = "regression"
        mode = self.results[mode_str]
        x_test = mode["x_test"][mode["features"]]
        y_test = mode["y_test"]
        y_pred = mode["model"].predict(x_test)
        final_score = mode["model"].score(x_test, y_test)
        # print(f"Final score on test set - {mode_str}: {final_score}")
        self.final_test_score_regression = final_score

        mode_str = "classification"
        mode = self.results[mode_str]
        x_test = mode["x_test"][mode["features"]]
        y_test = mode["y_test"]
        y_pred = mode["model"].predict(x_test)
        final_score = mode["model"].score(x_test, y_test)
        # print(f"Final score on test set - {mode_str}: {final_score}")
        self.final_test_score_classification = final_score

    def show_results(self, figsize=(12, 15)):
        this_wrapper = self

        fig, axs = plt.subplots(2, 2, figsize=figsize)

        mode = "regression"
        number_of_features = this_wrapper.results[mode]["number_of_features_tested"]
        number_of_features_used = this_wrapper.results[mode]["num_features_used"]
        scores = this_wrapper.results[mode]["scores"]
        ax = axs[0, 1]
        ax.scatter(number_of_features, scores, label="Score on validation set")
        final_scores = np.multiply(self.final_test_score_regression, np.ones([len(number_of_features),1]))
        ax.plot(number_of_features, final_scores, label="Eventual score on test set")
        ax.set(ylabel="Score", title=mode + " model improvement")
        ax.legend()

        ax = axs[0, 0]
        ax.scatter(number_of_features, number_of_features_used)
        ylabel = "Number of selected features"
        ax.set(ylabel=ylabel, title=mode + " feature filtering")


        mode = "classification"
        number_of_features = this_wrapper.results[mode]["number_of_features_tested"]
        number_of_features_used = this_wrapper.results[mode]["num_features_used"]
        scores = this_wrapper.results[mode]["scores"]
        ax = axs[1, 1]
        ax.scatter(number_of_features, scores, label="Score on validation set")
        final_scores = np.multiply(self.final_test_score_classification, np.ones([len(number_of_features), 1]))
        ax.plot(number_of_features, final_scores, label="Eventual score on test set")
        ax.set(ylabel="Score", title=mode + " model improvement")
        ax.legend()

        ax = axs[1, 0]
        ax.scatter(number_of_features, number_of_features_used)
        ylabel = "Number of selected features"
        ax.set(ylabel=ylabel, title=mode + " feature filtering")


        for ax in axs.flatten():
            ax.grid('minor')
            ax.set(xlabel="Potential Number of features")

        plt.show()


if __name__ == "__main__":
    save_point = Path(
        r"disjointification_model_point_bi_serial_fulldataset_500_feats\09_13_2023__01_12_39\09_13_2023__01_12_39.pkl")
    disj = disjointification.from_file(save_point)
    train_split, validation_split, test_split = 0.5, 0.3, 0.2

    wrapper = DisjointificationWrapper(disjointification_model=disj, train_size_split=train_split,
                                       validation_size_split=validation_split, test_size_split=test_split)
    wrapper.run()
    wrapper.show_results()