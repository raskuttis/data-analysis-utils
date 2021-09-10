"""
Purpose

Classes to compare different ML models, evaluate their efficacy and report
on best fit estimators

"""

import pandas as pd
import seaborn as sns
import numpy as np
import json
import logging

from sklearn import model_selection
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.pipeline import Pipeline

from .. import plotting_utilities


def get_feature_names(column_transformer):
    """
    Utility function to get feature names from all transformers, no matter
    how nested

    :param column_transformer: Nested Pipeline or Transformer object
    :return: List of feature names
    """

    # Turn lookup into function for better handling with pipeline later
    def get_names(trans):
        # >> Original get_feature_names() method
        if trans == 'drop' or (
                hasattr(column, '__len__') and not len(column)):
            return []
        if trans == 'passthrough':
            if hasattr(column_transformer, '_df_columns'):
                if ((not isinstance(column, slice))
                        and all(isinstance(col, str) for col in column)):
                    return column
                else:
                    return column_transformer._df_columns[column]
            else:
                indices = np.arange(column_transformer._n_features)
                return [f"x{i}" for i in indices[column]]
        if not hasattr(trans, 'get_feature_names'):
            # >>> Change: Return input column names if no method avaiable
            # Turn error into a warning
            logging.info("Transformer %s (type %s) does not "
                         "provide get_feature_names. "
                         "Will return input column names if available"
                          % (str(name), type(trans).__name__))
            # For transformers without a get_features_names method, use the input
            # names to the column transformer
            if column is None:
                return []
            else:
                return [name + "__" + f for f in column]

        return [f"{name}__{f}" for f in trans.get_feature_names()]

    ### Start of processing
    feature_names = []

    # Allow transformers to be pipelines. Pipeline steps are named differently, so preprocessing is needed
    if type(column_transformer) == Pipeline:
        l_transformers = [(name, trans, None, None) for step, name, trans in column_transformer._iter()]
    else:
        # For column transformers, follow the original method
        l_transformers = list(column_transformer._iter(fitted=True))

    for name, trans, column, _ in l_transformers:
        if type(trans) == Pipeline:
            # Recursive call on pipeline
            _names = get_feature_names(trans)
            # if pipeline has no transformer that returns names
            if len(_names) == 0:
                _names = [f"{name}__{f}" for f in column]
            feature_names.extend(_names)
        else:
            feature_names.extend(get_names(trans))

    return feature_names


class ModelReport(object):
    """
    Given a best-fit estimator, report on different features of the model to get
    an idea of how well it performs on test data, and what insights we can get
    from the model
    """

    def __init__(self, model, X_test, Y_test, fit=True, X_train=None, Y_train=None,
                 preprocess_pipe=None):
        """
        Init method for Model Report. Takes in best fit estimator, training
        data and test data

        :param model: E.g. sklearn model that we're reporting on
        :param X_test: Test data matrix
        :param Y_test: Test target variable
        :param fit: Boolean for whether the model has already been fit or not
        :param X_train: Training data, only used if the model isn't yet fit
        :param Y_train: Training target variable
        :param preprocess_pipe: A separate preprocessing pipeline if we want
        to concatenate that with the model
        """

        self.model = model
        if preprocess_pipe:
            self.model = Pipeline([("preprocess", preprocess_pipe), ("model", model)])
        # If the model isn't already fit, then fit it
        if not fit:
            self.model.fit(X_train, Y_train)

        # Get the predicted results from the model
        self.X_test = X_test
        self.Y_pred = self.model.predict(X_test)
        self.Y_prob = self.model.predict_proba(X_test)[:, 1]
        self.Y_actual = Y_test

    def plot_feature_importances(self, max_features=20):
        """
        Plot the feature importances in descending order for the model

        :param: Maximum number of features to plot
        :return: Matplotlib Axis object
        """

        # Check if there's a preprocess pipe and model in the existing model
        try:
            clf = self.model["model"]
            preprocess_pipe = self.model["preprocess"]
            importances = clf.feature_importances_
            feature_names = get_feature_names(preprocess_pipe)
        except Exception as error:
            print(f"Couldn't find feature names with error {error}")
            logging.info(f"Couldn't find feature names with error {error}")
            importances = self.model.feature_importances_
            feature_names = range(len(importances))

        importance_df = pd.DataFrame({"Feature": feature_names,
                                      "Importance": importances})
        importance_df = importance_df.sort_values(by=["Importance"], ascending=False)
        if max_features:
            importance_df = importance_df[:max_features]

        return sns.barplot(x="Importance", y="Feature", data=importance_df, palette="Blues_d")

    def plot_confusion_matrix(self):
        """
        Plot a confusion matrix between the predicted and actual results

        :return: Matplotlib Axis object
        """

        confusion = confusion_matrix(self.Y_actual, self.Y_pred)
        confusion_df = pd.DataFrame(confusion)
        confusion_df.index.name = "True"
        confusion_df.columns.name = "Predicted"
        return sns.heatmap(confusion_df, annot=True, fmt='.2f', cmap="YlGnBu")

    def get_classification_report(self):
        """
        Retrieves the classification report

        :return: tuple containing the accuracy score and a dataframe containing
        the classification report
        """

        report = classification_report(self.Y_actual, self.Y_pred, output_dict=True)
        accuracy = report.pop("accuracy")
        report = pd.DataFrame(report)

        return accuracy, report

    def plot_classification_report(self):
        """
        Plots the classification report table

        :return: Matplotlib Axis object
        """

        accuracy, report = self.get_classification_report()

        return plotting_utilities.plot_table(report, title="Classification Report")

    def get_classification_type(self, true_class, pred_class):
        """
        Retrieves the rows of the data matrix that correspond to different
        true and predicted classes e.g. false positives

        :param true_class: The real, observed class of the data
        :param pred_class: The predicted class
        :return: subset of X_test
        """

        indices = np.logical_and(self.Y_actual == true_class, self.Y_pred == pred_class)

        return self.X_test[indices]

    def plot_precision_recall(self):
        """
        Plot a precision recall curve

        :return: Matplotlib Axis object
        """

        precision, recall, _ = precision_recall_curve(self.Y_actual, self.Y_prob)
        pr_df = pd.DataFrame({"Precision": precision, "Recall": recall})

        return sns.lineplot(data=pr_df, x="Recall", y="Precision")

    def plot_roc(self):
        """
        Plots an ROC curve

        :return: Matplotlib Axis object
        """

        fpr, tpr, _ = roc_curve(self.Y_actual, self.Y_prob)
        roc_df = pd.DataFrame({"False Positive Rate": fpr, "True Positive Rate": tpr})

        return sns.lineplot(data=roc_df, x="False Positive Rate", y="True Positive Rate")


class ModelComparison(object):
    """
    Class to derive model comparison metrics between different sklearn models
    and plot them if necessary
    """

    def __init__(self, models, X_train, Y_train, cv=10, metrics=None, fit_metric="accuracy",
                 preprocess_pipe=None):
        """
        Init method for Model Comparison. Takes in a dictionary of models, training
        data and test data

        :param models: Dictionary of models to test including the model name and the
        parameters to fit over
        :param X_train: Training data
        :param Y_train: Training target variable
        :param cv: Number of folds to do cross-validation over
        :param metrics: Metrics to report on
        :param fit_metric: Metrics to use to choose the best fit estimator
        :param preprocess_pipe: A separate preprocessing pipeline if we want
        to concatenate that with the model
        """

        self.models = models
        self.folds = model_selection.KFold(n_splits=cv, shuffle=True)
        self.X_train = X_train
        self.Y_train = Y_train
        self.scores = None
        self.predictions = None
        self.metrics = metrics
        if not self.metrics:
            self.metrics = []
        self.fit_metric = fit_metric
        self.preprocess_pipe = preprocess_pipe
        self.best_model = None

    def get_fold_predictions(self, method="predict_proba"):
        """
        Function to get the predicted values over all the different folds

        :param method: Whether to use a probability estimate or a binary prediction
        :return: DataFrame containing the predicted target variable for each KFold
        """

        all_predictions = []

        for model_name, model_dict in self.models.items():
            clf = model_dict.get("Classifier")
            if self.preprocess_pipe:
                clf = Pipeline([("preprocess", self.preprocess_pipe), ("model", clf)])
            y_pred = []
            y_actual = []
            if hasattr(clf, method):
                predict_method = getattr(clf, method)
                for i, (train, test) in enumerate(self.folds.split(self.X_train)):
                    if isinstance(self.X_train, pd.DataFrame):
                        train_set = self.X_train.iloc[train]
                        test_set = self.X_train.iloc[test]
                    else:
                        train_set = self.X_train[train]
                        test_set = self.X_train[test]
                    clf.fit(train_set, self.Y_train[train])
                    Y_test = predict_method(test_set)[:, 1].tolist()
                    y_pred += Y_test
                    y_actual += self.Y_train[test].tolist()

                prediction_df = pd.DataFrame({"predicted": y_pred, "actual": y_actual})
                prediction_df["model"] = model_name
                all_predictions += [prediction_df]

        all_prediction_df = pd.concat(all_predictions)
        self.predictions = all_prediction_df

        return all_prediction_df

    def parse_grid_search_results(self, cv_results):
        """
        Function to parse out the results of a GridSearchCV into a more readable
        format

        :param cv_results: Output of GridSearchCV.cv_results_
        :return: DataFrame with metrics and fit times for each fold and
        parameter combination
        """

        # Get columns to split on and drop
        gs_results = pd.DataFrame(cv_results)
        non_split_cols = [col for col in gs_results.columns if not col.startswith("split")]
        drop_cols = [col for col in gs_results.columns if (col.startswith("mean") or
                                                           col.startswith("param_") or
                                                           (col.startswith("rank")
                                                            and col != f"rank_test_{self.fit_metric}") or
                                                           col.startswith("std"))]

        # Iterate over cross validations to parse out the DF
        all_gs_results = []
        for split in range(self.folds.n_splits):
            subset_df = gs_results[[col for col in gs_results.columns
                                    if col.startswith(f"split{split}")] +
                                   non_split_cols].copy()
            subset_df.columns = subset_df.columns.str.replace(f"split{split}_", "")

            # Generate Fake Fit and Score Time data
            subset_df["fit_time"] = np.random.normal(subset_df["mean_fit_time"],
                                                     subset_df["std_fit_time"])
            subset_df["score_time"] = np.random.normal(subset_df["mean_score_time"],
                                                       subset_df["std_score_time"])
            if self.preprocess_pipe:
                subset_df["params"] = subset_df["params"].apply(lambda x: {p_name.replace("model__", ""): p_value
                                                                           for p_name, p_value in x.items()})
            subset_df["params"] = subset_df["params"].apply(json.dumps)
            all_gs_results += [subset_df]
        scores_df = pd.concat(all_gs_results)
        scores_df = scores_df.drop(drop_cols, axis=1)
        scores_df = scores_df.rename(columns={f"rank_test_{self.fit_metric}": "rank"})

        return scores_df

    def get_best_model(self, fit=True):
        """
        Function to return the best fit model with the best fit parameters

        :param fit: Boolean for whether or not to fit the best model or just
        return it
        :return: Best sklearn estimator
        """

        # Get metrics if they don't already exist
        self.get_metrics([self.fit_metric])

        # Get the model with the best rank and best parameter rank
        best_scores = self.scores[(self.scores["rank"] == 1) &
                                  (self.scores["model_rank"] == 1)]
        best_model = best_scores["model"].unique()[0]
        best_model = self.models.get(best_model, {}).get("Classifier")
        best_params = json.loads(best_scores["params"].unique()[0])
        best_model.set_params(**best_params)
        if self.preprocess_pipe:
            best_model = Pipeline([("preprocess", self.preprocess_pipe), ("model", best_model)])

        # Save the model for the future
        self.best_model = best_model

        # Fit if necessary
        if fit:
            self.best_model.fit(self.X_train, self.Y_train)

        return self.best_model

    def get_metrics(self, metrics):
        """
        Function to get different metrics for each model, where the list of
        allowed metrics can be found here
        https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter

        :param metrics: List of metrics to report on
        :return: DataFrame with metrics and fit times for each fold and
        parameter combination
        """

        # Check if any of the metrics exist already
        if self.scores is not None:
            existing_metrics = self.scores["metrics"].unique()
            metrics = [metric for metric in metrics if metric not in existing_metrics]

        logging.info("Computing Metrics for previously uncomputed %s", json.dumps(metrics))

        # Get new metrics if needed
        if metrics:
            new_scores = []
            best_scores = {}
            for model_name, model_dict in self.models.items():
                clf = model_dict.get("Classifier")
                params = model_dict.get("Parameters", {})
                if self.preprocess_pipe:
                    clf = Pipeline([("preprocess", self.preprocess_pipe), ("model", clf)])
                    params = {f"model__{param_name}": param_value for param_name, param_value in params.items()}
                if params:
                    grid_search = GridSearchCV(clf, params, cv=self.folds, scoring=metrics,
                                               refit=self.fit_metric)
                    grid_search.fit(self.X_train, self.Y_train)
                    # Reshape the grid search results
                    scores_df = self.parse_grid_search_results(grid_search.cv_results_)
                    best_scores[model_name] = grid_search.best_score_
                else:
                    scores = cross_validate(clf, self.X_train, self.Y_train, cv=self.folds,
                                            scoring=metrics)
                    scores_df = pd.DataFrame(scores)
                    scores_df["params"] = "{}"
                    scores_df["params"] = scores_df["params"].apply(json.loads)
                    scores_df["rank"] = 1
                    best_scores[model_name] = np.mean(scores[f"test_{self.fit_metric}"])
                scores_df = scores_df.rename(columns={f"test_{metric}": metric for metric in metrics})
                scores_df["model"] = model_name
                new_scores += [scores_df]

            new_scores_df = pd.concat(new_scores)

            # Get ranking of different models
            best_ranks = {model: sorted(best_scores.values(), reverse=True).index(value) + 1
                          for model, value in best_scores.items()}
            best_ranks_df = pd.DataFrame.from_dict(best_ranks, orient="index"). \
                reset_index().rename(columns={"index": "model", 0: "model_rank"})
            new_scores_df = pd.merge(new_scores_df, best_ranks_df, how="left", on="model")
            new_scores_df = pd.melt(new_scores_df, id_vars=["model", "params", "rank", "model_rank"],
                                    var_name="metrics", value_name="values")

            if self.scores is not None:
                self.scores = pd.concat([self.scores, new_scores_df])
            else:
                self.scores = new_scores_df

            return new_scores_df

        else:
            logging.info("No new metrics detected, so not computing")
            return None

    def plot_metrics(self, metrics=None, x_var="model", model_name=None):
        """
        Function to plot different metrics for each model and parameter combination

        :param metrics: List of metrics to report on
        :param x_var: Whether to plot over model type, or parameters
        :param model_name: Name of model we want to plot over
        :return: Matplotlib Axis object
        """

        # Initialize metrics with the defaults
        if not metrics:
            metrics = self.metrics

        self.get_metrics(metrics)
        metrics_df = self.scores[self.scores["metrics"].isin(metrics)]

        # Choose either the best parameters for the model or the best fit model
        if x_var == "model":
            metrics_df = metrics_df[metrics_df["rank"] == 1]
        elif x_var == "params":
            if model_name:
                metrics_df = metrics_df[metrics_df["model"] == model_name]
            else:
                metrics_df = metrics_df[metrics_df["model_rank"] == 1]
        else:
            logging.warning(f"Invalid plotting variable {x_var} chosen")

        return sns.boxplot(x=x_var, y="values", hue="metrics", data=metrics_df)

    def plot_model_metrics(self):
        """
        Function to plot model metrics

        :return: Matplotlib Axis object
        """
        plot = self.plot_metrics()
        plot.set_ylim(0,1)
        return plot

    def plot_parameter_metrics(self, model_name=None):
        """
        Function to plot parameter metrics for a given model

        :param model_name: Name of the model to plot parameter scores for
        :return: Matplotlib Axis object
        """

        plot = self.plot_metrics(x_var="params", model_name=model_name)
        plot.set_ylim(0, 1)
        return plot

    def plot_times(self, times=None):
        """
        Function to plot times to compute different models

        :param times: List containing some combination of fit_time and score_time
        :return: Matplotlib Axis object
        """

        # Initialize times with the defaults
        if not times:
            times = ["fit_time"]

        return self.plot_metrics(metrics=times, x_var="model")

    # Decorator to iterate over models and get curves for each model
    def get_model_curves(curve_type):
        """
        Decorator function to iterate over models and parameters to combine the
        outputs of e.g. a P-R or ROC curve

        """
        def iter_models(self):

            if self.predictions is None:
                self.predictions = self.get_fold_predictions()

            all_curves = []
            for model_name in self.predictions["model"].unique():
                y_pred = self.predictions[self.predictions["model"] == model_name]["predicted"].tolist()
                y_actual = self.predictions[self.predictions["model"] == model_name]["actual"].tolist()

                curve_df = curve_type(self, y_actual, y_pred, model_name)
                all_curves += [curve_df]

            all_curve_df = pd.concat(all_curves)

            return all_curve_df

        return iter_models

    @get_model_curves
    def get_precision_recall(self, y_actual, y_pred, model_name):
        """
        Get precision recall curves for all models

        :param y_actual: Training target variables
        :param y_pred: Predicted training target variables
        :param model_name: Name of the model
        :return: Dataframe containing Precision and Recall
        """

        precision, recall, _ = precision_recall_curve(y_actual, y_pred)

        pr_df = pd.DataFrame({"Precision": precision, "Recall": recall})
        pr_df["model"] = model_name

        return pr_df

    def plot_precision_recall(self):
        """
        Plot a precision recall curve

        :return: Matplotlib Axis object
        """

        all_pr_df = self.get_precision_recall()
        return sns.lineplot(data=all_pr_df, x="Recall", y="Precision", hue="model")

    @get_model_curves
    def get_roc(self, y_actual, y_pred, model_name):
        """
        Gets an ROC curve

        :param y_actual: Training target variables
        :param y_pred: Predicted training target variables
        :param model_name: Name of the model
        :return: Dataframe containing FPR and TPR
        """

        fpr, tpr, _ = roc_curve(y_actual, y_pred)

        roc_df = pd.DataFrame({"FPR": fpr, "TPR": tpr})
        roc_df["model"] = model_name

        return roc_df

    def plot_roc(self):
        """
        Plot an ROC curve

        :return: Matplotlib Axis object
        """

        all_roc_df = self.get_roc()
        return sns.lineplot(data=all_roc_df, x="FPR", y="TPR", hue="model")
