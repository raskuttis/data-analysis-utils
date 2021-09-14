"""
Purpose

Classes to investigate correlations between features and target variables
in a training dataset

"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from scipy import stats


class FeatureCorrelator(object):
    """
    Class to investigate and plot Feature Correlations between different
    columns of a pandas dataframe
    """

    def __init__(self, feature_df):
        """
        Init method for Feature Correlator. Just takes in a dataframe with feature variables

        :param feature_df: DataFrame representing the training data matrix
        """

        self.feature_df = feature_df

    def plot_contingency_table(self, feature_a, feature_b):
        """
        Plot a contingency table on categorical features

        :param feature_a: Name of the first feature column
        :param feature_b: Name of the second feature column
        :return: Matplotlib Axis object
        """

        contingency = pd.crosstab(self.feature_df[feature_a], self.feature_df[feature_b])
        return sns.heatmap(contingency, annot=True, fmt='.2f', cmap="YlGnBu")

    def plot_histogram(self, feature_a, feature_b):
        """
        Plot a histogram representing percentages of one feature vs.
        another

        :param feature_a: Name of the first feature column
        :param feature_b: Name of the second feature column
        :return: Matplotlib Axis object
        """

        return sns.histplot(x=self.feature_df[feature_a], hue=self.feature_df[feature_b], stat="probability")

    def plot_violin(self, feature_a, feature_b):
        """
        Plot a violin plot to show how a continuous variable feature_a varies with a binary
        variable feature_b

        :param feature_a: Name of the first feature column
        :param feature_b: Name of the second feature column
        :return: Matplotlib Axis object
        """

        return sns.violinplot(data=self.feature_df, x=feature_b, y=feature_a)

    def plot_box(self, feature_a, feature_b):
        """
        Plot a box plot to show how a continuous variable feature_a varies with a binary
        variable feature_b

        :param feature_a: Name of the first feature column
        :param feature_b: Name of the second feature column
        :return: Matplotlib Axis object
        """

        return sns.boxplot(data=self.feature_df, x=feature_b, y=feature_a)

    def plot_regression(self, feature_a, feature_b):
        """
        Plot a regression for when we try to fit feature_b using feature_a

        :param feature_a: Name of the first feature column
        :param feature_b: Name of the second feature column
        :return: Matplotlib Axis object
        """

        # Plot Logistic Regression output
        plot = sns.regplot(data=self.feature_df, x=feature_a, y=feature_b,
                           y_jitter=.02, logistic=True, truncate=False)

        # Get some overall correlation metrics to measure the LR Accuracy
        # Logistic Regression Accuracy
        lr_classifier = LogisticRegression()
        featurea_vals = self.feature_df[feature_a].values.reshape(-1, 1)
        featureb_vals = self.feature_df[feature_b].values
        lr_classifier.fit(featurea_vals, featureb_vals)
        y_pred = lr_classifier.predict(featurea_vals)
        plt.text(0.6, 0.8, f"Accuracy = {accuracy_score(featureb_vals, y_pred):.4f}",
                 horizontalalignment='left', size='medium', color='black', weight='semibold')

        # Point-Biserial Correlation
        biserial_corr = stats.pointbiserialr(self.feature_df[feature_b], self.feature_df[feature_a])
        plt.text(0.6, 0.75, f"Correlation = {biserial_corr[0]:.4f}",
                 horizontalalignment='left', size='medium', color='black', weight='semibold')

        return plot


class SingleFeatureTargetCorrelator(FeatureCorrelator):
    """
    Class to investigate and plot Feature Correlations between different
    columns of a pandas dataframe
    """

    def __init__(self, feature_df, feature_name, target_name, feature_type=None, target_type=None,
                 data_range=None):
        """
        Init method for Feature Correlator. Takeakes in a dataframe with feature variables, as
        well as the feature and target variable names and types

        :param feature_df: DataFrame representing the training data matrix
        :param feature_name: Name of the column containing the feature
        :param target_name: Name of the column containing the target variable
        :param feature_type: Type of the feature e.g. Numerical, Categorical etc.
        :param target_type: Type of the Target variable e.g. Binary, Continuous
        :param data_range: Explicitly defined data range for the feature variable
        """

        feature_subset = feature_df[[feature_name, target_name]].dropna()
        super().__init__(feature_subset)
        self.feature = feature_name
        self.target = target_name

        # Get quartiles and IQR to define the data range
        feature_vals = feature_subset[self.feature]
        q1 = np.percentile(feature_vals, 25)
        q3 = np.percentile(feature_vals, 75)
        iqr = q3 - q1
        qmax = min(q3 + 4 * iqr, max(feature_vals))
        qmin = max(q1 - 4 * iqr, min(feature_vals))
        self.data_range = data_range
        if not self.data_range:
            self.data_range = (qmin, qmax)

        # Define the default plots based on whether the variable
        self.default_plots = []
        if feature_type == "Numerical" and target_type == "Binary":
            self.default_plots = ["plot_violin", "plot_box", "plot_histogram", "plot_regression"]
        elif feature_type == "Categorical" and target_type == "Binary":
            self.default_plots = ["plot_histogram", "plot_contingency_table"]

    def plot_contingency_table(self):
        """
        Plot a contingency between the feature and target

        :return: Matplotlib Axis object
        """

        return super().plot_contingency_table(self.feature, self.target)

    def plot_histogram(self):
        """
        Plot a histogram representing percentages of one categorical feature vs.
        another

        :return: Matplotlib Axis object
        """

        plot = super().plot_histogram(self.feature, self.target)
        plot.set_xlim(*self.data_range)
        return plot

    def plot_violin(self):
        """
        Plot a violin plot to show how a continuous variable varies with a binary target

        :return: Matplotlib Axis object
        """

        plot = super().plot_violin(self.feature, self.target)
        plot.set_ylim(*self.data_range)
        return plot

    def plot_box(self):
        """
        Plot a box plot to show how a continuous variable varies with a binary target

        :return: Matplotlib Axis object
        """

        plot = super().plot_box(self.feature, self.target)
        plot.set_ylim(*self.data_range)
        return plot

    def plot_regression(self):
        """
        Plot a regression for when we try to fit the target using the single feature

        :return: Matplotlib Axis object
        """

        plot = super().plot_regression(self.feature, self.target)
        plot.set_xlim(*self.data_range)
        return plot
