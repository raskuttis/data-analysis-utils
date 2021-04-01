"""
    Classes to take in a set of historical time series data for an individual, and a
    group and then build out a set of statistics for how anomalous the latest data point is
"""

# Module imports
import logging
import json
import traceback
import hashlib
import datetime
import pandas as pd

class IndividualProfile(object):
    """
    Class to take in a set of historical time series data for an individual, and
    then build out a set of statistics for how anomalous the latest data point is

        Attributes:
            latest_value (double): The latest value in the time series, for which we
                want to generate statistics
            time_series (Series): List of historical values in regularly spaced intervals
            profile_statistics (dict): Mapping from statistic name to computed
                statistic values
    """

    def __init__(self, latest_value, time_series):
        """
        Init method for Individual Profile. Just takes in the time series and the latest
        value

            Args:
                latest_value (double): The latest value in the time series, for which we
                    want to generate statistics
                time_series (list): List of historical values in regularly spaced intervals
                profile_statistics (dict): Mapping from statistic name to computed
                    statistic values

            Attribute Updates:
                latest_value, time_series, profile_statistics
        """

        # Initialize dict of Body Hashes
        self.latest_value = latest_value
        self.time_series = pd.Series(time_series)
        self.profile_statistics = {"Actual Value": latest_value}

    def build_profile(self, profile_specs):
        """
        Given a dictionary mapping profile names to functions and arguments, builds
            out the profile statistics dictionary

            Args:
                profile_specs (dict): Dictionary mapping statistics names to functions
                    and arguments for those functions

            Attribute Updates:
                profile_statistics

            Returns:
                Dict: Profile Statistics dictionary
        """

        # Iterate through statistic names in the input config
        for statistic_name, statistic_details in profile_specs.items():
            method_name = "profile_%s" % statistic_details.get("function", "")
            apply_method = getattr(self, method_name, None)
            arguments = statistic_details.get("arguments", {})
            if apply_method:
                logging.debug("Building out statistic using %s", method_name)
                self.profile_statistics[statistic_name] = apply_method(**arguments)
            else:
                logging.error("No valid statistic found for %s", method_name)

        return self.profile_statistics

    def profile_zscore(self, n_historical_values=None):
        """
        Function to calculate the z-score for the latest value compared to the historical
            time series

            Args:
                n_historical_values (int): Number of historical values to use to compute
                    the z-score. If no value is passed, we use all the data

            Returns:
                double: z-score
        """

        if n_historical_values:
            comparison_data = self.time_series[-n_historical_values:]
        else:
            comparison_data = self.time_series
        data_stddev = comparison_data.std()
        data_mean = comparison_data.mean()
        if data_stddev != 0:
            zscore = (self.latest_value - data_mean) / data_stddev
        else:
            zscore = 0
        
        return zscore

    def profile_ismax(self, n_historical_values=None):
        """
        Function to calculate whether the current value is a maximum compared to the rest
            of the time series

            Args:
                n_historical_values (int): Number of historical values to use to compute
                    the z-score. If no value is passed, we use all the data

            Returns:
                int: 1 or 0 for whether this point is a maximum now
        """

        if n_historical_values:
            comparison_data = self.time_series[-n_historical_values:]
        else:
            comparison_data = self.time_series
        if self.latest_value > comparison_data.max():
            return 1
        
        return 0

    def profile_abovepercentile(self, n_historical_values=None, percentile=0.99):
        """
        Function to calculate whether the current value is larger than a given percentile of
            the historical values

            Args:
                n_historical_values (int): Number of historical values to use to compute
                    the z-score. If no value is passed, we use all the data

            Returns:
                int: 1 or 0 for whether this point is above the percentile or not
        """

        if n_historical_values:
            comparison_data = self.time_series[-n_historical_values:]
        else:
            comparison_data = self.time_series
        if self.latest_value > comparison_data.quantile(percentile):
            return 1
        
        return 0

def main():
    """
        Main Function for event_hashes. Should contain  a bunch of testing functions
    """

if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        ERR_TRACEBACK = "; ".join(traceback.format_exc().split("\n"))
        logging.error("Exception: Function failed due to error %s with exception info %s",
                      err, ERR_TRACEBACK)
