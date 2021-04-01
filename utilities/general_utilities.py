"""
    Set of simple and very general utility functions for e.g. parsing yml
    config files
"""

# Module imports
from __future__ import absolute_import
from builtins import str
import collections
import datetime
import json
import logging
import os.path
import re
import traceback
import yaml
from yaml import load, dump
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
from yaml import load, dump

import six
from six.moves import range

# Global Default log formatter 
DEFAULT_LOG_FORMAT = '%(asctime)s [%(name)s] [%(levelname)s]  %(message)s'

'''
    FUNCTIONS
'''

# Set up YML representation of Ordered Dicts
def yml_dict_representer(dumper, data):
    """
    Ordered Dict Representer for YML
    """
    return dumper.represent_dict(six.iteritems(data))

def yml_dict_constructor(loader, node):
    """
    Ordered Dict Constructor for YML
    """
    return collections.OrderedDict(loader.construct_pairs(node))

def yml_nested_update(orig_dict, new_dict):
    """
    Function to update a nested yml without overwriting relevant keys
    """
    # Only update if new_dict and orig_dict are defined
    if new_dict:
        for key, val in new_dict.items():
            if isinstance(val, collections.Mapping):
                tmp = yml_nested_update(orig_dict.get(key, collections.OrderedDict()), val)
                orig_dict[key] = tmp
            elif isinstance(val, list):
                orig_dict[key] = (orig_dict.get(key, []) + val)
                orig_dict[key] = list(set(orig_dict.get(key, [])))
            else:
                orig_dict[key] = new_dict[key]
    return orig_dict

def parse_config(config_files, return_args=None, essential_keys=None):
    """
    Parse the config file into the full list of relevant arguments
    """

    # Make sure YML only reads in ordered dicts
    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    yaml.add_representer(collections.OrderedDict, yml_dict_representer)
    yaml.add_constructor(_mapping_tag, yml_dict_constructor)

    # Load YAML config file
    logging.info("Reading in configuration parameters from %s",
                 json.dumps(config_files))

    yml = collections.OrderedDict()
    for config_file in config_files:
        with open(config_file, "r") as yml_file:
            yml_update = yaml.load(yml_file)
            if yml_update:
                yml = yml_nested_update(yml, yml_update)

    # List of necessary keys to verify
    if not essential_keys:
        essential_keys = {}

    logging.info("Verifying that all essential config is contained in %s",
                 json.dumps(config_files))
    for config_key, key_list in six.iteritems(essential_keys):
        try:
            _ = yml[config_key]
        except Exception as err:
            logging.error("No %s arguments detected in config files %s",
                          config_key, json.dumps(config_files))
            raise ValueError("Exiting with error %s" % err)
        for sub_key in key_list:
            try:
                _ = yml[config_key][sub_key]
            except Exception as err:
                logging.error("No %s: %s arguments detected in config files %s",
                              config_key, sub_key, json.dumps(config_files))
                raise ValueError("Exiting with error %s" % err)
    logging.info("Verified that all essential config is contained in %s",
                 json.dumps(config_files))

    # Add the YAML output to the arguments if desired
    if return_args:
        # Iterate over arguments and add to them from config file
        arg_dict = return_args.__dict__
        for key, value in list(yml.items()):
            if isinstance(value, list):
                for sub_val in value:
                    arg_dict[key].append(sub_val)
            else:
                arg_dict[key] = value
        return return_args
    else:
        return yml

def write_jsonl(out_list, fname, write_flag="w"):
    """
    Writes a list of jsons to a file as a jsonl
    """
    outfile = fname
    logging.info("Writing %d lines to file %s, with method %s",
                 len(out_list), outfile, write_flag)
    with open(outfile, write_flag) as merged:
        for event in out_list:
            json.dump(event, merged)
            merged.write("\n")

    return None

def read_jsonl(fname):
    """
    Reads a jsonl file to a list of jsons as a generator
    """

    outfile = fname
    out_lines = 0
    logging.info("Reading lines from file %s", fname)
    with open(outfile) as merged:
        for line in merged:
            out_lines += 1
            yield json.loads(line)
    logging.info("Read %d lines from file %s", out_lines, fname)

def date_range(t_start, t_end, input_format="%Y-%m-%dT%H:%M:%S",
               output_format="%Y-%m-%dT%H:%M:%S", interval={"hours": 1},
               offset=None):
    """
    Generates a list of dates between the start and end date, with the given interval
    """
    # List of dates
    dates = {}

    # Starting and end times as datetimes
    if input_format not in ["datetime", "unix"]:
        date_start = datetime.datetime.strptime(t_start, input_format)
        date_end = datetime.datetime.strptime(t_end, input_format)
    elif input_format == "unix":
        date_start = datetime.datetime.fromtimestamp(t_start)
        date_end = datetime.datetime.fromtimestamp(t_end)
    else:
        date_start = t_start
        date_end = t_end
    logging.info("Generating datetime series between %s and %s",
                 date_start.strftime(output_format),
                 date_end.strftime(output_format))

    # Iterate over date variable, augmenting as we go
    date_current = date_start
    while date_current < date_end:
        if offset:
            date_offset = date_current + datetime.timedelta(**offset)
        else:
            date_offset = date_current
        if output_format != "datetime":
            date_current_str = date_current.strftime(output_format)
            date_offset_str = date_offset.strftime(output_format)
            dates[date_offset_str] = date_current_str
        else:
            dates[date_offset] = date_current
        date_current += datetime.timedelta(**interval)
    logging.info("Generated %d datetime values between %s and %s",
                 len(dates), date_start.strftime(output_format),
                 date_end.strftime(output_format))

    return dates

def convert_timestamp(string_time, input_format="%Y-%m-%dT%H:%M:%S",
                      output_format="ms"):
    """
    Converts an input string timestamp to the specified output
    """

    # Get timestamp as datetime
    timestamp = datetime.datetime.strptime(string_time, input_format)

    if output_format == "ms":
        return int((timestamp - datetime.datetime(1970, 1, 1)).total_seconds() * 1000)
    else:
        logging.error("Invalid output format %s provided", output_format)
        raise ValueError("Invalid output format provided")

def round_timestamp(string_time, input_format="ms",
                    round_ms=None):
    """
    Converts an input string timestamp in the specified format to a version
    rounded to the number of milliseconds specified in round_ms
    """

    # Convert the input if the input_format isn't ms
    if input_format != "ms":
        int_time = convert_timestamp(string_time, input_format=input_format)
    else:
        int_time = int(string_time)

    # Just return the converted time if no rounding is provided
    if not round_ms:
        return int_time

    # Otherwise calculate the quotient and remainder
    quotient, remainder = divmod(int_time, round_ms)

    # Augment the quotient by 1 if the remainder is more than half the rounding
    if remainder > round_ms / 2:
        return (quotient + 1) * round_ms
    else:
        return quotient * round_ms

def valid_date_fmt(value, fmt='%Y-%m-%dT%H:%M:%S.%fZ'):
    """
    Check if a string matches a specific datetime format
    Returns boolean result
    """
    try:
        dt = datetime.datetime.strptime(str(value), fmt)
    except:
        return False
    return True

def replace_json_values(input_json, replace_key=None, replace_val=None):
    """
    Replaces all the values in a JSON indexed by the replacement key with
    the replacement value
    """
    logging.debug("Replacing values indexed by %s with %s", replace_key,
                  json.dumps(replace_val))
    for input_key, input_value in list(input_json.items()):
        if input_key == replace_key:
            input_json[input_key] = replace_val
        elif type(input_value) == type(dict()):
            replace_json_values(input_value, replace_key=replace_key, replace_val=replace_val)
        elif type(input_value) == type(list()):
            for val in input_value:
                if type(val) == type(dict()):
                    replace_json_values(val, replace_key=replace_key, replace_val=replace_val)
                else:
                    pass
        else:
            pass

def strip_tags(text):
    """
    Strips html tags from the input string
    """
    ## Remove full tags
    try:
        rem_tags = re.sub("<[^<]+?>", "", text)
    except TypeError:
        rem_tags = re.sub("<[^<]+?>", "", text.decode("utf-8"))

    ## Remove broken tag from start of string
    split_first = rem_tags.split(">")
    if len(split_first) > 1:
        rem_first = ' '.join(split_first[1:])
    else:
        rem_first = split_first[0]

    ## Remove broken tag from end of string
    split_last = rem_first.split("<")
    if len(split_last) > 1:
        rem_last = ' '.join(split_last[:-1])
    else:
        rem_last = split_last[0]
        
    return rem_last

def configure_basic_logger(log_dir, log_filename_prefix, timestamp, enable_console=True, log_name='logger',
                           log_level=logging.DEBUG, log_formatter_string=DEFAULT_LOG_FORMAT,
                           log_retention_number=-1):
    '''Configures a basic logger object and returns it

    :param log_dir: the directory to write log files to
    :type log_dir: str
    :param log_filename_prefix: the prefix for the file to write logs to
    :type log_filename_prefix: str
    :param timestamp: timestamp to use at the end of the log file name
    :type timestamp: str
    :param enable_console: whether or not to write logs to the console
    :type enable_console: bool
    :param log_name: name of the logger object (optional)
    :type log_name: str
    :param log_level: logging level (e.g. logging.DEBUG, logging.INFO)
    :type log_level: str
    :param log_formatter_string: the string representing the log format
    :type log_formatter_string: str
    :param log_retention_number: the number of log files to retain including current log file;
        if negative, retains all
    :type log_retention_number: int
    :return: newly configured logger
    :rtype: logger
    '''

    # Instantiate logger
    root_logger = logging.getLogger(log_name)
    root_logger.setLevel(log_level)

    # Configure log file handler
    log_file_path = os.path.join(log_dir, log_filename_prefix + timestamp + '.log')
    log_file_handler = logging.FileHandler(log_file_path)
    log_file_handler.setFormatter(logging.Formatter(log_formatter_string))
    root_logger.addHandler(log_file_handler)

    # If enabled, configure console log handler
    if enable_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(log_formatter_string))
        root_logger.addHandler(console_handler)

    # Logger Configured
    root_logger.info('Logger configured - ' + log_file_path)

    # If log retention number is set to 0, change to 1 so current log file is retained
    if log_retention_number == 0:
        root_logger.warning('    Logger retention number cannot be 0. Defaulting to 1 to include current log')
        log_retention_number = 1

    # If log_retention_number is not negative, feature is enabled
    # Delete log files but retain the most recent ones up to the number
    if log_retention_number >= 0:
        root_logger.info('    Logger set to retain most recent ' + str(log_retention_number) + ' log file(s). Deleting the rest...')
        with rose_file_handler(log_dir, root_logger) as log_file_handler:
            log_file_handler.delete_directory_files(log_filename_prefix + '*', log_retention_number)
    else:
        root_logger.info(
            '    Logger set to retain all existing log file(s).')

    # Return the logger object
    return root_logger

def is_empty(obj):
    '''Returns true if object is empty or not defined

    :param obj: object to check
    :type obj: object
    :return: true if empty or not defined
    :rtype: bool
    '''
    return obj is None \
           or (isinstance(obj, six.string_types) and obj.strip() == '') \
           or (isinstance(obj, six.string_types) and obj.strip() == 'NULL') \
           or (isinstance(obj, collections.Iterable) and not obj) \
           or (isinstance(obj, dict) and not obj)

def normalize_string(a):
    '''Converts string into normal format:
        converts string to lowercase with no tabs or trailing spaces;
        converts numbers to strings

    :param a:
    :type a:
    :return:
    :rtype:
    '''
    if isinstance(a, six.string_types) or isinstance(a, str):
        return a.replace('\t', ' ').strip().lower()
    elif isinstance(a, int) or isinstance(a, float):
        return str(a)
    else:
        return None

def normalize_date(date, formats):
    '''Convert date to iso format

    :param date: attribute date value
    :type date: str
    :return: normalized date
    :rtype: str
    '''
    value = date
    for format in formats:
        try:
            value = datetime.datetime.strptime(value, format).isoformat()
            break
        except:
            pass
    else:
        raise ValueError("Could not convert [" + date + "] to iso format.")
    return value

def start_timer():
    '''Starts a timer for recording elapsed time

    :return: returns current time
    :rtype: datetime
    '''
    return datetime.datetime.now().replace(microsecond=0)


def get_elapsed(prefix, start):
    '''Returns message with elapsed time prefixed with the provided text

    :param prefix: Text to prefix the message
    :type prefix: str
    :param start: start time
    :type start: datetime
    :return: Elapsed time message
    :rtype: str
    '''
    return prefix + str(datetime.datetime.now() - start)

def merge_dicts(x, y, overwrite_lists=False):
    '''Merges two dictionaries
       If there are common keys with values that aren't lists, the values are overwritten using y's values
       If there are common keys and overwrite_lists = True, then the lists are overwritten using y's lists
       If there are common keys and overwrite_lists = False, then the lists are a combination of x's and y's lists

    :param x: first dictionary
    :type x: dict
    :param y: second dictionary
    :type y: dict
    :param union_lists: True if
    :return: merged dictionary
    :rtype: dict
    '''
    z = x.copy() # start with x's keys and values
    if overwrite_lists:
        z.update(y)  # modify z with y's keys and values
    else:
        # Combine all keys in x and y
        all_keys = list(set().union(list(x.keys()), list(y.keys())))

        # Loop through all keys
        # If it's in both dicts and the values are lists, then merge them
        # Otherwise, overwrite with y
        for key in all_keys:
            if key in x and key in y and x[key] and isinstance(x[key], list) and isinstance(y[key], list):
                merged_list = x[key] + y[key]
                z[key] = merged_list
            elif key in y:
                z[key] = y[key]
    return z

def main():
    """
        Main Function, which should contain a bunch of test functions for this module
    """

if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        ERR_TRACEBACK = "; ".join(traceback.format_exc().split("\n"))
        logging.error("Exception: Function failed due to error %s with exception info %s",
                      err, ERR_TRACEBACK)
