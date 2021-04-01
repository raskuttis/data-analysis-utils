"""
Script to handle the burnoff of data from a given file server or directory
"""

# Module imports
import logging
import traceback
import argparse
import datetime
import json
import os

from utilities.file_utilities import DirectoryFileHandler
from utilities.general_utilities import parse_config

def parse_args():
    """
    Parse the full list of input arguments and set up logging
    """
    parser = argparse.ArgumentParser(description=("Script to handle the burnoff of ",
                                                  "data from a directory"))
    parser.add_argument("-l", "--logFile", action="store", dest="log_file", required=False,
                        default=None, help="Filename for the output logs")
    parser.add_argument("-ll", "--logLevel", action="store", dest="log_level", required=False,
                        default="INFO", help="Level at which to output logs",
                        choices=["INFO", "DEBUG"])
    parser.add_argument("-m", "--mode", help="Whether to count or delete files",
                        choices=["count", "delete"])
    parser.add_argument("-d", "--deleteMode", help="Whether to remove the files or archive them",
                        dest="delete_mode", choices=["delete", "copy", "move", "zip"])
    parser.add_argument("-w", "--retentionWindow", action="store", dest="retention_window",
                        required=True,
                        default=None, help="Length of time to retain data e.g. 180d")
    parser.add_argument("-p", "--directories", help="List of directory paths to delete from",
                        nargs="+", default=None, required=True)
    parser.add_argument("-n", "--newDirectories", help="List of new directories to archive files to",
                        dest="new_directories", nargs="+", default=None)
    args = parser.parse_args()
    if args.delete_mode in ["copy", "move", "zip"] \
        and (len(args.directories) != len(args.new_directories)):
        parser.error("For non-delete actions we need to specify an \
                      archive directory for each input directory")

    # Logging config
    log_format = "%(asctime)s %(levelname)s {%(module)s} [%(funcName)s] %(message)s"
    if args.log_level == "INFO":
        logging.basicConfig(filename=args.log_file, level=logging.INFO,
                            format=log_format, datefmt="%Y-%m-%d %H:%M:%S")
    elif args.log_level == "DEBUG":
        logging.basicConfig(filename=args.log_file, level=logging.DEBUG,
                            format=log_format, datefmt="%Y-%m-%d %H:%M:%S")
    else:
        logging.error("Invalid logging level provided")
        raise ValueError("Invalid logging level provided")

    return args

def convert_retention_to_date(retention_window, time_format="%Y/%m/%d %H:%M:%S"):
    """
    Function to convert an input retention window e.g. 1d to a date

        Args:
            retention_window (str): Time window (in units like 1d, 5w) to go backwards
                from the current date
            time_format (str): datetime timeformat to output the final date in

        Returns:
            str: Starting date of the retention window in the format defined by
                time_format
    """

    conversion_units = {"s":"seconds", "m":"minutes", "h":"hours",
                        "d":"days", "w":"weeks"}

    retention_unit = conversion_units.get(retention_window[-1], None)
    if retention_unit:
        retention_count = int(retention_window[:-1])
        tdiff = datetime.timedelta(**{retention_unit: retention_count})
        current_date = datetime.datetime.now()
        start_date = (current_date - tdiff)
        if time_format == "ms":
            start_date = int((start_date - datetime.datetime(1970, 1, 1)).total_seconds() * 1000)
        else:
            start_date = start_date.strftime(time_format)
    else:
        logging.info("Invalid format for retention window %s provided, unit must be in %s",
                     retention_window, json.dumps(conversion_units))
        raise ValueError("Invalid date format for retention window provided")

    return start_date

def main(args):
    """
        Main Function
    """
    
    # Overarching date restriction queries from retention window
    start_date = convert_retention_to_date(args.retention_window, time_format="ms")

    # Iterate through the list of directories and delete from them
    for i, directory in enumerate(args.directories):
        dir_handler = DirectoryFileHandler(directory)
        if args.mode == "count":
            dir_handler.count_files(upper_st_mtime=start_date/1000.0)
        elif args.mode == "delete":
            if args.delete_mode == "delete":
                dir_handler.remove_files(upper_st_mtime=start_date/1000.0)
            elif args.delete_mode in ["copy", "move"]:
                new_directory = args.new_directories[i]
                dir_handler.move_files(new_directory, args.delete_mode,
                                       upper_st_mtime=start_date/1000.0)
            elif args.delete_mode in ["zip"]:
                new_directory = args.new_directories[i]
                current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
                tar_foldername, tar_filename = os.path.split(new_directory)
                tar_file_path = os.path.join(tar_foldername,
                                             "%s_%s.tar.gz" % (tar_filename, current_time))
                dir_handler.zip_files(tar_file_path,
                                      upper_st_mtime=start_date/1000.0)

if __name__ == "__main__":
    try:
        INPUT = parse_args()
        main(INPUT)
    except Exception as err:
        ERR_TRACEBACK = "; ".join(traceback.format_exc().split("\n"))
        logging.error("Exception: Function failed due to error %s with exception info %s",
                      err, ERR_TRACEBACK)
        raise ValueError("Exception: Function failed")
