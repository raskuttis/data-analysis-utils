"""
    Classes to handle the manipulation of files, moves, deletions etc. in as
    efficient a way as possible
"""

# Module imports
import logging
import os
import shutil
import datetime
import sys
import traceback
import json
import tarfile
import re

class LockFile(object):
    """
    Context manager that writes a lock file when open, and deletes it when closed
    """

    def __init__(self, lockfile, max_execution_seconds=None, process="Process"):
        """
        Init method for lockfile
        """
        self.lockfile = lockfile
        self.time_started = datetime.datetime.now()
        self.max_run_time = max_execution_seconds
        self.process = process
        self._do_cleanup = True

    def __enter__(self):
        """
        Check if the file already exist, if so exit the application
        """
        if os.path.isfile(self.lockfile):
            logging.info("New instance of %s started - Lockfile exists - exiting", self.process)
            self._do_cleanup = False
            sys.exit(1)
        else:
            with open(self.lockfile, "w") as lckfile:
                lckfile.write("%s locked since %s" % (self.process, str(self.time_started)))
            return self

    def check_timeout(self):
        """
        Check if the time since initiation has been exceeded
        """
        running_time = int((datetime.datetime.now() - self.time_started).total_seconds())
        if self.max_run_time:
            if running_time > self.max_run_time:
                logging.info("%s max run time %d seconds has been exceeded by %d seconds - exiting",
                             self.process, self.max_run_time, running_time - self.max_run_time)
                sys.exit(1)
            else:
                logging.info("%s has been running for %d seconds, %d left before max time - continuing",
                             self.process, running_time, self.max_run_time - running_time)
        else:
            logging.info("%s has been running for %d seconds and has no time limit - continuing",
                         self.process, running_time)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._do_cleanup:
            os.remove(self.lockfile)

class SingleFileHandler(object):
    """
        Class to handle the manipulation of files, moves, deletions etc. in as
        efficient a way as possible
    """

    def __init__(self, filename):
        """
        Init method for FileHandler. Primarily just maps the input file
        to a set of location
        """

        # Initialize list of files
        try:
            self.filename = filename
            self.file_locations = set([filename])
        except:
            logging.error("Input Filenames not passed in compatible format")
            raise ValueError("Input Filenames not passed in compatible format")

    def remove_file(self):
        """
        Deletes an input file from the original location
        """
        if os.path.isfile(self.filename):
            try:
                os.remove(self.filename)
                logging.debug("Successfully deleted file %s", self.filename)
            except Exception as err:
                logging.error("Failed to delete file %s with error %s",
                              self.filename, err)
        else:
            logging.debug("File %s doesn't exist", self.filename)
        # Update the location info
        self.file_locations.remove(self.filename)

    def copy_file(self, new_filename):
        """
        Copies an input file from the original location to a new location
        """
        if os.path.isfile(self.filename):
            try:
                if not os.path.exists(os.path.dirname(new_filename)):
                    logging.debug("Creating directory at %s",
                                  os.path.dirname(new_filename))
                    os.makedirs(os.path.dirname(new_filename))
                shutil.copyfile(self.filename, new_filename)
                logging.debug("Successfully copied file %s to %s",
                              self.filename, new_filename)
                self.file_locations.add(new_filename)
            except Exception as err:
                logging.error("Failed to copy file %s to %s with error %s",
                              self.filename, new_filename, err)
        else:
            logging.debug("File %s doesn't exist", self.filename)

    def move_file(self, new_filename):
        """
        Moves an input file from the original location to a new location
        """
        if os.path.isfile(self.filename):
            try:
                if not os.path.exists(os.path.dirname(new_filename)):
                    logging.debug("Creating directory at %s",
                                  os.path.dirname(new_filename))
                    os.makedirs(os.path.dirname(new_filename))
                shutil.move(self.filename, new_filename)
                logging.debug("Successfully moved file %s to %s",
                              self.filename, new_filename)
                self.file_locations.add(new_filename)
                self.file_locations.remove(self.filename)
            except Exception as err:
                logging.error("Failed to move file %s to %s with error %s",
                              self.filename, new_filename, err)
        else:
            logging.debug("File %s doesn't exist", self.filename)

    def search_file(self, regex_match, capture_group=0):
        """
        Searches over an input file's contents and extracts strings that
        match the given regular expression, returning them as a list
        """
        if os.path.isfile(self.filename):
            try:
                matches = []
                for line in open(self.filename, "r"):
                    for match in re.finditer(regex_match, line):
                        matches += [match.group(capture_group)]
                return matches
            except Exception as err:
                logging.error("Failed to move file %s to %s with error %s",
                              self.filename, new_filename, err)
        else:
            logging.debug("File %s doesn't exist", self.filename)

        return []

    def get_status_from_locations(self):
        """
        Gets the status of a file moved, deleted or copied from the location set
        """
        if not self.file_locations:
            return "deleted"
        elif len(self.file_locations) == 1:
            if self.filename == list(self.file_locations)[0]:
                return "static"
            return "moved"
        return "copied"

class DirectoryFileHandler(object):
    """
        Class to handle the manipulation of files, moves, deletions etc. for all
        files within a directory

        Attributes:
            directory_path: Path to the file directory
            directory_files: Dictionary mapping filenames to their creation dates
                and sizes
            valid_details: List of valid file details (i.e. attributes that exist
                on os_stat

    """

    def __init__(self, directory_path, recursive=False):
        """
        Init method for DirectoryHandler. Checks if the directory exists and gets
        a list of file details

            Args:
                directory_path (str): Path to the file directory
                recursive (bool): Whether to find files recursively or not

            Attribute Updates:
                directory_path, directory_files, valid_details
        """

        # Initialize the directory path
        self.directory_path = directory_path
        self.recursive = recursive
        if not os.path.isdir(self.directory_path):
            logging.error("Input Directory Path %s does not exist", self.directory_path)
            raise ValueError("Input Directory Path doesn't exist")

        # And get the details for all files in the directory
        self.valid_details = set()
        self.directory_files = self.get_file_details()

    def get_file_details(self):
        """
        Gets all the files within the directory and returns their details

            Returns:
                dict: Mapping from filename to a dictionary containing size,
                creation times and modification times

            Attribute Updates:
                valid_details

        """

        # Walk through the files in the directory and return a mapping from filename
        # to date and size
        directory_files = {}
        for root, dirs, files in os.walk(self.directory_path):
            for name in files:
                file_path = os.path.join(root, name)
                file_details = os.stat(file_path)
                file_dict = dict((detail_name, getattr(file_details, detail_name))
                                 for detail_name in dir(file_details)
                                 if not detail_name.startswith("__"))
                self.valid_details = self.valid_details.union(set(file_dict.keys()))
                directory_files[file_path] = file_dict
            # Only looking at top level if we aren't recursive
            if not self.recursive:
                break

        logging.info("Found %d files in directory %s", len(directory_files),
                     self.directory_path)

        return directory_files

    def parse_file_criteria(self, criteria, criteria_value):
        """
        Parses out a single matching criteria for files

            Args:
                criteria (str): This must be a string of the form
                    condition_filedetail, where condition is lower, upper
                    or equal and filedetail is any file parameter defined
                    in https://docs.python.org/3/library/os.html#os.stat_result
                criteria_value (int): For the given criteria, specifies a
                    value for which that should be compared against. E.g. for
                    criteria = lower_st_size and criteria_value = 1000, we'd
                    be looking for files larger than 1000 bytes

            Returns:
                matching_files (list): List of filepaths that match the
                criteria given
        """

        # Split out the criteria into it's two components
        split_criteria = criteria.split("_")
        if len(split_criteria) < 2:
            logging.error("Invalid criteria %s specified", criteria)
            return []

        # Check that the condition is in the right format
        condition = split_criteria[0]
        if condition not in ["lower", "upper", "equal"]:
            logging.error("Invalid criteria %s, condition must be one of " \
                          "lower, upper, equal")
            return []

        # Check that the file stat is a valid one
        file_detail = "_".join(split_criteria[1:])
        if file_detail not in self.valid_details:
            logging.error("Invalid criteria %s, file stat must be one of %s",
                          criteria, self.valid_details)
            return []
        
        # Iterate through the list of filenames and find those that match
        matching_files = []
        for file_path, file_dict in self.directory_files.items():
            file_value = file_dict.get(file_detail)
            if file_value:
                if (condition == "lower" and file_value > criteria_value) or \
                   (condition == "upper" and file_value < criteria_value) or \
                   (condition == "equal" and file_value == criteria_value):
                    matching_files += [file_path]

        return matching_files

    def count_files(self, **kwargs):
        """
        Counts input files from the directory, which match the set of input
        criteria

            Args:
                kwargs: Set of criteria, each of which must bot the form
                    criteria=criteria_value, where criteria is a string of the form
                    condition_filedetail, where condition is lower, upper
                    or equal and filedetail is any file parameter defined
                    in https://docs.python.org/3/library/os.html#os.stat_result
            
            Returns:
                matching_files (list): A list containing all files that match the
                    set of input criteria
        """

        # Iterate through the passed criteria to get a set of matching files
        matching_files = set()
        for criteria, criteria_value in kwargs.items():
            matching_files = matching_files.union(set(self.parse_file_criteria(criteria,
                                                                               criteria_value)))
        matching_files = list(matching_files)

        logging.info("Found %d of %d matching files from %s",
                     len(matching_files), len(self.directory_files), self.directory_path)
        logging.debug("Found the following matching files from %s: %s", self.directory_path,
                      json.dumps(matching_files))
        
        return matching_files


    def remove_files(self, **kwargs):
        """
        Deletes input files from the directory, which match the set of input
        criteria

            Args:
                kwargs: Set of criteria, each of which must bot the form
                    criteria=criteria_value, where criteria is a string of the form
                    condition_filedetail, where condition is lower, upper
                    or equal and filedetail is any file parameter defined
                    in https://docs.python.org/3/library/os.html#os.stat_result
        """

        matching_files = self.count_files(**kwargs)

        # Iterate through the files one by one and remove them
        logging.info("Deleting %d matching files from %s", len(matching_files),
                     self.directory_path)
        for file_path in matching_files:
            file_handler = SingleFileHandler(file_path)
            file_handler.remove_file()

    def move_files(self, new_directory, action, **kwargs):
        """
        Moves input files from the directory to a new directory, which match the set of input
        criteria

            Args:
                new_directory (str): Path to the new directory into which files
                    will be moved
                action (str): Whether to copy or move the files to the new
                    directory
                kwargs: Set of criteria, each of which must bot the form
                    criteria=criteria_value, where criteria is a string of the form
                    condition_filedetail, where condition is lower, upper
                    or equal and filedetail is any file parameter defined
                    in https://docs.python.org/3/library/os.html#os.stat_result
        """

        # Check whether the action is valid:
        if action not in ["copy", "move"]:
            logging.error("Invalid file action %s provided, so aborting", action)
            raise ValueError("Invalid file action provided")

        # Count the files to move
        matching_files = self.count_files(**kwargs)

        # Iterate through the files one by one and move them
        logging.info("Moving %d matching files from %s to %s", len(matching_files),
                     self.directory_path, new_directory)
        for file_path in matching_files:
            file_path_split = os.path.split(file_path)
            new_path = os.path.join(new_directory, file_path_split[1])
            file_handler = SingleFileHandler(file_path)
            if action in ["move"]:
                file_handler.move_file(new_path)
            elif action == "copy":
                file_handler.copy_file(new_path)

    def zip_files(self, tar_name, remove=True, **kwargs):
        """
        Zips input files from the directory to a new tarball, which match the set of input
        criteria

            Args:
                tar_name (str): Path to the new tarball we're creating
                remove (bool): Whether or not to remove the files after adding them
                kwargs: Set of criteria, each of which must bot the form
                    criteria=criteria_value, where criteria is a string of the form
                    condition_filedetail, where condition is lower, upper
                    or equal and filedetail is any file parameter defined
                    in https://docs.python.org/3/library/os.html#os.stat_result
        """

        # Count the files to move
        matching_files = self.count_files(**kwargs)

        # Iterate through the files one by one and move them
        logging.info("Zipping %d matching files from %s into %s", len(matching_files),
                     self.directory_path, tar_name)
        with tarfile.open(tar_name, mode="w:gz") as tarball:
            for file_path in matching_files:
                file_name = os.path.relpath(file_path, self.directory_path)
                tarball.add(file_path, arcname=file_name)
                if remove:
                    file_handler = SingleFileHandler(file_path)
                    file_handler.remove_file()

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
