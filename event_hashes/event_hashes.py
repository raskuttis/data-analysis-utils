"""
    Class to store a deduplication dictionary of all events, where the keys are a unique
    hash of the deduplication fields and the timestamp, and the value is an event
    dictionary containing all the fields needed to create a merged event
"""

# Module imports
import logging
import json
import traceback
import hashlib
import datetime

from string_methods import Jaccard, LSH, SimpleHash

class AllEventHashes(object):
    """
    Class to store a deduplication dictionary of all events, where the keys are a unique
    hash of the deduplication fields and the timestamp, and the value is an event
    dictionary containing all the fields needed to create a merged event

        Attributes:
            dedup_fields (list): List of event fields that need to be considered identical,
                modulo the hashing method, for an event to be considered a duplicate
            return_fields (list): List of event fields that should be pulled down
                from ES (i.e. group fields and dedup fields combined)
            compare_fields (list): If we're comparing string distances between events, fields
                that should be used in the comparison
            string_measure (string_method): String method to use for comparing distances
                between events. Options are currently Jaccard and MinHash
            time_range (int): Time in ms that events need to be bucketed into to be considered
                duplicates
            group_att (str): Name of attribute to add to duplicate events to store the
                Group Hash
    """

    def __init__(self, dedup_fields=None, return_fields=None, compare_fields=None,
                 string_measure=None, time_range=5*60*1000, group_att="Dedupe Group ID"):
        """
        Init method for AllEventHashes. Initializes the bodyhash dictionary, the
        fields used for deduplication

            Args:
                body_hashes (dict): Nested dictionary of Hash ID -> Time Window -> List of
                    events matching both Hash ID and Time Window, which is recursively
                    updated as we add more events
                dedup_fields (list): List of event fields that need to be considered identical,
                    modulo the hashing method, for an event to be considered a duplicate
                return_fields (list): List of event fields that should be pulled down
                    from ES (i.e. group fields and dedup fields combined)
                compare_fields (list): If we're comparing string distances between events, fields
                    that should be used in the comparison
                string_measure (string_method): String method to use for comparing distances
                    between events. Options are currently Jaccard and MinHash and if unset defaults
                    to Jaccard
                time_range (int): Time in ms that events need to be bucketed into to be considered
                    duplicates
                group_att (str): Name of attribute to add to duplicate events to store the
                    Group Hash

            Attribute Updates:
                body_hashes, dedup_fields, return_fields, compare_fields, string_measure,
                time_range, group_att
        """

        # Initialize dict of Body Hashes
        self.body_hashes = {}

        # Initialize relevant event fields to compare based on
        self.dedup_fields = dedup_fields
        self.return_fields = return_fields
        self.compare_fields = compare_fields

        # Initialize string measure
        if self.compare_fields and not string_measure:
            self.string_measure = Jaccard()
        else:
            self.string_measure = string_measure

        # Initialize time range
        self.time_range = time_range
        self.group_att = group_att

        # Modify the dedup fields to account for hasher definitions
        #self.instantiate_field_hashers(self.dedup_fields)

    def hasher_from_string(self, input_var, method):
        """
        Given a hashing method string, returns an instantiated hasher

            Args:
                input_var (str): String field that we want to hash
                method (dict): Dictionary of method and method attributes (either LSH or
                    SimpleHash) to produce bodyhash

            Returns:
                Hash: Instantiated hasher for input string
        """

        # Make sure the input is a string
        if isinstance(input_var, list):
            input_string = " ".join(input_var)
        elif isinstance(input_var, (str, unicode)):
            input_string = input_var
        else:
            input_string = str(input_var)

        # If no method defined, then return default
        if not method:
            return SimpleHash(input_string)
        elif isinstance(method, dict):
            if len(method) > 1:
                logging.error("Can't define multiple hashing methods")
                raise ValueError("Can't define multiple hashing methods")
            else:
                method_name, method_values = method.items()[0]
                if method_name == "LSH":
                    if method_values:
                        return LSH(input_string, **method_values)
                    else:
                        return LSH(input_string)
                elif method_name == "SimpleHash":
                    return SimpleHash(input_string)
                else:
                    logging.error("Invalid dedup method %s provided", method_name)
                    raise ValueError("Invalid dedup method")
        elif isinstance(method, (str, unicode)):
            if method == "LSH":
                return LSH(input_string)
            elif method == "SimpleHash":
                return SimpleHash(input_string)
            else:
                logging.error("Invalid dedup method %s provided", method)
                raise ValueError("Invalid dedup method")
        else:
            logging.error("Dedup method provided in incorrect format")
            raise ValueError("Dedup method provided in incorrect format")

        return SimpleHash(input_string)

    def extract_field_hash(self, event_dict, field, event_id=""):
        """
        Extracts the value of a field from a dictionary, iterating to lower levels if needed
        and then hashes each of the fields individually

            Args:
                event_dict (dict): Event dictionary that comes from ElasticEvent.event
                field (str): Name of field we're extracting from event_dict
                event_id (str): ID of event represented by event_dict (used only for
                    logging)

            Returns:
                list: List of instantiated hashers for event fields in field
        """

        full_field = []
        # If we just have a NoneType then return Nothing
        if not field:
            return []
        # If we have a list of fields then iterate over each field in turn and concatenate
        if isinstance(field, list):
            for sub_field in field:
                full_field += self.extract_field_hash(event_dict, sub_field,
                                                      event_id=event_id)
        # Otherwise if it's a dict, then iterate over the key-value pairs
        elif isinstance(field, dict):
            for super_field, sub_fields in field.iteritems():
                # If the key is method, then compute the hash right now over what's left of
                # the event_dict
                if super_field == "method":
                    return [self.hasher_from_string(event_dict, sub_fields)]
                elif super_field in event_dict:
                    if sub_fields:
                        full_field += self.extract_field_hash(event_dict[super_field],
                                                              sub_fields,
                                                              event_id=event_id)
                    else:
                        full_field += self.extract_field_hash(event_dict, super_field,
                                                              event_id=event_id)
                else:
                    logging.warning("Field %s doesn't exist on event with ID %s",
                                    super_field, event_id)
        # Otherwise if it's a string then just return the hashed field value
        elif isinstance(field, (str, unicode)):
            if field == "method":
                return [self.hasher_from_string(event_dict, None)]
            elif field in event_dict:
                return [self.hasher_from_string(event_dict[field], None)]
            else:
                logging.warning("Field %s doesn't exist on event with ID %s",
                                field, event_id)
                return []
        else:
            logging.warning("Dedup fields have been entered in the wrong format")
            raise ValueError

        return full_field

    def bodyhash_from_event(self, input_event):
        """
        Creates a unique hash for each event based on the deduplication fields

            Args:
                input_event (ElasticEvent): Input event to calculate body hashes for

            Returns:
                list: List of instantiated hashers (SimpleHash or LSH) for each
                    of the fields in dedup_fields
        """
        body_hash = self.extract_field_hash(input_event.event, self.dedup_fields,
                                            event_id=input_event.event["id"])
        return body_hash

    def in_time_range(self, timestamp, prev_timestamp):
        """
        Returns true if the timestamp is within time_range amount of milliseconds before
        or after prev_timestamp

            Args:
                timestamp (datetime): Input timestamp to check
                prev_timestamp (datetime): Comparison timestamp to check whether timestamp
                    is within time_range of

            Returns:
                bool: True if timestamp is within time_range of prev_timestamp and
                    False otherwise
        """
        return (prev_timestamp - self.time_range) < timestamp < (prev_timestamp + self.time_range)

    def in_bodyhashes(self, bodyhash):
        """
        Checks if the input bodyhash is in the dictionary of existing bodyhashes

            Args:
                bodyhash (list): List of instantiated hashers (SimpleHash or LSH) for each
                    of the fields in dedup_fields

            Returns:
                str: Printed bodyhash for matching event or else None if there is no match
        """

        for printed_hash, hash_dict in self.body_hashes.iteritems():
            comparison_hash = hash_dict.get("event_hash")
            if len(comparison_hash) != len(bodyhash):
                logging.debug("Mismatching hash lengths so can't compare")
            elif all([field_hash_a == field_hash_b for (field_hash_a, field_hash_b)
                      in zip(bodyhash, comparison_hash)]):
                logging.debug("Detected hash match")
                return printed_hash

        return None

    def add_event(self, merge_event):
        """
        Adds an event to the bodyhash dictionary. If the bodyhash already exists, and
        its within the timestamp window then merge the return fields into the dictionary
        otherwise add a new entry for this event

            Args:
                merge_event (ElasticEvent): Input event to add into bodyhashes dictionary

            Attribute Updates:
                body_hashes
        """

        # Get bodyhash from the event
        bodyhash = self.bodyhash_from_event(merge_event)
        printed_hash = "".join([hasher.print_hash() for hasher in bodyhash])
        essential_keys = ["id", "timestamp"]
        for key in essential_keys:
            if key not in merge_event.event:
                logging.error("Key %s not found in event %s, so skipping",
                              key, json.dumps(merge_event.event))
                return

        event_id = merge_event.unique_id
        event_timestamp = merge_event.event["timestamp"]
        if bodyhash:
            matched_printed_hash = self.in_bodyhashes(bodyhash)
            new_event_hash = EventHash(bodyhash, merge_event, return_fields=self.return_fields,
                                       compare_fields=self.compare_fields,
                                       group_att=self.group_att)
            if matched_printed_hash:
                # If the bodyhash is already there then check to see whether the timestamp is
                # within the prescribed range
                for original_timestamp, dupe_event in self.body_hashes.get(matched_printed_hash,
                                                                           {}).get("times", {}).iteritems():
                    # If timestamps overlap and the events aren't identical, then add
                    # them as dupes
                    if original_timestamp != "event_hash":
                        if self.in_time_range(event_timestamp, original_timestamp):
                            if event_id in dupe_event.event_ids:
                                logging.error("Detected Event ID %s twice, something's gone wrong",
                                              event_id)
                                dupe_event.n_exact_dupes += 1
                                break
                            else:
                                dupe_event.add_dupe(merge_event,
                                                    string_measure=self.string_measure)
                                break
                else:
                    # Otherwise, if timestamps don't overlap, add the new timestamp
                    self.body_hashes[matched_printed_hash]["times"][event_timestamp] = new_event_hash

            else:
                # Otherwise, if bodyhash isn't there, add it
                self.body_hashes[printed_hash] = {"times": {event_timestamp: new_event_hash},
                                                  "event_hash": bodyhash}

    def get_distances(self):
        """
        Returns a dict containing all the event id pairs and the distances extracted from
        the compare fields

            Returns:
                dict: Dictionary of distances where the keys are the string event IDs separated
                    by dashes and the values are the string distances between the two events
        """
        distance_dict = {}
        for _, body_hash_dict in self.body_hashes.iteritems():
            for _, dupe_event in body_hash_dict.get("times", {}).iteritems():
                for dupe_event_id, dupe_event_dict in dupe_event.event_ids.iteritems():
                    for match_event_id, match_distance in dupe_event_dict["distances"].iteritems():
                        if match_event_id != dupe_event_id:
                            distance_dict["%s-%s" % (dupe_event_id,
                                                     match_event_id)] = match_distance

        return distance_dict

    def get_dupes(self):
        """
        Returns a list of dicts containing ID, type and index for all the events that have
        a duplicate

           Returns:
                list: List of Event IDs for all events that have duplicates
        """

        dupe_id_list = []
        for _, body_hash_dict in self.body_hashes.iteritems():
            for _, dupe_event in body_hash_dict.get("times", {}).iteritems():
                if dupe_event.n_dupes > 1:
                    dupe_id_list += [json.loads(event_id) for event_id in dupe_event.event_ids]

        return dupe_id_list


    def get_group_id_attribute(self):
        """
        Returns a list of dicts containing ID, type, index and group ID as an attribute
        for all events that have a duplicate. Useful for passing to update the attributes
        in Elastic

            Returns:
                list: List of dicts containing ID, type, index and group ID as an attribute
                    for all events that have a duplicate
        """
        dupe_id_list = []
        for _, body_hash_dict in self.body_hashes.iteritems():
            for _, dupe_event in body_hash_dict.get("times", {}).iteritems():
                if dupe_event.n_dupes > 1:
                    for event_id in dupe_event.event_ids:
                        dupe_id_dict = json.loads(event_id)
                        dupe_id_dict["attributes"] = [{"type": "string",
                                                       "name": self.group_att,
                                                       "value": dupe_event.group_id}]
                        dupe_id_list += [dupe_id_dict]

        return dupe_id_list

    def get_base_events(self):
        """
        Returns a dict containing the first unique id and the base event for all events
        that have a duplicate. The unique ID is necessary to fetch that event.

            Returns:
                dict: Dictionary of events where the keys are the string event IDs for
                    a given bodyhash and the values are the event JSONs
        """
        base_dict = {}
        for _, body_hash_dict in self.body_hashes.iteritems():
            for _, dupe_event in body_hash_dict.get("times", {}).iteritems():
                if dupe_event.n_dupes > 1:
                    base_dict[dupe_event.event_ids.keys()[0]] = dupe_event.base_event

        return base_dict

    def describe(self, verbose=False):
        """
        Method to return a string containing some summary information of the event hash
        dictionary

            Args:
                verbose (bool): Flag for whether or not we want the string to be verbose or
                    not

            Returns:
                str: String summarizing the total number of events, merged events and duplicates
                    in the BodyHashes. If verbose it also prints out the number of duplicates
                    per individual BodyHash
        """
        n_exact_dupes = 0
        n_bulk = 0
        n_merge = 0
        n_non_merge = 0
        n_total = 0
        describe_str = "\n\n"
        for _, body_hash_dict in self.body_hashes.iteritems():
            for _, dupe_event in body_hash_dict.get("times", {}).iteritems():
                n_exact_dupes += dupe_event.n_exact_dupes
                if dupe_event.n_dupes > 1:
                    n_merge += 1
                    n_bulk += dupe_event.n_dupes
                    n_total += dupe_event.n_dupes
                    if verbose:
                        describe_str += "Event with hash %s and timestamp %s has %d " \
                                        "duplicates\n" % (dupe_event.print_bodyhash(),
                                                          dupe_event.get_timestamp(),
                                                          dupe_event.n_dupes)
                else:
                    n_non_merge += 1
                    n_total += 1

        describe_str += "\nFrom %d total events analysed, we have %d merged events (consisting of" \
                        " %d subevents) and %d unmerged events, with %d exact " \
                        "duplicates dropped\n\n" % (n_total, n_merge, n_bulk, n_non_merge,
                                                    n_exact_dupes)

        return describe_str

class EventHash(object):
    """
    Class to store all the information necessary to create, track, merge and publish
    events with the same bodyhash

        Attributes:
            n_dupes (int): Number of duplicate events for the given bodyhash, where duplicates
                are defined by the input bodyhash type
            n_exact_dupes (int): Number of exact duplicate events (primarily
                for debugging since there should be none)
            base_event (ElasticEvent): First event that matches this BodyHash
            event_id (str): Event ID that corresponds to base event
            original_timestamp (str): Timestamp that corresponds to base event
            event_ids (dict): Dictionary of event IDs mapped to their string distances from
                one another based on the compare fields
            bodyhash (list): List of instantiated hashers (SimpleHash or LSH) for each
                of the fields in dedup_fields
            group_id (str): Unique ID generated by printing each of the hashers in bodyhash
            return_fields (list): List of fields to return from each event for merging
            compare_fields (list): If we're comparing string distances between events, fields
                that should be used in the comparison
    """

    def __init__(self, bodyhash, init_event, return_fields=None,
                 compare_fields=None, group_att="Dedupe Group ID"):
        """
        Init method for EventHash. Initializes a number of counters and lists to
        track the unique ES events that belong to the same bodyhash

            Args:
                bodyhash (list): List of instantiated hashers (SimpleHash or LSH) for each
                    of the fields in dedup_fields
                init_event (ElasticEvent): First event that matches this BodyHash
                return_fields (list): List of fields to return from each event for merging
                compare_fields (list): If we're comparing string distances between events, fields
                    that should be used in the comparison
                group_att (str): Name of attribute to add to duplicate events to store the
                    Group Hash

            Attribute Updates:
                body_hash, base_event, n_dupes, n_exact_dupes, event_id, original_timestamp,
                event_ids, group_id, return_fields, compare_fields
        """

        # Counters for the number of duplicates, any exact duplicates for debugging
        self.n_dupes = 1
        self.n_exact_dupes = 0

        # List of event IDs that correspond to this bodyhash and timestamp
        self.base_event = init_event
        event_id = self.base_event.unique_id
        self.event_ids = {event_id: None}

        # Unique characteristics of this bodyhash
        self.bodyhash = bodyhash
        self.original_timestamp = init_event.event["timestamp"]
        self.group_id = self.set_group_id()

        # Add group ID to the subject
        if group_att:
            self.base_event.event["subject"] = "%s (%s = %s)" % (self.base_event.event.
                                                                 get("subject", ""),
                                                                 group_att, self.group_id)
            # Add attribute to the event
            self.base_event.add_attribute("string", group_att, self.group_id)

        # Output event, which consists of the fields we want to merge for this event
        self.return_fields = return_fields

        # If there are comparison fields, then store them. This can currently only
        # be done on high level fields e.g. not roles
        self.compare_fields = compare_fields

        # If there are comparison fields, then initialize a storage structure for the
        # distances between differen events in the hash
        if self.compare_fields:
            self.event_ids[event_id] = {"fields": None, "distances": None}
            self.event_ids[event_id]["fields"] = self.get_compare_fields(init_event)
            self.event_ids[event_id]["distances"] = {event_id: 0}

    def print_bodyhash(self):
        """
        Prints out a single bodyhash from the hash which is a list of hashers

            Returns:
                str: Unique ID generated by printing each of the hashers in bodyhash
        """

        return "".join([hasher.print_hash() for hasher in self.bodyhash])

    def set_group_id(self):
        """
        Creates a unique ID for this bodyhash and timestamp based on the hash of
        timestamp and bodyhash

            Returns:
                str: Unique ID generated by printing each of the hashers in bodyhash joined
                    together with the original_timestamp
        """
        str_time = datetime.datetime.fromtimestamp(self.original_timestamp
                                                   /1000).strftime("%Y-%m-%dT%H:%M:%S")
        group_id = self.print_bodyhash() + str_time
        return hashlib.sha1(group_id).hexdigest()

    def get_compare_fields(self, merge_event):
        """
        Extracts the string comparison fields from the input merge_event

            Args:
                merge_event (ElasticEvent): Input event to compare

            Returns:
                str: Join of all the values in the input merge_event taken from the
                    compare_fields
        """
        merge_event_str = []
        merge_event_id = merge_event.event.get("id", "unknown")
        for field in self.compare_fields:
            field_value = merge_event.event.get(field, None)
            merge_event_str += self.get_single_compare_field(field, field_value,
                                                             merge_event_id)

        return " ".join(merge_event_str)

    def get_single_compare_field(self, field, field_value, merge_event_id):
        """
        Extracts a single comparison field as a string from the input value. Used to
        deal with non-string based content

            Args:
                field (str): Name of field on event to extract. Needed if field_value
                    is a dict or list and we need to iterate through the event
                field (str/float/list): Value of field on event to extract
                merge_event_id (str): Event ID for event we're extracting (used purely
                    for logging)

            Returns:
                list: List of strings for field values
        """

        if isinstance(field_value, (str, unicode)):
            return [field_value]
        elif isinstance(field_value, (int, long, float, complex)):
            return [str(field_value)]
        elif isinstance(field_value, list):
            field_str = []
            for sub_field_value in field_value:
                field_str += self.get_single_compare_field(field, sub_field_value,
                                                           merge_event_id)
            return field_str
        elif isinstance(field_value, dict):
            field_str = []
            for sub_field, sub_field_value in field_value.iteritems():
                field_str += self.get_single_compare_field(sub_field, sub_field_value,
                                                           merge_event_id)
            return field_str
        else:
            logging.warning("String field %s not found on event with id %s", field,
                            merge_event_id)
            return []


    def add_dupe(self, merge_event, string_measure=Jaccard()):
        """
        Adds an event detected as a duplicate into the merged event

            Args:
                merge_event (ElasticEvent): Input event to add into bodyhash dictionary
                string_measure (string_method): String method to use for comparing distances
                    between events. Options are currently Jaccard and MinHash and if unset defaults
                    to Jaccard

            Attribute Updates:
                n_dupes, event_ids, base_event

        """
        self.n_dupes += 1
        self.event_ids[merge_event.unique_id] = None

        # Add the new event into the base event, merging the return fields
        self.base_event.add_event_fields(merge_event, merge_fields=self.return_fields)

        # If compare fields exists, then get distance of that field for this event
        # from every other previously ingested event
        if self.compare_fields:
            self.event_ids[merge_event.unique_id] = {"fields": None, "distances": None}
            merge_event_fields = self.get_compare_fields(merge_event)
            merge_eid = merge_event.unique_id
            self.event_ids[merge_eid]["fields"] = merge_event_fields
            self.event_ids[merge_eid]["distances"] = {merge_eid: 0}
            for event_id, event_compare_dict in self.event_ids.iteritems():
                event_fields = event_compare_dict["fields"]
                inter_event_distance = string_measure.get_distance(merge_event_fields,
                                                                   event_fields)
                self.event_ids[merge_eid]["distances"][event_id] = inter_event_distance
                event_compare_dict[merge_eid] = inter_event_distance

    def get_timestamp(self):
        """
        Method to return the timestamp in a readable string format

            Returns:
                str: Timestamp in the format "%Y-%m-%d %H:%m:%S"
        """
        return datetime.datetime.fromtimestamp(int(self.original_timestamp)
                                               / 1000).strftime("%Y-%m-%d %H:%M:%S")

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
