"""
Purpose

Classes to interact with AWS Comprehend for the purpose of detecting
PII, and then reshaping the output into a workable format
"""

import logging
import math
import boto3
import bisect
import datetime

from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


class ComprehendHandler:
    """
    Handles the submission of text to AWS Comprehend in a way
    that ensures we're not over-using the API
    """
    def __init__(self, starting_units=None, max_cost=0,
                 min_post_units=3, **kwargs):
        """
        :param starting_units: Number of units of text already submitted to
        Comprehend as a dictionary
        :param max_cost: Max allowable cost on AWS
        :param min_post_units: Minimum number of units to allow for a post
        to comprehend
        :param kwargs: Keywords associated with the AWS Credentials to
        initiate the session
        to Comprehend before halting things
        """

        # Initiate the comprehend client and the PII Detector
        self.session = boto3.Session(**kwargs)
        self.comprehend_client = self.session.client("comprehend")
        self.ce_client = self.session.client("ce")

        # Initiate the starting AWS usage
        if not starting_units:
            starting_units = self.get_comprehend_usage()
        self.running_cost = self.get_comprehend_cost(starting_units)
        self.units_submitted = starting_units
        self.max_cost = max_cost
        self.min_post_units = min_post_units

    def get_comprehend_usage(self, granularity="MONTHLY"):
        """
        Gets estimates of the AWS Comprehend usage broken down by the
        type of usage e.g. DetectPII etc.

        :param granularity: Whether to get cost for the past, day,
        month or year
        """

        # Define cost submission dictionary
        cost_dict = {"Granularity": granularity, "Metrics": ["UsageQuantity"],
                     "Filter": {"Dimensions": {"Key": "SERVICE", "Values": ["Amazon Comprehend"]}},
                     "GroupBy": [{"Type": "DIMENSION", "Key": "USAGE_TYPE"}]}
        end_date = datetime.date.today().strftime("%Y-%m-%d")
        start_date = datetime.date.today()
        if granularity in ["MONTHLY", "YEARLY"]:
            start_date = start_date.replace(day=1)
        if granularity in ["YEARLY"]:
            start_date = start_date.replace(month=1)
        start_date = start_date.strftime("%Y-%m-%d")
        cost_dict["TimePeriod"] = {"Start": start_date, "End": end_date}

        # And get comprehend usage
        logger.warning("You are making a submission to the AWS Cost Explorer "
                       "which will incur a cost of $0.01")
        cost_output = self.ce_client.get_cost_and_usage(**cost_dict)

        # Parse output into readable usage in terms of Units
        latest_costs = cost_output.get("ResultsByTime", [{}])[0]
        grouped_costs = latest_costs.get("Groups", [])
        usage = {}
        for cost in grouped_costs:
            cost_key = cost.get("Keys", [""])[0]
            split_cost_key = cost_key.split("-")
            if len(split_cost_key) != 2:
                logger.warning(f"Invalid Key {cost_key} found in AWS CE Response")
                continue
            cost_key = split_cost_key[1]
            cost_usage = cost.get("Metrics", {}).get("UsageQuantity", {}).get("Amount", 0)
            usage[cost_key] = usage.get(cost_key, 0) + int(cost_usage)

        return usage

    @staticmethod
    def get_comprehend_cost(usage, free_tier=True):
        """
        Given a usage dictionary, calculates the estimated total bill
        from AWS Comprehend. Based on https://aws.amazon.com/comprehend/pricing/

        :param usage: Output of get_comprehend_usage, lists the units
        submitted per comprehend usage_type
        :param free_tier: Whether or not we're in the AWS free tier
        :return total cost in dollars of using Comprehend

        """

        # Iterate over usage_types and get the cost
        cost = 0
        for usage_type, units in usage.items():
            if free_tier:
                units = max(units-5.0e4, 0)
            if usage_type in ["DetectKeyPhrases", "DetectSentiment", "DetectEntities",
                              "DetectDominantLanguage", ""]:
                cost += 0.0001 * min(units, 1.0e7) + 0.00005 * min(max(units - 1.0e7, 0), 4.0e7) + \
                    0.000025 * max(units - 5.0e7, 0)
            elif usage_type in ["DetectPiiEntities"]:
                cost += 0.0001 * min(units, 1.0e7) + 0.00005 * min(max(units-1.0e7, 0), 4.0e7) + \
                    0.000025 * min(max(units-5.0e7, 0), 5.0e7) + 0.000005 * max(units-1.0e8, 0)
            elif usage_type in ["DetectSyntax"]:
                cost += 0.00005 * min(units, 1.0e7) + 0.000025 * min(max(units-1.0e7, 0), 4.0e7) + \
                    0.0000125 * max(units-5.0e7, 0)
            elif usage_type in ["ContainsPiiEntities"]:
                cost += 0.000002 * min(units, 1.0e7) + 0.000001 * min(max(units-1.0e7, 0), 4.0e7) + \
                    0.0000005 * min(max(units-5.0e7, 0), 5.0e7) + 0.0000001 * max(units-1.0e8, 0)
            else:
                logger.warning(f"Invalid Usage Type {usage_type} detected")

        return cost

    def initialize_detector(self, usage_type):
        """
        Method to initialize a comprehend detector given the usage
        type

        :param usage_type: What service we're using with Comprehend.
        Options are: 'ContainsPiiEntities', 'DetectDominantLanguage',
        'DetectEntities', 'DetectKeyPhrases', 'DetectPiiEntities',
        'DetectSentiment', 'DetectSyntax'
        :return ComprehendDetect object

        """

        # Initialize the Detector for interacting with Comprehend
        if usage_type in ["DetectPiiEntities", "ContainsPiiEntities"]:
            comprehend_detector = PIIDetect(self.comprehend_client,
                                            hard_unit_limit=False)
        else:
            comprehend_detector = ComprehendDetect(self.comprehend_client)
        logging.info(f"Initialized Comprehend Client of type {usage_type}")

        return comprehend_detector

    def extract_batch(self, documents, detector, usage_type,
                      n_units=None, **kwargs):
        """
        Method that takes in a small batch of documents and a PII
        detector and then extracts the entities for that batch

        :param documents: List of documents to extract PII for
        :param detector: Object of type ComprehendDetect
        :param usage_type: What service we're using with Comprehend.
        :param n_units: Number of units involved with submitting this
        set of documents
        :return List of PII Entities for each document

        """

        # Get the number of units, if they're not passed
        if not n_units:
            n_units = detector.get_units(sum([len(text) for text in documents]))

        # First estimate the additional cost of submitting these documents
        estimated_cost = self.estimate_cost(n_units, usage_type)

        # Raise an error if the estimated cost is too much
        if estimated_cost + self.running_cost > self.max_cost:
            logger.exception(f"Comprehend Request exceeds maximum cost {self.max_cost}")
            raise ValueError(f"Comprehend Request exceeds maximum cost {self.max_cost}")

        # Otherwise continue submitting to Comprehend
        entities = detector.extract(documents, usage_type, **kwargs)

        # Update the usage and cost
        self.units_submitted[usage_type] = self.units_submitted.get(usage_type, 0) + n_units
        self.running_cost += estimated_cost

        return entities

    def extract_documents(self, documents, usage_type, **kwargs):
        """
        Method that takes in an iterator over documents and submits
        them in batches above the minimum post size to Comprehend

        :param documents: List of documents to extract PII for
        :param usage_type: What service we're using with Comprehend.
        Options are: 'ContainsPiiEntities', 'DetectDominantLanguage',
        'DetectEntities', 'DetectKeyPhrases', 'DetectPiiEntities',
        'DetectSentiment', 'DetectSyntax'
        :return Generator of PII Entities in each document

        """

        # Initialize the counters on document size
        submit_documents = []
        n_chars = 0

        # Initialize the Detector for interacting with Comprehend
        comprehend_detector = self.initialize_detector(usage_type)

        # Manually set the minimum post size to 1 document if we're using
        # Contains PII
        min_post_units = self.min_post_units
        if usage_type == "ContainsPiiEntities":
            min_post_units = 1

        # Iterate through the list of documents
        for text in documents:
            submit_documents += [text]
            n_chars += len(text)
            n_units = comprehend_detector.get_units(n_chars)
            if n_units < min_post_units:
                pass
            else:
                entities = self.extract_batch(submit_documents, comprehend_detector,
                                              usage_type, n_units=n_units, **kwargs)
                for entity in entities:
                    yield entity
                submit_documents = []
                n_chars = 0

        # And post whatever is left over as a final batch
        if n_chars > 0:
            entities = self.extract_batch(submit_documents, comprehend_detector,
                                          usage_type, n_units=n_units, **kwargs)
            for entity in entities:
                yield entity

    def estimate_cost(self, n_units, usage_type):
        """
        Method to estimate cost of submitting a set number of units
        to Comprehend, given the existing usage

        :param n_units: Number of units being submitted
        :param usage_type: What service we're using with Comprehend.
        :return estimated cost of submission

        """

        # First estimate the additional cost of submitting these documents
        current_usage = {usage_type: self.units_submitted.get(usage_type, 0)}
        est_usage = {usage_type: self.units_submitted.get(usage_type, 0) + n_units}
        estimated_cost = self.get_comprehend_cost(est_usage) - \
                         self.get_comprehend_cost(current_usage)

        return estimated_cost

    def estimate_usage(self, documents, usage_type):
        """
        Method to estimate the usage and cost of submitting documents to
        Comprehend

        :param documents: List of documents to extract PII for
        :param usage_type: What service we're using with Comprehend.
        :return number of units
        :return estimated cost of submission

        """

        # Initialize the counters on document size
        submit_documents = []
        n_chars = 0

        # Initialize the Detector for interacting with Comprehend
        comprehend_detector = self.initialize_detector(usage_type)

        # Iterate through the list of documents
        total_units = 0
        for text in documents:
            n_chars += len(text)
            n_units = comprehend_detector.get_units(n_chars)
            if n_units < self.min_post_units:
                pass
            else:
                total_units += n_units
                n_chars = 0

        # And add the last set of units
        total_units += n_units
        total_cost = self.estimate_cost(n_units, usage_type)

        return total_units, total_cost


class ComprehendDetect:
    """Encapsulates Comprehend detection functions."""
    def __init__(self, comprehend_client):
        """
        :param comprehend_client: A Boto3 Comprehend client.
        """
        self.comprehend_client = comprehend_client

    def detect_languages(self, text):
        """
        Detects languages used in a document.

        :param text: The document to inspect.
        :return: The list of languages along with their confidence scores.
        """
        try:
            response = self.comprehend_client.detect_dominant_language(Text=text)
            languages = response['Languages']
            logger.info("Detected %s languages.", len(languages))
        except ClientError:
            logger.exception("Couldn't detect languages.")
            raise
        else:
            return languages

    def detect_entities(self, text, language_code):
        """
        Detects entities in a document. Entities can be things like people and places
        or other common terms.

        :param text: The document to inspect.
        :param language_code: The language of the document.
        :return: The list of entities along with their confidence scores.
        """
        try:
            response = self.comprehend_client.detect_entities(
                Text=text, LanguageCode=language_code)
            entities = response['Entities']
            logger.info("Detected %s entities.", len(entities))
        except ClientError:
            logger.exception("Couldn't detect entities.")
            raise
        else:
            return entities

    def detect_key_phrases(self, text, language_code):
        """
        Detects key phrases in a document. A key phrase is typically a noun and its
        modifiers.

        :param text: The document to inspect.
        :param language_code: The language of the document.
        :return: The list of key phrases along with their confidence scores.
        """
        try:
            response = self.comprehend_client.detect_key_phrases(
                Text=text, LanguageCode=language_code)
            phrases = response['KeyPhrases']
            logger.info("Detected %s phrases.", len(phrases))
        except ClientError:
            logger.exception("Couldn't detect phrases.")
            raise
        else:
            return phrases

    def detect_pii(self, text, language_code):
        """
        Detects personally identifiable information (PII) in a document. PII can be
        things like names, account numbers, or addresses.

        :param text: The document to inspect.
        :param language_code: The language of the document.
        :return: The list of PII entities along with their confidence scores.
        """
        try:
            response = self.comprehend_client.detect_pii_entities(
                Text=text, LanguageCode=language_code)
            entities = response['Entities']
            logger.info("Detected %s PII entities.", len(entities))
        except ClientError:
            logger.exception("Couldn't detect PII entities.")
            raise
        else:
            return entities

    def contains_pii(self, text, language_code):
        """
        Detects whether a document contains personally identifiable information (PII).
        PII can be things like names, account numbers, or addresses.

        :param text: The document to inspect.
        :param language_code: The language of the document.
        :return: The list of PII types along with their confidence scores.
        """
        try:
            response = self.comprehend_client.contains_pii_entities(
                Text=text, LanguageCode=language_code)
            pii_types = response['Labels']
            logger.info("Detected %s types of PII.", len(pii_types))
        except ClientError:
            logger.exception("Couldn't detect PII entities.")
            raise
        else:
            return pii_types

    def detect_sentiment(self, text, language_code):
        """
        Detects the overall sentiment expressed in a document. Sentiment can
        be positive, negative, neutral, or a mixture.

        :param text: The document to inspect.
        :param language_code: The language of the document.
        :return: The sentiments along with their confidence scores.
        """
        try:
            response = self.comprehend_client.detect_sentiment(
                Text=text, LanguageCode=language_code)
            logger.info("Detected primary sentiment %s.", response['Sentiment'])
        except ClientError:
            logger.exception("Couldn't detect sentiment.")
            raise
        else:
            return response

    def detect_syntax(self, text, language_code):
        """
        Detects syntactical elements of a document. Syntax tokens are portions of
        text along with their use as parts of speech, such as nouns, verbs, and
        interjections.

        :param text: The document to inspect.
        :param language_code: The language of the document.
        :return: The list of syntax tokens along with their confidence scores.
        """
        try:
            response = self.comprehend_client.detect_syntax(
                Text=text, LanguageCode=language_code)
            tokens = response['SyntaxTokens']
            logger.info("Detected %s syntax tokens.", len(tokens))
        except ClientError:
            logger.exception("Couldn't detect syntax.")
            raise
        else:
            return tokens

    @staticmethod
    def get_units(text_length):
        """
        Given a piece of text, returns the number of units involved in
        submitting to AWS Comprehend

        :param text: The length of the document we're planning to send to Comprehend
        :return: Number of units
        """

        # Get number of characters in the text
        units = math.ceil(text_length / 100)
        logger.debug(f"There are {units} AWS units in the submitted text")

        return units


class PIIDetect(ComprehendDetect):
    """
    Encapsulates Comprehend detection functions for identifying PII
    """
    def __init__(self, comprehend_client, hard_unit_limit=True):
        """
        :param comprehend_client: A Boto3 Comprehend client.
        :param hard_unit_limit: Whether or not to stop submission
        if the number of units is less than 3
        """
        self.comprehend_client = comprehend_client
        self.hard_unit_limit = hard_unit_limit

    def extract(self, documents, usage_type, language_code="en", max_units=50000):
        """
        Detects personally identifiable information (PII) in a document,
        and outputs the entities in a readable format

        :param documents: List of documents to inspect
        :param usage_type: What service we're using with Comprehend.
        :param language_code: The language of the document e.g. "en"
        :param max_units: Maximum amount of units that we can submit
        :return: A mapping from PII type to the list of entities and
        snippets associated with them in the text
        """

        # Create a single text from all of the documents, as well as get
        # all the document offsets
        if usage_type == "DetectPiiEntities":
            text = "\n".join(document for document in documents)
            doc_offsets = [text.index(document) for document in documents]
        elif usage_type == "ContainsPiiEntities":
            if len(documents) > 1:
                raise ValueError(f"Can't submit more than 1 document at a time to "
                                 f"Contains PII")
            else:
                text = documents[0]
        else:
            raise ValueError(f"Invalid usage type {usage_type} provided")

        # First check that the text submission is valid
        if not self.check_text_submission(text, max_units):
            raise ValueError(f"Input text failed checks on submission length")

        # Then run PII detection and extraction
        logger.info(f"Submitting {len(documents)} documents for entity extraction")
        if usage_type == "DetectPiiEntities":
            pii_locs = self.detect_pii(text, language_code)
            pii_entities = self.output_pii_entities(text, doc_offsets, pii_locs)
        elif usage_type == "ContainsPiiEntities":
            pii_labels = self.contains_pii(text, language_code)
            pii_entities = self.output_pii_types(pii_labels)
        else:
            raise ValueError(f"Invalid usage type {usage_type} provided")

        return pii_entities

    def output_pii_entities(self, text, doc_offsets, entities):
        """
        Given a text and a set of entities output from detect_pii, returns
        the extracted entities and types of PII

        :param text: The document to inspect.
        :param entities: Output of detect_pii
        :param doc_offsets: Offsets within the text, to separate into
        different documents
        :return: A mapping from PII type to the list of entities and
        snippets associated with them in the text
        """

        # Iterate through entities to extract the PII from each
        entity_mappings = [{} for _ in doc_offsets]
        for entity in entities:
            pii_type = entity.get("Type")
            pii_score = entity.get("Score", 0)
            start = entity.get("BeginOffset", 0)
            finish = entity.get("EndOffset", 0)
            doc_index = bisect.bisect_left(doc_offsets, finish) - 1
            pii, pii_snippet = self.extract_snippet(text, start, finish)
            if pii_type:
                pii_dict = {"Score": pii_score, "PII": pii, "Snippet": pii_snippet,
                            "Start": start, "Finish": finish}
                entity_mapping = entity_mappings[doc_index]
                entity_mapping[pii_type] = entity_mapping.get(pii_type, []) + [pii_dict]
            else:
                logger.warning("Invalid PII Type found")

        return entity_mappings

    @staticmethod
    def output_pii_types(pii_labels):
        """
        Given a set of labels output from contains_pii, returns
        the types of PII

        :param pii_labels: Output of contains_pii
        :return: A mapping from PII type to the scores associated with
        each PII type
        """

        # Iterate through entities to extract the PII from each
        entity_mapping = {}
        for pii_dict in pii_labels:
            pii_type = pii_dict.get("Name")
            if pii_type:
                pii_output = {"Score": pii_dict.get("Score", 0)}
                entity_mapping[pii_type] = entity_mapping.get(pii_type, []) + [pii_output]
            else:
                logger.warning("Invalid PII Type found")

        return [entity_mapping]

    def check_text_submission(self, text, max_units=50000):
        """
        Given a piece of text, checks it's validity for submission to
        AWS comprehend against a variety of limits

        :param text: The document we're planning to send to Comprehend
        :param max_units: Maximum amount of units that we can submit
        :return: Boolean for whether it should be submitted or not
        """

        # First check that we're above the minimum number of units in
        # the text
        n_units = self.get_units(len(text))
        if n_units < 3:
            exception_string = f"Only {n_units} units in the text, which is " \
                               f"below the submission limit of 3"
            if self.hard_unit_limit:
                logger.exception(exception_string)
                return False
            else:
                logger.warning(exception_string)
        elif n_units > max_units:
            exception_string = f"The {n_units} units in the text, exceed the " \
                               f"maximum allowed amount remaining"
            logger.exception(exception_string)
            return False

        return True

    @staticmethod
    def extract_snippet(text, start, finish, snippet_size=50):
        """
        Given a piece of text and the start and finish of the identified
        word, returns a snippet from the text around that word

        :param text: The document with the snippet contained
        :param start: The starting character of the identified word
        :param finish: The ending character of the identified word
        :param snippet_size: The number of characters on either side
        of the word that we want to identify
        :return: A tuple containing the extracted word, and the
        snippet of text either side of it
        """

        # The extracted word is defined just by the start and end
        # points
        extracted_word = text[start:finish]

        # The extracted snippet is defined by the start and end plus
        # the snippet size
        snippet_start = max(start-snippet_size, 0)
        snippet_finish = min(finish+snippet_size,len(text))
        extracted_snippet = text[snippet_start:snippet_finish]

        return extracted_word, extracted_snippet