"""
    Set of classes, each of which can tokenize string fields and get distances between
    two strings
"""

# Module imports
import logging
import traceback
import random
import hashlib
import lxml.html
import lxml.etree

class StringMethods(object):
    """
    Class with a set of generic string methods that can be inherited by children classes

        Attributes:
            tokenizer (str): String denoting what type of tokenizing method to use. Currently,
                this only supports "basic"
    """

    def __init__(self, tokenizer="basic"):
        """
        Init method for StringMethods. Currently this just chooses a tokenizer function

            Args:
                tokenizer (str): String denoting what type of tokenizing method to use. Currently,
                    this only supports "basic", which is just splitting on whitespace

            Attribute Updates:
                tokenizer
        """
        self.tokenizer = tokenizer

    @staticmethod
    def clean_string(input_string, qtd_strip_limit=1000):
        """
        Cleans the input string, primarily performing HTML stripping

            Args:
                input_string (str): String that we want to clean up
                qtd_strip_limit (int): Length of string above which we start stripping out
                    quoted text

            Returns:
                str: Cleaned input string with HTML and quoted text stripped out
        """

        # First strip out QTD text, but only if the document is long enough
        lxml_document = lxml.html.document_fromstring(input_string)
        if len(input_string) > qtd_strip_limit:
            qtd_tags = set()
            for lxml_element in lxml_document.iter():
                if isinstance(lxml_element.tag, (str, unicode)):
                    if lxml_element.tag.startswith(("ro_disclaimer", "ro_quoted")):
                        qtd_tags.add(lxml_element.tag)
            lxml.etree.strip_elements(lxml_document, *list(qtd_tags))

        # Then extract text content
        lxml_string = lxml_document.text_content()
        lxml_string = lxml_string.replace("\n", " ").replace("\r", " ")
        if lxml_string:
            if not lxml_string.isspace():
                lxml_string = lxml.html.document_fromstring(lxml_string).text_content()

        return lxml_string

    def tokenize_string(self, input_string):
        """
        Tokenizes the input string according to the class method defined

            Args:
                input_string (str): String that we want to clean up

            Returns:
                list: List of string tokens that come from applying defined tokenizer
                    method
        """

        if self.tokenizer == "basic":
            out_string = input_string.lower()
            #out_string = re.sub(r"[^a-zA-Z0-9\s]", " ", out_string)
            tokens = [token for token in out_string.split(" ") if token != ""]
        else:
            logging.error("Invalid string tokenizer %s chosen", self.tokenizer)
            raise ValueError("Invalid string tokenizer %s chosen" % self.tokenizer)

        return tokens

    def generate_ngrams(self, input_string, n_gram_size=1):
        """
        Tokenizes the input string according to the class method defined and then splits
        into ngrams

            Args:
                input_string (str): String that we want to clean up
                n_gram_size (int): Number of tokens per n-gram to output

            Returns:
                list: List of string ngrams, each of length n_gram_size tokens
        """

        stripped_string = self.clean_string(input_string)
        tokens = self.tokenize_string(stripped_string)
        ngrams = zip(*[tokens[i:] for i in range(n_gram_size)])

        return [" ".join(ngram) for ngram in ngrams]


class Jaccard(StringMethods):
    """
    Class to get string distances based on Jaccard Similarity

        Attributes:
            tokenizer (str): String denoting what type of tokenizing method to use. Currently,
                this only supports "basic"
            n_gram_size (int): Number of tokens per n-gram to output
    """

    def __init__(self, tokenizer="basic", n_gram_size=1):
        """
        Init method for Jaccard class.

            Args:
                tokenizer (str): String denoting what type of tokenizing method to use. Currently,
                    this only supports "basic", which is just splitting on whitespace
                n_gram_size (int): Number of tokens per n-gram to output

            Attribute Updates:
                tokenizer, n_gram_size
        """

        StringMethods.__init__(self, tokenizer)
        self.n_gram_size = n_gram_size

    def get_distance(self, string_a, string_b):
        """
        Gets the distance between strings according to Jaccard Similarity

            Args:
                string_a (str): First string to compare
                string_b (str): Second string to compare

            Returns:
                float: Number between 0 and 1 representing the Jaccard distance between
                    string_a and string_b
        """

        event_set_a = set(self.generate_ngrams(string_a, n_gram_size=self.n_gram_size))
        event_set_b = set(self.generate_ngrams(string_b, n_gram_size=self.n_gram_size))

        intersect_cardinality = len(event_set_a.intersection(event_set_b))
        union_cardinality = len(event_set_a.union(event_set_b))
        if union_cardinality:
            return 1.0 - float(intersect_cardinality) / float(union_cardinality)
        else:
            return 0.0

        return 0.0

class MinHash(StringMethods):
    """
    Class to get string distances based on MinHash algorithm

        Attributes:
            tokenizer (str): String denoting what type of tokenizing method to use. Currently,
                this only supports "basic"
            n_gram_size (int): Number of tokens per n-gram to output
            seed (int): Random seed to use for generating masks to XOR with hashes and
                generate different minhashes
            masks (list): List of masks of length n_hashes to use to generate minhashes
    """

    def __init__(self, tokenizer="basic", n_gram_size=1, random_seed=42, n_hashes=200):
        """
        Init method for MinHash class.

            Args:
                tokenizer (str): String denoting what type of tokenizing method to use. Currently,
                    this only supports "basic", which is just splitting on whitespace
                n_gram_size (int): Number of tokens per n-gram to output
                random_seed (int): Random seed to use for generating masks to XOR with hashes and
                    generate different minhashes
                n_hashes (int): Number of minhashes to generate for each string

            Attribute Updates:
                tokenizer, n_gram_size, seed, masks
        """

        StringMethods.__init__(self, tokenizer)
        self.n_gram_size = n_gram_size
        self.seed = random_seed

        # Initialize random number generator
        random.seed(self.seed)

        # Get list of random integers
        self.masks = [random.randint(1, 2**160-1) for _ in xrange(n_hashes)]

    def hash_string(self, input_string):
        """
        Gets the set of hashes for all tokens in the input string

            Args:
                input_string (str): String that we want to hash

            Returns:
                list: List of minhashes for the input_string of length n_hashes
                list: List of n_grams that come from input_string
                list: List of hashed n_grams that come from input_string
        """

        n_grams = self.generate_ngrams(input_string, n_gram_size=self.n_gram_size)
        min_hashes = []
        hashes = [hashlib.sha1(token.encode('utf-8')).hexdigest() for token in n_grams]
        if hashes:
            for mask in self.masks:
                new_hashes = [int(hasha, 16) ^ mask for hasha in hashes]
                min_hashes += ["%x" % min(new_hashes)]

        return min_hashes, n_grams, hashes

    def get_distance(self, string_a, string_b):
        """
        Gets the distance between strings according to MinHash

            Args:
                string_a (str): First string to compare
                string_b (str): Second string to compare

            Returns:
                float: Number between 0 and 1 representing the MinHash distance between
                    string_a and string_b
        """

        hash_set_a, _, _ = self.hash_string(string_a)
        hash_set_b, _, _ = self.hash_string(string_b)
        hash_eq_set = [a == b for (a, b) in zip(hash_set_a, hash_set_b)]

        if len(hash_set_a) == 0:
            if len(hash_set_b) == 0:
                return 0.0
            else:
                return 1.0
        elif len(hash_set_b) == 0:
            return 1.0

        return 1.0 - float(sum(hash_eq_set)) / float(len(hash_eq_set))

class LSH(MinHash):
    """
    Class to hash an input string based on the LSH algorithm and compare them for equality

        Attributes:
            tokenizer (str): String denoting what type of tokenizing method to use. Currently,
                this only supports "basic"
            n_gram_size (int): Number of tokens per n-gram to output
            seed (int): Random seed to use for generating masks to XOR with hashes and
                generate different minhashes
            masks (list): List of masks of length n_hashes to use to generate minhashes
            n_hashes (int): Number of hashes in the minhash
            n_bands (int): Number of bands to split minhashes into in order to generate LSH
                signature
            hashes_per_band (int): Number of hashes per individual band (equal to
                n_hashes / n_bands)
            hash (list): LSH signature of input string, which is a list of hashes of length
                n_bands
    """

    def __init__(self, input_string, tokenizer="basic", n_gram_size=1, random_seed=42, n_hashes=200,
                 n_bands=20):
        """
        Init method for LSH class.

            Args:
                input_string (str): String that we want to hash
                tokenizer (str): String denoting what type of tokenizing method to use. Currently,
                    this only supports "basic", which is just splitting on whitespace
                n_gram_size (int): Number of tokens per n-gram to output
                random_seed (int): Random seed to use for generating masks to XOR with hashes and
                    generate different minhashes
                n_hashes (int): Number of minhashes to generate for each string
                n_bands (int): Number of bands to split minhashes into in order to generate LSH
                    signature

            Attribute Updates:
                tokenizer, n_gram_size, seed, masks, n_hashes, n_bands, hashes_per_band, hash
        """

        # Initialize the MinHash on top of which LSH is built
        MinHash.__init__(self, tokenizer, n_gram_size, random_seed, n_hashes)

        # Initialize the number of hashes and bands for comparison
        self.n_hashes = n_hashes
        self.n_bands = n_bands

        # Make sure that the number of bands factorizes into the number of minhashes
        self.hashes_per_band, remainder = divmod(self.n_hashes, self.n_bands)
        if remainder != 0:
            logging.error("The # bands in the LSH %d doesn't go into the # hashes %d",
                          self.n_bands, self.n_hashes)
            raise ValueError("Invalid # of bands for LSH")

        # And define the hash for this string
        self.hash = self.get_hash(input_string)

    def get_thresholds(self, dup_prob):
        """
        Function to get the Jaccard Similarity thresholds at different certainty levels

            Args:
                dup_prob (int): The probability we want to achieve of identifying near duplicates

            Returns:
                float: The similarity threshold that corresponds to the level of certainty
        """

        similarity = pow((1.0 - pow((1.0 - dup_prob), (1.0 / self.n_bands))),
                         (1.0 / self.hashes_per_band))
        logging.info("Documents with a Similarity of %f hava a %f probability of being duplicates",
                     similarity, dup_prob)

        return similarity

    def get_hash(self, string_a):
        """
        Gets the set of LSH hashes for the input string

            Args:
                string_a (str): String that we want to hash

            Returns:
                list: LSH signature of input string, which is a list of hashes of length
                    n_bands
        """

        # Get the list of minhashes for the string
        min_hashes, _, _ = self.hash_string(string_a)

        # Minhashes divided into blocks
        blocked_hashes = [min_hashes[ind:ind + self.hashes_per_band]
                          for ind in xrange(0, self.n_hashes, self.hashes_per_band)]
        joined_blocks = ["".join(block) for block in blocked_hashes]

        # And then hash the individual blocks
        hashed_blocks = [hashlib.sha1(block).hexdigest() for block in joined_blocks]

        return hashed_blocks

    def __eq__(self, hash_comparison):
        """
        Compares two signature hash lists defined by get_hash_list for equality

            Args:
                hash_comparison (LSH): Other LSH class type that we want to compare to this
                    one for equality

            Returns:
                bool: Flag for whether or not the other LSH is equal to this one
        """

        hash_list_a = self.hash
        hash_list_b = hash_comparison.hash
        return any([hash_a == hash_b for hash_a, hash_b in zip(hash_list_a, hash_list_b)])

    def print_hash(self):
        """
        Prints out a string representation of the hash. Note that for LSH equal hashes
        can print out different printed representations since LSH is a nonhashable type

            Returns:
                str: String that comes from joining hashes in the signature
        """

        return "".join(self.hash)

class SimpleHash(object):
    """
    Class to hash strings and compare them based on a simple sha encoding algorithm

        Attributes:
            hash (str): Hash of the input string
    """

    def __init__(self, input_string):
        """
        Init method for SimpleHash class.

            Args:
                input_string (str): String that we want to hash

            Attribute Updates:
                hash
        """

        # And define the hash for this string
        self.hash = self.get_hash(input_string)

    @staticmethod
    def get_hash(string_a):
        """
        Gets a single hash for the input string

            Args:
                string_a (str): String that we want to hash

            Returns:
                str: Hash of the input string
        """

        return hashlib.sha1(string_a.encode('utf-8')).hexdigest()

    def __eq__(self, hash_comparison):
        """
        Compares two hashes for equality

            Args:
                hash_comparison (SimpleHash): Other SimpleHash class type that we want to compare
                    to this one for equality

            Returns:
                bool: Flag for whether or not the other SimpleHash is equal to this one
        """

        hash_a = self.hash
        hash_b = hash_comparison.hash
        return hash_a == hash_b

    def print_hash(self):
        """
        Prints out a string representation of the hash

            Returns:
                str: Hash of the input string
        """

        return self.hash

def main():
    """
        Main Function for string_methods. Should contain  a bunch of testing functions
    """

if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        ERR_TRACEBACK = "; ".join(traceback.format_exc().split("\n"))
        logging.error("Exception: Function failed due to error %s with exception info %s",
                      err, ERR_TRACEBACK)
