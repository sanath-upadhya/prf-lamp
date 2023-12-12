import os 

NUMBER_OF_INPUTS=50

QUESTION_INPUT = 'input'
QUESTION_PROFILE = 'profile'
QUESTION_ID = 'id'

LLM_NAME = "google/flan-t5-large"

QUESTION_PROFILE_TEXT = 'text'
QUESTION_PROFILE_TITLE = 'title'
QUESTION_PROFILE_ID = 'id'

TREC_DOC_OPEN = '<DOC>'
TREC_DOC_CLOSE = '</DOC>'
TREC_DOCNO_OPEN = '<DOCNO>'
TREC_DOCNO_CLOSE = '</DOCNO>'
TREC_TEXT_OPEN = '<TEXT>'
TREC_TEXT_CLOSE = '</TEXT>'
TREC_CORPUS_FILENAME = 'trec_format_doc_corpus'

CREATE_INDEX_GALAGO_BIN = '/Users/swatiupadhya/Downloads/CS646/galago/galago-3.16-bin/bin/galago'
CREATE_INDEX_BUILD = 'build'
CREATE_INDEX_CURRENT_PATH = os.getcwd()
CREATE_INDEX_FILETYPE = '--fileType=trectext'
CREATE_INDEX_STEMMEDPOSTINGS = '--stemmedPostings=true'
CREATE_INDEX_STEMMER = '--stemmer+krovetz'
CREATE_INDEX_INPUTPATH = '--inputPath='
CREATE_INDEX_INDEXPATH = '--indexPath='
CREATE_INDEX_INDEX_FOLDERNAME = 'sample_index'

BATCH_SEARCH_FILENAME = "batch_search.json"
BATCH_SEARCH_OUTPUT_FILENAME = "output"
BATCH_SEARCH_REQUESTED = "requested"
BATCH_SEARCH_INDEX = "index"
BATCH_SEARCH_RELEVANCE_MODEL = "relevanceModel"
BATCH_SEARCH_FBDOCS = "fbDocs"
BATCH_SEARCH_FBTERMS = "fbTerm"
BATCH_SEARCH_FB_ORIGIN_WEIGHT = "fbOrigWeight" 
BATCH_SEARCH_QUERIES = "queries"
BATCH_SEARCH_NUMBER = "number"
BATCH_SEARCH_TEXT = "text"
BATCH_SEARCH_RM = "#rm"
BATCH_SEARCH_COMMAND = "batch-search"

BATCH_SEARCH_RM3 = "org.lemurproject.galago.core.retrieval.prf.RelevanceModel3"
BATCH_SEARCH_RM1 = "org.lemurproject.galago.core.retrieval.prf.RelevanceModel1"
BATCH_SEARCH_MODEL = BATCH_SEARCH_RM1
BATCH_SEARCH_REQUESTED_VALUE = 2
BATCH_SEARCH_FBDOCS_VALUE = 1
BATCH_SEARCH_FBTERMS_VALUE = 0
BATCH_SEARCH_FB_ORIGIN_WEIGHT_VALUE = 0.25


NEWLINE = '\n'
BACKSLASH = '/'
FLOWER_BRACKET_OPEN = '{'
FLOWER_BRACKET_CLOSE = '}'
SQUARE_BRACKET_OPEN = '['
SQUARE_BRACKET_CLOSE =']'
BRACKET_OPEN = '('
BRACKET_CLOSE = ')'
COLON = ':'
COMMA = ','
OPEN_QUOTE = "\""
CLOSE_QUOTE = "\""
REDIRECT_OUTPUT = "rubbish.txt"

#CHATGPT_INPUT_QUERY = "The headline must be in JSON format with headline generated in the field \"HEAD\"."
CHATGPT_INPUT_QUERY = ""
CHATGPT_INPUT_QUERY_HEADLINE = "The user has previously generated the headline "
CHATGPT_INPUT_QUERY_ARTICLE = " for the article "
