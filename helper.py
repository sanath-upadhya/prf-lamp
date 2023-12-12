import random
import constants
import re
import subprocess
import os
import shutil
import openai
import sys
from rouge_score import rouge_scorer
import nltk
import evaluate
import numpy as np
from datasets import load_dataset
from transformers import T5Tokenizer, DataCollatorForSeq2Seq
from transformers import T5ForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

def get_random_index(data):
    index_dict = {}
    count = 0
    print("Size of data is ")
    print(len(data))
    while True:
        rand_int = random.randint(0, len(data)-1)
        if rand_int in index_dict:
            pass
        else:
            index_dict[rand_int] = True
            count = count+1

            if (count >= constants.NUMBER_OF_INPUTS):
                break
    
    return index_dict

def create_trec_format_profile(profile):
    final_output = ""
    for ind_profile in profile:
        ind_profile_id = ind_profile[constants.QUESTION_PROFILE_ID]
        ind_profile_text = ind_profile[constants.QUESTION_PROFILE_TEXT]
        ind_profile_title = ind_profile[constants.QUESTION_PROFILE_TITLE]
        #print("New output is ")
        temp_result = (constants.TREC_DOC_OPEN + constants.NEWLINE + 
                       constants.TREC_DOCNO_OPEN + ind_profile_id + constants.TREC_DOCNO_CLOSE + constants.NEWLINE + 
                       constants.TREC_TEXT_OPEN + constants.NEWLINE +
                       ind_profile_title + " " + ind_profile_text + constants.NEWLINE +
                       constants.TREC_TEXT_CLOSE + constants.NEWLINE +
                       constants.TREC_DOC_CLOSE + constants.NEWLINE)
        #print(temp_result)
        final_output = final_output + temp_result
    
    return final_output

def write_to_file(filedata, filename):
    f = open(filename, "w")
    f.write(filedata)
    f.close()

def create_batch_search_number_line(key, value):
    final_output = constants.OPEN_QUOTE + key + constants.CLOSE_QUOTE + constants.COLON + constants.OPEN_QUOTE + value + constants.CLOSE_QUOTE + constants.COMMA + constants.NEWLINE
    return final_output

def create_batch_search_text_line(key, value):
    final_output = constants.OPEN_QUOTE + key + constants.CLOSE_QUOTE + constants.COLON + constants.OPEN_QUOTE + constants.BATCH_SEARCH_RM + constants.BRACKET_OPEN + value + constants.BRACKET_CLOSE + constants.CLOSE_QUOTE + constants.NEWLINE
    return final_output

def create_batch_search_line(key, value, no_comma, quotes_around_value):
    final_output = ""
    if no_comma:
        if quotes_around_value:
            final_output = constants.OPEN_QUOTE + key + constants.CLOSE_QUOTE + constants.COLON + constants.OPEN_QUOTE + value + constants.CLOSE_QUOTE + constants.NEWLINE
        else:
            final_output = constants.OPEN_QUOTE + key + constants.CLOSE_QUOTE + constants.COLON + value + constants.NEWLINE
    else:
        if quotes_around_value:
            final_output = constants.OPEN_QUOTE + key + constants.CLOSE_QUOTE + constants.COLON + constants.OPEN_QUOTE + value + constants.CLOSE_QUOTE +constants.COMMA + constants.NEWLINE
        else:
            final_output = constants.OPEN_QUOTE + key + constants.CLOSE_QUOTE + constants.COLON + value + constants.COMMA + constants.NEWLINE
    return final_output

def create_batch_search_json(id, input):
    final_output = ""
    #Need to remove all non-alphanureic characters as Galago throws an error
    real_input = re.sub(r'[^A-Za-z0-9 ]+', '', input)
    final_output = (final_output + constants.FLOWER_BRACKET_OPEN + constants.NEWLINE +
                    create_batch_search_line(constants.BATCH_SEARCH_REQUESTED, str(constants.BATCH_SEARCH_REQUESTED_VALUE), False, False) + 
                    create_batch_search_line(constants.BATCH_SEARCH_INDEX, get_index_location(), False, True) +
                    create_batch_search_line(constants.BATCH_SEARCH_RELEVANCE_MODEL, constants.BATCH_SEARCH_MODEL, False, True) +
                    create_batch_search_line(constants.BATCH_SEARCH_FBDOCS, str(constants.BATCH_SEARCH_FBDOCS_VALUE), False, False) +
                    create_batch_search_line(constants.BATCH_SEARCH_FBTERMS, str(constants.BATCH_SEARCH_FBTERMS_VALUE), False, False) +
                    create_batch_search_line(constants.BATCH_SEARCH_FB_ORIGIN_WEIGHT, str(constants.BATCH_SEARCH_FB_ORIGIN_WEIGHT_VALUE), False, False) + 
                    create_batch_search_line(constants.BATCH_SEARCH_QUERIES, constants.SQUARE_BRACKET_OPEN, True, False) +
                    constants.FLOWER_BRACKET_OPEN + constants.NEWLINE +
                    create_batch_search_number_line(constants.BATCH_SEARCH_NUMBER, str(id)) +
                    create_batch_search_text_line(constants.BATCH_SEARCH_TEXT, real_input) +
                    constants.FLOWER_BRACKET_CLOSE + constants.NEWLINE +
                    constants.SQUARE_BRACKET_CLOSE + constants.NEWLINE +
                    constants.FLOWER_BRACKET_CLOSE + constants.NEWLINE
                    )

    write_to_file(final_output, constants.BATCH_SEARCH_FILENAME)

def set_java_env_variable():
    command = []
    command.append("export")
    java_home = "JAVA_HOME=`/usr/libexec/java_home -v 1.8`"
    command.append(java_home)
    final_command = ""
    for ind_command in command:
        final_command = final_command + " " + ind_command
    
    print(final_command)
    os.system(final_command)
    #os.system("java -version")

    #command.append(java_home)
    #subprocess.run(command)

def create_index():
    set_java_env_variable()
    command = []
    command.append(constants.CREATE_INDEX_GALAGO_BIN)
    command.append(constants.CREATE_INDEX_BUILD)
    command.append(constants.CREATE_INDEX_FILETYPE)
    
    location_of_inputpath = constants.CREATE_INDEX_CURRENT_PATH + constants.BACKSLASH + constants.TREC_CORPUS_FILENAME
    final_inputpath_string = constants.CREATE_INDEX_INPUTPATH + location_of_inputpath
    command.append(final_inputpath_string)

    location_of_indexpath = constants.CREATE_INDEX_CURRENT_PATH + constants.BACKSLASH + constants.CREATE_INDEX_INDEX_FOLDERNAME
    final_indexpath_string = constants.CREATE_INDEX_INDEXPATH + location_of_indexpath
    command.append(final_indexpath_string)

    command.append(constants.CREATE_INDEX_STEMMEDPOSTINGS)
    command.append(constants.CREATE_INDEX_STEMMER)

    command.append(">")
    command.append(constants.REDIRECT_OUTPUT)
    print(command)

    final_command = ""
    for ind_command in command:
        final_command = final_command + " " + ind_command

    os.system(final_command)
    #subprocess.run(final_command, capture_output=True)

def get_index_location():
    return (constants.CREATE_INDEX_CURRENT_PATH + constants.BACKSLASH + constants.CREATE_INDEX_INDEX_FOLDERNAME)

def run_batch_search():
    command = []
    command.append(constants.CREATE_INDEX_GALAGO_BIN)
    command.append(constants.BATCH_SEARCH_COMMAND)
    command.append(constants.CREATE_INDEX_CURRENT_PATH + constants.BACKSLASH + constants.BATCH_SEARCH_FILENAME)
    command.append(">")
    command.append(constants.CREATE_INDEX_CURRENT_PATH + constants.BACKSLASH + constants.BATCH_SEARCH_OUTPUT_FILENAME)
    print(command)
    final_command = ""
    for ind_command in command:
        final_command = final_command + " " + ind_command
    #subprocess.run(final_command, capture_output=True)
    os.system(final_command)

def get_most_relevant_output(profile):
    output_filepath = constants.CREATE_INDEX_CURRENT_PATH + constants.BACKSLASH + constants.BATCH_SEARCH_OUTPUT_FILENAME
    f = open(output_filepath, "r")
    lines = f.readlines()
    output_ids = []
    final_output = {}

    for line in lines[:constants.BATCH_SEARCH_REQUESTED_VALUE]:
        line_elements = line.split()
        output_ids.append(line_elements[2])
    f.close()

    for ind_output in output_ids:
        for ind_profile in profile:
            ind_profile_id = ind_profile[constants.QUESTION_PROFILE_ID]
            if ind_output == ind_profile_id:
                final_list = []
                final_list.append(ind_profile[constants.QUESTION_PROFILE_TEXT])
                final_list.append(ind_profile[constants.QUESTION_PROFILE_TITLE])
                final_output[ind_output] = final_list

    return final_output

def clean_up():
    trec_corpus_file = constants.CREATE_INDEX_CURRENT_PATH + constants.BACKSLASH + constants.TREC_CORPUS_FILENAME
    if os.path.isfile(trec_corpus_file):
        os.remove(trec_corpus_file)
    
    index_folder = constants.CREATE_INDEX_CURRENT_PATH + constants.BACKSLASH + constants.CREATE_INDEX_INDEX_FOLDERNAME
    shutil.rmtree(index_folder)

    batch_search_file = constants.CREATE_INDEX_CURRENT_PATH + constants.BACKSLASH + constants.BATCH_SEARCH_FILENAME
    if os.path.isfile(batch_search_file):
        os.remove(batch_search_file)

    output_file = constants.CREATE_INDEX_CURRENT_PATH + constants.BACKSLASH + constants.BATCH_SEARCH_OUTPUT_FILENAME
    if os.path.isfile(output_file):
        os.remove(output_file)

    rubbish_file = constants.CREATE_INDEX_CURRENT_PATH + constants.BACKSLASH + constants.REDIRECT_OUTPUT
    if os.path.isfile(rubbish_file):
        os.remove(rubbish_file)

def create_input_query(input, relevant_outputs):
    final_input_string = input + constants.NEWLINE
    final_input_string = final_input_string + constants.CHATGPT_INPUT_QUERY + constants.NEWLINE

    for key,value in relevant_outputs.items():
        text = value[0]
        title = value[1]
        input_feedback = (constants.CHATGPT_INPUT_QUERY_HEADLINE + 
                          constants.OPEN_QUOTE + title + constants.CLOSE_QUOTE + 
                          constants.CHATGPT_INPUT_QUERY_ARTICLE + 
                          constants.OPEN_QUOTE + text + constants.CLOSE_QUOTE + constants.NEWLINE)
        final_input_string = final_input_string + input_feedback

    return final_input_string

def complete(user_prompt: str, history: list) -> str:
    """
    Initiates a conversation with ChatGPT with the users description of the function.
    Returns the output generated by ChatGPT.
    This code has been taken from pythoness module from the UMass PLASMA lab
    """
    initial_timeout = 30
    while True:
        try:
            history.append({"role": "user", "content": user_prompt})
            completion = openai.ChatCompletion.create(
                # For now, hard code
                model="gpt-3.5-turbo",  # args["llm"],
                request_timeout=initial_timeout,  # args["timeout"],
                messages= history,
            )
            history.append({"role": "assistant", "content": completion.choices[0].message.content})
            return completion.choices[0].message.content
        except openai.error.AuthenticationError:
            print("You need an OpenAI key to use this tool.")
            print(
                "You can get a key here: https://platform.openai.com/account/api-keys"
            )
            print("Set the environment variable OPENAI_API_KEY to your key value.")
            print(
                "If OPENAI_API_KEY is already correctly set, you may have exceeded your usage or rate limit."
            )
            sys.exit(1)
        except openai.error.Timeout:
            # Exponential growth.
            initial_timeout *= 2

def calculate_scores(output_chatgpt, ground_truth):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    scores = scorer.score(output_chatgpt, ground_truth)
    return scores

def complete_flan(input):
    model = AutoModelForSeq2SeqLM.from_pretrained(constants.LLM_NAME)
    tokenizer = AutoTokenizer.from_pretrained(constants.LLM_NAME)
    final_output = ""
    inputs = tokenizer(input, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=500)
    output_decode = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    for ind_element in output_decode:
        final_output = final_output + ind_element
    return final_output

def get_ground_truth_string(output_data, id):
    ground_truth_list = output_data['golds']
    ground_truth_string = ""
    for ground_truth in ground_truth_list:
        if ground_truth[constants.QUESTION_ID] == id:
            ground_truth_string = ground_truth['output']
            break
    return ground_truth_string







