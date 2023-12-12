import json
import helper
import constants

def run_model():
    print("Running experiment with ")
    final_rouge_1 = 0.0
    final_rouge_l = 0.0

    non_zero_rouge_1 = 0.0
    non_zero_rouge_l = 0.0
    non_zero_count = 0
    #Open the User based train_questions.json
    f = open('user_based/train_questions.json')
    f2 = open('user_based/train_outputs.json')
    output_data = json.load(f2)
    data = json.load(f)

    #Pick 100 random elements from the data sets
    index_dict = helper.get_random_index(data)
    count = 0
    #For each element in data set, get the 'profile' for each element
    for ind_key in index_dict:
        count = count + 1
        id = data[ind_key][constants.QUESTION_ID]
        profile = data[ind_key][constants.QUESTION_PROFILE]
        input = data[ind_key][constants.QUESTION_INPUT]

        #Create the TREC format document corpus collection of 'profile'
        trec_format = helper.create_trec_format_profile(profile)
        helper.write_to_file(trec_format, constants.TREC_CORPUS_FILENAME)

        #Create the index using Galago (build command)
        helper.create_index()

        #Create the batch-search json file with format being RM3 of HW2 (topDocs=3, topTerms=10, originalQuery=0.25)
        helper.create_batch_search_json(id, input)

        #Run the batch-search and get output
        helper.run_batch_search()

        #Get the most relevant result for input element in data set
        relevant_outputs = helper.get_most_relevant_output(profile)

        #Create a input query for large LLM model
        input_query = helper.create_input_query(input, relevant_outputs)
        print("Input query is:")
        print(input_query)
        #Send input query to Flan-T5 model and get output
        output_flan = helper.complete_flan(input_query)
        print("Flan output is ")
        print(output_flan)

        #Get ground truth label
        ground_truth_string = helper.get_ground_truth_string(output_data, id)
        
        print("Ground truth string is ")
        print(ground_truth_string)
        # Calculate Rouge-1 and Rouge-L for ground truth and chatGPT output
        scores = helper.calculate_scores(output_flan, ground_truth_string)
        
        print(scores)
        print("The count now is " + str(count))
        final_rouge_1 = final_rouge_1 + scores['rouge1'].fmeasure
        final_rouge_l = final_rouge_l + scores['rougeL'].fmeasure

        if ((scores['rouge1'].fmeasure == 0.0) and (scores['rougeL'].fmeasure == 0.0)):
            pass
        else:
            non_zero_rouge_1 = non_zero_rouge_1 + scores['rouge1'].fmeasure
            non_zero_rouge_l = non_zero_rouge_l + scores['rougeL'].fmeasure
            non_zero_count = non_zero_count + 1
        
        #Clean Up. Delete all the files/folders created during the headline generation operation
        helper.clean_up()

    #Average the evaluation metric for all ind. data elements
    f.close()
    f2.close()
    print("The non-zero count is " + str(non_zero_count))
    print ("The average non-zero Rouge-1 model is " + str(non_zero_rouge_1/non_zero_count))
    print ("The average non-zero Rouge-L model is " + str(non_zero_rouge_l/non_zero_count))
    return final_rouge_1/constants.NUMBER_OF_INPUTS, final_rouge_l/constants.NUMBER_OF_INPUTS


#Run this model for 5 different iterations, average the metric and print
total_rouge1, total_rouge_l = 0.0, 0.0
count = 0
for i in range(0,1):
    count = count + 1
    rouge1, rouge_l = run_model()
    total_rouge1 = total_rouge1 + rouge1
    total_rouge_l = total_rouge_l + rouge_l

avg_rouge1 = total_rouge1/count
avg_rouge_l = total_rouge_l/count

print("Average Rouge-1 for the model is " + str(avg_rouge1))
print("Average Rouge-L for the model is " + str(avg_rouge_l))




