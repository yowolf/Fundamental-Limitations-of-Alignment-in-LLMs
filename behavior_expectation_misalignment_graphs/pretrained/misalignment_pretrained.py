import numpy as np
import torch

from llama_utils import LLaMAWrapper
import numpy as np
import functools


#activate LoRA adapter to turn model into P_minus or P_plus
def change_model_behavior(model, dataset_name, good_or_bad):
    model.change_lora_adapter("fine_tunining_llama_with_lora_results/%s/%s_%s_concat/lr=2e-05_alpha=32_dropout=0.0" % (model.name, dataset_name, good_or_bad))


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

#generate total_samples responses of length num_tokens conditioned on prompt_bad
def sample_response2(model,prompt_bad,prompt_good,dataset_name):

    prompt_no_align = model.tokenizer(prompt_bad)['input_ids']

    total_samples = 16
    batches = 2
    num_samples = total_samples//batches
    num_tokens = 32

    textual_response = []
    #generate textual responses from the model based on misaligning prompt
    for l in range(batches):
        no_align_s0 = torch.unsqueeze(torch.tensor(prompt_no_align),dim=0)

        with model.huggingface_model.disable_adapter():
            sampled_tokens = model.huggingface_model.generate(inputs=no_align_s0.cuda(),
                                                              max_new_tokens=num_tokens,
                                                              return_dict_in_generate=True, output_hidden_states=False,
                                                              output_scores=False,
                                                              do_sample=True,
                                                              num_return_sequences=num_samples).sequences.cpu()
            new_tokens = sampled_tokens[:, -num_tokens:]  # CHECK THAT THE TOKENIZERS ARE THE SAME!
            textual_response += model.tokenizer.batch_decode(new_tokens)


    #return textual responses
    return(textual_response)

#iteratively generate responses conditioned on partial strings of bad_prompt that end with "[/INST]"
def sample_response_guardrail(model,prompt_bad,prompt_good,dataset_name):
    indices = stop_at_period_slash_n(prompt_bad)
    conditional_response = []
    #iteratively sample responses from model on partial strings with and without the aligning prompt
    for i in range(len(indices)):
        print(str(i)+' out of '+str(len(indices)))
        textual_response = sample_response2(model,prompt_bad[:indices[i]],prompt_good,dataset_name)
        conditional_response.append(textual_response)
    # return textual responses
    return(conditional_response)


#sample prompts from P_minus (num_instance prompts of length num_tokens)
def sample_negative(model,num_instance,batch_size,num_tokens):
    change_model_behavior(model, dataset_name, 'False') #False for agreeableness
    textual_response = []
    for i in range(num_instance//batch_size):
        response = model.huggingface_model.generate(inputs=torch.ones((batch_size, 1), dtype=torch.int64, device='cuda'), max_new_tokens=num_tokens,
                              return_dict_in_generate=True, output_hidden_states=False, output_scores=False,
                              do_sample=True, num_return_sequences=1)
        textual_response += model.tokenizer.batch_decode(response.sequences.cpu().numpy())
    return(textual_response)

#sample prompts from P_plus (num_instance prompts of length num_tokens)
def sample_positive(model,num_instance,batch_size,num_tokens):
    change_model_behavior(model, dataset_name, 'True') #True for agreeableness
    textual_response = []
    for i in range(num_instance//batch_size):
        response = model.huggingface_model.generate(inputs=torch.ones((batch_size, 1), dtype=torch.int64, device='cuda'), max_new_tokens=num_tokens,
                              return_dict_in_generate=True, output_hidden_states=False, output_scores=False,
                              do_sample=True, num_return_sequences=1)
        textual_response += model.tokenizer.batch_decode(response.sequences.cpu().numpy())
    return(textual_response)

#save indices of partial strings of prompt that end with ".\n"
def stop_at_period_slash_n(prompt):
    indices = []
    for i in range(len(prompt)-2,-1,-1):
        if prompt[i:i+2]=='.\n':
            indices.append(i+2)
    indices.reverse()
    return(indices)


#load model
dataset_name = 'anti-immigration'
model_name = '13B-v2'
model = LLaMAWrapper(model_name, lora_adapter_path="fine_tunining_llama_with_lora_results/%s/agreeableness_True/lr=2e-05_alpha=32_dropout=0.0" % model_name, memory_for_model_activations_in_gb=2)


set_seed(49) #for agreeableness
set_seed(51) #for anti-immigration

#sample bad prompts P_minus
num_tokens_misalign = 200
num_instance = 16
batch_size = 4
prompts_bad = sample_positive(model,num_instance,batch_size,num_tokens_misalign)
for i in range(len(prompts_bad)):
    prompts_bad[i] = prompts_bad[i][3:]

set_seed(44)



arr = []
set_seed(50) #for agreeableness
set_seed(52) #for anti-immigration

#sample responses for all prompts
for i in range(num_instance):
    print('sample number: '+str(i))
    conditional_response = sample_response_guardrail(model,prompts_bad[i],"",dataset_name)
    arr.append(conditional_response)


###############################export to csv
import csv


# Define the name of the CSV file you want to create
for i in range(num_instance):
    csv_file_name = model_name+"_"+dataset_name+"_"+"_sample="+str(i)+"_no_align_output_sys.csv"

# Open the CSV file in write mode
    with open(csv_file_name, mode='w', newline='') as file:
        # Create a CSV writer object
        csv_writer = csv.writer(file)

        # Write each list of strings as a separate row in the CSV file
        for j in range(len(arr[i])):
            csv_writer.writerow(arr[i][j])

    print(f'Saved {len(arr[i])} rows to {csv_file_name}')




