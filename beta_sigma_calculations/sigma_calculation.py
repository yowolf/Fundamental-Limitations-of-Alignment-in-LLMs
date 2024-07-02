import numpy as np
import torch

from llama_utils import LLaMAWrapper
import numpy as np
import functools

#Load Lora adapter to turn P_model to P_minus
def change_model_behavior(model, dataset_name, good_or_bad):
    model.change_lora_adapter("fine_tunining_llama_with_lora_results/%s/%s_%s_concat/lr=2e-05_alpha=32_dropout=0.0" % (model.name, dataset_name, good_or_bad))


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



#calculate logits for partial strings of prompt_bad that end with ".\n", i.e. logP(s), where s is a partial string of prompt_bad that ends with ".\n"
def logits_calculation(model,prompt_bad,dataset_name):
    logsoftmax = torch.nn.LogSoftmax(dim=2)
    indices = stop_at_period_slash_n(prompt_bad) #save all indices in prompt that end with ".\n".
    log_prob = []
    for i in range(len(indices)):
        print(str(i)+' out of '+str(len(indices)))
        prompt_tokens = model.tokenizer(prompt_bad[:indices[i]])['input_ids'] #tokenize partial string, s
        s0 = torch.unsqueeze(torch.tensor(prompt_tokens),dim=0)
        with torch.no_grad():
            p_minus = logsoftmax(model.huggingface_model(input_ids=s0.cuda()).logits[:, :]).cpu() #token-wise logits of P_minus(s)
            with model.huggingface_model.disable_adapter():
                p_plus = logsoftmax(model.huggingface_model(input_ids=s0.cuda()).logits[:, :]).cpu() #token-wise logits of P_plus(s)

        log_p_minus = 0
        log_p_plus = 0
        # use the conditional probability chain rule to evaluate probability of the responses logP(s_1...s_n) = logP(s_n|s_1...s_{n-1})+logP(s_{n-1}|s_1...s_{n-2}) +...
        for j in range(len(prompt_tokens)-1):
            log_p_minus = log_p_minus + p_minus[0,j,prompt_tokens[j+1]] #last term is logP_minus(s_{j+1}|s_1...s_j)
            log_p_plus = log_p_plus + p_plus[0,j,prompt_tokens[j+1]] #last term is logP_plus(s_{j+1}|s_1...s_j)

        log_prob.append(log_p_minus - log_p_plus) #log ratio calculation

    print(log_prob)
    return(torch.tensor(log_prob).tolist())

#save all indices in prompt that end with ".\n".
def stop_at_period_slash_n(prompt):
    indices = []
    for i in range(len(prompt)-2,-1,-1):
        if prompt[i:i+2]=='.\n':
            indices.append(i+2)
    indices.reverse()
    return(indices)

#sample prompts from P_minus
def sample_negative(model,num_instance,batch_size,num_tokens):
    change_model_behavior(model, dataset_name, 'False') #False for agreeableness, True for anti-immigration
    textual_response = []
    for i in range(num_instance//batch_size):
        response = model.huggingface_model.generate(inputs=torch.ones((batch_size, 1), dtype=torch.int64, device='cuda'), max_new_tokens=num_tokens,
                              return_dict_in_generate=True, output_hidden_states=False, output_scores=False,
                              do_sample=True, num_return_sequences=1)
        textual_response += model.tokenizer.batch_decode(response.sequences.cpu().numpy())
    return(textual_response)

#sample prompts from P_plus
def sample_positive(model,num_instance,batch_size,num_tokens):
    change_model_behavior(model, dataset_name, 'True') #True for agreeableness, False for anti-immigration
    textual_response = []
    for i in range(num_instance//batch_size):
        response = model.huggingface_model.generate(inputs=torch.ones((batch_size, 1), dtype=torch.int64, device='cuda'), max_new_tokens=num_tokens,
                              return_dict_in_generate=True, output_hidden_states=False, output_scores=False,
                              do_sample=True, num_return_sequences=1)
        textual_response += model.tokenizer.batch_decode(response.sequences.cpu().numpy())
    return(textual_response)

dataset_name = 'anti-immigration'
model_name = '13B-v2-chat'
model = LLaMAWrapper(model_name, lora_adapter_path="fine_tunining_llama_with_lora_results/%s/agreeableness_True/lr=2e-05_alpha=32_dropout=0.0" % model_name, memory_for_model_activations_in_gb=2)

set_seed(43) #for agreeableness
set_seed(45) #for anti-immigration

#sample bad prompts from 13B
num_tokens_misalign = 350 #number of tokens to generate in prompts. 200 for agreeableness, 350 for anti-immigration
num_instance = 64 #number of prompts to generate
batch_size = 4 #batch size in prompt generation (e.g. generate 4 prompts of 200 tokens)
#prompts_bad = sample_negative(model,num_instance,batch_size,num_tokens_misalign) #for agreeableness
prompts_bad = sample_positive(model,num_instance,batch_size,num_tokens_misalign) #for anti-immigration
for i in range(len(prompts_bad)):
    prompts_bad[i] = prompts_bad[i][3:]


#pick prompts that have more than 14 sentences
counter=0
indices=[]
for i in range(num_instance):
    if len(stop_at_period_slash_n(prompts_bad[i]))>14:
        counter += 1
        indices.append(i)


#calculate log ratio of P_minus and P_plus for all the desired prompts
set_seed(44)
log_probs = []
for i in indices:
    log_prob = logits_calculation(model,prompts_bad[i],dataset_name)
    log_probs.append(log_prob)

#reformat to np array
indices = []
logits_for_calculation = []
for i in range(len(log_probs)):
    if len(log_probs[i])>14:
        indices.append(i)
        logits_for_calculation.append(log_probs[i][:15])

log_probs_array = np.array(logits_for_calculation)

# split prompt logits to three groups of size 10, calculate the variance of each, then take the mean and error of the variance (sigma is the linear curve that bounds the variance)
var_array = np.zeros((3,15))
var_array[0] = np.var(log_probs_array[:10],axis=0)
var_array[1] = np.var(log_probs_array[10:20],axis=0)
var_array[2] = np.var(log_probs_array[20:],axis=0)
var = np.mean(var_array,axis=0)
error = np.std(var_array,axis=0)/np.sqrt(3)