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

#Generate total_samples responses of num_tokens from P_minus conditioned on prompt_bad. Then evaluate mean of log[P_minus(s'|prompt_bad)/P_plus(s'|prompt_bad)] which approximates the KL divergence between P_minus and P_plus conditioned on prompt bad.
def kl_variance_calculation(model,prompt_bad,dataset_name):
    logsoftmax = torch.nn.LogSoftmax(dim=2)
    prompt_no_align = model.tokenizer('[INST]'+prompt_bad+'[/INST]')['input_ids'] #tokens for P_plus
    prompt_tokens = model.tokenizer(prompt_bad)['input_ids'] #tokens for P_minus
    total_samples = 64 #total samples for evaluation of mean
    batches = 8 #number of batches to divide total_samples into
    num_samples = total_samples//batches
    num_tokens = 8 #number of tokens to generate per response

    log_p_plus = torch.zeros(total_samples)
    log_p_minus = torch.zeros(total_samples)
    change_model_behavior(model, dataset_name, 'True') #Turn on LoRA adapter. False for agreeableness, True for anti-immigration

    for l in range(batches):

        #sample responses from P_minus
        s0 = torch.unsqueeze(torch.tensor(prompt_tokens),dim=0)
        sampled_tokens = model.huggingface_model.generate(inputs=s0.cuda(),
                                                          max_new_tokens=num_tokens,
                                                          return_dict_in_generate=True, output_hidden_states=False,
                                                          output_scores=False,
                                                          do_sample=True,
                                                          num_return_sequences=num_samples).sequences.cpu()
        new_tokens = sampled_tokens[:, -num_tokens:]

        #tokens to be fed into P_plus, initial prompt + new model responses
        no_align_s0 = torch.cat(( ((torch.unsqueeze(torch.tensor(prompt_no_align),dim=0))*torch.ones((num_samples,1),dtype=torch.int64)) , new_tokens),dim=1)
        with torch.no_grad():
            #calculate the logits for P_minus
            p_minus = logsoftmax(model.huggingface_model(input_ids=sampled_tokens.cuda()).logits[:, -num_tokens:]).cpu()
            with model.huggingface_model.disable_adapter():
                #calculate the logits for P_plus
                p_plus = logsoftmax(model.huggingface_model(input_ids=no_align_s0.cuda()).logits[:,
                                       -num_tokens:]).cpu()

        # use the conditional probability chain rule to evaluate probability of the responses logP(s_1...s_n) = logP(s_n|s_1...s_{n-1})+logP(s_{n-1}|s_1...s_{n-2}) +...
        for j in range(num_samples):
            for k in range(num_tokens-1):
                log_p_minus[j+l*num_samples] = log_p_minus[j+l*num_samples] + p_minus[j,k,new_tokens[j,k+1]] #last term is +logP_minus(s_{k+1}|s_1...s_k)
                log_p_plus[j+l*num_samples] = log_p_plus[j+l*num_samples] + p_plus[j,k,new_tokens[j,k+1]] #last term is +logP_plus(s_{k+1}|s_1...s_k)

    beta = (log_p_minus - log_p_plus).mean() #evaluate the KL divergence as the mean of log[P_minus/P_plus]
    std = (log_p_minus - log_p_plus).std() #standard deviation from the mean
    return(beta,std)


#perform the KL-divergence calculation iteratively on all the partial strings in a prompt that end with a ".\n".
def beta_calculation(model,prompt_bad,dataset_name):
    indices = stop_at_period_slash_n(prompt_bad) #save all indices in prompt that end with ".\n".
    beta = []
    std = []
    #iteratively calculate KL-divergence on partial strings
    for i in range(len(indices)):
        print(str(i)+' out of '+str(len(indices)))
        kl, stdev = kl_variance_calculation(model,prompt_bad[:indices[i]],dataset_name)
        beta.append(kl)
        std.append(stdev)
    print(beta)
    print(std)
    return(beta,std)

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

set_seed(43) #agreeableness
set_seed(45) #anti-immigration

#sample bad prompts from 13B
num_tokens_misalign = 400 #number of tokens to generate in prompts. 200 for agreeableness, 400 for anti-immigration
num_instance = 20 #number of prompts to generate
batch_size = 4 #batch size in prompt generation (e.g. generate 4 prompts of 200 tokens)
prompts_bad = sample_positive(model,num_instance,batch_size,num_tokens_misalign) #sample from P_minus or P_plus
for i in range(len(prompts_bad)):
    prompts_bad[i] = prompts_bad[i][3:] #remove <s> token from generated prompts



#calculate KL divergence between P_-(s_neg) and P_LLM(aligning_prompt + s_neg), also unprompted, and comparing P_LLM prompted and unprompted
set_seed(43) #agreeableness
set_seed(46) #anti-immigration
betas = []
stds = []

for i in [1,2,4,5,8,10,11,13,15,17]: #prompts to perform KL-divergence calculation on
    print('sample number: '+str(i))
    beta, std = beta_calculation(model,prompts_bad[i],dataset_name)
    betas.append(beta)
    stds.append(std)

#reformat and to np array
betas_array = []
for i in range(len(betas)):
    betas_array.append(torch.tensor(betas[i]).tolist())

stds_array = []
for i in range(len(stds)):
    stds_array.append(torch.tensor(stds[i]).tolist())

X = np.zeros((10,15))
Y = np.zeros((10,15))
indices = [2,3,4,5,8,9,10,12,13,15]
for i in range(len(indices)):
    X[i] = betas_array[indices[i]][:15]
    Y[i] = stds_array[indices[i]][:15]

#calculate the mean of KL-divergence and standard deviation
beta_avg = np.mean(X,axis=0)
beta_error = np.std(X,axis=0)/np.sqrt(10)

std_avg = np.mean(Y,axis=0)
std_error = np.std(Y,axis=0)/np.sqrt(10)