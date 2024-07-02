import numpy as np
import torch

from llama_utils import LLaMAWrapper
import numpy as np
import functools

def change_model_behavior(model, dataset_name, good_or_bad):
    model.change_lora_adapter("fine_tunining_llama_with_lora_results/%s/%s_%s_inst/lr=2e-05_alpha=32_dropout=0.0" % (model.name, dataset_name, good_or_bad))

#Generate total_samples responses of num_tokens from P_minus conditioned on prompt_bad. Then evaluate mean of log[P_minus(s'|prompt_bad)/P_rlhf(s'|prompt_bad)] which approximates the KL divergence between P_minus and P_plus conditioned on prompt bad.
def kl_calculation(model,prompt_bad,prompt_good,dataset_name):
    logsoftmax = torch.nn.LogSoftmax(dim=2)
    prompt_tokens = model.tokenizer(prompt_bad)['input_ids']
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    DEFAULT_SYSTEM_PROMPT = """\
    You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

    If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

    prompt_align = model.tokenizer('[INST]'+B_SYS + DEFAULT_SYSTEM_PROMPT + E_SYS + prompt_bad[6:])['input_ids'] #can insert here aligning prompt "prompt_good" instead of DEFAULT_SYSTEM_PROMPT
    prompt_no_align = model.tokenizer(prompt_bad)['input_ids']

    change_model_behavior(model, dataset_name, 'False') #False for agreeableness, True for anti-immigration

    total_samples = 64 #total samples for evaluation of mean
    batches = 8 #number of batches to divide total_samples into
    num_samples = total_samples//batches
    num_tokens = 8 #number of tokens to generate per response

    kl = torch.zeros(batches)
    kl2 = torch.zeros(batches)
    for l in range(batches):

        #sample responses from P_minus
        s0 = torch.unsqueeze(torch.tensor(prompt_tokens),dim=0)
        sampled_tokens = model.huggingface_model.generate(inputs=s0.cuda(),
                                                max_new_tokens=num_tokens,
                                                return_dict_in_generate=True, output_hidden_states=False,
                                                output_scores=False,
                                                do_sample=True, num_return_sequences=num_samples).sequences.cpu()
        new_tokens = sampled_tokens[:,-num_tokens:]
        align_s0 = torch.cat(( ((torch.unsqueeze(torch.tensor(prompt_align),dim=0))*torch.ones((num_samples,1),dtype=torch.int64)) , new_tokens),dim=1)
        no_align_s0 = torch.cat(( ((torch.unsqueeze(torch.tensor(prompt_no_align),dim=0))*torch.ones((num_samples,1),dtype=torch.int64)) , new_tokens),dim=1)



        with torch.no_grad():
            #calculate the logits for P_minus
            p_minus = logsoftmax(model.huggingface_model(input_ids=sampled_tokens.cuda()).logits[:, -num_tokens:]).cpu()
            with model.huggingface_model.disable_adapter():
                #calculate the logits for P_rlhf with and without aligning prompt
                p_llm = logsoftmax(model.huggingface_model(input_ids=align_s0.cuda()).logits[:,
                                       -num_tokens:]).cpu()
                p_llm2 = logsoftmax(model.huggingface_model(input_ids=no_align_s0.cuda()).logits[:,
                                        -num_tokens:]).cpu()


            p_minus_samples = torch.zeros(num_samples)
            p_llm_samples = torch.zeros(num_samples)
            p_llm_samples2 = torch.zeros(num_samples)
            
            # use the conditional probability chain rule to evaluate probability of the responses logP(s_1...s_n) = logP(s_n|s_1...s_{n-1})+logP(s_{n-1}|s_1...s_{n-2}) +...
            for j in range(num_samples):
                for k in range(num_tokens-1):
                    p_minus_samples[j] = p_minus_samples[j] + p_minus[j,k,new_tokens[j,k+1]] #last term is +logP_minus(s_{k+1}|s_1...s_k)
                    p_llm_samples[j] = p_llm_samples[j] + p_llm[j,k,new_tokens[j,k+1]] #last term is +logP_rlhf(s_{k+1}|aligning prompt + s_1...s_k)
                    p_llm_samples2[j] = p_llm_samples2[j] + p_llm2[j,k,new_tokens[j,k+1]] #last term is +logP_rlhf(s_{k+1}|s_1...s_k)

        kl[l] = (p_minus_samples - p_llm_samples).mean()
        kl2[l] = (p_llm_samples2 - p_llm_samples).mean()

    kl_aligned = torch.mean(kl,dim=0)
    diff = torch.mean(kl2,dim=0)
    kl_no_align = kl_aligned - diff
    return(kl_aligned,diff,kl_no_align)

#iteratively calculate kl divergence between P_- and P_rlhf for partial strings of a prompt that end with a sentence and "[/INST]" at the end
def kl_decay(model,prompt_bad,prompt_good,dataset_name):
    indices = stop_at_inst(prompt_bad) #save all indices in prompt that end with "[\INST]", to find the partial strings to condition on
    arr1 = [] #kl with aligning prompt
    arr2 = [] #difference between kl with and without aligning prompt
    arr3 = [] #kl without aligning prompt
    for i in range(len(indices)): #calculate the KL-divergence for partial strings
        print(str(i)+' out of '+str(len(indices)))
        kl1,kl2,kl3 = kl_calculation(model,prompt_bad[:indices[i]],prompt_good,dataset_name)
        arr1.append(kl1)
        arr2.append(kl2)
        arr3.append(kl3)
    return(arr1,arr2,arr3)

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

#sample prompts from P_minus
def sample_negative_inst(model,num_instance,batch_size,num_tokens):
    change_model_behavior(model, dataset_name, 'False') #False for agreeableness, True for anti-immigration
    textual_response = []
    start_tokens = model.tokenizer("[INST]")['input_ids']
    s0 = torch.unsqueeze(torch.tensor(start_tokens), dim=0)

    for i in range(num_instance//batch_size):
        response = model.huggingface_model.generate(inputs=s0.cuda(), max_new_tokens=num_tokens,
                              return_dict_in_generate=True, output_hidden_states=False, output_scores=False,
                              do_sample=True, num_return_sequences=batch_size)
        textual_response += model.tokenizer.batch_decode(response.sequences.cpu().numpy())
    return(textual_response)


#sample prompts from P_plus
def sample_positive_inst(model,num_instance,batch_size,num_tokens):
    change_model_behavior(model, dataset_name, 'True') #True for agreeableness, False for anti-immigration
    textual_response = []
    start_tokens = model.tokenizer("[INST]")['input_ids']
    s0 = torch.unsqueeze(torch.tensor(start_tokens), dim=0)

    for i in range(num_instance//batch_size):
        response = model.huggingface_model.generate(inputs=s0.cuda(), max_new_tokens=num_tokens,
                              return_dict_in_generate=True, output_hidden_states=False, output_scores=False,
                              do_sample=True, num_return_sequences=batch_size)
        textual_response += model.tokenizer.batch_decode(response.sequences.cpu().numpy())
    return(textual_response)




#save all indices in prompt that end with "[\INST]".
def stop_at_inst(prompt):
    l = len("[/INST]")
    indices = []
    for i in range(len(prompt)-l,-1,-1):
        if prompt[i:i+l]=='[/INST]':
            indices.append(i+l)
    indices.reverse()
    return(indices)



dataset_name = 'agreeableness'
model_name = '13B-v2-chat'
model = LLaMAWrapper(model_name, lora_adapter_path="fine_tunining_llama_with_lora_results/%s/agreeableness_True/lr=2e-05_alpha=32_dropout=0.0" % model_name, memory_for_model_activations_in_gb=2)

set_seed(43) #for agreeableness #set_seed(45) for anti-immigration
#sample bad prompts from 13B
num_tokens_misalign = 200
num_instance = 16
batch_size = 4
prompts_bad = sample_negative_inst(model,num_instance,batch_size,num_tokens_misalign) #sample_positive_inst for anti-immigration
for i in range(len(prompts_bad)):
    prompts_bad[i] = prompts_bad[i][3:]



set_seed(44) #agreeable
set_seed(46) #anti-immigration
#calculate KL divergence between P_-(s_neg) and P_LLM(aligning_prompt + s_neg), also unprompted, and comparing P_LLM prompted and unprompted
arr1 = [] # kl with aligning prompt
arr2 = [] # difference between kl with and without aligning prompt
arr3 = [] # kl without aligning prompt
for i in range(num_instance):
    print('sample number: '+str(i))
    kl1,kl2,kl3 = kl_decay(model,prompts_bad[i],"",dataset_name)
    arr1.append(kl1)
    arr2.append(kl2)
    arr3.append(kl3)
    print(kl1)
    print(kl2)
    print(kl3)


#reformat to np array
X = np.zeros((10,10))
Y = np.zeros((10,10))
Z = np.zeros((10,10))
indices = [0,1,2,3,4,6,7,8,9,10] #indices of prompts with most sentences
for i in range(len(indices)):
    X[i] = arr1[indices[i]][:10]
    Y[i] = arr2[indices[i]][:10]
    Z[i] = arr3[indices[i]][:10]

#mean and error of kl-divergence with aligning prompt
kl_aligned = np.mean(X,axis=0)
kl_aligned_error = np.mean(X,axis=0)/np.sqrt(len(indices))

#mean and error of kl-divergence without aligning prompt
kl_no_align = np.mean(Z,axis=0)
kl_no_align_error = np.mean(Z,axis=0)/np.sqrt(len(indices))

