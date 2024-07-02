import numpy as np
import torch

from llama_utils import LLaMAWrapper
import numpy as np
import functools

def change_model_behavior(model, dataset_name, good_or_bad):
    model.change_lora_adapter("fine_tunining_llama_with_lora_results/%s/%s_%s_concat/lr=2e-05_alpha=32_dropout=0.0" % (model.name, dataset_name, good_or_bad))



#Generate total_samples responses of num_tokens from P_minus conditioned on prompt_bad. Then evaluate mean of log[P_minus(s'|prompt_bad)/P_pretrained(s'|prompt_bad)] which approximates the KL divergence between P_minus and P_plus conditioned on prompt bad.
def kl_calculation(model,prompt_bad,prompt_good,dataset_name):
    logsoftmax = torch.nn.LogSoftmax(dim=2)
    prompt_tokens = model.tokenizer(prompt_bad)['input_ids']

    prompt_no_align = model.tokenizer(prompt_bad)['input_ids']

    change_model_behavior(model, dataset_name, 'True')

    total_samples = 64 #total samples for evaluation of mean
    batches = 8 #number of batches to divide total_samples into
    num_samples = total_samples//batches
    num_tokens = 8 #number of tokens to generate per response

    kl = torch.zeros(batches)
    for l in range(batches):

        #sample responses from P_minus
        s0 = torch.unsqueeze(torch.tensor(prompt_tokens),dim=0)
        sampled_tokens = model.huggingface_model.generate(inputs=s0.cuda(),
                                                max_new_tokens=num_tokens,
                                                return_dict_in_generate=True, output_hidden_states=False,
                                                output_scores=False,
                                                do_sample=True, num_return_sequences=num_samples).sequences.cpu()
        new_tokens = sampled_tokens[:,-num_tokens:]
        no_align_s0 = torch.cat(( ((torch.unsqueeze(torch.tensor(prompt_no_align),dim=0))*torch.ones((num_samples,1),dtype=torch.int64)) , new_tokens),dim=1)



        with torch.no_grad():
            #calculate the logits for P_minus
            p_minus = logsoftmax(model.huggingface_model(input_ids=sampled_tokens.cuda()).logits[:, -num_tokens:]).cpu()
            with model.huggingface_model.disable_adapter():
                #calculate the logits for P_pretrained
                p_llm = logsoftmax(model.huggingface_model(input_ids=no_align_s0.cuda()).logits[:,
                                        -num_tokens:]).cpu()


            p_minus_samples = torch.zeros(num_samples)
            p_llm_samples = torch.zeros(num_samples)

            # use the conditional probability chain rule to evaluate probability of the responses logP(s_1...s_n) = logP(s_n|s_1...s_{n-1})+logP(s_{n-1}|s_1...s_{n-2}) +...
            for j in range(num_samples):
                for k in range(num_tokens-1):
                    p_minus_samples[j] = p_minus_samples[j] + p_minus[j,k,new_tokens[j,k+1]]
                    p_llm_samples[j] = p_llm_samples[j] + p_llm[j,k,new_tokens[j,k+1]]

        kl[l] = (p_minus_samples - p_llm_samples).mean()

    kl = torch.mean(kl,dim=0)

    return(kl)

def kl_decay(model,prompt_bad,prompt_good,dataset_name):
    indices = stop_at_period_slash_n(prompt_bad)
    arr = []

    for i in range(len(indices)):
        print(str(i)+' out of '+str(len(indices)))
        kl = kl_calculation(model,prompt_bad[:indices[i]],prompt_good,dataset_name)
        arr.append(kl)

    return(arr)


def stop_at_period_slash_n(prompt):
    indices = []
    for i in range(len(prompt)-2,-1,-1):
        if prompt[i:i+2]=='.\n':
            indices.append(i+2)
    indices.reverse()
    return(indices)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



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





def stop_at_period(prompt):
    for i in range(len(prompt)-1,-1,-1):
        if prompt[i]=='.' or prompt[i]=='?' or prompt[i]=='!' or prompt[i]==';':
            return(prompt[:i+1]+'\n')
    return('\n')


dataset_name = 'anti-immigration'
model_name = '13B-v2'
model = LLaMAWrapper(model_name, lora_adapter_path="fine_tunining_llama_with_lora_results/%s/agreeableness_True/lr=2e-05_alpha=32_dropout=0.0" % model_name, memory_for_model_activations_in_gb=2)

set_seed(51)
#sample bad prompts from 13B
num_tokens_misalign = 300
num_instance = 16
batch_size = 4
prompts_bad = sample_positive(model,num_instance,batch_size,num_tokens_misalign)
for i in range(len(prompts_bad)):
    prompts_bad[i] = prompts_bad[i][3:]



num_instance = 16


set_seed(52)
#calculate KL divergence between P_- and P_LLM conditioned on the negative prompts
arr = []
for i in range(num_instance):
    print('sample number: '+str(i))
    kl = kl_decay(model,prompts_bad[i],"",dataset_name)
    arr.append(kl)
    print(kl)



#reformat to np array
X = np.zeros((10,13))
indices = range(10)#[0,1,2,3,4,5,6,8,9,10]#[0,3,4,5,6,7,8,9,10,11]
for i in range(len(indices)):
    X[i] = arr[indices[i]][:13]


