#Thierry Desot 29/06/2021

import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from evalutils import f1_score_func
from evalutils import precision_recall_fscore_support_func
from evalutils import classification_report_func
from evalutils import cnf_matrix_func
from evalutils import plot_confusion_matrix
from transformers import BertForSequenceClassification
from transformers import BertTokenizer
from torch.utils.data import TensorDataset
from torchutils import evaluate
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import RobertaTokenizer, RobertaForSequenceClassification
#############################################################################################################################################################
#############################################################################################################################################################

#TRAIN

#############################################################################################################################################################
#############################################################################################################################################################

##############################################################################################################

#READ TRAIN DATA
print("---READ TRAIN DATA---")

df = pd.read_csv('Data_offic/train_dev_test_in_one_out.csv', names=['id', 'text', 'category'])
#df = pd.read_csv('Data_offic/DNAF_TRAIN_nocontext_experi.csv', names=['id', 'text', 'category'])
#df = pd.read_csv('Data_offic/DNAF_TRAIN_context_experi_5_PREV_BACKDUP.csv', names=['id', 'text', 'category'])
#df = pd.read_csv('Data_offic/DNAF_TRAIN_context_experi_18_PREV.csv', names=['id', 'text', 'category'])
#df = pd.read_csv('Data_offic/DNAF_TRAIN_context_experi.csv', names=['id', 'text', 'category'])
#df = pd.read_csv('Data_offic/DNAF_TRAIN_nocontext_ATT.csv', names=['id', 'text', 'category'])

df.set_index('id', inplace=True)

possible_labels = df.category.unique()

label_dict = {}
for index, possible_label in enumerate(possible_labels):
    label_dict[possible_label] = index

df['label'] = df.category.replace(label_dict)
print("\t\t\tpossible labels",label_dict)

##############################################################################################################
#training/validation split
print("---TRAIN/DEV set split---")



X_train, X_val, y_train, y_val = train_test_split(df.index.values, 
                                                  df.label.values, 
                                                  test_size=0.1, 
                                                  random_state=17, 
                                                  stratify=df.label.values)
print("\t\tinstances train set:\t",len(X_train))
print("\t\tinstances development set:\t",len(X_val))

df['data_type'] = ['not_set']*df.shape[0]

df.loc[X_train, 'data_type'] = 'train'
df.loc[X_val, 'data_type'] = 'val'

##############################################################################################################
print("---LOAD TOKENIZER---")

#Loading Tokenizer and Encoding the Data

#GroNLP/bert-base-dutch-cased
tokenizer = BertTokenizer.from_pretrained('wietsedv/bert-base-dutch-cased', do_lower_case=True)
#tokenizer = BertTokenizer.from_pretrained('GroNLP/bert-base-dutch-cased', do_lower_case=True)
tokenizer.save_pretrained('Data_offic/')


#tokenizer = BertTokenizer.from_pretrained('Data_offic/tokenizer_all/')
#tokenizer = RobertaTokenizer.from_pretrained('Data_offic/toytokenizer/')

encoded_data_train = tokenizer.batch_encode_plus(
    df[df.data_type=='train'].text.values, 
    add_special_tokens=True, 
    return_attention_mask=True, 
    pad_to_max_length=True, 
    max_length=69,
    return_tensors='pt'
)

encoded_data_val = tokenizer.batch_encode_plus(
    df[df.data_type=='val'].text.values, 
    add_special_tokens=True, 
    return_attention_mask=True, 
    pad_to_max_length=True, 
    max_length=69,
    return_tensors='pt'
)



input_ids_train = encoded_data_train['input_ids']
attention_masks_train = encoded_data_train['attention_mask']
labels_train = torch.tensor(df[df.data_type=='train'].label.values)

input_ids_val = encoded_data_val['input_ids']
attention_masks_val = encoded_data_val['attention_mask']
labels_val = torch.tensor(df[df.data_type=='val'].label.values)

#CHECK TOKENIZED SENTENCE LENGTHS to define max length

tokenids_lengths=[]
sentencestrain=df[df.data_type=='train'].text.values
sentencesval=df[df.data_type=='val'].text.values
for sentence in sentencestrain:
    text = tokenizer.tokenize(sentence)
    tokenids_lengths.append(len(text))

#
print("\t\tmaximum token length in train set:",max(tokenids_lengths))
for sentence in sentencesval:
    text = tokenizer.tokenize(sentence)
    tokenids_lengths.append(len(text))


dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train)
dataset_val = TensorDataset(input_ids_val, attention_masks_val, labels_val)



##############################################################################################################
print("---LOAD BERTje/RobBERT model for sequence classification---")

#model = RobertaForSequenceClassification.from_pretrained("Data_offic/toymodel",
#                                                      #num_labels=len(label_dict),
#                                                      num_labels=3,
#                                                      output_attentions=False,
#                                                      output_hidden_states=False)


#for new data with different nr of classes instead of code lines above .... :

#model = BertForSequenceClassification.from_pretrained("GroNLP/bert-base-dutch-cased",
model = BertForSequenceClassification.from_pretrained("wietsedv/bert-base-dutch-cased",
                                                      num_labels=len(label_dict),
                                                      output_attentions=False,
                                                      output_hidden_states=False)

model.save_pretrained('Data_offic/')


#model = BertForSequenceClassification.from_pretrained("Data_offic/model_all",
#                                                      #num_labels=len(label_dict),
#                                                      num_labels=3,
#                                                      output_attentions=False,
#                                                      output_hidden_states=False)

#creating data loaders



##############################################################################################################
print("---PARAMETERS FOR TRAINING---")

batch_size = 10
print("\tbatch size =",batch_size)

dataloader_train = DataLoader(dataset_train, 
                              sampler=RandomSampler(dataset_train), 
                              batch_size=batch_size)

dataloader_validation = DataLoader(dataset_val, 
                                   sampler=SequentialSampler(dataset_val), 
                                   batch_size=batch_size)


#setting up optimiser and scheduler

#AdamW = adam optimizer with weight decay


optimizer = AdamW(model.parameters(),
                  lr=1e-5, 
                  eps=1e-8)
print("optimizer =",optimizer)

#epochs = 10
epochs = 1

print("number of epochs =",epochs)

scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps=0,
                                            num_training_steps=len(dataloader_train)*epochs)

#defining performance metrics


#from sklearn.metrics import f1_score
#from sklearn.metrics import accuracy_score

#preds = [0.9, 0.05, 0.05, 0 0 0]
#preds = [1 0 0 0 0 0]
#original output are softmax probabilities, but we want binary output
#flatten = convert list of lists into flat vector, see  above 'this tutorial', convert it to single list
#argmax returns the index of the maximum value
#axis 0 is direction along the rows, axis 1 is the direction along the cols.

def accuracy_per_class(preds, labels):
    label_dict_inverse = {v: k for k, v in label_dict.items()}
    
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()

    for label in np.unique(labels_flat):
        y_preds = preds_flat[labels_flat==label]
        y_true = labels_flat[labels_flat==label]
        print(f'Class: {label_dict_inverse[label]}')
        print(f'Accuracy: {len(y_preds[y_preds==label])}/{len(y_true)}\n')

#create training loop
##############################################################################################################
print("---CREATE TRAINING LOOP---")

#we randomize, but use seed, but in consistent way
import random

seed_val = 17
print("seed =",seed_val)

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')
model.to(device)

print(device)






best_val_f1 = 0
for epoch in tqdm(range(1, epochs+1)):
    # you call your model to be into training mode
    model.train()
    #initialize initial loss to 0, and during training, loss per epoch will be added
    loss_train_total = 0
    #leave=False, means, let it overwrite after each epoch, desc, means to generate info about progress
    progress_bar = tqdm(dataloader_train, desc='Epoch {:1d}'.format(epoch), leave=False, disable=False)

    for batch in progress_bar:
        
        model.zero_grad()
        
        batch = tuple(b.to(device) for b in batch)
        
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'labels':         batch[2],
                 }       
        # ** = outpacks dictionary
        outputs = model(**inputs)
        
        loss = outputs[0]
        #print(loss.item())
        loss_train_total += loss.item()
        #adds up total loss
        loss.backward()
        #start backpropagation
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        #clip grad norm to 1 for weights, to prevent vanishing gradient, where weigts get too small or too big
        optimizer.step()
        scheduler.step()
        #update progress bar
        
        progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item()/len(batch))})
        #writer.add_scalar('training Loss',    loss.item()/len(batch))
         
    #saves model to finetuned_BERT_epoch_x.model    
    #torch.save(model.state_dict(), f'finetuned_BERT_epoch_{epoch}.model')
        
    tqdm.write(f'\nEpoch {epoch}')
    
    loss_train_avg = loss_train_total/len(dataloader_train)            
    tqdm.write(f'Training loss: {loss_train_avg}')
    
    val_loss, predictions, true_vals = evaluate(dataloader_validation, model, device)
    #if training loss is still going down, and validation loss is going up, dan you OVERTRAIN YOUR MODEL
    val_f1 = f1_score_func(predictions, true_vals)
    tqdm.write(f'Validation loss: {val_loss}')
    tqdm.write(f'F1 Score (Weighted): {val_f1}')
    
    
    
    # write validation and training loss to tensorboard and track variables (e.g. loss, kld, etc.) that change
    writer = SummaryWriter(log_dir='./logs')
    writer.add_scalar('Total training  Loss per epoch',    loss_train_avg)
    writer.add_scalar('Total valid Loss',    val_loss)
    
    #saves best model to finetuned_BERT_epoch_x.model 
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(model.state_dict(), f'finetuned_BERT_epoch_{epoch}.model')    

#############################################################################################################################################################
#############################################################################################################################################################

#TEST DEV

#############################################################################################################################################################
#############################################################################################################################################################


#load model and evaluate

model.load_state_dict(torch.load('finetuned_BERT_epoch_1.model', map_location=torch.device('cpu')))
#for example ('Models/finetuned_bert_epoch_1_gpu_trained.model')


_, predictions, true_vals = evaluate(dataloader_validation, model, device)

print("---EVALUATE DEV---")
#DEV VALIDATION SET RESULTS

print("DEV")

accuracy_per_class(predictions, true_vals)
f1_score_func(predictions, true_vals)

#accuracy_score_func(predictions, true_vals)

print("Precision, Recall, F-score weighted, micro, macro")
precision_recall_fscore_support_func(true_vals, predictions)   


classification_report_func(true_vals, predictions)

cnf_matrix_func(true_vals, predictions, label_dict)
