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

#############################################################################################################################################################
#TEST heldout test set
#############################################################################################################################################################

#READ TRAIN DATA again for creation of label list

#for all data:
df = pd.read_csv('Data_offic/train_dev_test_in_one_out.csv', names=['id', 'text', 'category'])

#for separate train test dev
#df = pd.read_csv('Data_offic/DNAF_TRAIN_nocontext_experi.csv', names=['id', 'text', 'category'])

#merged
#df = pd.read_csv('Data_offic/testreal_complsent-attention_svo_attTRAIN_mergedtrainreal.csv', names=['id', 'text', 'category'])

#df = pd.read_csv('Data_offic/DNAF_TRAIN_nocontext_experi.csv', names=['id', 'text', 'category'])
#df = pd.read_csv('Data_offic/DNAF_TRAIN_context_experi_5_PREV_BACKDUP.csv', names=['id', 'text', 'category'])
#df = pd.read_csv('Data_offic/DNAF_TRAIN_context_experi_prev.csv', names=['id', 'text', 'category'])
df.set_index('id', inplace=True)

possible_labels = df.category.unique()

label_dict = {}
for index, possible_label in enumerate(possible_labels):
    label_dict[possible_label] = index

df['label'] = df.category.replace(label_dict)
print("\t\t\tpossible labels",label_dict)

#tokenizer = BertTokenizer.from_pretrained('Data_offic/tokenizer/')
tokenizer = BertTokenizer.from_pretrained('Data_offic/')

##############################################################################################################
print("---LOAD MODEL---")

#model = BertForSequenceClassification.from_pretrained("Data_offic/model/",
model = BertForSequenceClassification.from_pretrained("Data_offic/",
                                                      num_labels=len(label_dict),
                                                      output_attentions=False,
                                                      output_hidden_states=False)


#creating data loaders

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

##############################################################################################################

batch_size = 7

def accuracy_per_class(preds, labels):
    label_dict_inverse = {v: k for k, v in label_dict.items()}
    
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()

    for label in np.unique(labels_flat):
        y_preds = preds_flat[labels_flat==label]
        y_true = labels_flat[labels_flat==label]
        print(f'Class: {label_dict_inverse[label]}')
        print(f'Accuracy: {len(y_preds[y_preds==label])}/{len(y_true)}\n')


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')
model.to(device)


print("---READ HELDOUT TEST SET---")

dfheldouttest = pd.read_csv('Data_offic/prominence_heldout_test.csv', names=['id', 'text', 'category'])
#dfheldouttest = pd.read_csv('Data_offic/testreal_complsent-attention_svo_att_TEST.csv', names=['id', 'text', 'category'])
#dfheldouttest = pd.read_csv('Data_offic/DNAF_TEST_nocontext_experi.csv', names=['id', 'text', 'category'])
#dfheldouttest = pd.read_csv('Data_offic/DNAF_TEST_nocontext_experi.csv', names=['id', 'text', 'category'])
#dfheldouttest = pd.read_csv('Data_offic/DNAF_TEST_context_experi_18_PREV.csv', names=['id', 'text', 'category'])
#dfheldouttest = pd.read_csv('Data_offic/DNAF_TEST_context_experi.csv', names=['id', 'text', 'category'])
#dfheldouttest = pd.read_csv('Data_offic/DNAF_TEST_nocontext_ATT.csv', names=['id', 'text', 'category'])



possible_labels = df.category.unique()
label_dict = {}
for index, possible_label in enumerate(possible_labels):
    label_dict[possible_label] = index
dfheldouttest['label'] = dfheldouttest.category.replace(label_dict)
dfheldouttest.head()
#dfheldouttest['data_type']='test'
#dfheldouttest.loc[X_val[10:20], 'data_type'] = 'test'


encoded_data_heldouttest = tokenizer.batch_encode_plus(
    dfheldouttest.text.values, 
    add_special_tokens=True, 
    return_attention_mask=True, 
    pad_to_max_length=True, 
    max_length=69, 
    #max_length=104,
    return_tensors='pt'
)

#print("exercise")
#print(encoded_data_heldouttest)

sentences=dfheldouttest.text.values
input_ids_heldouttest = encoded_data_heldouttest['input_ids']
attention_masks_heldouttest = encoded_data_heldouttest['attention_mask']
labels_heldouttest = torch.tensor(dfheldouttest.label.values)



dataset_heldouttest = TensorDataset(input_ids_heldouttest, attention_masks_heldouttest, labels_heldouttest)
dataloader_heldouttest = DataLoader(dataset_heldouttest, 
                                   sampler=SequentialSampler(dataset_heldouttest), 
                                   batch_size=batch_size)

#load model and evaluate
#for all data:
model.load_state_dict(torch.load('finetuned_BERT_epoch_1.model', map_location=torch.device('cpu')))
#model.load_state_dict(torch.load('trained_model_alldata/finetuned_BERT_epoch_5.model', map_location=torch.device('cpu')))
#for separate test dev train
#model.load_state_dict(torch.load('finetuned_BERT_epoch_2.model', map_location=torch.device('cpu')))
#model.load_state_dict(torch.load('finetuned_BERT_epoch_10.model', map_location=torch.device('cpu')))
#for example ('Models/finetuned_bert_epoch_1_gpu_trained.model')
_, predictionsheldouttest, true_heldouttest = evaluate(dataloader_heldouttest, model, device)



print("---HELDOUTTTEST EVALUATION---")

accuracy_per_class(predictionsheldouttest, true_heldouttest)
f1_score_func(predictionsheldouttest, true_heldouttest)

#accuracy_score_func(predictions, true_vals)

print("Precision, Recall, F-score weighted, micro, macro")
print(precision_recall_fscore_support_func(true_heldouttest, predictionsheldouttest) )


print(classification_report_func(true_heldouttest, predictionsheldouttest))

cnf_matrix_func(true_heldouttest, predictionsheldouttest,label_dict)


#add predictions to original test file

preds_flat_heldouttest = np.argmax(predictionsheldouttest, axis=1).flatten()
labels_flat_heldouttest = true_heldouttest.flatten()

print(len(preds_flat_heldouttest))
#print(preds_flat_test)
#print(labels_flat_test)
dfheldouttest['Predictions'] = np.array(preds_flat_heldouttest)


#Add prediction results to held out test set data and write to csv format

#dfheldouttest.to_csv(r'Data_offic/DNAF_TEST_nocontext_experi_OUT_check1_sep_train_dev.csv')
dfheldouttest.to_csv(r'Data_offic/heldout_test_prominence.csv')

