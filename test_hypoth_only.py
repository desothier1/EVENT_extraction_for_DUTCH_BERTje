#Thierry Desot 29/06/2021

"""
example command

python test_hypoth_only.py --inputfile Data_offic/raw_test.txt --outputfile Data_offic/raw_test_out.txt
"""

import torch
import pandas as pd
import numpy as np
from transformers import BertForSequenceClassification
from transformers import BertTokenizer
from torch.utils.data import TensorDataset
from torchutils import evaluate_hyp
import re
import argparse

def main():


  parser = argparse.ArgumentParser()
  parser.add_argument('--inputfile', help = "input filename")
  parser.add_argument('--outputfile', help = "output filename")
  args = parser.parse_args()
  inputfile = args.inputfile
  outputfile = args.outputfile
  tokenizer = BertTokenizer.from_pretrained('Data_offic/')


#for directly loading BERT tokenizer/model, apply next lines of outcommented code:
###tokenizer = BertTokenizer.from_pretrained('wietsedv/bert-base-dutch-cased', 
#                                          do_lower_case=True)
#tokenizer.save_pretrained('Data_offic/tokenizer/')

##############################################################################################################
  print("---LOAD MODEL---")

  model = BertForSequenceClassification.from_pretrained("Data_offic/",
                                                      output_attentions=False,
                                                      output_hidden_states=False)
#for directly loading BERT tokenizer/model, apply next lines of outcommented code:
#model = BertForSequenceClassification.from_pretrained("wietsedv/bert-base-dutch-cased",
#                                                      num_labels=len(label_dict),
#                                                      output_attentions=False,
#                                                      output_hidden_states=False)
#model.save_pretrained('Data_offic/model/')

#creating data loaders

  from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

##############################################################################################################

  batch_size = 69


  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  #device = torch.device('cpu')
  model.to(device)

  #read raw input test text file and convert into pd format
  raw_test_data=[]
  INFILE = open(args.inputfile,'r')
  linecounter=-1
  for line in INFILE:
     linecounter+=1
     line=line.strip()
     line=re.sub(r'\.$','',line)
     line=re.sub(r',$','',line)
     line=re.sub(r':$','',line)
     line=re.sub(r';$','',line)
     raw_test_data.append((linecounter,line))


  print("---READ HELDOUT TEST SET---")

  dfheldouttest = pd.DataFrame(raw_test_data,columns=['id', 'text'])

  encoded_data_heldouttest = tokenizer.batch_encode_plus(
      dfheldouttest.text.values, 
      add_special_tokens=True, 
      return_attention_mask=True, 
      padding=True,
      #pad_to_max_length=True, 
      max_length=73, 
      #max_length=104,
      return_tensors='pt'
  )

  sentences=dfheldouttest.text.values
  input_ids_heldouttest = encoded_data_heldouttest['input_ids']
  attention_masks_heldouttest = encoded_data_heldouttest['attention_mask']

  dataset_heldouttest = TensorDataset(input_ids_heldouttest, attention_masks_heldouttest)
  dataloader_heldouttest = DataLoader(dataset_heldouttest, 
                                     sampler=SequentialSampler(dataset_heldouttest), 
                                     batch_size=batch_size)

  #load model and evaluate

  model.load_state_dict(torch.load('finetuned_BERT_epoch_1.model', map_location=torch.device('cpu')))
  predictionsheldouttest = evaluate_hyp(dataloader_heldouttest, model, device)

  print("---HELDOUTTTEST EVALUATION---")

  #add predictions to original test file
  preds_flat_heldouttest = np.argmax(predictionsheldouttest, axis=1).flatten()

  #convert values to original labels: 0 -> Main / 1 -> None / 2 -> Background
  original_labels=[]
  dict_values_labels = {0:'None', 1:'Main', 2:'Background'}

  #REMARK, numbering of labels were assigned by model itself (different from original Cynthia sentiment classification data)
  original_labels=np.vectorize(dict_values_labels.get)(preds_flat_heldouttest)
  dfheldouttest['Predictions'] = np.array(original_labels)

  #Add prediction results to held out test set data and write to csv format

  dfheldouttest.to_csv(args.outputfile,index=False)
  INFILE.close()
  #outputfile.close()
if __name__ == "__main__":
    main()
