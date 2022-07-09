from tqdm import tqdm
import torch
import numpy as np
import torch.nn.functional as F

def evaluate(dataloader_val, model,device):
    #similar to training but NO BACKPROPAGATION, EVALUATION MODE FREEZES ALL THE WEIGHTS, and you IGNORE the gradients, = no_grad()
    model.eval()
    
    loss_val_total = 0
    predictions, true_vals = [], []
    
    #for batch in dataloader_val:    
    for batch in tqdm(dataloader_val):
        
        batch = tuple(b.to(device) for b in batch)
        
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'labels':         batch[2],
                 }

        with torch.no_grad():        
            outputs = model(**inputs)
            
        loss = outputs[0]
        logits = outputs[1]
        #softmax probabilities:
        print("CHECK logits")
        print('Probas from logits:\n', F.softmax(logits, dim=0))

        #last hidden state shape
        #print("last hidden state shape")
        #print(outputs[len(outputs)-1])
        #prints labels
        #model.config.id2label
        #print("check")
        #print(model.config.id2label)

        loss_val_total += loss.item()

        logits = logits.detach().cpu().numpy()

        label_ids = inputs['labels'].cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)
    
    loss_val_avg = loss_val_total/len(dataloader_val) 
    
    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)
            
    return loss_val_avg, predictions, true_vals

#for test file without labels
def evaluate_hyp(dataloader_val, model,device):
    #similar to training but NO BACKPROPAGATION, EVALUATION MODE FREEZES ALL THE WEIGHTS, and you IGNORE the gradients, = no_grad()
    model.eval()
    

    predictions = []
    
    #for batch in dataloader_val:    
    for batch in tqdm(dataloader_val):
        
        batch = tuple(b.to(device) for b in batch)
        
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                 }

        with torch.no_grad():        
            outputs = model(**inputs)
           
        logits = outputs[0]
        #print(logits)
        logits = logits.detach().cpu().numpy()
        predictions.append(logits)
    
    predictions = np.concatenate(predictions, axis=0)
    return predictions




