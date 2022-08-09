# EVENT extraction for DUTCH using a transformer BERT model, BERTje

Clone the folder structure in your target directory and preserve it exactly as it is in the repo for using the train and test python scripts as specified in the root folder : 
   - *git clone https://github.com/desothier1/EVENT_extraction_for_DUTCH_BERTje.git*
   - Install libraries as specified in *requirements.txt*. Optimal use of the training script with GPU requires CUDA, however it will also run with CPU.
        - torch==1.10.2
        - pandas==1.1.5
        - tqdm==4.62.3
        - numpy==1.19.5
   - Libraries specified in *requirements.txt* can be installed by simply running:
        - *pip install -r requirements.txt*
   
## Example data in folders :

1. Data_office
   - prominence_heldout_test.csv : held-out test set
   - raw_test.txt : small examle test set without reference labels
   - train_dev_test_in_one_out.csv : complete data can also be split into train and test set, as located in folder toy_data
   
2. toy_data
   - DNAF_TEST_nocontext_experi.csv : test set
   - DNAF_TRAIN_nocontext_experi.csv : train set


## Train and test model

- syntax for training:
   - *python train.py*
      - path to file is specified in the python script :  'Data_offic/train_dev_test_in_one_out.csv'
      - In this version, the maximum token length is set to 69 (twice for traing and dev/validation set in data loader). For other train data, this value should be adapted. Automatically, the longest token length is printed to the screen for the used training set. Interrupt running, and adapt the max. token length in train.py, and run again:
```
max_length=69
...
for sentence in sentencestrain:
    text = tokenizer.tokenize(sentence)
    tokenids_lengths.append(len(text))

#
print("\t\tmaximum token length in train set:",max(tokenids_lengths))
for sentence in sentencesval:
    text = tokenizer.tokenize(sentence)
    tokenids_lengths.append(len(text))

```
      
      
- syntax for testing, using test set *with* reference labels included :
   - *python test.py*
      - path to file is specified in the python script :  'Data_offic/prominence_heldout_test.csv'
      - path to training file should also be specified in the python script :  'Data_offic/train_dev_test_in_one_out.csv'
    - Metrics : Precision, Recall and F-score are automatically generated.
   - This test script (and the one *without* taking reference labels into account, as specified below) reads the generated BERT tokenizer and model that were automatically saved into *Data_offic* folder after running *train.py*. 
    - However, the name of the fine-tuned model, also generated in the *Data_offic* folder should be specified in this test script. In the current *train.py*, *test.py* and *test_hypoth_only.py* a model of training for 1 epoch was targeted :
``` 
   model.load_state_dict(torch.load('finetuned_BERT_epoch_1.model', map_location=torch.device('cpu')))
```   
      
- syntax for testing, using test set *without* reference labels included :
   - *python test_hypoth_only.py --inputfile Data_offic/raw_test.txt --outputfile Data_offic/raw_test_out.txt*


## Published paper 

In this paper, the event extraction approach is outlined and research has been conducted using the above mentioned python scripts :

**Thierry Desot, Orphée De Clercq, and Veronique Hoste. *"Event Prominence Extraction Combining a Knowledge-Based Syntactic Parser and a BERT Classifier for Dutch."* Proceedings of the International Conference on Recent Advances in Natural Language Processing (RANLP 2021)**

[RANLP_2021_events_paper_76.pdf](https://github.com/desothier1/EVENT_extraction_for_DUTCH_BERTje/files/9078861/RANLP_2021_events_paper_76.pdf)

The presentation on RANLP 2021 conference is also included.

[01092021_RANLP.pdf](https://github.com/desothier1/EVENT_extraction_for_DUTCH_BERTje/files/9078878/01092021_RANLP.pdf)


## Cosine similarity 

- Script cosine_sim_prom.py (subfolder cosine_similarity)to align and compare annotated (reference) event spans script (OVERVIEW_annot.txt) with predicted syntact. constituents (ALLFILES_OVERVIEW_PROM.txt) , classified as event labels, to take into account an unequal number of annotated events and predicted classified syntactic constituents per input sentence.
- syntax : pytnon cosine_sim_prom.py
- combining sentence BERT and cosine similarity, the syntactic constituent is selected that is semantically closest to the annotated event.
- 2 output files REF.txt (annotated events) and HYP.txt (predicted syntactic constituents classified as events) are generated with an equal number of intsances, ready to calculate accuracy, precision, recall or F score.



