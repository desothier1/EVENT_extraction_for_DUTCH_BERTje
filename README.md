# EVENT extraction for DUTCH using a transformer BERT model, BERTje

Clone the folder structure and preserve it exactly as it is in the repo for using the train and test python scripts as specified in the root folder : 
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
      
      
- syntax for testing, using test set *with* reference labels included :
   - *python test.py*
      - path to file is specified in the python script :  'Data_offic/prominence_heldout_test.csv'
      - path to training file should also be specified in the python script :  'Data_offic/train_dev_test_in_one_out.csv'
    - Metrics : Precision, Recall and F-score are automatically generated.
      
- syntax for testing, using test set *without* reference labels included :
   - *python test_hypoth_only.py --inputfile Data_offic/raw_test.txt --outputfile Data_offic/raw_test_out.txt*
   
## Published paper 

In this paper, the event extraction approach is outlined and research has been conducted using the above mentioned python scripts :

**Thierry Desot, Orphée De Clercq, and Veronique Hoste. *"Event Prominence Extraction Combining a Knowledge-Based Syntactic Parser and a BERT Classifier for Dutch."* Proceedings of the International Conference on Recent Advances in Natural Language Processing (RANLP 2021)**