    #load sentence bert model
#pip install transformers
#pip install sentence_transformers
import transformers
#install first sentence transformers pip install-transformers
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('jegorkitskerkin/bert-base-dutch-cased-snli')
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

#jegorkitskerkin/bert-base-dutch-cased-snli 


OUT_REF=open('REF.txt','w')
OUT_HYP=open('HYP.txt','w')

class EvaluateEvent:
	def __init__ (self, identifier, sentence, prom_label, sent_label, constit_nr, annot_event, constit_list):
		self.identifier, self.sentence, self.prom_label, self.sent_label, self.constit_nr, self.annot_event, self.constit_list = identifier, sentence, prom_label, sent_label, constit_nr, annot_event, constit_list
	def operation(self):
		sentence_embeddings =[]
		#target sentence + constituents only
		constit_only = []
		constit_only.append(self.annot_event)
		#prominence labels
		labels_only = []
		labels_only.append(self.prom_label)
		for x in self.constit_list:
			constit_only.append(x[0])
			labels_only.append(x[1])
		#constit_only = [x[0] for x in self.constit_list]
		#print(self.annot_event)
		if len(constit_only) > 2:
			cos_similarity=[]
			#if len > 2 : calculate similarity target sentence and other (= more than one) syntact. constit.

			#print("len > 2")	
			#print(constit_only)
			#print(labels_only)
			sentence_embeddings = model.encode(constit_only)
			#print(sentence_embeddings.shape)
			#calculate cos similarity between TARGET annotated event, as 1st element in list, and other elements (synt. constit) in list
			cos_similarity = cosine_similarity([sentence_embeddings[0]],sentence_embeddings[1:])
			#print(type(cos_similarity))
			#convert numpy array to list
			cos_similarity2 = cos_similarity.tolist()
			#print(type(cos_similarity2))
			#print(cos_similarity2)
			#flatten list
			flat_cos_similarity = [item for sublist in cos_similarity2 for item in sublist]
			print(flat_cos_similarity)
			#select max value + max index, in order to select, label corresponding to synt. constit. with highest similarity to target sentence
			max_value = max(flat_cos_similarity)
			max_index = flat_cos_similarity.index(max_value)
			# add 1 to max_index, as in sentence list, target sentence is included with synt. constit. sentences
			max_index = max_index + 1
			#print(max_value,max_index)
			#print("FINSEL")
			#print(constit_only[max_index],labels_only[max_index])

			#RETURN REF + HYP PROM LABELS
			print(self.annot_event,self.constit_list,self.prom_label,labels_only[max_index])
			return self.prom_label,labels_only[max_index]
		else:
			#if len <= 2, it's about target sentence + 1 synt. constit, print sentences and labels

			#print("len <= 2")			
			#print(constit_only)
			#print(labels_only)
			# ref and hyp label to calculate F score
			#ref_label = labels_only[0]
			hyp_label = labels_only[1]

			#RETURN REF + HYP PROM LABELS
			print(self.annot_event,self.constit_list,self.prom_label,hyp_label)
			return self.prom_label,hyp_label

#open annottfile
ANNOTFILE=open('OVERVIEW_annot.txt','r')
CONSTITFILE=open('ALLFILES_OVERVIEW_PROM.txt','r')

ANNOTLIST=[]
for line in ANNOTFILE:
	line=line.strip()
	#split line
	annotsegs=line.split('\t')
	assert len(annotsegs) == 6
	#append segs to ANNOTLIST
	ANNOTLIST.append(annotsegs)


CONSTITLIST=[]
for line in CONSTITFILE:
	line=line.strip()
	#split line
	if not "EMPTY" in line:
		constitsegs=line.split('\t')
	
		assert len(constitsegs) == 5
		#append segs to ANNOTLIST
		CONSTITLIST.append(constitsegs)

ANNOTLIST_WITH_TUPLES = []
for annotitem in ANNOTLIST:
	constit_label_list=[]
#	#print(annotitem)
	for constititem in CONSTITLIST:
#	print(constititem)
	
	#for constititem in CONSTITLIST:
		#if filenames are equal and sentence numbers are equal
		if constititem[0] == annotitem[0]:
				if annotitem[4] == constititem[1]:
					#add constituent and constit number of files with predicted synt. constituents
					constit_label_list.append((constititem[3],constititem[4]))
				
	#print(constit_label_list)
	annotitem.append(constit_label_list)
	ANNOTLIST_WITH_TUPLES.append(annotitem)	


REFERENCES=[]
HYPS=[]
for elem in ANNOTLIST_WITH_TUPLES:
	#print(elem)
	elem_obj=EvaluateEvent(elem[0],elem[1],elem[2],elem[3],elem[4],elem[5],elem[6])
	REF,HYP = elem_obj.operation()
	print(REF,HYP)
	REFERENCES.append(REF)
	HYPS.append(HYP)
	OUT_REF.write("%s\n"%(REF))
	OUT_HYP.write("%s\n"%(HYP))

"""

example output

['2570493', '6', 'Main', 'positive', '3', 'zich te laten vaccineren', [('Vanaf 1 januari krijgt het zorgpersoneel drie maanden de tijd', 'Main'), ('om zich te laten vaccineren', 'Main')]]

"""




