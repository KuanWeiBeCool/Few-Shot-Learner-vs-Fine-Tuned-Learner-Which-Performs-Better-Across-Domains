import pandas as pd
from pandas import DataFrame as df
import numpy as np
import math

import json
import pandas as pd

def convert_dataset(path, new_path):
	'''
	Make the dataset compatible for REDN model.
	'''
	data = json.load(open(path))
	five_classes = {key:data[key] for key in ["gene_found_in_organism", "occurs_in", "causative_agent_of", "classified_as", "gene_plays_role_in_process"]}
	new_data = {'relation':[], 'token':[], 'h':[], 't':[]}
	for label in five_classes.keys():
		for item in five_classes[label]:
			new_data['relation'].append(label)
			new_data['token'].append(item['tokens'])
			new_data['h'].append({'name': item['h'][0], 'id': item['h'][1], 'pos':[item['h'][-1][0][0], item['h'][-1][0][-1] + 1]})
			new_data['t'].append({'name': item['t'][0], 'id': item['t'][1], 'pos':[item['t'][-1][0][0], item['t'][-1][0][-1] + 1]})
			new_data_df = pd.DataFrame(new_data)
			new_data_df_shuffled = new_data_df.sample(frac=1).reset_index(drop=True)
			new_data_df_shuffled.to_json(new_path, orient='records', lines=True)

def semeval_partition(train_size):
	train = pd.read_json('/content/drive/MyDrive/Final Project/REDN-master/Datasets/semeval/train.txt', lines = True)
	dev = pd.read_json('/content/drive/MyDrive/Final Project/REDN-master/Datasets/semeval/dev.txt', lines = True)
	test = pd.read_json('/content/drive/MyDrive/Final Project/REDN-master/Datasets/semeval/test.txt', lines = True)

	result = [train,dev,test]
	result_df = pd.concat(result, ignore_index = True)
	# combine original train, dev, test together and have a mixed dataframe

	a = result_df.groupby(['relation'], as_index = False)# Prepare dataframe for each SemEval relation

	ce1 = a.get_group('Cause-Effect(e1,e2)')
	ce2 = a.get_group('Cause-Effect(e2,e1)')
	cause_effect = [ce1, ce2]
	cause_effect_df = pd.concat(cause_effect, ignore_index = True)
	cause_effect_df = cause_effect_df.sample(frac=1).reset_index(drop=True) # dataframe for this relation only

	cw1 = a.get_group('Component-Whole(e1,e2)')
	cw2 = a.get_group('Component-Whole(e2,e1)')
	component_whole = [cw1, cw2]
	component_whole_df = pd.concat(component_whole, ignore_index = True)
	component_whole_df = component_whole_df.sample(frac=1).reset_index(drop=True)  # dataframe for this relation only


	cc1 = a.get_group('Content-Container(e1,e2)')
	cc2 = a.get_group('Content-Container(e2,e1)')
	content_container = [cc1, cc2]
	content_container_df = pd.concat(content_container, ignore_index = True)
	content_container_df = content_container_df.sample(frac=1).reset_index(drop=True)  # dataframe for this relation only

	ed1 = a.get_group('Entity-Destination(e1,e2)')
	ed2 = a.get_group('Entity-Destination(e2,e1)')
	entity_destination = [ed1, ed2]
	entity_destination_df = pd.concat(entity_destination, ignore_index = True) 
	entity_destination_df = entity_destination_df.sample(frac=1).reset_index(drop=True)  # dataframe for this relation only

	eo1 = a.get_group('Entity-Origin(e1,e2)')
	eo2 = a.get_group('Entity-Origin(e2,e1)')
	entity_origin = [eo1, eo2]
	entity_origin_df = pd.concat(entity_origin, ignore_index = True)
	entity_origin_df = entity_origin_df.sample(frac=1).reset_index(drop=True)  # dataframe for this relation only

	ia1 = a.get_group('Instrument-Agency(e1,e2)')
	ia2 = a.get_group('Instrument-Agency(e2,e1)')
	instrument_agency = [ia1, ia2]
	instrument_agency_df = pd.concat(instrument_agency, ignore_index = True)
	instrument_agency_df = instrument_agency_df.sample(frac=1).reset_index(drop=True)  # dataframe for this relation only

	mc1 = a.get_group('Member-Collection(e1,e2)')
	mc2 = a.get_group('Member-Collection(e2,e1)')
	member_collection = [mc1, mc2]
	member_collection_df = pd.concat(member_collection, ignore_index = True)
	member_collection_df = member_collection_df.sample(frac=1).reset_index(drop=True)  # dataframe for this relation only

	mt1 = a.get_group('Message-Topic(e1,e2)')
	mt2 = a.get_group('Message-Topic(e2,e1)')
	message_topic = [mt1, mt2]
	message_topic_df = pd.concat(message_topic, ignore_index = True)
	message_topic_df = message_topic_df.sample(frac=1).reset_index(drop=True)

	other_df = a.get_group('Other').reset_index(drop=True)

	pp1 = a.get_group('Product-Producer(e1,e2)')
	pp2 = a.get_group('Product-Producer(e2,e1)')
	product_producer = [pp1, pp2]
	product_producer_df = pd.concat(product_producer, ignore_index = True)
	product_producer_df = product_producer_df.sample(frac=1).reset_index(drop=True)


	relations = [cause_effect_df, component_whole_df , content_container_df, entity_destination_df , entity_origin_df , instrument_agency_df, member_collection_df, message_topic_df, other_df, product_producer_df]


	df = relations[0]
	tr_df = df.iloc[:train_size] # training_df
	d_t_df = df.iloc[train_size + 1:] # dev_test_df
	d_df = d_t_df.iloc[:100] # dev_df
	te_df = d_t_df.iloc[101:] # test_df


	for relation_df in relations[1:]:

	  train_df = relation_df.iloc[:train_size]
	  dev_test_df = relation_df.iloc[train_size + 1:]
	  dev_df = dev_test_df.iloc[:100]
	  test_df = dev_test_df.iloc[101:]

	  tr_df = tr_df.append(train_df, ignore_index = True)
	  d_df = d_df.append(dev_df, ignore_index = True)
	  te_df = te_df.append(test_df, ignore_index = True)

	tr_df.to_json('/content/drive/MyDrive/Final Project/REDN-master/Datasets/semeval/semeval_5cls/Second/20%_train/train.json', orient='records', lines=True) 
	d_df.to_json('/content/drive/MyDrive/Final Project/REDN-master/Datasets/semeval/semeval_5cls/Second/20%_train/dev.json', orient='records', lines=True) 
	te_df.to_json('/content/drive/MyDrive/Final Project/REDN-master/Datasets/semeval/semeval_5cls/Second/20%_train/test.json', orient='records', lines=True) 
	
	
	
def nyt_partition(train_percent, dev_percent):	
	train = pd.read_json('/content/drive/MyDrive/Final Project/REDN-master/Datasets/nyt10/nyt10_train.txt', lines=True)
	dev = pd.read_json('/content/drive/MyDrive/Final Project/REDN-master/Datasets/nyt10/nyt10_val.txt', lines = True)
	test = pd.read_json('/content/drive/MyDrive/Final Project/REDN-master/Datasets/nyt10/nyt10_test.txt', lines = True)

	result = [train,dev,test]
	result_df = pd.concat(result, ignore_index = True)
	
	a = result_df.groupby(['relation'], as_index = False)
	all_relations = a.first().iloc[:,0]
	df_group = []

	for relation in all_relations:
	relation_df = a.get_group(relation).reset_index(drop=True)
	df_group.append(relation_df)
	
	result_train_df = pd.DataFrame(columns=df_group[1].columns)
	result_dev_df = pd.DataFrame(columns=df_group[1].columns)
	result_test_df = pd.DataFrame(columns=df_group[1].columns)

	for df in df_group:
		num_data = len(df)

		if (num_data <= 5):
		  # Put all the instances into train
		  result_train_df = result_train_df.append(df, ignore_index = True)
		  continue

		num_train = math.ceil(num_data * train_percent)
		num_dev = math.ceil(num_data * dev_percent)
		
		train_df = df.iloc[:num_train]
		dev_test_df = df.iloc[num_train + 1:]
		dev_df = dev_test_df.iloc[:num_dev]
		test_df = dev_test_df.iloc[num_dev + 1:]

		result_train_df = result_train_df.append(train_df, ignore_index = True)
		result_dev_df = result_dev_df.append(dev_df, ignore_index = True)
		result_test_df = result_test_df.append(test_df, ignore_index = True)

	result_train_df.to_json('/content/drive/MyDrive/Final Project/REDN-master/Datasets/nyt10/0.5%_train/train.json', orient='records', lines=True) 
	result_dev_df.to_json('/content/drive/MyDrive/Final Project/REDN-master/Datasets/nyt10/0.5%_train/dev.json', orient='records', lines=True) 
	result_test_df.to_json('/content/drive/MyDrive/Final Project/REDN-master/Datasets/nyt10/0.5%_train/test.json', orient='records', lines=True) 
	


def new_nyt_partition(train_percent, dev_percent, test_percent):	
	df = pd.read_json('/content/drive/MyDrive/550_Folder/Final Project/REDN-master/Datasets/First/nyt10_5cls_first.json', orient='records', lines=True)

	a = df.groupby(['relation'], as_index = False)
	all_relations = a.first().iloc[:,0]
	df_group = []

	for relation in all_relations:
	relation_df = a.get_group(relation).reset_index(drop=True)
	df_group.append(relation_df)
	
	result_train_df = pd.DataFrame(columns=df_group[1].columns)
	result_dev_df = pd.DataFrame(columns=df_group[1].columns)
	result_test_df = pd.DataFrame(columns=df_group[1].columns)

	for df in df_group:
		num_data = len(df)

		num_train = math.ceil(num_data * train_percent)
		num_dev = math.ceil(num_data * dev_percent)
  		num_test = math.ceil(num_data * test_percent)

		train_df = df.iloc[:num_train]
		dev_test_df = df.iloc[num_train + 1:]
		dev_df = dev_test_df.iloc[:num_dev]
		test_df = dev_test_df.iloc[-num_test:]

		result_train_df = result_train_df.append(train_df, ignore_index = True)
		result_dev_df = result_dev_df.append(dev_df, ignore_index = True)
		result_test_df = result_test_df.append(test_df, ignore_index = True)

	result_train_df.to_json('/content/drive/MyDrive/Final Project/REDN-master/Datasets/nyt10_new/First/5%_train/train.json', orient='records', lines=True) 
	result_dev_df.to_json('/content/drive/MyDrive/Final Project/REDN-master/Datasets/nyt10_new/First/5%_train/dev.json', orient='records', lines=True) 
	result_test_df.to_json('/content/drive/MyDrive/Final Project/REDN-master/Datasets/nyt10_new/First/5%_train/test.json', orient='records', lines=True) 
	

def pubmed_partition(train_percent, dev_percent, test_percent):	
	df = pd.read_json('/content/drive/MyDrive/550_Folder/Final Project/REDN-master/Datasets/pubmed/First/pubmed_5cls_first.json', orient='records', lines=True)

	a = df.groupby(['relation'], as_index = False)
	all_relations = a.first().iloc[:,0]
	df_group = []

	for relation in all_relations:
	relation_df = a.get_group(relation).reset_index(drop=True)
	df_group.append(relation_df)
	
	result_train_df = pd.DataFrame(columns=df_group[1].columns)
	result_dev_df = pd.DataFrame(columns=df_group[1].columns)
	result_test_df = pd.DataFrame(columns=df_group[1].columns)

	for df in df_group:
		num_data = len(df)

		num_train = math.ceil(num_data * train_percent)
		num_dev = math.ceil(num_data * dev_percent)
   		num_test = math.ceil(num_data * test_percent)

		train_df = df.iloc[:num_train]
		dev_test_df = df.iloc[num_train + 1:]
		dev_df = dev_test_df.iloc[:num_dev]
		test_df = dev_test_df.iloc[-num_test:]

		result_train_df = result_train_df.append(train_df, ignore_index = True)
		result_dev_df = result_dev_df.append(dev_df, ignore_index = True)
		result_test_df = result_test_df.append(test_df, ignore_index = True)

	result_train_df.to_json('/content/drive/MyDrive/Final Project/REDN-master/Datasets/pubmed/First/5%_train/train.json', orient='records', lines=True) 
	result_dev_df.to_json('/content/drive/MyDrive/Final Project/REDN-master/Datasets/pubmed/First/5%_train/dev.json', orient='records', lines=True) 
	result_test_df.to_json('/content/drive/MyDrive/Final Project/REDN-master/Datasets/pubmed/First/5%_train/test.json', orient='records', lines=True) 

if __name__ == '__main__':
	
	convert_dataset('/content/drive/MyDrive/550_Folder/Final Project/FewRel-master/data/val_pubmed.json', 'REDN-master/Datasets/pubmed_5cls_first.json')
	# train_size: 0.5% = 5; 1% = 10, 2% = 20, ...
	# Ex: train_size = 10 inst./cls = 1% of SemEval
	semeval_partition(10)
	
	# Ex: 0.5% training + 10% dev
	nyt_partition(.005, .1)
 
  	# Ex: 5% training + 10% dev + 70% test
	new_nyt_partition(.05, .1, .7)
 
  	# Ex: 5% training + 10% dev + 70% test
	pubmed_partition(.05, .1, .7)
