import string
import re
import torch
import numpy as np
import pandas as pd

from nltk.corpus import stopwords
from typing import List, Tuple
from collections import Counter

# global variables/constant
N_CLASSES = 19

def get_data(path: str) -> List[str]:
    """
    Readlines of the data given path and return a list of strings.
	
	Argument
        path (str): the path to the csv data.
    """
    f = open(path, "r")
    data = f.readlines()
    f.close()
    return data

def _clean_caption(caption: str) -> str:
    """
    Perform cleaning caption.
    Reference: https://www.analyticsvidhya.com/blog/2022/01/text-cleaning-methods-in-nlp/
	
	Argument:
        caption (str): the string caption
    """
    # for removing punctuations
    translator = str.maketrans("", "", string.punctuation)

    caption = caption.lower()
    caption = caption.translate(translator)
    caption = re.sub('\d+', '', caption)
    caption = " ".join(caption.split())

    return caption

def load_data(data: List[str], has_label: bool=True) -> pd.DataFrame:
    """
    Perform text cleaning and turn the csv file data to pandas DataFrame type.
	
	Argument:
        data (List[str]): the list of strings split by "\\n" character of the csv file.
		has_label (bool): to differentiate from training and testing set.
    """
    # store the data into different lists
    image_ids = []
    captions = []

    # changing the label to one-hot encoding
    n_instances = len(data) - 1 # excl. first line
    labels = np.zeros((n_instances, N_CLASSES))

    # iterate each line
    for i, line in enumerate(data):

        # ignore the first line
        if (i == 0): continue
        
        caption = line[line.find('\"')+1:-1]
        caption = _clean_caption(caption)

        # split them to three different data: image id, label and caption
        line = line.split(",")
        image_id = line[0]

        # store them to lists
        image_ids.append(image_id)
        captions.append(caption)
        
        # create a one-hot encoded label
        if (has_label):
            label = line[1]
            multi_labels = label.split(" ")
            for label in multi_labels:
                labels[i - 1][int(label) - 1] = 1

    # create a pandas dataframe
    data = {
        'image_id': image_ids,
        'caption': captions,
    }
    if (has_label):
        for label in range(N_CLASSES):
            data[f'class {label + 1}'] = labels.T[label]
    df = pd.DataFrame(data)

    return df

def remove_class(df: pd.DataFrame, class_no: int) -> pd.DataFrame:
    """
    Since there is a data imbalance, we can remove instances with a large number of only a single class.
	
	Argument:
        df (pd.DataFrame): the data frame of the csv file.
		class_no (int): the class index to be removed.
    """
    # get the instances with ONLY class no given
    df_n = df[df[f'class {class_no}'] == 1.0]
    for n in range(2, N_CLASSES + 1):
        df_n = df_n[df_n[f'class {n}'] == 0.0]

    # remove instances with ONLY class no given
    df_no_n = pd.concat([df, df_n]).drop_duplicates(keep=False)

    return df_no_n

def _preprocess_string(s: str) -> str:
	"""
	Preprocess the string by removing numbers and non-alphabetical characters.
	
	Argument:
        s (str): the string to be preprocessed.
	"""
	s = re.sub(r"[^\w\s]", '', s)
	s = re.sub(r"\s+", '', s)
	s = re.sub(r"\d", '', s)
	return s

def _padding(sentences: List[int], seq_len: int) -> np.ndarray:
	"""
	Add and extra pad on the left hand side to ensure shape consistency.
	
	Argument:
        sentences (List(int)): a list of sentences that have been tokenized.
		seq_len (int): the resultant width of the array.
	"""
	features = np.zeros((len(sentences), seq_len), dtype=int)
	for i, review in enumerate(sentences):
		if (len(review) != 0):
			features[i, -len(review):] = np.array(review)[:seq_len]
	return features

def tokenize(x_train: List[str]) -> Tuple[np.ndarray, dict]:
	"""
	Tokenize into an np array with word corresponded to indexes.
	
	Argument:
        x_train (List(str)): the list of captions.
	"""
	word_list = []	
	stop_words = set(stopwords.words('english'))
	for sentence in x_train:
		for word in sentence.lower().split():
			word = _preprocess_string(word)
			if word not in stop_words and word != '':
				word_list.append(word)

	corpus = Counter(word_list)
	corpus_ = sorted(corpus, key=corpus.get, reverse=True)
	onehot_dict = {w: i+1 for i,w in enumerate(corpus_)}

	# tokenize
	final_list_train = []
	for sentence in x_train:

		ls = []
		for word in sentence.lower().split():
			word = _preprocess_string(word)
			if (word in onehot_dict.keys()):
				idx = onehot_dict[word]
				ls.append(idx)
		final_list_train.append(ls)

	# find the maximum length of the sentence
	max_len = 0
	for sentence in final_list_train:
		max_len = len(sentence) if (len(sentence) > max_len) else max_len
	print("Max Sentence Length:", max_len)

	# padding
	final_list_train = _padding(final_list_train, max_len)
	return final_list_train, onehot_dict

def count_class(t_data: pd.DataFrame):
	"""
	Count the frequency of each class.

	Argument:
		train_data (pd.DataFrame): the csv data.
	"""
	counter = {}
	for n in range(19):
		data = t_data[t_data['class ' + str(n + 1)] == 1.]
		data = data['class ' + str(n + 1)]
		freq = data.count()
		counter['class ' + str(n + 1)] = freq
	return counter

def calculate_pos_weights(class_counts: List[int], data) -> torch.Tensor:
	"""
	Calculates the positive weights of the data.
	Reference: https://stackoverflow.com/questions/57021620/how-to-calculate-unbalanced-weights-for-bcewithlogitsloss-in-pytorch
	
	Argument:
		class_counts (List[int]): the frequency of each class.
		data (pd.DataFrame): the csv file.
	"""
	pos_weights = np.zeros(len(class_counts))
	neg_counts = [len(data) - pos_count for pos_count in class_counts]
	for cdx, (pos_count, neg_count) in enumerate(zip(class_counts, neg_counts)):
		pos_weights[cdx] = neg_count / (pos_count + 1e-5)
		if (pos_count == 0):
			pos_weights[cdx] = 0
	return torch.from_numpy(pos_weights).float()
