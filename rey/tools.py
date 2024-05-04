import string
import re
import numpy as np
import pandas as pd

from typing import List

# global variables/constant
N_CLASSES = 19

def get_data(path: str) -> List[str]:
    """
    Readlines of the data given path and return a list of strings.
    """
    f = open(path, "r")
    data = f.readlines()
    f.close()
    return data

def _clean_caption(caption: str) -> str:
    """
    Perform cleaning caption.
    Reference: https://www.analyticsvidhya.com/blog/2022/01/text-cleaning-methods-in-nlp/
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
    """
    #TODO: some captions contain "ski's, someone's": how to make use of this?

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
    """
    # get the instances with ONLY class no given
    df_n = df[df[f'class {class_no}'] == 1.0]
    for n in range(2, N_CLASSES + 1):
        df_n = df_n[df_n[f'class {n}'] == 0.0]

    # remove instances with ONLY class no given
    df_no_n = pd.concat([df, df_n]).drop_duplicates(keep=False)

    return df_no_n
