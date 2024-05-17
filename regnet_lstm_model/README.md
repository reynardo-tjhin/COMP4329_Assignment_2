# Model Architecture 2: RegNet + LSTM Model

## Structure of the Folder

The structure of this folder is:
- `data`: deals with creating a `Dataset` class to handle getting the images, texts and labels.
- `models`: this folder contains the basic CNN model, the popular models such as RegNet and LSTM model.
- `notebooks`: this folder explores the standalone models, graphs, combined model, etc. This folder is mainly used to try different ideas to perform the classification.
- `tools`: this folder contains files to help with tokenizing the textual data, getting metrics, preprocessing the data, etc.

## Running the Training

To run the training of the architecture, please go the Notebook directory and find the "Combined Model.ipynb" file. Please first change the directory of the file to where the data and the csv files are located. The combined model is instantiated in the notebook and the cell will need to be run before running the training. Please run all the cells to run the training. If the training is run successfully, the notebook will produce a "combined_model_1.pth" file in this current repository. The submission "csv" files will also be created. There are two of these files, one that predict the training dataset and another is the one that needs to be submitted.

## Running the Inference

If you are not interested in training, inference can directly be performed. Please head to the "Combined Model.ipynb" file in this current folder that contains this README.md. Please first change the directory of the file to where the data and the csv files are located. Run the whole notebook and it will produce a csv file that needs to be submitted. The file will be named "submission_combined.csv". The notebook uses the "regnet_lstm_model.pth" model.