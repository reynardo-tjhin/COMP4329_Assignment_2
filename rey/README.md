# COMP4329 Assignment 2

## Available Models

| Model | Acc@1 | Params | File Size |
|---|---|---|---|
| DenseNet121_Weights.IMAGENET1K_V1 | 74.434 | 8.0M | 30.8 MB |
| EfficientNet_B0_Weights.IMAGENET1K_V1 | 77.692 | 5.3M | 20.5 MB |
| EfficientNet_B1_Weights.IMAGENET1K_V1 | 78.642 | 7.8M | 30.1 MB |
| EfficientNet_B1_Weights.IMAGENET1K_V2 | 79.838 | 7.8M | 30.1 MB |
| EfficientNet_B2_Weights.IMAGENET1K_V1 | 80.608 | 9.1M | 35.2 MB |
| MNASNet1_3_Weights.IMAGENET1K_V1 | 76.506 | 6.3M | 24.2 MB |
| MobileNet_V3_Large_Weights.IMAGENET1K_V2 | 75.274 | 5.5M | 21.1 MB |
| RegNet_X_1_6GF_Weights.IMAGENET1K_V2 | 79.668 | 9.2M | 35.3 MB |

## File Structure

- cnn.ipynb: Jupyter Notebook containing the CNN model for image classification
- combined.ipynb: Jupyter Notebook containing the combined model of CNN and LSTM in an attempt for multi-class label classification
- dataset.py: Python file to create custom Dataset class for PyTorch
- main.py: [currently not updated yet]
- Model Summary.ipynb: Jupyter Notebook to view the model architecture
- nlp.ipynb: Jupyter Notebook containing the LSTM model for text classification\
- tools.sh: Python file to read file, get data, etc.
- run.sh: to run main.py