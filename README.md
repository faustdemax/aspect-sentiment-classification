# NLP Assignment – CentraleSupélec 2025

## Contributors

- Faustine de Maximy  
- Cyrielle Théobald  
- Matthieu Neau  

---

## CLASSIFIER DESCRIPTION

We implemented an aspect-based sentiment classifier to predict the polarity (positive, negative, neutral) of a sentence with respect to a given target term and aspect category.

Our approach is based on fine-tuning the pre-trained model `roberta-base` (authorized by the assignment), which performed better than both BERT and DeBERTa on our dataset.

### Model Architecture

- Model: `facebook/roberta-base`
- Type: encoder-only transformer (classification head, 3 classes)
- Input format:  
  `Sentence: [sentence] Aspect: [aspect] Term: [term]`
- Tokenization: `RobertaTokenizer`
- Training: 3 epochs, batch size 8, learning rate = 2e-5, linear scheduler
- Loss function: weighted `CrossEntropyLoss` to address label imbalance (majority class = positive)

### Training

- Training data: `traindata.csv` only, as required
- Dev set (`devdata.csv`) used only for validation after training

### Evaluation

We evaluated the classifier using the provided `tester.py` script (not modified).  
Executed with:

`python tester.py -g 0 -n 1`

---

## RESULTS ON DEV SET

- Average accuracy: **86.7%**
- Training time: ~50 seconds per run on Colab GPU

---

## BASELINES AND OTHER MODELS TESTED

We also tested several other models from the authorized list:

- `bert-base-uncased`: ~84.2% accuracy
- `bert-large-uncased`: ~84.7%
- `microsoft/deberta-v3-base`: ~84.6%
- `microsoft/deberta-v3-large`: not used due to 14GB memory limit

These models underperformed compared to `roberta-base`, which proved more stable and robust for this task.

---

## NOTES

- No unauthorized libraries, models, or scripts were used.
- `classifier.py` strictly follows the template and required method signatures.
- The model runs entirely offline if the pretrained weights are already cached.
- The data is not included in the repository.

