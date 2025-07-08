from typing import List
import csv
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification, get_scheduler
from torch.optim import AdamW
from tqdm import tqdm
from collections import Counter

class AspectSentimentDataset(Dataset):
    def __init__(self, filename, tokenizer, max_len=128):
        self.samples = []
        self.tokenizer = tokenizer
        self.label_map = {"positive": 0, "negative": 1, "neutral": 2}
        with open(filename, 'r', encoding='utf-8') as f:
            for row in csv.reader(f, delimiter="\t"):
                if len(row) != 5:
                    continue
                label, aspect, term, offsets, sentence = row
                self.samples.append((label.strip(), aspect.strip(), term.strip(), sentence.strip()))

        # Affiche la répartition des labels (sanity check)
        label_ids = [self.label_map[s[0]] for s in self.samples]
        print("Label counts:", Counter(label_ids))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        label, aspect, term, sentence = self.samples[idx]
        text = f"Sentence: {sentence} Aspect: {aspect} Term: {term}"
        encoding = self.tokenizer(text, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()
        label_id = self.label_map[label]
        return input_ids, attention_mask, label_id


class Classifier:
    def __init__(self, ollama_url: str):
        self.model_name = "FacebookAI/roberta-base"
        self.tokenizer = RobertaTokenizer.from_pretrained(self.model_name)
        self.model = RobertaForSequenceClassification.from_pretrained(self.model_name, num_labels=3)

    def train(self, train_filename: str, dev_filename: str, device: torch.device):
        self.model.to(device)
        dataset = AspectSentimentDataset(train_filename, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
        optimizer = AdamW(self.model.parameters(), lr=2e-5)

        # Compute class weights
        labels = [s[0] for s in dataset.samples]
        label_map = {"positive": 0, "negative": 1, "neutral": 2}
        label_ids = [label_map[l] for l in labels]
        counts = Counter(label_ids)
        total = sum(counts.values())
        weights = [total / counts[i] for i in range(3)]
        class_weights = torch.tensor(weights, dtype=torch.float).to(device)
        loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

        # Scheduler
        num_training_steps = len(dataloader) * 3
        lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

        self.model.train()
        for epoch in range(3):
            print(f"Epoch {epoch+1}/3")
            epoch_loss = 0.0
            progress_bar = tqdm(dataloader, desc="Training", leave=False)
            for input_ids, attention_mask, labels in progress_bar:
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)

                outputs = self.model(input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs.logits, labels)
                epoch_loss += loss.item()

                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                progress_bar.set_postfix(loss=loss.item())
            print(f"Epoch {epoch+1} done. Mean Loss: {epoch_loss / len(dataloader):.4f}")

    def predict(self, data_filename: str, device: torch.device) -> List[str]:
        self.model.eval()
        self.model.to(device)
        dataset = AspectSentimentDataset(data_filename, self.tokenizer)
        loader = DataLoader(dataset, batch_size=32, shuffle=False)
        preds = []
        inv_label_map = {0: "positive", 1: "negative", 2: "neutral"}

        with torch.no_grad():
            for input_ids, attention_mask, _ in loader:
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                outputs = self.model(input_ids, attention_mask=attention_mask)
                pred_labels = torch.argmax(outputs.logits, dim=1)
                preds.extend([inv_label_map[p.item()] for p in pred_labels])

        return preds
