from logicedu import get_logger, get_unique_labels, get_metrics, multi_acc, MNLIDataset
import argparse
from torch.utils.data import Dataset, TensorDataset, DataLoader
import pandas as pd
from torch import nn
from transformers import AutoTokenizer, AutoModel, AutoConfig
import torch
import torch.nn.functional as F
import time
from parabart import ParaBart
from collections import Counter

torch.manual_seed(0)


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class CustomDataset(Dataset):
    def __init__(self, inputs, labels, outputs):
        self.outputs = outputs
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx], self.outputs[idx]


def build_embeddings(model, tokenizer, sents):
    embeddings = torch.ones((len(sents), model.config.d_model))
    for i, sent in enumerate(sents):
        sent_inputs = tokenizer(sent, return_tensors="pt")
        sent_token_ids = sent_inputs['input_ids']
        sent_embed = model.encoder.embed(sent_token_ids.to(device))
        embeddings[i] = sent_embed
    return embeddings.to(device)


class SiameseDataset:
    def __init__(self, train_ds, dev_ds, test_ds, label_col_name, map, tokenizer_path):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.map = map
        self.label_col_name = label_col_name
        self.mappings = pd.read_csv("../../data/mappings.csv")
        self.unique_labels = get_unique_labels(pd.concat([train_ds, dev_ds, test_ds]), self.label_col_name)[0]
        print("unique_labels=", self.unique_labels)
        # self.label_dict = {'entailment': 0, 'contradiction': 1}
        self.train_ds = self.init_ds(train_ds)
        self.dev_ds = self.init_ds(dev_ds)
        self.test_ds = self.init_ds(test_ds)

    def init_ds(self, ds):
        inputs = []
        labels = []
        outputs = []
        for i, row in ds.iterrows():
            for label in self.unique_labels:
                # encoded_input = self.tokenizer(row['source_article'], padding=True, truncation=True,
                #                                return_tensors='pt')
                inputs.append(row['source_article'])
                # print("label=", label)
                if self.map == 'base':
                    modified_label = label
                elif self.map == 'simplify':
                    modified_label = \
                        list(self.mappings[self.mappings['Original Name'] == label]['Understandable Name'])[0]
                elif self.map == 'description':
                    modified_label = list(self.mappings[self.mappings['Original Name'] == label]['Description'])[0]
                elif self.map == 'logical-form':
                    modified_label = list(self.mappings[self.mappings['Original Name'] == label]['Logical Form'])[0]
                # encoded_label = self.tokenizer(modified_label, padding=True, truncation=True, return_tensors='pt')
                labels.append(modified_label)
                if label == row[self.label_col_name]:
                    outputs.append(1)
                else:
                    outputs.append(0)
        counts = Counter(outputs)
        print(counts)
        return CustomDataset(inputs, labels, outputs)

    def custom_collate_fn(self, data_samples):
        inputs = [sample[0] for sample in data_samples]
        labels = [sample[1] for sample in data_samples]
        outputs = [sample[2] for sample in data_samples]
        return inputs, labels, outputs

    def get_data_loaders(self, batch_size=32, shuffle=True):

        train_loader = DataLoader(
            self.train_ds,
            shuffle=shuffle,
            batch_size=batch_size,
            collate_fn=self.custom_collate_fn
        )

        val_loader = DataLoader(
            self.dev_ds,
            shuffle=shuffle,
            batch_size=batch_size,
            collate_fn=self.custom_collate_fn

        )

        test_loader = DataLoader(
            self.test_ds,
            shuffle=False,
            batch_size=len(self.unique_labels),
            collate_fn=self.custom_collate_fn
        )

        return train_loader, val_loader, test_loader


class Classifier(nn.Module):

    def __init__(self, hidden_layer_size=256, input_size=768 * 2, no_of_hidden_layer=1):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_layer_size)
        self.hidden_layers = []
        for i in range(no_of_hidden_layer):
            self.hidden_layers.append(nn.Linear(hidden_layer_size, hidden_layer_size).to(device))
        self.fc2 = nn.Linear(hidden_layer_size, 2)

    def forward(self, x):
        # print(x.shape)
        x = F.relu(self.fc1(x))
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        x = self.fc2(x)
        return x


class SiameseNetwork(nn.Module):
    def __init__(self, model_path, hidden_layer_size, number_of_hidden_layers, device, config, tokenizer):
        super(SiameseNetwork, self).__init__()
        self.tokenizer = tokenizer
        self.number_of_hidden_layers = number_of_hidden_layers
        self.hidden_layer_size = hidden_layer_size
        self.encoder1 = ParaBart(config)
        self.encoder1.load_state_dict(torch.load(model_path, map_location=device))
        self.encoder2 = ParaBart(config)
        self.encoder2.load_state_dict(torch.load(model_path, map_location=device))
        self.classifier = Classifier(input_size=1536, hidden_layer_size=self.hidden_layer_size,
                                     no_of_hidden_layer=self.number_of_hidden_layers)

    def forward(self, x, y):
        # print(x)
        embeddings1 = build_embeddings(self.encoder1, self.tokenizer, x)
        embeddings2 = build_embeddings(self.encoder2, self.tokenizer, y)
        embeddings = torch.cat((embeddings1, embeddings2), dim=1)
        # print(embeddings.shape)
        output = self.classifier(embeddings)
        # print(output.shape)
        return output


def train(train_loader, val_loader, model, optimizer, save_path, epochs=10, pos_weight=12):
    min_val_loss = float('inf')
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([1.0, pos_weight]))
    loss_fn.to(device)
    for epoch in range(epochs):
        start = time.time()
        total_train_loss = 0
        total_train_acc = 0
        total_train_prec = 0
        total_train_rec = 0
        for i, (inputs, hypothesis, label) in enumerate(train_loader):
            if i % 10 == 0:
                logger.debug('%d %d', epoch, i)
            preds = model(inputs, hypothesis)
            # print(preds,torch.tensor(outputs))
            loss = loss_fn(preds, torch.tensor(label).to(device))
            # print("I am here at make dot")
            # make_dot(loss, params=dict(model.named_parameters())).render(directory='graph').replace('\\', '/')
            acc, prec, rec = multi_acc(preds, torch.tensor(label).to(device), flip=False)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            total_train_acc += acc
            total_train_prec += prec
            total_train_rec += rec
            # print("one backprop done")
        train_acc = total_train_acc / len(train_loader)
        train_loss = total_train_loss / len(train_loader)
        train_prec = total_train_prec / len(train_loader)
        train_rec = total_train_rec / len(train_loader)
        total_val_acc = 0
        total_val_loss = 0
        total_val_prec = 0
        total_val_rec = 0
        with torch.no_grad():
            for i, (inputs, hypothesis, label) in enumerate(val_loader):
                preds = model(inputs, hypothesis)
                loss = loss_fn(preds, torch.tensor(label).to(device))
                acc, prec, rec = multi_acc(preds, torch.tensor(label).to(device), flip=False)
                total_val_loss += loss.item()
                total_val_acc += acc
                total_val_prec += prec
                total_val_rec += rec
            val_acc = total_val_acc / len(val_loader)
            val_loss = total_val_loss / len(val_loader)
            val_rec = total_val_rec / len(val_loader)
            val_prec = total_val_prec / len(val_loader)
            flag = True
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                logger.info("saving model")
                torch.save(model.state_dict(), save_path)
            else:
                flag = False
            end = time.time()
            hours, rem = divmod(end - start, 3600)
            minutes, seconds = divmod(rem, 60)

            logger.info(
                f'Epoch {epoch + 1}: train_loss: {train_loss:.4f} train_acc: {train_acc:.4f} train_prec: {train_prec:.4f} train_rec: {train_rec:.4f}| val_loss: {val_loss:.4f} val_acc: {val_acc:.4f} val_prec: {val_prec:.4f} val_rec: {val_rec:.4f}')
            logger.info("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
            if not flag:
                break


def eval1(model, test_loader, logger):
    with torch.no_grad():
        all_preds = []
        all_labels = []
        for batch_idx, (inputs, hypothesis, label) in enumerate(test_loader):
            logger.debug("%d", batch_idx)
            preds = model(inputs, hypothesis)
            y_pred = torch.log_softmax(preds, dim=1).argmax(dim=1)
            all_preds.append(y_pred)
            all_labels.append(torch.tensor(label).to(device))
        all_preds = torch.stack(all_preds)
        all_labels = torch.stack(all_labels)
        return get_metrics(all_preds, all_labels, sig=False)


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    logger = get_logger()
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--tokenizer", help="tokenizer path")
    parser.add_argument("-m", "--model", help="model path")
    parser.add_argument("-s", "--savepath", help="path to save trained model")
    parser.add_argument("-w", "--weight", help="Weight of positive class")
    parser.add_argument("-mp", "--map", help="Map labels to this category")
    parser.add_argument("-hi", "--hidden", help="Size of hidden layer")
    parser.add_argument("-n", "--number", help="Number of hidden layer")
    logger.info("device = %s", device)
    args = parser.parse_args()
    logger.debug("%s", args)
    fallacy_train = pd.read_csv('../../data/edu_train.csv')
    fallacy_dev = pd.read_csv('../../data/edu_dev.csv')
    fallacy_test = pd.read_csv('../../data/edu_test.csv')
    fallacy_ds = SiameseDataset(fallacy_train, fallacy_dev, fallacy_dev, 'updated_label', 'logical-form',
                                args.tokenizer)
    config = AutoConfig.from_pretrained(args.tokenizer)
    config.word_dropout = 0.2
    config.extra_pos_embeddings = 2
    model = SiameseNetwork(args.model, int(args.hidden), int(args.number), device, config, fallacy_ds.tokenizer).to(
        device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    train_loader, val_loader, test_loader = fallacy_ds.get_data_loaders()
    train(train_loader, val_loader, model, optimizer, args.savepath, pos_weight=int(args.weight))
    model.load_state_dict(torch.load(args.savepath))
    eval1(model, test_loader, logger)
    scores = eval1(model, test_loader, logger)
    logger.info("micro f1: %f macro f1:%f precision: %f recall: %f exact match %f", scores[4], scores[5], scores[1],
                scores[2], scores[3])
