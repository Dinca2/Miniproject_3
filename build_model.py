
import torch
import csv
import os
import pandas as pd
import numpy as np
from torch import nn
from torchvision import transforms
from transformers import ViTForImageClassification, ViTImageProcessor, TrainingArguments, Trainer
from sklearn.metrics import f1_score, confusion_matrix
from datasets import load_dataset
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

#makes label csv files to make it easier to access/organize the labels
def make_labels(labels, data_path):
    header = {"filename": "", "label": "","encode_label":0}
    
    #encodes labels as labels need to float values for the data collator
    number = [n for n in range(0, len(labels))]
    encode_labels = dict(zip(labels,number))
    
    for label in labels: #each micro organism
        for sub in next(os.walk(os.path.join(data_path + label + "/")))[1]: #each train/test sub directory
            with open(data_path +label + '/' + sub +'/' 'labels.csv','w', newline='') as f: #opens csv file
                w = csv.DictWriter(f, header.keys())
                w.writeheader() #creates column names
                
                #only gets image files
                for file in [i for i in os.listdir(os.path.join(data_path + label + "/" + sub + '/')) if 
                             (i[-3:].lower() == 'jpg' or i[-3:].lower() =='png' or i[-4:].lower() =='jpeg')]: 
                    row = {"filename":file, "label": label,"encode_label":encode_labels[label]}
                    w.writerow(row)

#concatenates all "label" csv's
def get_labels(labels, data_path, sub):
    all_labels = pd.DataFrame()
    for label in labels:
        path = os.path.join(data_path + label + "/" + sub + '/')
        for file in [i for i in os.listdir(path) if i == 'labels.csv']:
            temp = pd.read_csv(path+file)
            all_labels = pd.concat([all_labels, temp['encode_label']], ignore_index=True)
    all_labels.columns = ['label']
    return all_labels['label']

#renames image files to include their label so that it's easier to identify what the image is of
def rename_images(labels, data_path):
    for label in labels:
        for sub in next(os.walk(os.path.join(data_path + label + "/")))[1]:
            path_to_file = os.path.join(data_path + label + "/" + sub + '/')
            for file in os.listdir(path_to_file):
                os.rename(path_to_file + file, path_to_file +label+file)

#computes metrics for trainer function
def compute_metrics(logits_and_labels):
    logits, labels = logits_and_labels
    predictions = np.argmax(logits, axis=-1)
    acc = np.mean(predictions == labels)
    f1 = f1_score(labels, predictions, average = 'micro')
    return {'accuracy': acc, 'f1_score': f1}

#what transformations are done to the images
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(232),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(232),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

#image datasets have to have a "pixel_values" feature so this converts each image into its pixel values after the transformations are done to it
def train_transforms(examples):
    examples["pixel_values"] = [data_transforms['train'](img.convert("RGB")) for img in examples["image"]]
    del examples["image"]
    return examples

def test_transforms(examples):
    examples["pixel_values"] = [data_transforms['test'](img.convert("RGB")) for img in examples["image"]]
    del examples["image"]
    return examples

#defualt collator had issues with the label type so this overides the collate function in trainer
#to convert the labels into the right type
def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

#same with the collate function, the trainer function had trouble with the label type so this
#overides the default compute loss function to convert the label into a long tensor
def compute_loss(model, inputs, return_outputs=False):
    labels = inputs.pop("labels")
    labels = labels.type(torch.LongTensor)
    outputs = model(**inputs)
    logits = outputs.get("logits")
    loss_fct = nn.CrossEntropyLoss()
    loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
    return (loss, outputs) if return_outputs else loss

data_path = './Micro_Organism/'
labels = os.listdir(data_path)
#rename_images(labels, data_path)
#make_labels(labels, data_path)

#shuffle is false by defualt, don't make it true as the labels depend on it being in same order
data = load_dataset("imagefolder", data_dir=data_path) #createes dataset of all the images
train_labels = get_labels(labels, data_path, '/train/') #gets labels for the training data
test_labels = get_labels(labels, data_path, '/test/') #gets labels for the testing data

#removes and readds label
data['train'] = data['train'].remove_columns('label').add_column('label', train_labels)
data['test']= data['test'].remove_columns('label').add_column('label', test_labels)

test_set = data['test']
split_train = data['train'].train_test_split(test_size=0.1)
train_set = split_train['train']
val_set = split_train['test']

train_set.set_transform(train_transforms)
val_set.set_transform(test_transforms)
test_set.set_transform(test_transforms)

checkpoint = 'google/vit-base-patch16-224'
model = ViTForImageClassification.from_pretrained(checkpoint, ignore_mismatched_sizes=True)
processor = ViTImageProcessor.from_pretrained(checkpoint)

training_args = TrainingArguments(
    output_dir="./output",
    save_strategy="epoch",
    evaluation_strategy="epoch",
    learning_rate=0.0001,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    remove_unused_columns=False
)

trainer = Trainer(model,
                  training_args,
                  train_dataset = train_set,
                  eval_dataset = val_set,
                  tokenizer=processor,
                  data_collator=collate_fn,
                  compute_metrics=compute_metrics,
)
trainer.compute_loss=compute_loss
trainer.train()

decode_labels = {n:label for n,label in enumerate(labels)}
outputs = trainer.predict(test_set)
print(outputs.metrics)

y_true = outputs.label_ids
y_pred = outputs.predictions.argmax(1)

cm = confusion_matrix(y_true, y_pred)
print(cm)

torch.save(model.state_dict(), "./model.pth")
torch.save(model, "./model_complete.pth")

#Questions
#