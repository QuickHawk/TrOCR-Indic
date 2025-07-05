import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from transformers import (
    VisionEncoderDecoderModel,
    AutoImageProcessor,
    AutoTokenizer
)

import evaluate
import pandas

from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split

from data_sets import SyntheticDataset
from utils import load_and_preprocess_data, send_notification, load_english_data, load_icdar_data

intialization_progress = tqdm(total=7, desc="Initializing", dynamic_ncols=True, leave=False)

intialization_progress.write(f"Setting Hyperparameters...")
# Hyperparameters
IMG_SIZE = (32, 128)    # Not used
DATASET_URL = "C:\\Users\\Aarya\\Documents\\Synthetic IndicSTR12 Dataset\\synthetic"

ENCODER_SHORT_NAME = "deit"
DECODER_SHORT_NAME = "indicbart"

ENCODER_MODEL_NAME = "facebook/deit-base-distilled-patch16-224"
# ENCODER_MODEL_NAME = "microsoft/swin-tiny-patch4-window7-224"
DECODER_MODEL_NAME = "ai4bharat/IndicBART"

# FRACTION = 0.01
FRACTION = 1
intialization_progress.update(1)

intialization_progress.write(f"Loading Processor...")
processor = AutoImageProcessor.from_pretrained(ENCODER_MODEL_NAME, use_fast=True)
intialization_progress.update(1)

# processor.size = {"height": IMG_SIZE[0], "width": IMG_SIZE[1]}
intialization_progress.write(f"Loading Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(DECODER_MODEL_NAME, use_fast=True)
intialization_progress.update(1)

intialization_progress.write(f"Loading Model...")
# model = VisionEncoderDecoderModel.from_pretrained(r"checkpoints\deit-indicbart")
intialization_progress.update(1)
# model.encoder.config.image_size = IMG_SIZE

model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
    ENCODER_MODEL_NAME, DECODER_MODEL_NAME,
    decoder_add_cross_attention=True,
)
# model.load_state_dict(torch.load(r"checkpoints\swin-indicbart-model_2_14000.pth"))
# model.load_state_dict(torch.load(r"checkpoints\best\deit-indicbart-model_0.1672.pth"))
model.load_state_dict(torch.load(r"last_model.pth"))

bos_id = tokenizer._convert_token_to_id_with_added_voc("<s>")
eos_id = tokenizer._convert_token_to_id_with_added_voc("</s>")
pad_id = tokenizer._convert_token_to_id_with_added_voc("<pad>")

model.config.decoder_start_token_id = tokenizer._convert_token_to_id_with_added_voc("<2en>")
model.config.bos_token_id = bos_id
model.config.eos_token_id = eos_id
model.config.pad_token_id = pad_id
model.config.vocab_size = model.config.decoder.vocab_size

model.config.max_length = 256
model.config.early_stopping = True
model.config.no_repeat_ngram_size = 3
model.config.length_penalty = 2.0
model.config.num_beams = 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)

intialization_progress.write("Loading CER Metric...")
cer_metric = evaluate.load("cer")

def compute_cer(pred_ids, labels_ids):

    labels_ids[labels_ids == -100] = tokenizer.pad_token_id

    pred_list = []
    label_list = []

    for pred_id, label_id in zip(pred_ids, labels_ids):
        label_str = tokenizer.decode(label_id, skip_special_tokens=True)
        pred_str = tokenizer.decode(pred_id, skip_special_tokens=True)

        if label_str.strip() == "" or len(label_str.strip()) == 0:
            continue

        label_list.append(label_str)
        pred_list.append(pred_str)
        
    cer = cer_metric.compute(predictions=pred_list, references=label_list)

    return {"cer": cer}
intialization_progress.update(1)

# Load data
intialization_progress.write("Loading Data...")
# data = load_and_preprocess_data(DATASET_URL)
# data = load_english_data()
# data = pandas.read_csv("synthetic.csv")
intialization_progress.update(1)

best_loss = float('inf')
loss_history = []
cer_history = []

START = 1
EPOCHS = 25

intialization_progress.write("Splitting Data...")
# train_data, test_data = train_test_split(frac_data, test_size=0.2, shuffle=True)
corpus_dir = r'data_sets/corpus'
font_url = r'data_sets/fonts'
train_dataset = SyntheticDataset(corpus_dir=corpus_dir, fonts_dir=font_url, processor=processor, tokenizer=tokenizer)
# train_dataset = IndicSTR(DATASET_URL, train_data, processor=processor, tokenizer=tokenizer)

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=True)
intialization_progress.update(1)

intialization_progress.close()

for epoch in range(START, EPOCHS + 1):

    model.train()
    train_loss = 0

    progress_bar = tqdm(train_dataloader, desc="Training", dynamic_ncols=True, leave=False)
    progress_bar.set_description(f"Epoch {epoch}/{EPOCHS}")

    for idx, batch in enumerate(progress_bar):

        try:

            for k,v in batch.items():
                batch[k] = v.to(device)

            outputs = model(**batch)

            loss = outputs.loss

            logits = outputs.logits
            preds = logits.argmax(-1)
            additional_loss = torch.sqrt(torch.square(batch['labels'][:, 0] - preds[:, 0]).sum()) * 10

            loss = loss + additional_loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss.item()
            progress_bar.set_postfix({"loss": train_loss / ((idx + 1) * train_dataloader.batch_size)})

            if idx % 1000 == 0:
                torch.save(model.state_dict(), rf"checkpoints\{ENCODER_SHORT_NAME}-{DECODER_SHORT_NAME}-model_" + str(epoch) + "_" + str(idx) + ".pth")
                torch.save(model.state_dict(), rf"last_model.pth")
                try:
                    send_notification("*Training in progress*\n*Epoch:* `" + str(epoch) + "` | *Batch:* `" + str(idx) + "` | *Loss:* `" + str(train_loss / ((idx + 1) * train_dataloader.batch_size)) + "`")

                except Exception as e:
                    print(f"Error during notification at epoch {epoch}, batch {idx}: {e}")

        except Exception as e:
            print(f"Error at epoch {epoch}, batch {idx}: {e}")
            continue

    avg_loss = train_loss / len(train_dataloader)
    loss_history.append(avg_loss)

    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), fr"checkpoints\{ENCODER_SHORT_NAME}-{DECODER_SHORT_NAME}-model_{str(epoch)}.pth")
        try:
            send_notification("Model saved at epoch " + str(epoch) + " with loss " + str(avg_loss))
        except Exception as e:
            print(f"Error during notification at epoch {epoch}: {e}")

    try:
        send_notification("Training Complete \nModel has been trained for " + str(epoch) + " epochs.")# \nCER: " + str(valid_cer / len(test_dataloader)))
    except Exception as e:
        print(f"Error during notification at epoch {epoch}: {e}")

torch.save(model.state_dict(), fr"checkpoints\{ENCODER_SHORT_NAME}-{DECODER_SHORT_NAME}-model_final.pth")
