from transformers import VisionEncoderDecoderModel, AutoImageProcessor, AutoTokenizer
import evaluate
import torch
from tqdm.auto import tqdm
from utils import load_and_preprocess_data, send_notification, load_english_data
from tabulate import tabulate
from indicnlp.transliterate.unicode_transliterate import UnicodeIndicTransliterator
from data_sets import IndicSTR
from torch.utils.data import DataLoader

DATASET_URL = "C:\\Users\\Aarya\\Documents\\Synthetic IndicSTR12 Dataset\\synthetic"
# DATASET_URL = "."
# 
# ENCODER_MODEL_NAME = "facebook/deit-base-distilled-patch16-224"
ENCODER_MODEL_NAME = "microsoft/swin-tiny-patch4-window7-224"
DECODER_MODEL_NAME = "ai4bharat/IndicBART"

processor = AutoImageProcessor.from_pretrained(ENCODER_MODEL_NAME, use_fast=True)
tokenizer = AutoTokenizer.from_pretrained(DECODER_MODEL_NAME, use_fast=True)

model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
    ENCODER_MODEL_NAME, DECODER_MODEL_NAME,
    decoder_add_cross_attention=True,
)

# model.load_state_dict(torch.load(r"checkpoints\deit-indicbart-model_3_16000.pth"))
model.load_state_dict(torch.load(r"checkpoints\best\swin-indicbart-model.pth"))
# model.load_state_dict(torch.load(r"last_model.pth"))

bos_id = tokenizer._convert_token_to_id_with_added_voc("<s>")
eos_id = tokenizer._convert_token_to_id_with_added_voc("</s>")
pad_id = tokenizer._convert_token_to_id_with_added_voc("<pad>")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

cer_metric = evaluate.load("cer")
wer_metric = evaluate.load("wer")

def compute_cer(pred_ids, labels_ids, lang):

    labels_ids[labels_ids == -100] = tokenizer.pad_token_id

    pred_list = []
    label_list = []

    for pred_id, label_id in zip(pred_ids, labels_ids):
        label_str = tokenizer.decode(label_id, skip_special_tokens=True)
        pred_str = tokenizer.decode(pred_id, skip_special_tokens=True)

        if label_str.strip() == "" or len(label_str.strip()) == 0:
            continue

        # lang = tokenizer.decode(pred_id[1])[2: -1]

        if lang != "en":
            label_str_transformed = UnicodeIndicTransliterator.transliterate(label_str, "hi", lang)
            pred_str_transformed = UnicodeIndicTransliterator.transliterate(pred_str, "hi", lang)

        else:
            label_str_transformed = label_str
            pred_str_transformed = pred_str

        # print(label_str_transformed, pred_str_transformed)
        # print(lang, label_str, pred_str, pred_str_transformed)

        label_list.append(label_str_transformed)
        pred_list.append(pred_str_transformed)
        
    cer = cer_metric.compute(predictions=pred_list, references=label_list)
    wer = wer_metric.compute(predictions=pred_list, references=label_list)

    return {"cer": cer, "wer": wer}


data = load_and_preprocess_data(DATASET_URL)
# data = load_english_data(test=True)

table = [
    ['lang', 'cer', 'wer']
]

lang = data['lang'].unique()

model.eval()

for l in lang:
    valid_cer = 0.0
    valid_wer = 0.0

    lang_data = data[data['lang'] == l]

    dataset = IndicSTR(DATASET_URL, lang_data, processor=processor, tokenizer=tokenizer)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, drop_last=True)

    with torch.no_grad():
        progress = tqdm(dataloader, desc=f"Evaluating {l}", dynamic_ncols=True, leave=False)
        for idx, batch in enumerate(progress):
            # run batch generation
            try:
                outputs = model.generate(
                    batch["pixel_values"].to(device),
                    use_cache=True,
                    num_beams=4,
                    max_length=128,
                    min_length=1,
                    early_stopping=True,
                    pad_token_id=pad_id,
                    bos_token_id=bos_id,
                    eos_token_id=eos_id,
                    decoder_start_token_id=tokenizer._convert_token_to_id_with_added_voc("<2en>")
                )

                # compute metrics
                metrics = compute_cer(pred_ids=outputs, labels_ids=batch["labels"], lang = l)
                valid_cer += metrics['cer']
                valid_wer += metrics['wer']
                progress.set_postfix_str(f"CER: {valid_cer / (idx + 1):.4f} | WER: {valid_wer / (idx + 1):.4f}")

            except Exception as e:
                print(f"Error at epoch {idx}: {e}")
                continue

    table.append([l, valid_cer / len(dataloader), valid_wer / len(dataloader)])
    send_notification(f"*Evaluation in progress*\n*Language:* `{l}` | *CER:* `{valid_cer / len(dataloader)}` | *WER:* `{valid_wer / len(dataloader)}`")

table_print = tabulate(table, headers="firstrow", tablefmt="github")
print(table_print)
send_notification(table_print)