from PIL import Image
import requests
from io import BytesIO
import pandas
import codecs
import os
from tqdm.auto import tqdm
import unicodedata
from indicnlp import common
from indicnlp.transliterate.unicode_transliterate import UnicodeIndicTransliterator

common.set_resources_path(os.path.join(os.path.dirname(__file__), "indicnlp_resources"))

def send_notification(message, photo: Image = None):    

    # Replace with your bot token and chat ID
    BOT_TOKEN = ""
    CHAT_ID = ""
    MESSAGE = message

    # Send message
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    params = {"chat_id": CHAT_ID, "text": MESSAGE, "parse_mode": "markdown"}

    files = None

    response = requests.get(url, params=params)

    # Print response (optional)
    # print(response.json())
    
    if photo is not None:
        image_bytes = BytesIO()
        photo.save(image_bytes, format="png")  # Convert to JPEG or PNG
        image_bytes.seek(0)  # Move pointer to the beginning

        # Telegram API endpoint
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"

        # Send image as a file
        files = {"photo": ("image.jpg", image_bytes, "image/png")}
        data = {"chat_id": CHAT_ID}

        response = requests.post(url, data=data, files=files)
        
    # print("Notification Sent!!")

def load_indic_dataset(url: str) -> pandas.DataFrame:
    
    data = {
        'file_name': [],
        'text': []
    }
    
    for folder in os.listdir(url):
        if folder != '.DS_Store':
            for file in os.listdir(os.path.join(url, folder)):
                if file.endswith('_gt.txt'):
                    cur_file_name = file.split("_")[0]
                    
                    with codecs.open(os.path.join(url, folder, file), 'r', encoding='utf-8') as file:
                        lines = file.readlines()
                        
                        for line in lines:
                            line = line.strip()
                            f_path, text = line.rsplit('\t', 1)
                            stitched_path = f_path.replace('\t', '_')
                            
                            file_path = os.path.join(url, folder, "cropped_images", f"{cur_file_name}_{stitched_path}.jpeg")
                            if os.path.exists(file_path):
                                data["file_name"].append(os.path.join(folder, "cropped_images", f"{cur_file_name}_{stitched_path}.jpeg"))
                                data["text"].append(text)
                                
    return pandas.DataFrame(data)



def load_data(DATASET_URL: str) -> pandas.DataFrame:
    '''
    Load SynthIndicSTR12 Dataset
    '''
    all_data = []

    # EXCLUDE_LANGUAGES = ["mod_meitei_manipuri", "mod_marathi"] # Exclude these languages for XLM-Roberta
    # EXCLUDE_LANGUAGES = ["mod_bengali", "mod_gujarati", "mod_hindi", "mod_meitei_manipuri", "mod_kannada", "mod_assamese", "mod_odia"]  # Exclude these languages for mBART
    EXCLUDE_LANGUAGES = [
        # "mod_assamese",
        # "mod_bengali",
        # "mod_gujarati",
        # "mod_hindi",
        # "mod_kannada",
        # "mod_malayalam",
        # # "mod_marathi",
        "mod_meitei_manipuri",
        # "mod_odia",
        # "mod_punjabi",
        # "mod_tamil",
        # "mod_telugu",
        # "mod_urdu"
    ]

    for lang_folder in tqdm(os.listdir(DATASET_URL), desc="Loading data", dynamic_ncols=True, leave=True):
        if not os.path.isdir(os.path.join(DATASET_URL, lang_folder)):
            continue

        if lang_folder in EXCLUDE_LANGUAGES:
            continue

        sub_folder = os.listdir(os.path.join(DATASET_URL, lang_folder))[0]
        sub_folder_path = os.path.join(DATASET_URL, lang_folder, sub_folder)

        # with codecs.open(os.path.join(sub_folder_path, f"{sub_folder}_gt.txt"), 'r', encoding='utf-8') as file:
        #     for _ in range(10000):
        #         lines = file.readline()
        #         filename, text = lines.split("\t")
        #         if text is "" or len(text) == 0:
        #             continue
                
        #         full_path = os.path.join(sub_folder_path, *filename.split("/"))
        #         all_data.append([str(full_path), text[:-1]])

        lang_data = pandas.read_csv(os.path.join(sub_folder_path, f"{sub_folder}_gt.txt"), sep="\t", header = None, encoding='utf-8')
        lang_data.columns = ['file_name', 'text']

        lang_data['file_name'] = str(sub_folder_path) + '/' + lang_data['file_name']     

        all_data.append(lang_data)

    data = pandas.concat(all_data, ignore_index=True)
    # data = pandas.DataFrame(all_data, columns=["file_name", "text"])

    return data

def load_and_preprocess_data(DATASET_URL):
    all_data = []

    # EXCLUDE_LANGUAGES = [
    #     "mod_meitei_manipuri",
    #     "mod_urdu"
    # ]

    EXCLUDE_LANGUAGES = [
        # "mod_assamese",
        # "mod_bengali",
        # "mod_gujarati",
        # "mod_hindi",
        # "mod_kannada",
        # "mod_malayalam",
        # "mod_marathi",
        "mod_meitei_manipuri",
        # "mod_odia",
        # "mod_punjabi",
        # "mod_tamil",
        # "mod_telugu",
        "mod_urdu"
    ]

    LANGUAGE_MAP = {
        "mod_assamese": "as",
        "mod_bengali": "bn",
        "mod_gujarati": "gu",
        "mod_hindi": "hi",
        "mod_kannada": "kn",
        "mod_malayalam": "ml",
        "mod_marathi": "mr",
        "mod_odia": "or",
        "mod_punjabi": "pa",
        "mod_tamil": "ta",
        "mod_telugu": "te",
    }

    FILTERED_LANGUAGES = [lang for lang in os.listdir(DATASET_URL) if lang not in EXCLUDE_LANGUAGES and os.path.isdir(os.path.join(DATASET_URL, lang))]

    for lang_folder in tqdm(FILTERED_LANGUAGES, desc="Loading data", dynamic_ncols=True, leave=True):

        sub_folder = os.listdir(os.path.join(DATASET_URL, lang_folder))[0]
        sub_folder_path = os.path.join(DATASET_URL, lang_folder, sub_folder)

        lang_data = pandas.read_csv(os.path.join(sub_folder_path, f"{sub_folder}_gt.txt"), sep="\t", header = None, encoding='utf-8')
        lang_data.columns = ['file_name', 'text']

        lang_data = lang_data.sample(frac=0.001)
         
        # print(lang_data['text'])
        # print(type(lang_data['text']))

        lang_data['text'] = lang_data['text'].apply(lambda x: UnicodeIndicTransliterator.transliterate(unicodedata.normalize("NFC", x), LANGUAGE_MAP[lang_folder], 'hi'))

        lang_data['file_name'] = str(sub_folder_path) + '\\' + lang_data['file_name']     
        lang_data['text'] = '<2' + LANGUAGE_MAP[lang_folder] + "> " + lang_data['text'] + ' </s>'
        # lang_data['text'] = lang_data['text'] + ' </s>'

        # lang_data = lang_data.iloc[:1000]
        lang_data['lang'] = LANGUAGE_MAP[lang_folder]

        all_data.append(lang_data)

    data = pandas.concat(all_data, ignore_index=True)

    return data

def load_english_data(test = False):
    
    if test:
        CSV_URL = r"data\icdar2013\test_data.csv"
        FOLDER_PATH = r"data\icdar2013\test_cropped_images"
    else:
        CSV_URL = r"data\icdar2013\train_data.csv"
        FOLDER_PATH = r"data\icdar2013\train_cropped_images"
    
    data = pandas.read_csv(CSV_URL, sep=",", encoding='utf-8')

    data['text'] = '<2en> ' + data["text"] + ' </s>'
    data['file_name'] = str(FOLDER_PATH) + '\\' + data['file_name']
    # data['lang'] = 'en'
    
    return data    
    
def load_icdar_data(test = False):
    
    if test:
        CSV_URL = r"data\icdar2013\test_data.csv"
        FOLDER_PATH = r"data\icdar2013\test_cropped_images"
    else:
        CSV_URL = r"D:\MT23MCS002\Project\Detection\datasets\ICDAR 2013\train_data_cleaned.csv"
        FOLDER_PATH = r"D:\MT23MCS002\Project\Detection\datasets\ICDAR 2013\train_cropped_images"
    
    data = pandas.read_csv(CSV_URL, sep=",", encoding='utf-8')

    data['text'] = '<2en> ' + data["text"] + ' </s>'
    data['file_name'] = str(FOLDER_PATH) + '\\' + data['file_name']
    # data['lang'] = 'en'
    
    return data