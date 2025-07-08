# TrOCR-Indic

This project is part of Project Phase - II of Graduate Program in Visvesvaraya National Institute of Technology, Nagpur.

This model utilizes the trocr approach to predict the **Indic Texts** from **cropped_images**. 

The model is available in [HuggingFace](https://huggingface.co/QuickHawk/trocr-indic).

## Model Details

The model follows the TrOCR approach of training OCR for Scene Texts. Since, there is scarcity for generalized model for majority of Indian Languages, this model serves it replacement.

![TrOCR_Architecture.jpg](https://cdn-uploads.huggingface.co/production/uploads/6868f8219c4cd7445653ada1/d6d9a0UVlL8EleZrC_ts9.jpeg)
*Courtesty: TrOCR - [original paper](https://huggingface.co/papers/2109.10282)*

The model is trained for the following languages:

- Assamese
- Bengali
- Gujarati
- Hindi
- Kannada
- Malayalam
- Marathi
- Odia
- Punjabi
- Telugu
- Tamil

### Model Description

**IMPORTANT**
Although the model is trained on these languages due to limitations of IndicBART, the model is trained with only Devnagiri Scripts. 

The output is in the following format: 
```
<LANGUAGE TOKEN> <TEXT TOKENS> <EOS TOKEN>
```

The following flowchart gives a better picture on the approach of training and inference regarding this model.

![Reworked_Implementation](https://cdn-uploads.huggingface.co/production/uploads/6868f8219c4cd7445653ada1/1KiAan55GWl9tZNOTuMs0.png)

**Datasets used:** [IndicSTR12](https://cvit.iiit.ac.in/research/projects/cvit-projects/indicstr)

### Results

| Metric | Assamese | Bengali | Gujarati | Hindi | Kannada | Malayalam | Marathi | Odia | Punjabi | Tamil | Telugu |
|--------|----------|---------|----------|-------|---------|-----------|---------|------|---------|-------|--------|
| CER    | 0.069    | 0.133   | 0.058    | 0.075 | 0.212   | 0.154     | 0.082   | 0.120 | 0.097   | 0.122 | 0.220  |
| WER    | 0.205    | 0.395   | 0.192    | 0.283 | 0.576   | 0.519     | 0.312   | 0.375 | 0.304   | 0.409 | 0.612  |

Well, the model isn't perfect. But it's a start.

## Limitations

The main limitation comes from IndicBART which is primarily trained on IndicTexts. 

### Recommendations

Since the TrOCR is modular in approach one can just swap out the IndicBART model and train it with new model. Must keep in mind about the preprocessing and outputs.
