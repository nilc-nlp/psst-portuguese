# Psst Portuguese

Code for the preprocessing, training and evaluation of the psst-portuguese model, available at the [NILC NLP Hugging Face](https://huggingface.co/nilc-nlp). Some scripts require the creation of a .env with your hugging face and wandb keys.

## Preprocessing pipeline

1. Run "parse-textgrid.py" to convert the (audio, textgrid) pairs from the original datasets into a csv file.
2. Run "cut_audio.py" with the csv file from (1) to cut the (audio, textgrid) pairs into small audio segments
3. Run "concat_audio_clean.py" to remove unwanted elements from the audio segments transcriptions and concatenate the audios.

## Training pipeline

1. Convert the results from the preprocessing pipeline into a Hugging Face Dataset.
2. With the dataset, run "finetune_whisper.py" with the desired arguments.

## Evaluation pipeline

1. Run "psst_inference.py" to generate the transcriptions from the test split.
2. Run "psst_metrics.py" to generate metrics based on the transcriptions

## Citation

The paper for this project is not yet publicly available. If you want to use this code, please cite using the GitHub link or the arxiv link 
https://arxiv.org/abs/2607.07408 .