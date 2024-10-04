from transformers import AutoTokenizer, T5EncoderModel

tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-base")
text_encoder = T5EncoderModel.from_pretrained("google-t5/t5-base")

tokenizer.save_pretrained("components/tokenizer/t5-base")
text_encoder.save_pretrained("components/text-encoder/t5-base")

from datasets import load_dataset

dataset = load_dataset("openmodelinitiative/initial-test-dataset")