from diffusers import AutoencoderKL
from transformers import AutoTokenizer, T5EncoderModel

tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-base")
text_encoder = T5EncoderModel.from_pretrained("google-t5/t5-base")

tokenizer.save_pretrained("../components/tokenizer/t5-base")
text_encoder.save_pretrained("../components/text-encoder/t5-base")

vae = AutoencoderKL.from_pretrained("wolfgangblack/flux_vae")
vae.save_pretrained("../components/vae/flux_schnell_vae")