# ReALLM: A general framework for LLM compression and fine-tuning

This repository is the official code for paper, https://arxiv.org/abs/2405.13155.

We introduce ReALLM, a novel approach for compression and memory-efficient adaptation of pre-trained language models that encompasses most of the post-training quantization and fine-tuning methods for a budget of <4 bits.
Pre-trained matrices are decomposed into a high-precision low-rank component and a vector-quantized latent representation (using an autoencoder).
During the fine-tuning step, only the low-rank components are updated. Our results show that pre-trained matrices exhibit different patterns.
ReALLM adapts the shape of the encoder (small/large embedding, high/low bit VQ, etc.) to each matrix.
ReALLM proposes to represent each matrix with a small embedding on b bits and a neural decoder model $\mathcal{D}_\phi$ with its weights on $b_\phi$ bits.
The decompression of a matrix requires only one embedding and a single forward pass with the decoder.
Our weight-only quantization algorithm yields the best results on language generation tasks (C4 and WikiText-2) for a budget of 3 bits without any training.
With a budget of 2 bits, ReALLM achieves state-of-the art performance after fine-tuning on a small calibration dataset.

## Quantization
* initializing the weights of the model with vector quantization rank=64 2bits (and bucket=2 hard-coded)
```
python run_vqdora.py --results_path=$MODEL_FOLDER/VQ_dora_2b_bucket2_rank64 --lora_model_name=nf2 --lora_num_ranks=64
```
* HNeRV training for layer q at depth 0 of Mistral7b (quantpshuffle layers hard-coded in 8 bits, parameter `wbits` in `model_all.Conv2d_nf`)
```
python train_nerv_all.py --vid=q0_512 --conv_type=['convnext', 'quantpshuffle'] --enc_dim=64_16 --enc_strds=[2,2,2,2,2] --dec_strds=[2,2,2,2,2] --modelsize=7.2
```
## Finetuning
* finetuning the model by bloc
```
python run_per_bloc.py --checkpoint_dir=$MODEL_FOLDER/VQ_dora_2b_bucket2_rank64
```

* finetuinng the model globally
```
python run_clm.py --checkpoint_dir=$MODEL_FOLDER/VQ_dora_2b_bucket2_rank64 --lora_model_name=nf2 --lora_num_ranks=64
```


## Perplexity evaluation
* evaluation model without finetuning
```
python eval_perplexity.py --checkpoint_dir=$MODEL_FOLDER/VQ_dora_2b_bucket2_rank64
```

* evaluation model with bloc finetuning
```
python eval_perplexity.py --checkpoint_dir=$MODEL_FOLDER/VQ_dora_2b_bucket2_rank64/finetune_bloc
```

* evaluation model with global finetuning
```
python eval_perplexity.py --checkpoint_dir=$MODEL_FOLDER/VQ_dora_2b_bucket2_rank64/finetune_bloc/finetune_global
```

## Citation
If you use this code, please cite the following
```
@misc{leconte2024reallm,
      title={ReALLM: A general framework for LLM compression and fine-tuning}, 
      author={Louis Leconte and Lisa Bedin and Van Minh Nguyen and Eric Moulines},
      year={2024},
      eprint={2405.13155},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
## References
* paper: https://arxiv.org/abs/2405.13155
* HNeRV: https://github.com/haochen-rye/HNeRV
* LQ-LoRA: https://github.com/HanGuo97/lq-lora/tree/main
* AQLM: https://github.com/Vahe1994/AQLM
