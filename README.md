# ReALLM

https://arxiv.org/abs/2405.13155

## model weights
* initializing the weights of the model with VQ DORA rank=64 2bits (and bucket=2 hard-coded)
```
python run_vqdora.py --results_path=$MODEL_FOLDER/VQ_dora_2b_bucket2_rank64 --lora_model_name=nf2 --lora_num_ranks=64
```

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