# [EMNLP 2024] Low-Resource Machine Translation through the Lens of Personalized Federated Learning 

This repository contains code for paper [Low-Resource Machine Translation through the Lens of Personalized Federated Learning](https://arxiv.org/abs/2406.12564)

## Using optimizer

The optimizer could be found in ```pipeline_src/optimizers.py```. To add this into your code you just need to import the optimizer and correctly provide the losses to it during training. Below is the example of training with Indonesian and Javanese languages.
Our code also requires accelerate to run.

```python
from pipeline_src.optimizers import MeritFedParallelMD
from accelerate import Accelerator

# Init the accelerator
accelerator = Accelerator()

config = {
  'lr': <learning rate of main optimizer>,
  'npeers': <number of datasets>,
  'mdlr_': <learning rate for auxiliary optimizer>,
  'mdniters_' = <number of auxiliary optimizer iterations>,
  'drop_threshold' = <threshold to drop worker. Could be any value if not using>
}
device = <your device>

model = <your HF model>
# wrap the model with accelerator
model = accelerator.prepare_model(model)
weight_name_map = <mapping for dataset indexes and their names> # for example {0: indonesian, 1: javanese}
train_loader, val_loader = <your dataloaders>
# wrap dataloaders with accelerator
train_loader = accelerator.prepare(train_loader)
val_loader = accelerator.prepare(val_loader)

optimizer = MeritFedA(
  model.parameters(), config, val_loader=val_loader, model=model, accelerator=accelerator
)

# During training we need to register each worker grad

# First we calculate loss on the Indonesian Data
w_id = 0 # We have set id 0 to indonesian
output = model.forward(indonesian_input)
loss = output["loss"]
loss.backward()
# We step with providing id of data, model and validation loader to perform auxiliary optimization
# double optimizer class since first is wrapper of accelerate
optimizer.optimizer.register_worker_grad(w_id)
# we perform the zero grad here
optimizer.zero_grad()

# Next we do the same with second language, javanese in our example
w_id = 1 # Javanese has id = 1
output = model.forward(javanese_input)
loss = output["loss"]
loss.backward()
# We step the same way but with new id
optimizer.optimizer.register_worker_grad(w_id)
# WE DO NOT PERFORM ZERO GRAD AT LAST WORKER!

# Finally we make a step once all gradients are registered
optimizer.step()
# Then we zero gradients only after the step performed
optimizer.zero_grad()
```

## Paper reproducement

### Getting started

install requierements and set up environmental variables. Since the code heavily relies on the CUDA, Apex and Accelerate, it is highly recommended, to run it with [Docker Container](https://hub.docker.com/layers/vityavitalich/mtfl/apex/images/sha256-7d28a74d840ab4c3cd0f17a7bad5dc82ab70e4b6bfd496bd79e6c01b24677e45?context=repo), that already contains all dependencies.

Otherwise, one may install packages with following commands, however may experience errors due to CUDA installation errors.

python version = 3.10.12
1. ```pip install -r requirements.txt```

Further, to run any training, code requires to have the following environment variables to be defined

1. ```export HF_TOKEN=<your token>```
2. ```export SAVING_DIR=<path to your cache directory>```
3. ```export WANDB_API_KEY=<wandb api key for logging>```

### Training Models

All training is performed via ```train.py``` script, the only thing to modify is training config, located at ```configs/train.yml```. The config has the following fields, divided by groups.

Environment parameters:
- SEED: random seed for whole script, we have used 57,58 and 59
- CUDA_VISIBLE_DEVICES: visible devices on your machine

Model parameters:
- EPOCHS: number of epochs to perform. Could be overwritten by MAX_STEPS
- MAX_STEPS: maximum backward steps to perform. Overwrites EPOCHS. Set to -1 if disable this parameter and only consider EPOCHS
- BATCH_SIZE: batch size
- LR: learning rate in any optimizer
- MIN_LR: minimum learning rate in scheduler
- MAX_SEQ_LEN: maximum sequence length, set to 256 to reproduce results
- DATA_PATH: path to hf repo with data or to local dir with hf format dataset
- MODEL_TYPE: we experimented with different model types, please set to M2M100
- MODEL_CHECKPOINT: HF format model checkpoint from local folder or HF Hub
- DATA_PARALLEL: Whether use data parallelism or not
- TARGET_LANG: target language. 'en' for Indonesian experiments, 'fi' for Finno-Ugric


Logging Parameters:
- LOG_PRED_EVERY: Logging predictions to WandB every LOG_PRED_EVERY iterations
- RUN_NAME: Name of WandB run
- VAL_EVERY_EPOCH: Validation performed and loss reported every VAL_EVERY_EPOCH epochs
- SAVE_EVERY: Model is saved every SAVE_EVERY epoch
- SAVE_STRATEGY: Whether save after epoch of after some steps passed. Possible values: 'epochs' or 'steps'
- COMPUTE_METRICS_EVERY: Target metrics are computed every COMPUTE_METRICS_EVERY epochs

Generation Parameters:
- NO_REPEAT_NGRAM: generation parameter, set to 0 to reproduce results
- SAMPLING: generation parameter, set to false to reproduce results
- TEMPERATURE: generation parameter, set to 1 to reproduce results
- TOP_K: generation parameter, not used if Sampling disabled
- NUM_BEAMS: generation parameter, set to 4 to reproduce results

Personalized Federated Learning Parameters:
- FL: parameter capable of turning off and on the Personalized Federated Learning
- FL_LR: Learning rate for auxillary optimizer in FL setting, not used if FL disabled
- FL_NITERS: Number of steps for auxillary optimizer in FL setting, not used if FL disabled
- AUX_ADAM: Whether to perform auxiliary optimization with Adam and not Mirror Descent. Not recommended, since showed worse and unstable performance
- FL_BETA_1: Beta_1 parameter for Auxiliary Adam optimizer.
- DROP_THRESHOLD: Threshold to use for dropping worker, corresponding to MeritFed_Drop
- ENABLE_FL_EVERY: On which epochs to perform MeritFed, corresponding to MeritFed_Cycle

Adaptive Batch Parameters
- ADAPTIVE_BATCH_SIZE: parameter capable of turning off and on the Adaptive Batch
- TOTAL_BATCH_SIZE: parameter of ADAPTIVE_BATCH, not used if ADAPTIVE_BATCH disabled
- MIN_BATCH_SIZE: parameter of ADAPTIVE_BATCH, not used if ADAPTIVE_BATCH disabled
- MAX_BATCH_SIZE: parameter of ADAPTIVE_BATCH, not used if ADAPTIVE_BATCH disabled


### Infering models

Infering is possible with setting training parameters in such setting that model will not be trained, but will be inferred. One needs to set

- MAX_STEPS = 1
- LR = 0
- COMPUTE_METRICS_EVERY = 1


### Data

All the used data is located in [drive](https://drive.google.com/drive/folders/1XTkXwDBhcKLxVatkLTWfK11ksyLzKSAI?usp=sharing) with compressed files. If the data with no target language in train is not present, it could be easily obtained by dropping target language from train. And if the only train data for language is not present, it could be obtained by dropping all other languages from full dataset. The validation and test sets are fixed as they are in provided datasets.
