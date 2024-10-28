# RLHF Training

This part of codes is for the RLHF training, including both the DPO and PPO algorithms.

## Model Checkpoints
- [SFT model](https://huggingface.co/RLHFlow/LLaMA3-SFT), also check more SFT checkpoints [Here](https://huggingface.co/collections/RLHFlow/sft-models-66eda119ea7d19a23904da28)
- [Reward model](https://huggingface.co/sfairXC/FsfairX-LLaMA3-RM-v0.1), also check more reward models [Here](https://huggingface.co/collections/RLHFlow/rlhflow-reward-models-669ecdd1c7e62283cb54b5fd)

## DPO Installation instructions

It is recommended to have two separate environments for **inference** and **training**, respectively, so that we can do online iterative training.

**Inference Environment**

```sh
conda create -n vllm python=3.10.9
conda activate vllm
pip install datasets

# The following code is tested for CUDA12.0-12.2, and CUDA12.6
# To develop llama-3, mistral, gemma-1, 1.1, 2, deepseek you can consider the following vllm version
pip install vllm==0.5.4

pip install accelerate==0.33.0
pip install deepspeed==0.14.5
pip install transformers==4.43.4
pip install numpy==1.26.4 #Note that the numpy version should be `numpy<2.0`.  `Numpy 2.0` will encounter unexpected issues!!!
```

**Training Environment**

```sh
conda create -n rlhflow python=3.10.9
conda activate rlhflow

git clone https://github.com/huggingface/alignment-handbook.git
cd ./alignment-handbook/
git checkout 27f7dbf00663dab66ad7334afb7a1311fa251f41
pip3 install torch==2.1.2 torchvision torchaudio
python -m pip install .
pip install flash-attn==2.6.3
pip install accelerate==0.33.0
pip install huggingface-hub==0.24.7
```

You also need to install the wandb to record the training and login with your huggingface account so that you have access to the LLaMA3 models.

```sh
pip install wandb==0.17.7

wandb login
huggingface-cli login
```

## PPO Installation instructions

```sh
conda create -n ppo_train python=3.10.9
conda activate ppo_train
conda install cuda -c nvidia/label/cuda-12.2.0
conda install nvidia/label/cuda-12.2.0::cuda-nvcc
pip3 install torch==2.1.2 torchvision torchaudio
pip install openrlhf[vllm]
pip3 install torch==2.1.2 torchvision torchaudio
```

## Get Started for DPO training
We present a step-by-step guidance in this section. 

### Step 1 Data Generation
We have prepared some prompt sets on huggingface.
- UltraFeedback RLHFlow/ultrafeedback_iter1, RLHFlow/ultrafeedback_iter2, RLHFlow/ultrafeedback_iter3
- RLHFlow/iterative-prompt-v1-iter1-20K, RLHFlow/iterative-prompt-v1-iter2-20K, RLHFlow/iterative-prompt-v1-iter3-20K ...

To accelerate data generation, we use the VLLM. We prepare two ways of using VLLM to inference for a more robust implementation, where you can try them out and choose the one that fits with your environment best. We use LLaMA3-8B as an example. 

You may create a test_gen.sh file, and copy the following contents into the file and run ``bash test_gen.sh''.

```sh
# First approach: initialize 4 VLLM processes and split the prompt set to the 4 agents
# The generated samples will be stored at output_dir + local_index + ".jsonl

my_world_size=8 # how many gpu you use
infer_model=RLHFlow/LLaMA3-SFT
prompt_dir=RLHFlow/test_generation_2k
mkdir data
output_dir=./data/gen_data

conda activate vllm
CUDA_VISIBLE_DEVICES=0 python ./generation/get_hf2.py --model_name_or_path ${infer_model} --dataset_name_or_path ${prompt_dir} --output_dir ${output_dir} --K 4 --temperature 1.0 --local_index 0 --my_world_size ${my_world_size}  &
CUDA_VISIBLE_DEVICES=1 python ./generation/get_hf2.py --model_name_or_path ${infer_model} --dataset_name_or_path ${prompt_dir} --output_dir ${output_dir} --K 4 --temperature 1.0 --local_index 1 --my_world_size ${my_world_size}  &
CUDA_VISIBLE_DEVICES=2 python ./generation/get_hf2.py --model_name_or_path ${infer_model} --dataset_name_or_path ${prompt_dir} --output_dir ${output_dir} --K 4 --temperature 1.0 --local_index 2 --my_world_size ${my_world_size}  &
CUDA_VISIBLE_DEVICES=3 python ./generation/get_hf2.py --model_name_or_path ${infer_model} --dataset_name_or_path ${prompt_dir} --output_dir ${output_dir} --K 4 --temperature 1.0 --local_index 3 --my_world_size ${my_world_size}  &
CUDA_VISIBLE_DEVICES=4 python ./generation/get_hf2.py --model_name_or_path ${infer_model} --dataset_name_or_path ${prompt_dir} --output_dir ${output_dir} --K 4 --temperature 1.0 --local_index 4 --my_world_size ${my_world_size}  &
CUDA_VISIBLE_DEVICES=5 python ./generation/get_hf2.py --model_name_or_path ${infer_model} --dataset_name_or_path ${prompt_dir} --output_dir ${output_dir} --K 4 --temperature 1.0 --local_index 5 --my_world_size ${my_world_size}  &
CUDA_VISIBLE_DEVICES=6 python ./generation/get_hf2.py --model_name_or_path ${infer_model} --dataset_name_or_path ${prompt_dir} --output_dir ${output_dir} --K 4 --temperature 1.0 --local_index 6 --my_world_size ${my_world_size}  &
CUDA_VISIBLE_DEVICES=7 python ./generation/get_hf2.py --model_name_or_path ${infer_model} --dataset_name_or_path ${prompt_dir} --output_dir ${output_dir} --K 4 --temperature 1.0 --local_index 7 --my_world_size ${my_world_size}  &

# then, we merge the 8 datasets into one dataset.
wait
python ./generation/merge_data.py --base_path ${output_dir} --output_dir ./data/gen_data.json --num_datasets ${my_world_size}
```

We can also use API server to generate new responses.

```sh
mkdir data
conda activate vllm

# register the api server
bash ./generation/register_server.sh RLHFlow/LLaMA3-SFT

# start to generate
python ./generation/gen_hf.py --ports 8000 8001 8002 8003 8004 8005 8006 8007 --tokenizer RLHFlow/LLaMA3-SFT --dataset_name_or_path RLHFlow/test_generation_2k --output_dir ./data/gen_data.jsonl --K 4 --temperature 1.0
```

### Step 2 Data Annotation
Then, we call the reward/preference model trained in step 2 to rank the generated responses. 

```sh
accelerate launch ./annotate_data/get_rewards.py --dataset_name_or_path ./data/gen_data.jsonl --output_dir ./data/data_with_rewards.jsonl --K 4
```
If you encounter error ``TypeError: Got unsupported ScalarType BFloat16'', considering adjusting your transformer version.

### Step 3 Training

```sh
conda activate rlhflow
accelerate launch --config_file ./configs/zero2.yaml dpo_iteration/run_dpo.py ./config/training.yaml
```
If you encounter ``RuntimeError: CUDA error: invalid device ordinal, CUDA kernel errors might be asynchronously reported at some other API call'', you need to adjust num_of_process in the config file according to your GPUs.


## Get Started for PPO training
We have summarized the training details in run_ppo.sh and you can run the experiment as follows.

```sh
bash run_ppo.sh
```
