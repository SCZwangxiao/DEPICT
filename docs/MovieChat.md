# Setting the environment
- Create a new conda environment:
```bash
conda create -m moviechat python=3.8
```
- Install moviechat:
```bash
pip install MovieChat==0.6.3
```
- Download vicuna-7b-v0:
```bash
cd MODEL_PATH # your path to store model checkpoints
git lfs clone https://huggingface.co/ZzZZCHS/vicuna-7b-v0
```

# Inference
- Setting variables:
```bash
rank=__node_id__
world_size=__number_of_nodes__
num_gpus=__number_of_gpus_per_node__
MODEL_PATH=__your_path_to_store_model_checkpoints__

anno_file='data/depict/annotations/test.json'
video_root='data/depict/videos'
output_dir='results'
```

- Baseline performance and ablation studies on different modalities:
```bash
# performance of modality ['V', 'V+T', 'V+ASR', 'V+T+ASR']
python inference/run_inference_moviechat.py \
--anno_file=$anno_file --video_root=$video_root \
--output_dir=$output_dir \
--num_gpus=$num_gpus --rank=$rank --world_size=$world_size

# performance of modality ['T', 'ASR', 'T+ASR']
python tools/data/bilibili/run_inference_hfnlp.py \
--anno_file=$anno_file --video_root=$video_root \
--output_dir=$output_dir \
--llm_model_path=$MODEL_PATH/'vicuna-7b-v0' \
--num_gpus=8 --rank=$rank --world_size=$world_size --num_workers=8
```

- Impact of sampled frames on performance:
```bash
for n_samples in 1 2 4 8 16 32 64;do
    python inference/run_inference_moviechat.py \
    --anno_file=$anno_file --video_root=$video_root \
    --output_dir=$output_dir \
    --n_samples=$n_samples --ablation_types=['V'] --model_suffix='f_'$n_samples \
    --rank=$rank --world_size=$world_size --num_workers=8
done
```