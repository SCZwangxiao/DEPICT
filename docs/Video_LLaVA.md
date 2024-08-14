# Setting the environment
- Follow the instructions in [Video-LLaVA](https://github.com/PKU-YuanGroup/Video-LLaVA) to prepare invironments.

- Download Video-LLaVA checkpoint:
```bash
cd MODEL_PATH # your path to store model checkpoints
git lfs clone https://huggingface.co/LanguageBind/Video-LLaVA-7B-hf
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
python tools/data/bilibili/run_inference_videollava.py \
--anno_file=$anno_file --video_root=****$video_root --output_dir=$output_dir \
--hf_model_path=$MODEL_PATH/"Video-LLaVA-7B-hf" \
--num_frames=8 --num_gpus=$num_gpus --rank=$rank --world_size=$world_size --num_workers=8

# performance of modality ['T', 'ASR', 'T+ASR']
python inference/run_inference_hfnlp.py \
--anno_file=$anno_file --video_root=$video_root \ 
--output_dir=$output_dir \
--llm_model_path=$MODEL_PATH/"vicuna-7b-v1.5" \
--max_asr_length=2048 --num_gpus=$num_gpus --rank=$rank --world_size=$world_size --num_workers=8
```