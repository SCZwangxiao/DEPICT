# Setting the environment
- Follow the instructions in [LLaMA-VID](https://github.com/DAMO-NLP-SG/Video-LLaMA) to prepare invironments.

- Download [EVA-ViT-G]("https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/eva_vit_g.pth")  from BLIP2's repo.

- Download CLIP models:
```bash
cd MODEL_PATH
git lfs clone https://huggingface.co/openai/clip-vit-large-patch14
```

- Download vicuna-v1.5:
```bash
cd MODEL_PATH # your path to store model checkpoints
git lfs clone https://huggingface.co/lmsys/vicuna-7b-v1.5
git lfs clone https://huggingface.co/lmsys/vicuna-13b-v1.5
```

- Download LLaMA-VID models:
```bash
cd MODEL_PATH
git lfs clone https://huggingface.co/YanweiLi/llama-vid-7b-full-224-video-fps-1
git lfs clone https://huggingface.co/YanweiLi/llama-vid-13b-full-224-video-fps-1
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
python inference/run_inference_llamavid.py \
--anno_file=$anno_file --video_root=$video_root \
--output_dir=$output_dir \
--model_path=$MODEL_PATH/"llama-vid-7b-full-224-video-fps-1" \
--clip_path=$MODEL_PATH/"eva_vit_g.pth" \
--clip_processor_path=$MODEL_PATH/"clip-vit-large-patch14" \
--num_gpus=$num_gpus --rank=$rank --world_size=$world_size \
--num_workers=8

python inference/run_inference_llamavid.py \
--anno_file=$anno_file --video_root=$video_root \
--output_dir=$output_dir \
--model_path=$MODEL_PATH/"llama-vid-13b-full-224-video-fps-1" \
--clip_path=$MODEL_PATH/"eva_vit_g.pth" \
--clip_processor_path=$MODEL_PATH/"clip-vit-large-patch14" \
--num_gpus=$num_gpus --rank=$rank --world_size=$world_size \
--num_workers=8

# performance of modality ['T', 'ASR', 'T+ASR']
python inference/run_inference_hfnlp.py \
--anno_file=$anno_file --video_root=$video_root \ 
--output_dir=$output_dir \
--llm_model_path=$MODEL_PATH/"vicuna-7b-v1.5" \
--max_asr_length=2048 --num_gpus=8 --rank=$rank --world_size=$world_size --num_workers=8

python inference/run_inference_hfnlp.py \
--anno_file=$anno_file --video_root=$video_root \ 
--output_dir=$output_dir \
--llm_model_path=$MODEL_PATH/"vicuna-13b-v1.5" \
--max_asr_length=2048 --num_gpus=8 --rank=$rank --world_size=$world_size --num_workers=8
```