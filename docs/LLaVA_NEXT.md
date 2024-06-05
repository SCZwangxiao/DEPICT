# Setting the environment
- Follow the instructions in [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT) to prepare invironments.

- Download LLaVA-NeXT checkpoint:
```bash
cd MODEL_PATH
git lfs clone https://huggingface.co/lmms-lab/LLaVA-NeXT-Video-7B
git lfs clone https://huggingface.co/lmms-lab/LLaVA-NeXT-Video-7B-DPO
git lfs clone https://huggingface.co/lmms-lab/LLaVA-NeXT-Video-34B
git lfs clone https://huggingface.co/lmms-lab/LLaVA-NeXT-Video-34B-DPO
```

- Download CLIP ViT-L/14 336 with checkpoint:
```bash
cd MODEL_PATH
git lfs clone https://huggingface.co/openai/clip-vit-large-patch14-336
```

- Download Yi-1.5-34B with checkpoint:
```bash
cd MODEL_PATH
git lfs clone https://huggingface.co/01-ai/Yi-1.5-34B-Chat
```

- Fix typos in `LLaVA-NeXT/llavavid/mm_utils.py`. Specifically, replace:
```python
if (output_ids[0, -keyword_id.shape[0]:] == keyword_id).all():
    return True
```
with:
```python
if output_ids[0].shape[0] < keyword_id.shape[0]:
    # Generated length too short
    return False
elif (output_ids[0, -keyword_id.shape[0]:] == keyword_id).all():
    return **True**
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
python inference/run_inference_llavanext.py \
--anno_file=$anno_file --video_root=$video_root \
--output_dir=$output_dir \
--model_path=$MODEL_PATH/"LLaVA-NeXT-Video-7B" \
--clip_path=$MODEL_PATH/"clip-vit-large-patch14-336" \
--num_gpus=$num_gpus --rank=$rank --world_size=$world_size \
--num_workers=8

python inference/run_inference_llavanext.py \
--anno_file=$anno_file --video_root=$video_root \
--output_dir=$output_dir \
--model_path=$MODEL_PATH/"LLaVA-NeXT-Video-7B-DPO" \
--clip_path=$MODEL_PATH/"clip-vit-large-patch14-336" \
--num_gpus=$num_gpus --rank=$rank --world_size=$world_size \
--num_workers=8

python inference/run_inference_llavanext.py \
--anno_file=$anno_file --video_root=$video_root \
--output_dir=$output_dir \
--model_path=$MODEL_PATH/"LLaVA-NeXT-Video-34B" \
--clip_path=$MODEL_PATH/"clip-vit-large-patch14-336" \
--num_gpus=$num_gpus --rank=$rank --world_size=$world_size \
--num_workers=8

python inference/run_inference_llavanext.py \
--anno_file=$anno_file --video_root=$video_root \
--output_dir=$output_dir \
--model_path=$MODEL_PATH/"LLaVA-NeXT-Video-34B-DPO" \
--clip_path=$MODEL_PATH/"clip-vit-large-patch14-336" \
--num_gpus=$num_gpus --rank=$rank --world_size=$world_size \
--num_workers=8


# performance of modality ['T', 'ASR', 'T+ASR']
python inference/run_inference_hfnlp.py \
--anno_file=$anno_file --video_root=$video_root \ 
--output_dir=$output_dir \
--llm_model_path=$MODEL_PATH/"vicuna-7b-v1.5" \
--max_asr_length=2048 --num_gpus=$num_gpus --rank=$rank --world_size=$world_size --num_workers=8

python inference/run_inference_hfnlp.py \
--anno_file=$anno_file --video_root=$video_root --output_dir=$output_dir \
--llm_model_path=$MODEL_PATH/"Yi-1.5-34B-Chat" \
--num_gpus=$num_gpus --rank=$rank --world_size=$world_size --num_workers=8
```

- Impact of sampled frames on performance:
```bash
for num_frames in 1 2 4 8 16 32 64;do
    python inference/run_inference_llavanext.py \
    --anno_file=$anno_file --video_root=$video_root --output_dir=$output_dir \
    --model_path=$MODEL_PATH/"LLaVA-NeXT-Video-7B" \
    --clip_path=$MODEL_PATH/"clip-vit-large-patch14-336" \
    --num_frames=$num_frames --ablation_types=['V'] --model_suffix='f_'$num_frames \
    --num_gpus=$num_gpus --rank=$rank --world_size=$world_size --num_workers=8
done
```