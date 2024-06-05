# Setting the environment
- Follow the instructions in [VideoChat2](https://github.com/OpenGVLab/Ask-Anything/tree/main/video_chat2) to prepare invironments.

- Download the pretrained Q-Former:
```bash
cd MODEL_PATH # your path to store model checkpoints
wget https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/videochat2/umt_l16_qformer.pth
```

- Download Vicuna-7b-v0 with checkpoint:
```bash
cd MODEL_PATH
git lfs clone https://huggingface.co/ZzZZCHS/vicuna-7b-v0
wget https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/videochat2/videochat2_7b_stage3.pth
```

- Download Mistral-7B with checkpoint:
```bash
cd MODEL_PATH
git lfs clone https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2
git lfs clone https://huggingface.co/OpenGVLab/VideoChat2_stage3_Mistral_7B
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
python inference/run_inference_videochat2.py \
--anno_file=$anno_file --video_root=$video_root --output_dir=$output_dir \
--config_file="Ask-Anything/video_chat2/configs/config.json" \
--vit_blip_model_path=$MODEL_PATH/"umt_l16_qformer.pth" \
--llm_model_path=$MODEL_PATH/"vicuna-7b-v0" \
--videochat2_model_path=$MODEL_PATH/"videochat2_7b_stage3.pth" \
--num_frames=8 --num_gpus=$num_gpus --rank=$rank --world_size=$world_size --num_workers=8

python inference/run_inference_videochat2.py \
--anno_file=$anno_file --video_root=$video_root --output_dir=$output_dir \
--config_file="Ask-Anything/video_chat2/configs/config_mistral.json" \
--vit_blip_model_path=$MODEL_PATH/"umt_l16_qformer.pth" \
--llm_model_path=$MODEL_PATH/"Mistral-7B-Instruct-v0.2" \
--videochat2_model_path=$MODEL_PATH/"VideoChat2_stage3_Mistral_7B/videochat2_mistral_7b_stage3.pth" \
--num_frames=8 --num_gpus=$num_gpus --rank=$rank --world_size=$world_size --num_workers=8


# performance of modality ['T', 'ASR', 'T+ASR']
python inference/run_inference_hfnlp.py \
--anno_file=$anno_file --video_root=$video_root \ 
--output_dir=$output_dir \
--llm_model_path=$MODEL_PATH/"vicuna-7b-v1.5" \
--max_asr_length=2048 --num_gpus=$num_gpus --rank=$rank --world_size=$world_size --num_workers=8

python inference/run_inference_hfnlp.py \
--anno_file=$anno_file --video_root=$video_root \
--output_dir=$output_dir \
--llm_model_path=$MODEL_PATH/"Mistral-7B-Instruct-v0.2" \
--max_asr_length=2048 --num_gpus=$num_gpus --rank=$rank --world_size=$world_size --num_workers=8
```

- Impact of sampled frames on performance:
```bash
for num_frames in 1 2 4 8 16 32 64;do
    python inference/run_inference_videochat2.py \
    --anno_file=$anno_file --video_root=$video_root \
    -output_dir=$output_dir \
    --config_file="Ask-Anything/video_chat2/configs/config_mistral.json" \
    --vit_blip_model_path=$MODEL_PATH/"umt_l16_qformer.pth" \
    --llm_model_path=$MODEL_PATH/"Mistral-7B-Instruct-v0.2" \
    --videochat2_model_path=$MODEL_PATH/"VideoChat2_stage3_Mistral_7B/videochat2_mistral_7b_stage3.pth" \
    --num_frames=$num_frames --ablation_types=['V'] --model_suffix='f_'$num_frames \
    --num_gpus=$num_gpus --rank=$rank --world_size=$world_size --num_workers=8
done
```