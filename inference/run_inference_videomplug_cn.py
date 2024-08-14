import re
import os
from tqdm import tqdm

import torch
import mmengine
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

from mplug_owl_video.modeling_mplug_owl import MplugOwlForConditionalGeneration
from mplug_owl_video.processing_mplug_owl import MplugOwlImageProcessor, MplugOwlProcessor


class BilibiliDataset:
    def __init__(self, anno_file, video_root, processor) -> None:
        self.annos = mmengine.load(anno_file)
        self.video_root = video_root
        self.processor = processor

    def __len__(self):
        return len(self.annos)

    def __getitem__(self, idx):
        anno = self.annos[idx]
        BVid = anno['BVid']
        anno['videopath'] = os.path.join(self.video_root, f'{BVid}.mp4')
        return self.processor(anno)


class MPLUGProcessor:
    PROMPT_V = '''The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
    Human: <|video|>
    Human: 请直接阐述视频所讲述的内容。
    AI: '''

    PROMPT_ASR = '''The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
    Human: 字幕：{asr_results}。请依据上述信息，直接阐述视频所讲述的内容。
    AI: '''

    PROMPT_T_ASR = '''The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
    Human: 标题：{video_title}。字幕：{asr_results}。请依据上述信息，直接阐述视频所讲述的内容。
    AI: '''

    PROMPT_V_T = '''The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
    Human: <|video|>
    Human: 标题：{video_title}。请依据上述信息，直接阐述视频所讲述的内容。
    AI: '''

    PROMPT_V_ASR = '''The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
    Human: <|video|>
    Human: 字幕：{asr_results}。请依据上述信息，直接阐述视频所讲述的内容。
    AI: '''

    PROMPT_V_T_ASR = '''The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
    Human: <|video|>
    Human: 字幕：{asr_results}。标题：{video_title}。请依据上述信息，直接阐述视频所讲述的内容。
    AI: '''

    def __init__(self, pretrained_ckpt, num_frames, max_asr_length=1024, ablation_types=['V', 'ASR', 'V+T', 'V+ASR', 'V+T+ASR']) -> None:
        image_processor = MplugOwlImageProcessor.from_pretrained(pretrained_ckpt)
        tokenizer = AutoTokenizer.from_pretrained(pretrained_ckpt)
        processor = MplugOwlProcessor(image_processor, tokenizer)
        self.processor = processor
        self.num_frames = num_frames
        self.max_asr_length = max_asr_length
        self.ablation_types = ablation_types

    def __call__(self, anno):
        videopath = anno['videopath']
        video_title = anno['video_title_zh']
        asr_results = anno['asr_results_zh']
        asr_results = ','.join([ele['asr_result_zh'] for ele in asr_results]) if asr_results else ""
        asr_results = asr_results[:self.max_asr_length]

        results = dict()
        for ablation_type in self.ablation_types:
            if ablation_type == 'V':
                prompt = self.PROMPT_V
            elif ablation_type == 'ASR':
                prompt = self.PROMPT_ASR.format(asr_results=asr_results)
            elif ablation_type == 'T+ASR':
                prompt = self.PROMPT_T_ASR.format(video_title=video_title, asr_results=asr_results)
            elif ablation_type == 'V+T':
                prompt = self.PROMPT_V_T.format(video_title=video_title)
            elif ablation_type == 'V+ASR':
                prompt = self.PROMPT_V_ASR.format(asr_results=asr_results)
            elif ablation_type == 'V+T+ASR':
                prompt = self.PROMPT_V_T_ASR.format(video_title=video_title, asr_results=asr_results)
            else:
                print(ablation_type)
                raise NotImplementedError
            prompts = [prompt]
            video_list = [videopath]
            if 'V' not in ablation_type:
                inputs = self.processor(text=prompts, return_tensors='pt')
            else:
                inputs = self.processor(text=prompts, videos=video_list, num_frames=self.num_frames, return_tensors='pt')
            inputs = {k: v.bfloat16() if v.dtype == torch.float else v for k, v in inputs.items()}
            results[ablation_type] = inputs

        return results, anno


if __name__ == "__main__":
    num_frames = 16
    num_workers = 4
    max_asr_length = 1024

    anno_file = 'data/bilibili/annotations/sample.json'
    video_root = 'data/bilibili/videos'

    pretrained_ckpt = "/mmu_nlp_ssd/wangxiao24/models/huggingface/mplug-youku-bloomz-7b"
    generate_kwargs = {
        'do_sample': True,
        'top_k': 5,
        'max_length': 1024
    }

    model = MplugOwlForConditionalGeneration.from_pretrained(
        pretrained_ckpt,
        torch_dtype=torch.bfloat16,
        device_map={'': 0},
    )

    tokenizer = AutoTokenizer.from_pretrained(pretrained_ckpt)
    processor = MPLUGProcessor(pretrained_ckpt=pretrained_ckpt, 
                               num_frames=num_frames, 
                               max_asr_length=max_asr_length, 
                               ablation_types=['V', 'ASR', 'T+ASR', 'V+T', 'V+ASR', 'V+T+ASR'])

    dataset = BilibiliDataset(anno_file, video_root, processor)
    dataloader = DataLoader(dataset=dataset, num_workers=num_workers, batch_size=None)


    inference_results = {}
    for sample in tqdm(dataloader):
        results, anno = sample
        BVid = anno['BVid']
        summarization = anno['summarization_zh']
        type2results = dict()

        for ablation_type, inputs in results.items():
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            with torch.no_grad():
                res = model.generate(**inputs, **generate_kwargs)
            sentence = tokenizer.decode(res.tolist()[0], skip_special_tokens=True)
            type2results[ablation_type] = dict(
                pred=sentence,
                gt=summarization,
            )

        inference_results[BVid] = type2results
    mmengine.dump(inference_results, 'inference_results_sample.json')