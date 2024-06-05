import sys
sys.path.append('/home/wangxiao24/dev_videochat/tools/data/bilibili/LLaVA-NeXT')
import os
import time
from tqdm import tqdm
import multiprocess as mp

import cv2
import fire
import torch
import mmengine
import numpy as np
from decord import VideoReader, cpu
from torch.utils.data import DataLoader

from llavavid.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llavavid.conversation import conv_templates, SeparatorStyle
from llavavid.model.builder import load_pretrained_model
from llavavid.utils import disable_torch_init
from llavavid.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
# pip installtransformers==4.37.2 
# pip install accelerate -U


class BilibiliDataset:
    def __init__(self, anno_file, video_root, processor, rank, world_size) -> None:
        self.video_root = video_root
        self.processor = processor

        self.annos = []
        all_annos = mmengine.load(anno_file)
        for idx, anno in enumerate(all_annos):
            if idx % world_size == rank:
                self.annos.append(anno)

    def __len__(self):
        return len(self.annos)

    def __getitem__(self, idx):
        anno = self.annos[idx]
        BVid = anno['BVid']
        anno['video_path'] = os.path.join(self.video_root, f'{BVid}.mp4')
        return self.processor(anno)


class VideoLLMProcessor:
    QUESTION_T = '''Title: {video_title}. Please summarize the content of the video based on given information in English.'''

    QUESTION_ASR = '''Subtitles: {asr_results}. Please summarize the content of the video based on given information in English.'''

    QUESTION_T_ASR = '''Title: {video_title}. Subtitles: {asr_results}. Please summarize the content of the video based on given information in English.'''

    QUESTION_V = '''Please summarize the content of the video based on given information.'''

    QUESTION_V_T = '''Title: {video_title}. Please summarize the content of the video based on given information.'''

    QUESTION_V_ASR = '''Subtitles: {asr_results}. Please summarize the content of the video based on given information.'''

    QUESTION_V_T_ASR = '''Subtitles: {asr_results}. Title: {video_title}. Please summarize the content of the video based on given information.'''

    def __init__(self, 
                 num_frames, 
                 model_path, 
                 model_base,
                 overwrite_config,
                 conv_mode,
                 max_asr_length=1024, 
                 mm_use_im_start_end=False, 
                 ablation_types=['V', 'ASR', 'V+T', 'V+ASR', 'V+T+ASR']) -> None:
        self.num_frames = num_frames
        self.image_processor, self.tokenizer = self.get_tokenizer_processor(model_path, model_base, overwrite_config, mm_use_im_start_end)
        self.conv_mode = conv_mode
        self.max_asr_length = max_asr_length
        self.mm_use_im_start_end = mm_use_im_start_end
        self.ablation_types = ablation_types

    @staticmethod
    def get_tokenizer_processor(model_path, model_base, overwrite_config, mm_use_im_start_end):
        # Initialize the model
        model_name = get_model_name_from_path(model_path)
        # Set model configuration parameters if they exist
        tokenizer, model_check, image_processor, _ = load_pretrained_model(
            model_path, model_base, model_name, device_map='cpu', overwrite_config=overwrite_config)
        assert mm_use_im_start_end == model_check.config.mm_use_im_start_end
        del model_check
        return image_processor, tokenizer

    @staticmethod
    def load_video(video_path, num_frames):
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frame_num = len(vr)
        uniform_sampled_frames = np.linspace(0, total_frame_num + 1, num_frames + 2, dtype=int)
        uniform_sampled_frames = uniform_sampled_frames[1:-1]
        frame_idx = uniform_sampled_frames.tolist()
        spare_frames = vr.get_batch(frame_idx).asnumpy()
        return spare_frames

    def __call__(self, anno):
        video_path = anno['video_path']
        video_title = anno['video_title']
        asr_results = anno['asr_results']
        asr_results = asr_results if asr_results else ""
        asr_results = asr_results[:self.max_asr_length]

        # try:
        results = dict()
        # Load video
        video_tensor = self.load_video(video_path, self.num_frames)
        video_tensor = self.image_processor.preprocess(video_tensor, return_tensors="pt")["pixel_values"].half()
        video_tensor = [video_tensor]
        for ablation_type in self.ablation_types:
            if ablation_type == 'T':
                question = self.QUESTION_T.format(video_title=video_title)
            elif ablation_type == 'ASR':
                question = self.QUESTION_ASR.format(asr_results=asr_results)
            elif ablation_type == 'T+ASR':
                question = self.QUESTION_T_ASR.format(video_title=video_title, asr_results=asr_results)
            elif ablation_type == 'V':
                question = self.QUESTION_V
            elif ablation_type == 'V+T':
                question = self.QUESTION_V_T.format(video_title=video_title)
            elif ablation_type == 'V+ASR':
                question = self.QUESTION_V_ASR.format(asr_results=asr_results)
            elif ablation_type == 'V+T+ASR':
                question = self.QUESTION_V_T_ASR.format(video_title=video_title, asr_results=asr_results)
            else:
                print(ablation_type)
                raise NotImplementedError

            qs = question
            if 'V' in ablation_type: # Add Visual tokens
                if self.mm_use_im_start_end:
                    qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
                else:
                    qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
                video = video_tensor
            else:
                video = None

            conv = conv_templates[self.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0)
            attention_masks = input_ids.ne(self.tokenizer.pad_token_id).long()
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

            inputs = dict(video=video,
                            input_ids=input_ids,
                            attention_masks=attention_masks,
                            question=question,
                            stop_str=stop_str)
            results[ablation_type] = inputs
        # except:
        #     results = None

        return results, anno


class VideoLLMWorker(mp.Process):
    def __init__(self, 
                 task_queue, 
                 result_queue, 
                 device, 
                 model_path, 
                 model_base, 
                 overwrite_config):
        super().__init__()
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.device = device
        self.model_path = model_path
        self.model_base = model_base
        self.overwrite_config = overwrite_config

    @staticmethod
    def build_model(device, model_path, model_base, overwrite_config):
        print('Init model in %s' % device)
        model_name = get_model_name_from_path(model_path)
        tokenizer, model, _, _ = load_pretrained_model(model_path, model_base, model_name, device_map=device, overwrite_config=overwrite_config)
        model = model.to(device) # `device_map` cannot work on vision_tower
        return tokenizer, model

    @staticmethod
    def inference(model, tokenizer, stop_str, video, question, input_ids, attention_masks):
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        cur_prompt = question
        with torch.inference_mode():
            model.update_prompt([[cur_prompt]])
            output_ids = model.generate(inputs=input_ids, 
                                        images=video, 
                                        attention_mask=attention_masks, 
                                        modalities="video", 
                                        do_sample=True, 
                                        temperature=0.2, 
                                        max_new_tokens=1024, use_cache=True, stopping_criteria=[stopping_criteria])

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        if outputs.endswith(stop_str):
            outputs = outputs[: -len(stop_str)]
        answer = outputs.strip()
        return answer

    def run(self):
        tokenizer, model = self.build_model(self.device, self.model_path, self.model_base, self.overwrite_config)

        while True:
            # print(self.device, 'Get a sample')
            task = self.task_queue.get()
            if task is None:
                break

            task_id, sample = task

            results, anno = sample
            BVid = anno['BVid']
            summarization = anno['summarization']
            type2results = dict(BVid=BVid, success=True)
            if results is not None:
                try:
                    for ablation_type, inputs in results.items():
                        video = inputs['video']
                        video = [video[0].to(self.device)] if video else None
                        input_ids = inputs['input_ids'].to(self.device)
                        attention_masks = inputs['attention_masks'].to(self.device)
                        question = inputs['question']
                        stop_str = inputs['stop_str']
                        with torch.no_grad():
                            answer = self.inference(model, tokenizer, stop_str, video, question, input_ids, attention_masks)
                            # print(answer)
                        type2results[ablation_type] = dict(
                            pred=answer,
                            gt=summarization,
                        )
                except Exception as e:
                    print(e)
                    msg = str(e)
                    type2results['success'] = False
                    type2results['message'] = msg
            else:
                type2results['success'] = False
                type2results['message'] = 'Data loading failure'

            self.result_queue.put((task_id, type2results))


def parse_args(
    anno_file,
    video_root,
    output_dir,
    model_path,
    clip_path,
    num_frames=8,
    ablation_types=['T', 'ASR', 'T+ASR', 'V', 'V+T', 'V+ASR', 'V+T+ASR'],
    model_suffix='',
    num_gpus=1,
    rank=0,
    world_size=1,
    num_workers=8
):
    return (anno_file, 
            video_root, 
            output_dir, 
            model_path, 
            clip_path, 
            num_frames,
            ablation_types,
            model_suffix,
            num_gpus, 
            rank, 
            world_size, 
            num_workers)


if __name__ == "__main__":
    mp.set_start_method('spawn')

    (anno_file, 
     video_root, 
     output_dir, 
     model_path, 
     clip_path, 
     num_frames,
     ablation_types,
     model_suffix,
     num_gpus, 
     rank, 
     world_size, 
     num_workers) = fire.Fire(parse_args)

    model_base = None
    if clip_path is not None:
        overwrite_config = dict(mm_vision_tower=clip_path)
    else:
        overwrite_config = None

    conv_mode = "vicuna_v1"
    mm_use_im_start_end = False
    max_asr_length = 1024

    model_name = os.path.basename(model_path)
    if model_suffix and len(model_suffix):
        model_name = model_name + '_' + model_suffix
    dataset_name = os.path.splitext(os.path.basename(anno_file))[0]

    manager = mp.Manager()
    task_queue = manager.Queue()
    result_queue = manager.Queue()
    procs = []
    for gpu_id in range(num_gpus):
        device = f'cuda:{gpu_id}'
        proc = VideoLLMWorker(task_queue, result_queue, device, model_path, model_base, overwrite_config)
        proc.start()
        procs.append(proc)

    processor = VideoLLMProcessor(
        num_frames, 
        model_path, 
        model_base,
        overwrite_config,
        conv_mode,
        max_asr_length=1024, 
        mm_use_im_start_end=False,
        ablation_types=ablation_types)
    dataset = BilibiliDataset(anno_file, video_root, processor, rank, world_size)
    dataloader = DataLoader(dataset=dataset, num_workers=num_workers, batch_size=None)

    for task_id, sample in tqdm(enumerate(dataloader), total=len(dataloader)):
        while True:
            if task_queue.qsize() < num_gpus:
                task_queue.put((task_id, sample))
                break
            else:
                time.sleep(0.1)
    for _ in range(num_gpus):
        task_queue.put(None)

    num_tasks = len(dataloader)
    finished_tasks = 0
    print('Getting finished tasks')
    inference_results = {}
    while finished_tasks < num_tasks:
        task_id, type2results = result_queue.get()
        BVid = type2results.pop('BVid')
        inference_results[BVid] = type2results
        finished_tasks += 1

    for proc in procs:
        proc.join()

    mmengine.dump(inference_results, 
                  os.path.join(output_dir, f'inference_results_{dataset_name}_{model_name}_{rank}.json'),
                  indent=2)