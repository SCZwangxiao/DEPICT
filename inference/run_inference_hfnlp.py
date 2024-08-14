import sys
import os
import io
import time
import copy
from tqdm import tqdm
import multiprocess as mp

import cv2
import fire
import torch
import mmengine
import numpy as np
from PIL import Image
from transformers import pipeline
from torch.utils.data import DataLoader


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


class LLMProcessor:
    QUESTION_T = '''Title: {video_title}. Please summarize the content of the video based on given information in English: '''

    QUESTION_ASR = '''Subtitles: {asr_results}. Please summarize the content of the video based on given information in English: '''

    QUESTION_T_ASR = '''Title: {video_title}. Subtitles: {asr_results}. Please summarize the content of the video based on given information in English: '''

    def __init__(self, 
                 max_asr_length=1024, 
                 ablation_types=['T', 'ASR', 'T+ASR']) -> None:
        self.max_asr_length = max_asr_length
        self.ablation_types = ablation_types

    def __call__(self, anno):
        video_title = anno['video_title']
        asr_results = anno['asr_results']
        asr_results = asr_results if asr_results else ""
        asr_results = asr_results[:self.max_asr_length]

        results = dict()
        for ablation_type in self.ablation_types:
            if ablation_type == 'T':
                question = self.QUESTION_T.format(video_title=video_title)
            elif ablation_type == 'ASR':
                question = self.QUESTION_ASR.format(asr_results=asr_results)
            elif ablation_type == 'T+ASR':
                question = self.QUESTION_T_ASR.format(video_title=video_title, asr_results=asr_results)
            else:
                print(ablation_type)
                raise NotImplementedError

            inputs = dict(question=question)
            results[ablation_type] = inputs

        return results, anno


class LLMWorker(mp.Process):
    def __init__(self, 
                 task_queue, 
                 result_queue, 
                 llm_model_path,
                 device):
        super().__init__()
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.llm_model_path = llm_model_path
        self.device = device

    @staticmethod
    def build_model(llm_model_path, device):
        print('Init model in %s' % device)
        if 'Yi' in llm_model_path:
            use_fast = False
        else:
            use_fast = True
        pipe = pipeline("text-generation", 
                model=llm_model_path,
                device=device,
                torch_dtype=torch.float16,
                use_fast=use_fast)

        return pipe

    @staticmethod
    def inference(pipe, question):
        response = pipe(question, max_new_tokens=512, do_sample=True, return_full_text=False)
        pred_answer = response[0]['generated_text']
        return pred_answer

    def run(self):
        pipe = self.build_model(self.llm_model_path, self.device)

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
                        question = inputs['question']
                        with torch.no_grad():
                            pred_answer = self.inference(pipe, question)
                        type2results[ablation_type] = dict(
                            pred=pred_answer,
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
    # Data
    anno_file,
    video_root,
    output_dir,
    # Model
    llm_model_path,
    # Infer
    max_asr_length=1024,
    num_gpus=1,
    rank=0,
    world_size=1,
    num_workers=8
):
    return (anno_file, 
            video_root, 
            output_dir, 
            llm_model_path,
            max_asr_length,
            num_gpus, 
            rank, 
            world_size, 
            num_workers)


def get_model_name(llm_model_path):
    llm_name = os.path.basename(llm_model_path).lower()
    return llm_name


if __name__ == "__main__":
    mp.set_start_method('spawn')

    (anno_file, 
     video_root, 
     output_dir, 
     llm_model_path,
     max_asr_length,
     num_gpus, 
     rank, 
     world_size, 
     num_workers) = fire.Fire(parse_args)

    model_name = get_model_name(llm_model_path)
    dataset_name = os.path.splitext(os.path.basename(anno_file))[0]

    manager = mp.Manager()
    task_queue = manager.Queue()
    result_queue = manager.Queue()
    procs = []
    for gpu_id in range(num_gpus):
        device = f'cuda:{gpu_id}'
        proc = LLMWorker(task_queue, result_queue, llm_model_path, device)
        proc.start()
        procs.append(proc)

    processor = LLMProcessor(
        max_asr_length, 
        ablation_types=['T', 'ASR', 'T+ASR'])
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