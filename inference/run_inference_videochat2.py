import sys
sys.path.append('/home/wangxiao24/dev_videochat/tools/data/bilibili/Ask-Anything/video_chat2')
import os
import io
import time
import copy
from tqdm import tqdm
import multiprocess as mp
# from IPython.display import Video, HTML

import cv2
import fire
import torch
import mmengine
import numpy as np
from decord import VideoReader, cpu
from torch.utils.data import DataLoader
from transformers import StoppingCriteria, StoppingCriteriaList
import numpy as np
from PIL import Image
from decord import VideoReader, cpu
from dataset.video_transforms import (
    GroupNormalize, GroupScale, GroupCenterCrop, 
    Stack, ToTorchFormatTensor
)
import torchvision.transforms as T
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import matplotlib.pyplot as plt
from peft import get_peft_model, LoraConfig, TaskType

from utils.config import Config
from models import VideoChat2_it_vicuna, VideoChat2_it_mistral
from utils.easydict import EasyDict


def get_prompt(conv):
    ret = conv.system + conv.sep
    for role, message in conv.messages:
        if message:
            ret += role + ": " + message + conv.sep
        else:
            ret += role + ":"
    return ret


def get_prompt2(conv):
    ret = conv.system + conv.sep
    count = 0
    for role, message in conv.messages:
        count += 1
        if count == len(conv.messages):
            ret += role + ": " + message
        else:
            if message:
                ret += role + ": " + message + conv.sep
            else:
                ret += role + ":"
    return ret


def get_context_emb(conv, model, img_list, answer_prompt=None, print_res=False):
    if answer_prompt:
        prompt = get_prompt2(conv)
    else:
        prompt = get_prompt(conv)
    if print_res:
        print(prompt)
    if '<VideoHere>' in prompt:
        prompt_segs = prompt.split('<VideoHere>')
    else:
        prompt_segs = prompt.split('<ImageHere>')
    assert len(prompt_segs) == len(img_list) + 1, "Unmatched numbers of image placeholders and images."
    with torch.no_grad():
        if hasattr(model, 'llama_model'):
            seg_tokens = [
                model.llama_tokenizer(
                    seg, return_tensors="pt", add_special_tokens=i == 0).to(model.device).input_ids
                # only add bos to the first seg
                for i, seg in enumerate(prompt_segs)
            ]
            seg_embs = [model.llama_model.base_model.model.model.embed_tokens(seg_t) for seg_t in seg_tokens]
        elif hasattr(model, 'mistral_model'):
            seg_tokens = [
                model.mistral_tokenizer(
                    seg, return_tensors="pt", add_special_tokens=i == 0).to(model.device).input_ids
                # only add bos to the first seg
                for i, seg in enumerate(prompt_segs)
            ]
            seg_embs = [model.mistral_model.base_model.model.model.embed_tokens(seg_t) for seg_t in seg_tokens]

    mixed_embs = [emb for pair in zip(seg_embs[:-1], img_list) for emb in pair] + [seg_embs[-1]]
    mixed_embs = torch.cat(mixed_embs, dim=1)
    return mixed_embs


def ask(text, conv):
    conv.messages.append([conv.roles[0], text + '\n'])


class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True
        return False

def answer(conv, model, img_list, do_sample=True, max_new_tokens=200, num_beams=1, min_length=1, top_p=0.9,
               repetition_penalty=1.0, length_penalty=1, temperature=1.0, answer_prompt=None, print_res=False):
    stop_words_ids = [
        torch.tensor([835]).to(model.device),
        torch.tensor([2277, 29937]).to(model.device)]  # '###' can be encoded in two different ways.
    stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
    
    conv.messages.append([conv.roles[1], answer_prompt])
    embs = get_context_emb(conv, model, img_list, answer_prompt=answer_prompt, print_res=print_res)
    with torch.no_grad():
        if hasattr(model, 'llama_model'):
            outputs = model.llama_model.generate(
                inputs_embeds=embs,
                max_new_tokens=max_new_tokens,
                stopping_criteria=stopping_criteria,
                num_beams=num_beams,
                do_sample=do_sample,
                min_length=min_length,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                temperature=temperature,
            )
        elif hasattr(model, 'mistral_model'):
            outputs = model.mistral_model.generate(
                inputs_embeds=embs,
                max_new_tokens=max_new_tokens,
                stopping_criteria=stopping_criteria,
                num_beams=num_beams,
                do_sample=do_sample,
                min_length=min_length,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                temperature=temperature,
            )
    output_token = outputs[0]
    if output_token[0] == 0:  # the model might output a unknow token <unk> at the beginning. remove it
            output_token = output_token[1:]
    if output_token[0] == 1:  # some users find that there is a start token <s> at the beginning. remove it
            output_token = output_token[1:]
    if hasattr(model, 'llama_model'):
        output_text = model.llama_tokenizer.decode(output_token, add_special_tokens=False)
    elif hasattr(model, 'mistral_model'):
        output_text = model.mistral_tokenizer.decode(output_token, add_special_tokens=False)
    output_text = output_text.split('###')[0]  # remove the stop sign '###'
    output_text = output_text.split('Assistant:')[-1].strip()
    conv.messages[-1][1] = output_text
    return output_text, output_token.cpu().numpy()


def get_index(num_frames, num_segments):
    seg_size = float(num_frames - 1) / num_segments
    start = int(seg_size / 2)
    offsets = np.array([
        start + int(np.round(seg_size * idx)) for idx in range(num_segments)
    ])
    return offsets


def load_video(video_path, num_segments=8, return_msg=False, resolution=224):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    num_frames = len(vr)
    frame_indices = get_index(num_frames, num_segments)

    # transform
    crop_size = resolution
    scale_size = resolution
    input_mean = [0.48145466, 0.4578275, 0.40821073]
    input_std = [0.26862954, 0.26130258, 0.27577711]

    transform = T.Compose([
        GroupScale(int(scale_size), interpolation=InterpolationMode.BICUBIC),
        GroupCenterCrop(crop_size),
        Stack(),
        ToTorchFormatTensor(),
        GroupNormalize(input_mean, input_std) 
    ])

    images_group = list()
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].numpy())
        images_group.append(img)
    torch_imgs = transform(images_group)
    if return_msg:
        fps = float(vr.get_avg_fps())
        sec = ", ".join([str(round(f / fps, 1)) for f in frame_indices])
        # " " should be added in the start and end
        msg = f"The video contains {len(frame_indices)} frames sampled at {sec} seconds."
        return torch_imgs, msg
    else:
        return torch_imgs


def get_sinusoid_encoding_table(n_position=784, d_hid=1024, cur_frame=8, ckpt_num_frame=4, pre_n_position=784): 
    ''' Sinusoid position encoding table ''' 
    # TODO: make it with torch instead of numpy 
    def get_position_angle_vec(position): 
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)] 

    # generate checkpoint position embedding
    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(pre_n_position)]) 
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2]) # dim 2i 
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2]) # dim 2i+1 
    sinusoid_table = torch.tensor(sinusoid_table, dtype=torch.float, requires_grad=False).unsqueeze(0)

    print(f"n_position: {n_position}")
    print(f"pre_n_position: {pre_n_position}")

    if n_position != pre_n_position:
        T = ckpt_num_frame # checkpoint frame
        P = 14 # checkpoint size
        C = d_hid
        new_P = int((n_position // cur_frame) ** 0.5) # testing size
        if new_P != 14:
            print(f'Pretraining uses 14x14, but current version is {new_P}x{new_P}')
            print(f'Interpolate the position embedding')
            sinusoid_table = sinusoid_table.reshape(-1, T, P, P, C)
            sinusoid_table = sinusoid_table.reshape(-1, P, P, C).permute(0, 3, 1, 2)
            sinusoid_table = torch.nn.functional.interpolate(
                sinusoid_table, size=(new_P, new_P), mode='bicubic', align_corners=False)
            # BT, C, H, W -> BT, H, W, C ->  B, T, H, W, C
            sinusoid_table = sinusoid_table.permute(0, 2, 3, 1).reshape(-1, T, new_P, new_P, C)
            sinusoid_table = sinusoid_table.flatten(1, 3)  # B, THW, C

    if cur_frame != ckpt_num_frame:
        print(f'Pretraining uses 4 frames, but current frame is {cur_frame}')
        print(f'Interpolate the position embedding')
        T = ckpt_num_frame # checkpoint frame
        new_T = cur_frame # testing frame
        # interpolate
        P = int((n_position // cur_frame) ** 0.5) # testing size
        C = d_hid
        sinusoid_table = sinusoid_table.reshape(-1, T, P, P, C)
        sinusoid_table = sinusoid_table.permute(0, 2, 3, 4, 1).reshape(-1, C, T)  # BHW, C, T
        sinusoid_table = torch.nn.functional.interpolate(sinusoid_table, size=new_T, mode='linear')
        sinusoid_table = sinusoid_table.reshape(1, P, P, C, new_T).permute(0, 4, 1, 2, 3) # B, T, H, W, C
        sinusoid_table = sinusoid_table.flatten(1, 3)  # B, THW, C

    return sinusoid_table


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
    QUESTION_V = '''Please summarize the content of the video based on given information.'''

    QUESTION_V_T = '''Title: {video_title}. Please summarize the content of the video based on given information.'''

    QUESTION_V_ASR = '''Subtitles: {asr_results}. Please summarize the content of the video based on given information.'''

    QUESTION_V_T_ASR = '''Subtitles: {asr_results}. Title: {video_title}. Please summarize the content of the video based on given information.'''

    def __init__(self, 
                 num_frames, 
                 resolution=224,
                 max_asr_length=1024, 
                 ablation_types=['V', 'ASR', 'V+T', 'V+ASR', 'V+T+ASR']) -> None:
        self.num_frames = num_frames
        self.resolution = resolution
        self.max_asr_length = max_asr_length
        self.ablation_types = ablation_types

    def __call__(self, anno):
        video_path = anno['video_path']
        video_title = anno['video_title']
        asr_results = anno['asr_results']
        asr_results = asr_results if asr_results else ""
        asr_results = asr_results[:self.max_asr_length]

        try:
            results = dict()
            for ablation_type in self.ablation_types:
                if ablation_type == 'V':
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
                # Load video
                vid, msg = load_video(video_path, num_segments=self.num_frames, return_msg=True, resolution=self.resolution)
                # print(msg)

                # The model expects inputs of shape: T x C x H x W
                TC, H, W = vid.shape
                video = vid.reshape(1, TC//3, 3, H, W)

                inputs = dict(question=question)
                results[ablation_type] = inputs
            results['video'] = video
        except:
            results = None

        return results, anno


class VideoLLMWorker(mp.Process):
    def __init__(self, 
                 task_queue, 
                 result_queue, 
                 model_cfg,
                 resolution,
                 num_frames,
                 device):
        super().__init__()
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.model_cfg = model_cfg
        self.resolution = resolution
        self.num_frames = num_frames
        self.device = device

    @staticmethod
    def build_model(cfg, device, resolution, num_frames):
        print('Init model in %s' % device)
        # Load ckpt from stage 2
        cfg.model.vision_encoder.num_frames = 4 # constant for init
        if 'llama_model_path' in cfg['model']:
            llm_name = os.path.basename(cfg['model']['llama_model_path']).lower()
        elif 'mistral_model_path' in cfg['model']:
            llm_name = os.path.basename(cfg['model']['mistral_model_path']).lower()
        else:
            raise NotImplementedError

        if 'vicuna' in llm_name:
            model = VideoChat2_it_vicuna(config=cfg.model)
        elif 'mistral' in llm_name:
            model = VideoChat2_it_mistral(config=cfg.model)
        else:
            raise NotImplementedError

        # Add lora to run stage3 model
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, inference_mode=False, 
            r=16, lora_alpha=32, lora_dropout=0.
        )
        if 'llama_model_path' in cfg['model']:
            model.llama_model = get_peft_model(model.llama_model, peft_config)
        elif 'mistral_model_path' in cfg['model']:
            model.mistral_model = get_peft_model(model.mistral_model, peft_config)

        state_dict = torch.load(cfg['model']['videochat2_model_path'], "cpu")

        if 'model' in state_dict.keys():
            msg = model.load_state_dict(state_dict['model'], strict=False)
        else:
            msg = model.load_state_dict(state_dict, strict=False)
        print(msg)

        # Add new position embedding
        new_pos_emb = get_sinusoid_encoding_table(
            n_position=(resolution // 16)**2 * num_frames, 
            cur_frame=num_frames)
        model.vision_encoder.encoder.pos_embed = new_pos_emb

        model = model.eval().to(device)
        return model

    @staticmethod
    def prepare_context(model, video, device):
        video = video.to(device)
        img_list = []
        with torch.no_grad():
            image_emb, _ = model.encode_img(video, "Watch the video and answer the question.")
        img_list.append(image_emb)
        return img_list

    @staticmethod
    def inference(model, question, img_list):
        chat = EasyDict({
            "system": "",
            "roles": ("Human", "Assistant"),
            "messages": [],
            "sep": "###"
        })

        chat.messages.append([chat.roles[0], f"<Video><VideoHere></Video>\n"])
        ask(question, chat)

        pred_answer = answer(conv=chat, model=model, do_sample=False, img_list=img_list, max_new_tokens=256, print_res=False)[0]
        # print(pred_answer)
        return pred_answer

    def run(self):
        model = self.build_model(self.model_cfg, self.device, self.resolution, self.num_frames)

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
                video = results.pop('video')
                img_list = self.prepare_context(model, video, self.device)
                try:
                    for ablation_type, inputs in results.items():
                        question = inputs['question']
                        with torch.no_grad():
                            pred_answer = self.inference(model, question, img_list)
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
    config_file,
    vit_blip_model_path,
    llm_model_path,
    videochat2_model_path,
    # Infer
    ablation_types=['V', 'V+T', 'V+ASR', 'V+T+ASR'],
    model_suffix='',
    num_frames=8,
    num_gpus=1,
    rank=0,
    world_size=1,
    num_workers=8
):
    return (anno_file, 
            video_root, 
            output_dir, 
            config_file,
            vit_blip_model_path,
            llm_model_path,
            videochat2_model_path,
            ablation_types,
            model_suffix,
            num_frames,
            num_gpus, 
            rank, 
            world_size, 
            num_workers)


def get_model_name(llm_model_path):
    llm_name = os.path.basename(llm_model_path).lower()
    return f'VideoChat2_{llm_name}'


def get_updated_config(config_file, vit_blip_model_path, llm_model_path, videochat2_model_path):
    cfg = Config.from_file(config_file)
    cfg['model']['vit_blip_model_path'] = vit_blip_model_path
    if 'llama_model_path' in cfg['model']:
        cfg['model']['llama_model_path'] = llm_model_path
    elif 'mistral_model_path' in cfg['model']:
        cfg['model']['mistral_model_path'] = llm_model_path
    else:
        raise NotImplementedError
    cfg['model']['videochat2_model_path'] = videochat2_model_path
    return cfg


if __name__ == "__main__":
    mp.set_start_method('spawn')

    (anno_file, 
     video_root, 
     output_dir, 
     config_file,
     vit_blip_model_path,
     llm_model_path,
     videochat2_model_path,
     ablation_types,
     model_suffix,
     num_frames,
     num_gpus, 
     rank, 
     world_size, 
     num_workers) = fire.Fire(parse_args)

    resolution = 224
    max_asr_length = 1024

    model_name = get_model_name(llm_model_path)
    if model_suffix and len(model_suffix):
        model_name = model_name + '_' + model_suffix
    dataset_name = os.path.splitext(os.path.basename(anno_file))[0]
    cfg = get_updated_config(config_file, vit_blip_model_path, llm_model_path, videochat2_model_path)

    manager = mp.Manager()
    task_queue = manager.Queue()
    result_queue = manager.Queue()
    procs = []
    for gpu_id in range(num_gpus):
        device = f'cuda:{gpu_id}'
        proc = VideoLLMWorker(task_queue, result_queue, cfg, resolution, num_frames, device)
        proc.start()
        procs.append(proc)

    processor = VideoLLMProcessor(
        num_frames, 
        resolution,
        max_asr_length, 
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