import os
from tqdm import tqdm
from PIL import Image
import multiprocess as mp

import cv2
import fire
import torch
import mmengine
from torch.utils.data import DataLoader
from MovieChat.processors.video_processor import AlproVideoEvalProcessor
from MovieChat.models.chat_model import Chat
from MovieChat.models.moviechat import MovieChat


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


class MovieChatProcessor:
    QUESTION_V = '''Please summarize the content of the video based on given information.'''

    QUESTION_V_T = '''Title: {video_title}. Please summarize the content of the video based on given information.'''

    QUESTION_V_ASR = '''Subtitles: {asr_results}. Please summarize the content of the video based on given information.'''

    QUESTION_V_T_ASR = '''Subtitles: {asr_results}. Title: {video_title}. Please summarize the content of the video based on given information.'''

    def __init__(self, max_asr_length=1024, ablation_types=['V', 'ASR', 'V+T', 'V+ASR', 'V+T+ASR']) -> None:
        self.max_asr_length = max_asr_length
        self.ablation_types = ablation_types

    def __call__(self, anno):
        video_title = anno['video_title']
        asr_results = anno['asr_results']
        asr_results = asr_results if asr_results else ""
        asr_results = asr_results[:self.max_asr_length]

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
            inputs = dict(question=question)
            results[ablation_type] = inputs

        return results, anno


class VideoLLMWorker(mp.Process):
    def __init__(self, task_queue, result_queue, device, n_samples, fragment_video_dir):
        super().__init__()
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.device = device
        self.n_samples = n_samples
        self.fragment_video_dir = fragment_video_dir

    @staticmethod
    def build_model(device, n_samples):
        print('Initializing Chat')
        moviechat_model = MovieChat.from_config(device=device).to(device)
        vis_processor_cfg = {'name': 'alpro_video_eval', 'n_frms': 8, 'image_size': 224}
        frame_processor = AlproVideoEvalProcessor.from_config(vis_processor_cfg)
        chat = Chat(moviechat_model, frame_processor, device=device)
        chat.n_samples = n_samples
        print('Initialization Finished')
        return chat

    @staticmethod
    def prepare_context(chat, fragment_video_dir, video_path, device):
        chat.model.long_memory_buffer = [] # Clear long memory cache
        fragment_video_path = os.path.join(fragment_video_dir, os.path.basename(video_path))
        os.system(f'cp {video_path} {fragment_video_path}')

        middle_video = False # True->Breakpoint mode, False->Global mode
        cur_min = 0 # Change it when Breakpoint mode
        cur_sec = 0 # Change it when Breakpoint mode

        cap = cv2.VideoCapture(video_path)
        cur_fps = cap.get(cv2.CAP_PROP_FPS)
        cap.set(cv2.CAP_PROP_POS_FRAMES, cur_fps)
        ret, frame = cap.read()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        # import pdb;pdb.set_trace()
        image = chat.image_vis_processor(pil_image).unsqueeze(0).unsqueeze(2).half().to(device)
        cur_image = chat.model.encode_image(image)

        img_list = []
        msg = chat.upload_video_without_audio(
            video_path=video_path, 
            fragment_video_path=fragment_video_path,
            cur_min=cur_min, 
            cur_sec=cur_sec, 
            cur_image=cur_image, 
            img_list=img_list, 
            middle_video=middle_video,
            question=None
        )
        return msg, img_list

    @staticmethod
    def inference(chat, msg, img_list, question):
        answer = chat.answer(
        img_list=img_list,
        input_text=question,
        msg=msg,
        num_beams=1,
        temperature=1.0,
        max_new_tokens=300,
        max_length=2000)[0]
        return answer

    def run(self):
        chat = self.build_model(self.device, self.n_samples)

        while True:
            task = self.task_queue.get()
            if task is None:
                break

            task_id, sample = task

            results, anno = sample
            BVid = anno['BVid']
            video_path = anno['video_path']
            summarization = anno['summarization']
            type2results = dict(BVid=BVid, success=True)
            try:
                msg, img_list = self.prepare_context(chat, self.fragment_video_dir, video_path, self.device)
                for ablation_type, inputs in results.items():
                    question = inputs['question']
                    with torch.no_grad():
                        answer = self.inference(chat, msg, img_list, question)
                    type2results[ablation_type] = dict(
                        pred=answer,
                        gt=summarization,
                    )
            except Exception as e:
                msg = str(e)
                type2results['success'] = False
                type2results['message'] = msg

            self.result_queue.put((task_id, type2results))


def parse_args(
    anno_file,
    video_root,
    output_dir,
    n_samples=4,
    ablation_types=['V', 'V+T', 'V+ASR', 'V+T+ASR'],
    model_suffix='',
    num_gpus=1,
    rank=0,
    world_size=1,
):
    return anno_file, video_root, output_dir, n_samples, ablation_types, model_suffix, num_gpus, rank, world_size


if __name__ == "__main__":
    mp.set_start_method('spawn')

    anno_file, video_root, output_dir, n_samples, ablation_types, model_suffix, num_gpus, rank, world_size = fire.Fire(parse_args)

    num_workers = 4
    max_asr_length = 1024
    model_name = 'moviechat'
    if model_suffix and len(model_suffix):
        model_name = model_name + '_' + model_suffix
    dataset_name = os.path.splitext(os.path.basename(anno_file))[0]
    fragment_video_dir = '.cache'

    os.makedirs(fragment_video_dir, exist_ok=True)

    manager = mp.Manager()
    task_queue = manager.Queue()
    result_queue = manager.Queue()
    procs = []
    for gpu_id in range(num_gpus):
        device = f'cuda:{gpu_id}'
        proc = VideoLLMWorker(task_queue, result_queue, device, n_samples, fragment_video_dir)
        proc.start()
        procs.append(proc)

    processor = MovieChatProcessor(
        max_asr_length=max_asr_length, 
        ablation_types=ablation_types)
    dataset = BilibiliDataset(anno_file, video_root, processor, rank, world_size)
    dataloader = DataLoader(dataset=dataset, num_workers=num_workers, batch_size=None)

    for task_id, sample in enumerate(dataloader):
        task_queue.put((task_id, sample))
    for _ in range(num_gpus):
        task_queue.put(None)

    num_tasks = len(dataloader)
    finished_tasks = 0
    pbar = tqdm(total=num_tasks)
    inference_results = {}
    while finished_tasks < num_tasks:
        task_id, type2results = result_queue.get()
        BVid = type2results.pop('BVid')
        inference_results[BVid] = type2results
        finished_tasks += 1
        pbar.update(1)
    pbar.close()

    for proc in procs:
        proc.join()

    mmengine.dump(inference_results, 
                  os.path.join(output_dir, f'inference_results_{dataset_name}_{model_name}_{rank}.json'),
                  indent=2)