import re
import os
from glob import glob
from tqdm import tqdm

import mmengine
import pandas as pd
from rouge import Rouge
import multiprocess as mp


def rouge_score(prediction, ground_truth, **kwargs):
    rouge = Rouge()
    try:
        scores = rouge.get_scores([prediction], [ground_truth], avg=True)
    except:
        return 0.0
    return scores["rouge-l"]["f"]


class EvaluatorWorker(mp.Process):
    def __init__(self, task_queue, result_queue, lang):
        super().__init__()
        assert lang in ['en', 'zh']
        self.task_queue = task_queue
        self.result_queue = result_queue

    def run(self):
        while True:
            task = self.task_queue.get()
            if task is None:
                break
            model_name, resfiles = task

            num_samples = 0
            num_success = 0
            type2id2score = {}
            for resfile in resfiles:
                results = mmengine.load(resfile)
                for BVid, type2results in results.items():
                    num_samples += 1
                    success = type2results.pop('success')
                    if success:
                        num_success += 1
                        for type, res in type2results.items():
                            score = rouge_score(res['pred'], res['gt']) * 100
                            id2score = type2id2score.get(type, {})
                            id2score[BVid] = score
                            type2id2score[type] = id2score

            eval_results = dict(
                type2id2score=type2id2score,
                num_samples=num_samples,
                success_rate=num_success / num_samples
            )

            self.result_queue.put((model_name, eval_results))



if __name__ == "__main__":
    output_dir = 'results'
    dataset_name = 'test'
    num_workers = 16

    # Load all files of inference results
    all_result_files = glob(os.path.join(output_dir, "*.json"))
    model_name2resfiles = {}
    for resultfile in all_result_files:
        model_name, rank = re.findall(f'{output_dir}/inference_results_{dataset_name}_(.+)_(\d+).json', resultfile)[0]
        resfiles = model_name2resfiles.get(model_name, [])
        resfiles.append(resultfile)
        model_name2resfiles[model_name] = resfiles

    # Lauch evaluator workers
    manager = mp.Manager()
    task_queue = manager.Queue()
    result_queue = manager.Queue()
    procs = [
        EvaluatorWorker(task_queue, result_queue, lang='en')
        for _ in range(num_workers)
    ]
    for proc in procs:
        proc.start()

    for model_name, resfiles in model_name2resfiles.items():
        task_queue.put((model_name, resfiles))
    for _ in range(num_workers):
        task_queue.put(None)

    # Fetch evaluation results
    model_name2eval_results = {}
    pbar = tqdm(total=len(model_name2resfiles))
    while len(model_name2eval_results) < len(model_name2resfiles):
        model_name, eval_results = result_queue.get()
        model_name2eval_results[model_name] = eval_results
        pbar.update(1)
    pbar.close()

    for proc in procs:
        proc.join()

    # Output evaluation results
    model_name_list = []
    V_list = []
    V_T_list = []
    V_ASR_list = []
    V_T_AST_list = []
    T_list = []
    ASR_list = []
    T_ASR_list = []

    for model_name, eval_results in model_name2eval_results.items():
        print('%d samples evaluated in %s' % (eval_results['num_samples'], model_name))
        assert eval_results['success_rate'] >= 0.9, 'success of %s < 0.9!' % model_name
        model_name_list.append(model_name)
        V_list.append(None)
        V_T_list.append(None)
        V_ASR_list.append(None)
        V_T_AST_list.append(None)
        T_list.append(None)
        ASR_list.append(None)
        T_ASR_list.append(None)
        for ablation_type, id2score in eval_results['type2id2score'].items():
            score = sum(id2score.values()) / len(id2score)
            if ablation_type == 'V':
                V_list[-1] = score
            elif ablation_type == 'V+T':
                V_T_list[-1] = score
            elif ablation_type == 'V+ASR':
                V_ASR_list[-1] = score
            elif ablation_type == 'V+T+ASR':
                V_T_AST_list[-1] = score
            elif ablation_type == 'T':
                T_list[-1] = score
            elif ablation_type == 'ASR':
                ASR_list[-1] = score
            elif ablation_type == 'T+ASR':
                T_ASR_list[-1] = score
            else:
                raise NotImplementedError

    df = pd.DataFrame({
        'model_name': model_name_list,
        'V': V_list,
        'V+T': V_T_list,
        'V+ASR': V_ASR_list,
        'V+T+ASR': V_T_AST_list,
        'T': T_list,
        'ASR': ASR_list,
        'T+ASR': T_ASR_list
    })
    df.to_csv(f'result_{dataset_name}.csv', index=False)