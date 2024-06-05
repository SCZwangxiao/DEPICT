import os
import re
import glob
import json
from tqdm import tqdm


def load_asr_results(asr_file):
    if asr_file is not None and os.path.exists(asr_file):
        asr_results = []
        asr_lines = open(asr_file, 'r').readlines()
        for idx in range(0, len(asr_lines) - 1, 4):
            if idx + 3 > len(asr_lines) - 1:
                break
            asr_results.append(dict(
                timestamp=asr_lines[idx+1].rstrip(),
                asr_result=asr_lines[idx+2].rstrip()
            ))
    else:
        asr_results = None
    return asr_results


def filter_summarization(summarization, comment_user):
    # @AI视频小助理 1
    res = re.findall(r'(.+)--(以上|本)内容由AI视频小助理生成，关注解锁AI助理.*', summarization, re.DOTALL)
    if len(res):
        return res[0][0]

    # @AI视频小助理 2
    res = re.findall(r'(.+)--(以上|本)(内容由模型基于视频内容|内容基于视频内容由模型)生成，仅供参考.*', summarization, re.DOTALL)
    if len(res):
        return res[0][0]

    # @AI视频小助理 3
    res = re.findall(r'--?本内容由.+大佬下凡召唤，热心市民@AI视频小助理闪现赶来\n\n?(.+)(实名羡慕up这溢出屏幕的才华.+)?', summarization, re.DOTALL)
    if len(res):
        return res[0][0]

    # @AI视频小助理 4
    if '@' not in summarization and 'AI' not in summarization and '模型' not in summarization:
        # No any AI symbols
        # print('No symbols', summarization)
        return summarization

    if comment_user == '有趣的程序员':
        results = ''
        for summ in summarization.split('\n\n'):
            if '内容总结' in summ or '时间线' in summ:
                results += summ + '\n'
        return results.rstrip()

    # @AI全文总结1
    res = re.findall(r'.*(课代表总结|课代表回顾|视频回顾|视频总结|课代表精华概览|概述).?:?\n?(.+)\n?--(本消息)?由.+ 召唤(成功)?.*', summarization, re.DOTALL)
    if len(res):
        return res[0][1]

    # @AI全文总结2
    res = re.findall(r'.*(课代表总结|课代表回顾|视频回顾|视频总结|课代表精华概览|概述).?:?\n?(.+)', summarization, re.DOTALL)
    if len(res):
        return res[0][1]

    # @课代表猫
    res = re.findall(r'(喵~)?(.+)\n本喵由.+ 召唤，.+，点赞关注即可领养一只同款AI喵哦！', summarization, re.DOTALL)
    if len(res):
        return res[0][1]

    # Unknown pattern
    print('Unknown pattern in:', summarization)
    print('Comment user', comment_user)
    # exit()

    return None


def get_video_file_index(video_cache_dir):
    BVid2video_file = {}
    BVid2asr_file = {}
    all_video_paths = glob.glob(os.path.join(video_cache_dir, '*/*/Videos', '*.mp4'))
    for video_path in all_video_paths:
        BVid = re.findall(f'{video_cache_dir}/(.+)/(.+)/Videos/.+.mp4', video_path)[0][0]
        BVid2video_file[BVid] = video_path
    all_asr_paths = glob.glob(os.path.join(video_cache_dir, '*/*/Videos', '*.srt'))
    for asr_path in all_asr_paths:
        BVid = re.findall(f'{video_cache_dir}/(.+)/(.+)/Videos/.+.srt', asr_path)[0][0]
        BVid2asr_file[BVid] = asr_path
    return BVid2video_file, BVid2asr_file


if __name__ == "__main__":
    crawl_results = "crawl_results/summary_all_current_11642.json"
    video_cache_dir = "video_cache"
    dataset_dir = "data/bilibili_dev1w"
    anno_filename = 'dev1w.json'

    os.makedirs(os.path.join(dataset_dir, 'annotations'), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, 'videos'), exist_ok=True)

    BVid2video_file, BVid2asr_file = get_video_file_index(video_cache_dir)

    annotations = []
    data = json.load(open(crawl_results, 'r'))
    wo_asr_cnt = 0
    wo_video_cnt = 0
    wo_sum_tpl = 0
    for d in tqdm(data):
        BVid = d['video_BV']
        video_type = d['video_type']
        video_title = d['video_title']
        video_duration = d['video_duration']
        video_owner = d['video_owner']
        pv = d['video_view']
        like = d['video_like']
        coin = d['video_coin']
        summarization = d['comment'][0]['comment_content']

        summarization = filter_summarization(summarization, d['comment'][0]['comment_user'])
        if summarization is None: # filter failure
            wo_sum_tpl += 1
            continue
        assert type(summarization) == str, summarization

        if BVid in BVid2video_file:
            raw_video_path = BVid2video_file[BVid]
            raw_asr_path = BVid2asr_file.get(BVid, None)
            asr_results = load_asr_results(raw_asr_path)
            if asr_results is None:
                wo_asr_cnt += 1
            annotations.append(dict(
                BVid=BVid,
                video_type_zh=video_type,
                video_title_zh=video_title,
                video_duration=video_duration,
                video_owner=video_owner,
                pv=pv,
                like=like,
                coin=coin,
                summarization_zh=summarization,
                asr_results_zh=asr_results
            ))
            # copy and rename file
            tgt_video_dir = os.path.join(dataset_dir, 'videos')
            tgt_video_path_old_name = os.path.join(tgt_video_dir, raw_video_path.split('/')[-1])
            tgt_video_path = os.path.join(tgt_video_dir, f'{BVid}.mp4')
            if not os.path.exists(tgt_video_path):
                os.system(f'cp "{raw_video_path}" {tgt_video_dir}')
                os.system(f'mv "{tgt_video_path_old_name}" {tgt_video_path}')
        else:
            wo_video_cnt += 1

    tgt_anno_file = os.path.join(dataset_dir, 'annotations', anno_filename)
    json.dump(annotations, open(tgt_anno_file, 'w'), indent=2)

    print('%.1f%% samples have unknown summarization template' % (wo_sum_tpl / len(data) * 100))
    print('%.1f%% samples have no video' % (wo_video_cnt / len(data) * 100))
    print('%.1f%% valid samples have no asr results' % (wo_asr_cnt / len(annotations) * 100))