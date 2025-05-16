import os.path
import sys
import json
import numpy as np
import argparse
sys.path.append('..')
from easyeditor import (
    FTHyperParams, 
    IKEHyperParams, 
    KNHyperParams, 
    MEMITHyperParams, 
    ROMEHyperParams, 
    LoRAHyperParams,
    MENDHyperParams,
    SERACHparams,
    AlphaEditHyperParams,
    GraceHyperParams,
    WISEHyperParams,
    PMETHyperParams,
    )
from easyeditor import BaseEditor
from easyeditor.models.ike import encode_ike_facts
from sentence_transformers import SentenceTransformer
from easyeditor import KnowEditDataset
import torch
import gc
import nltk
nltk.data.path.append('/fs-computility/ai-shen/shared/share/nltk_data')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--editing_method', required=True, type=str)
    parser.add_argument('--hparams_dir', required=True, type=str)
    parser.add_argument('--data_dir', required=True, type=str)
    parser.add_argument('--output_dir', default='./outputs', type=str)
    parser.add_argument('--ds_size', default=None, type=int)
    parser.add_argument('--start_index', default=None, type=int)
    parser.add_argument('--end_index', default=None, type=int)
    parser.add_argument('--datatype', default=None,type=str)
    parser.add_argument('--sequential_edit', action='store_true')
    parser.add_argument('--train_data_path', type=str)
    parser.add_argument('--evaluation_type', default='LLM-judge', type=str)
    parser.add_argument('--api_key', default="dummy", type=str)

    args = parser.parse_args()

    if args.editing_method == 'FT':
        editing_hparams = FTHyperParams
    elif args.editing_method == 'IKE':
        editing_hparams = IKEHyperParams
    elif args.editing_method == 'ICE':
        editing_hparams = IKEHyperParams
    elif args.editing_method == 'KN':
        editing_hparams = KNHyperParams
    elif args.editing_method == 'MEMIT':
        editing_hparams = MEMITHyperParams
    elif args.editing_method == 'ROME':
        editing_hparams = ROMEHyperParams
    elif args.editing_method == 'LoRA':
        editing_hparams = LoRAHyperParams
    elif args.editing_method == 'SERAC':
        editing_hparams = SERACHparams
    elif args.editing_method == 'MEND':
        editing_hparams = MENDHyperParams
    elif args.editing_method == 'AlphaEdit':
        editing_hparams = AlphaEditHyperParams
    elif args.editing_method == 'GRACE':
        editing_hparams = GraceHyperParams
    elif args.editing_method == 'WISE':
        editing_hparams = WISEHyperParams
    elif args.editing_method == 'PMET':
        editing_hparams = PMETHyperParams
    else:
        raise NotImplementedError

    datas = KnowEditDataset(args.data_dir, size=args.ds_size, start_index=args.start_index, end_index=args.end_index)
    
    hparams = editing_hparams.from_hparams(args.hparams_dir)
    # specify real-world evaluation and provide the api key for LLM-as-a-Judge
    
    hparams.evaluation_type = args.evaluation_type
    hparams.api_key = args.api_key
    
    #editor = BaseEditor.from_hparams(hparams)
    
    # def get_model_params(model):
    #     return {name: param.detach().cpu().numpy() for name, param in model.named_parameters()}
    # def compare_params(previous_params, current_params):
    #     # 比较两个参数字典中的参数
    #     for name in previous_params:
    #         if name in current_params:
    #             if not np.array_equal(previous_params[name], current_params[name]):
    #                 return True  # 如果发现任何参数不同，则返回True
    #     return False
    # previous_params = get_model_params(editor.model)

    # previous_model = editor.model
    if not args.sequential_edit: # single edit
        all_metrics = []  # 其中每一个也是列表
        for index,event in enumerate(datas):  # 每一个event有多个prompt进行顺序编辑
            # current_params = get_model_params(editor.model)

            # if compare_params(previous_params, current_params):  # 如果模型的参数发生了变化
            #     editor.model=previous_model
            #     current_params = get_model_params(editor.model)
            #     if compare_params(previous_params, current_params):  # 如果模型的参数发生了变化
            #         print("还是变化了")
            if 'editor' in locals():
                del editor
                torch.cuda.empty_cache()
                gc.collect()
            editor = BaseEditor.from_hparams(hparams)
            prompts = event["prompts"]
            target_new = event["target_new"]
            subjects = event["subjects"]

            local_fact_question = event["local_fact_question"] # 再改为一个列表套列表
            local_fact_answer = event["local_fact_answer"] # 再改为一个列表套列表

            fact_question = event["fact_question"]  # 再改为一个列表套列表
            fact_answer = event["fact_answer"] # 再改为一个列表套列表


            locality_prompts = [None]*len(prompts)
            locality_prompts[-1] = local_fact_question if len(local_fact_question)>0 else None

            locality_ans = [None]*len(prompts)
            locality_ans[-1] = local_fact_answer if len(local_fact_answer)>0 else None 

            assert len(prompts) == len(locality_prompts) == len(locality_ans) 
            
            locality_inputs = {
                'Locality_Fact':{
                    'prompt': locality_prompts,
                    'ground_truth': locality_ans
                }
            }

            portability_prompts = [None]*len(prompts)
            portability_prompts[-1] = fact_question if len(fact_question)>0 else None

            portability_ans = [None]*len(prompts)
            portability_ans[-1] = fact_answer if len(fact_answer)>0 else None

            assert len(prompts) == len(portability_prompts) == len(portability_ans) 
            
            portability_inputs = {
                'Portability_Fact':{
                    'prompt': portability_prompts,
                    'ground_truth': portability_ans
                }
            }
            if args.editing_method=="WISE":
                def extract_fact_local_qas(data_item):
                    local_qas = data_item.get("fact", {}).get("local_qas", [])
                    result = []
                    for qa in local_qas:
                        question = qa.get("question", "")
                        answer = qa.get("answer", {}).get("name", "")
                        result.append(f"{question} {answer}")
                    return result
                loc_data = json.load(open('/fs-computility/ai-shen/songxin/EasyEdit_for_concept/examples/dataset/ELKEN/train_filtered.json', 'r', encoding='utf-8'))
                all_results = []
                for item in loc_data:
                    all_results.extend(extract_fact_local_qas(item))
                loc_prompts = all_results[:len(prompts)]
                metrics, edited_model, _ = editor.edit(
                    prompts=prompts,
                    target_new=target_new,
                    subject=subjects,
                    locality_inputs=locality_inputs,
                    portability_inputs=portability_inputs,
                    loc_prompts=loc_prompts,
                    sequential_edit=True,
                    test_generation=False,
                )
                all_metrics.append(metrics)
            else:
                metrics, edited_model, _ = editor.edit(
                    prompts=prompts,
                    target_new=target_new,
                    subject=subjects,
                    locality_inputs=locality_inputs,
                    portability_inputs=portability_inputs,
                    sequential_edit=True,
                    test_generation=False,
                )
                all_metrics.append(metrics)

        os.makedirs(args.output_dir, exist_ok=True)
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        output_file = os.path.join(
            args.output_dir,
            f'{args.datatype}_{hparams.model_name.split("/")[-1]}_{args.editing_method}_N={args.ds_size}_Sequential={args.sequential_edit}_{timestamp}.json'
        )

        with open(output_file, 'w', encoding='utf-8') as f:
            for batch in all_metrics:
                json.dump(batch, f, ensure_ascii=False)
                f.write('\n')
        print("See results at: ", output_file)

        def summary_metrics_for_LLM_judge(all_metrics):

            mean_metrics = dict()
            for eval in ["pre", "post"]:
                if eval not in all_metrics[0]:
                    continue
                mean_metrics[eval] = dict()
                for key in ["rewrite_acc", "rephrase_acc", "rewrite_contain_acc", "rephrase_contain_acc", "fluency", 'rewrite_ppl', 'ood_acc']:
                    if key in all_metrics[0][eval].keys():
                        if key == "fluency":  # 对 fluency 字典值进行均值计算
                            fluency_values = [
                                metric[eval][key]['ngram_entropy'] if 'ngram_entropy' in metric[eval][key] else np.nan
                                for metric in all_metrics if key in metric[eval]
                            ]
                            fluency_values = [val for val in fluency_values if not np.isnan(val)]  # 移除无效值
                            mean_metrics[eval][key] = np.mean(fluency_values) if fluency_values else np.nan
                        else:  # 对其他评估指标进行均值计算
                            mean_metrics[eval][key] = np.mean(
                                [np.mean(metric[eval][key]) for metric in all_metrics if key in metric[eval]]
                            )
                for key in ["locality", "portability"]:
                    if key in all_metrics[-1][eval].keys() and all_metrics[-1][eval][key] != {}:
                        mean_metrics[eval][key] = dict()
                        sub_keys = set()
                        for metric in all_metrics:
                            if key in metric[eval]:
                                sub_keys.update(metric[eval][key].keys())

                        for sub_key in sub_keys:
                            if not sub_key.endswith("_acc"):
                                continue

                            per_sample_means = []
                            for metric in all_metrics:
                                if key in metric[eval] and sub_key in metric[eval][key]:
                                    vals = metric[eval][key][sub_key]
                                    if isinstance(vals, list):
                                        if len(vals) == 0:
                                            continue
                                        # 若是 list of list，flatten
                                        if isinstance(vals[0], list):
                                            vals = [v for v in vals if isinstance(v, list) and len(v) > 0]
                                            vals = [item for sublist in vals for item in sublist]
                                        vals = [v for v in vals if isinstance(v, (float, int))]
                                        if len(vals) > 0:
                                            per_sample_means.append(np.mean(vals))

                            if len(per_sample_means) > 0:
                                mean_metrics[eval][key][sub_key] = np.mean(per_sample_means)


            return mean_metrics

            # 读取json数据
        all_metrics=[]
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                metric = summary_metrics_for_LLM_judge(data)
                all_metrics.append(metric)

        # 初始化累加值和计数器
        total_rewrite_acc = 0
        total_rewrite_contain_acc = 0
        count_rewrite_acc = 0
        count_rewrite_contain_acc = 0

        total_Locality_Fact_acc = 0
        total_Locality_Fact_contain_acc = 0
        count_Locality_Fact_acc = 0
        count_Locality_Fact_contain_acc = 0

        total_Portability_Fact_acc = 0
        total_Portability_Fact_contain_acc = 0
        count_Portability_Fact_acc = 0
        count_Portability_Fact_contain_acc = 0

        for record in all_metrics:
            post = record['post']

            if 'rewrite_acc' in post:
                total_rewrite_acc += post['rewrite_acc']
                count_rewrite_acc += 1
            if 'rewrite_contain_acc' in post:
                total_rewrite_contain_acc += post['rewrite_contain_acc']
                count_rewrite_contain_acc += 1

            locality = post.get('locality', {})
            if 'Locality_Fact_acc' in locality:
                total_Locality_Fact_acc += locality['Locality_Fact_acc']
                count_Locality_Fact_acc += 1
            if 'Locality_Fact_contain_acc' in locality:
                total_Locality_Fact_contain_acc += locality['Locality_Fact_contain_acc']
                count_Locality_Fact_contain_acc += 1

            portability = post.get('portability', {})
            if 'Portability_Fact_acc' in portability:
                total_Portability_Fact_acc += portability['Portability_Fact_acc']
                count_Portability_Fact_acc += 1
            if 'Portability_Fact_contain_acc' in portability:
                total_Portability_Fact_contain_acc += portability['Portability_Fact_contain_acc']
                count_Portability_Fact_contain_acc += 1

        # 安全计算平均值，避免除以零
        def safe_div(a, b):
            return a / b if b != 0 else 0

        print(f"Average rewrite_acc: {safe_div(total_rewrite_acc, count_rewrite_acc)}")
        print(f"Average rewrite_contain_acc: {safe_div(total_rewrite_contain_acc, count_rewrite_contain_acc)}")
        print(f"Average Locality_Fact_acc: {safe_div(total_Locality_Fact_acc, count_Locality_Fact_acc)}")
        print(f"Average Locality_Fact_contain_acc: {safe_div(total_Locality_Fact_contain_acc, count_Locality_Fact_contain_acc)}")
        print(f"Average Portability_Fact_acc: {safe_div(total_Portability_Fact_acc, count_Portability_Fact_acc)}")
        print(f"Average Portability_Fact_contain_acc: {safe_div(total_Portability_Fact_contain_acc, count_Portability_Fact_contain_acc)}")




    elif args.sequential_edit: # sequential edit
        all_prompts=[]
        all_target_new = []
        all_subjects = []
        all_locality_prompts = []
        all_locality_ans = []
        all_portability_prompts = []
        all_portability_ans = []
        prompt_len=[]
        editor = BaseEditor.from_hparams(hparams)
        for index,event in enumerate(datas):  # 每一个event有多个prompt进行顺序编辑
            
            prompts = event["prompts"]
            prompt_len.append(len(prompts))
            all_prompts.extend(event["prompts"])
            all_target_new.extend(event["target_new"])
            all_subjects.extend(event["subjects"])

            local_fact_question = event["local_fact_question"] # 再改为一个列表套列表
            local_fact_answer = event["local_fact_answer"] # 再改为一个列表套列表

            fact_question = event["fact_question"]  # 再改为一个列表套列表
            fact_answer = event["fact_answer"] # 再改为一个列表套列表


            locality_prompts = [None]*len(prompts)
            locality_prompts[-1] = local_fact_question if len(local_fact_question)>0 else None
            all_locality_prompts.extend(locality_prompts)

            locality_ans = [None]*len(prompts)
            locality_ans[-1] = local_fact_answer if len(local_fact_answer)>0 else None 
            all_locality_ans.extend(locality_ans)

            portability_prompts = [None]*len(prompts)
            portability_prompts[-1] = fact_question if len(fact_question)>0 else None
            all_portability_prompts.extend(portability_prompts)

            portability_ans = [None]*len(prompts)
            portability_ans[-1] = fact_answer if len(fact_answer)>0 else None
            all_portability_ans.extend(portability_ans)

        assert len(all_prompts) == len(all_locality_prompts) == len(all_locality_ans) 
        
        locality_inputs = {
            'Locality_Fact':{
                'prompt': all_locality_prompts,
                'ground_truth': all_locality_ans
            }
        }

        assert len(all_prompts) == len(all_portability_prompts) == len(all_portability_ans) 
        
        portability_inputs = {
            'Portability_Fact':{
                'prompt': all_portability_prompts,
                'ground_truth': all_portability_ans
            }
        }
        if args.editing_method=="WISE":
            def extract_fact_local_qas(data_item):
                local_qas = data_item.get("fact", {}).get("local_qas", [])
                result = []
                for qa in local_qas:
                    question = qa.get("question", "")
                    answer = qa.get("answer", {}).get("name", "")
                    result.append(f"{question} {answer}")
                return result
            loc_data = json.load(open('/fs-computility/ai-shen/songxin/EasyEdit_for_concept/examples/dataset/ELKEN/train_filtered.json', 'r', encoding='utf-8'))
            all_results = []
            for item in loc_data:
                all_results.extend(extract_fact_local_qas(item))
            loc_prompts = all_results[:len(all_prompts)]

            metrics, edited_model, _ = editor.edit(
                prompts=all_prompts,
                target_new=all_target_new,
                subject=all_subjects,
                locality_inputs=locality_inputs,
                portability_inputs=portability_inputs,
                loc_prompts=loc_prompts,
                sequential_edit=True,
                test_generation=False,
            )
        else:
            metrics, edited_model, _ = editor.edit(
                prompts=all_prompts,
                target_new=all_target_new,
                subject=all_subjects,
                locality_inputs=locality_inputs,
                portability_inputs=portability_inputs,
                sequential_edit=True,
                test_generation=False,
            )

        os.makedirs(args.output_dir, exist_ok=True)
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        output_file = os.path.join(
            args.output_dir,
            f'{args.datatype}_{hparams.model_name.split("/")[-1]}_{args.editing_method}_N={args.ds_size}_Sequential={args.sequential_edit}_{timestamp}.json'
        )
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(metrics,f, ensure_ascii=False, indent=4)

        print("See results at: ", output_file)

        print("需要重新计算rewrite_acc: ")

        data = metrics
        for eval in ["pre", "post"]:
            if eval not in data[0]:
                continue
            for key in ["rewrite_acc", "rephrase_acc", "rewrite_contain_acc", "rephrase_contain_acc", "fluency", 'rewrite_ppl', 'ood_acc']:
                if key in data[0][eval].keys():
                    acc = [item[eval][key] for item in data]
                    
                    slices = []
                    start = 0
                    for length in prompt_len:
                        end = start + length
                        slices.append(acc[start:end])
                        start = end

                    group_means = [np.mean(s) for s in slices]
                    final_mean = np.mean(group_means)
                    print(f"{key}_acc:{final_mean}")















































    # def summary_metrics_for_LLM_judge(all_metrics):

    #     mean_metrics = dict()
    #     for eval in ["pre", "post"]:
    #         if eval not in all_metrics[0]:
    #             continue
    #         mean_metrics[eval] = dict()
    #         for key in ["rewrite_acc", "rephrase_acc", "rewrite_contain_acc", "rephrase_contain_acc", "fluency", 'rewrite_ppl', 'ood_acc']:
    #             if key in all_metrics[0][eval].keys():
    #                 if key == "fluency":  # 对 fluency 字典值进行均值计算
    #                     fluency_values = [
    #                         metric[eval][key]['ngram_entropy'] if 'ngram_entropy' in metric[eval][key] else np.nan
    #                         for metric in all_metrics if key in metric[eval]
    #                     ]
    #                     fluency_values = [val for val in fluency_values if not np.isnan(val)]  # 移除无效值
    #                     mean_metrics[eval][key] = np.mean(fluency_values) if fluency_values else np.nan
    #                 else:  # 对其他评估指标进行均值计算
    #                     mean_metrics[eval][key] = np.mean(
    #                         [np.mean(metric[eval][key]) for metric in all_metrics if key in metric[eval]]
    #                     )
    #         for key in ["locality", "portability"]:
    #             if key in all_metrics[-1][eval].keys() and all_metrics[-1][eval][key] != {}:
    #                 mean_metrics[eval][key] = dict()
    #                 sub_keys = set()
    #                 for metric in all_metrics:
    #                     if key in metric[eval]:
    #                         sub_keys.update(metric[eval][key].keys())

    #                 for sub_key in sub_keys:
    #                     if not sub_key.endswith("_acc"):
    #                         continue

    #                     per_sample_means = []
    #                     for metric in all_metrics:
    #                         if key in metric[eval] and sub_key in metric[eval][key]:
    #                             vals = metric[eval][key][sub_key]
    #                             if isinstance(vals, list):
    #                                 if len(vals) == 0:
    #                                     continue
    #                                 # 若是 list of list，flatten
    #                                 if isinstance(vals[0], list):
    #                                     vals = [v for v in vals if isinstance(v, list) and len(v) > 0]
    #                                     vals = [item for sublist in vals for item in sublist]
    #                                 vals = [v for v in vals if isinstance(v, (float, int))]
    #                                 if len(vals) > 0:
    #                                     per_sample_means.append(np.mean(vals))

    #                     if len(per_sample_means) > 0:
    #                         mean_metrics[eval][key][sub_key] = np.mean(per_sample_means)


    #     # mean_metrics["time"] = np.mean([metric["time"] for metric in all_metrics])
    #     #print("Metrics Summary: ", mean_metrics)
    #     return mean_metrics

    #     # 读取json数据
    # all_metrics=[]
    # with open(output_file, 'r', encoding='utf-8') as f:
    #     for line in f:
    #         data = json.loads(line)
    #         metric = summary_metrics_for_LLM_judge(data)
    #         all_metrics.append(metric)

    # total_rewrite_acc = 0
    # total_rewrite_contain_acc = 0
    # total_Locality_Fact_acc = 0
    # total_Locality_Fact_contain_acc = 0
    # total_Portability_Fact_acc = 0
    # total_Portability_Fact_contain_acc = 0

    # # 记录数据的个数
    # num_records = len(all_metrics)

    # # 遍历数据并累加相应的值
    # for record in all_metrics:
    #     post = record['post']
        
    #     total_rewrite_acc += post['rewrite_acc']
    #     total_rewrite_contain_acc += post['rewrite_contain_acc']
        
    #     # locality部分
    #     locality = post.get('locality', {})
    #     total_Locality_Fact_acc += locality.get('Locality_Fact_acc', 0)
    #     total_Locality_Fact_contain_acc += locality.get('Locality_Fact_contain_acc', 0)
        
    #     # portability部分
    #     portability = post.get('portability', {})
    #     total_Portability_Fact_acc += portability.get('Portability_Fact_acc', 0)
    #     total_Portability_Fact_contain_acc += portability.get('Portability_Fact_contain_acc', 0)

    # # 计算平均值
    # avg_rewrite_acc = total_rewrite_acc / num_records
    # avg_rewrite_contain_acc = total_rewrite_contain_acc / num_records
    # avg_Locality_Fact_acc = total_Locality_Fact_acc / num_records
    # avg_Locality_Fact_contain_acc = total_Locality_Fact_contain_acc / num_records
    # avg_Portability_Fact_acc = total_Portability_Fact_acc / num_records
    # avg_Portability_Fact_contain_acc = total_Portability_Fact_contain_acc / num_records

    # # 输出结果
    # print(f"Average rewrite_acc: {avg_rewrite_acc}")
    # print(f"Average rewrite_contain_acc: {avg_rewrite_contain_acc}")
    # print(f"Average Locality_Fact_acc: {avg_Locality_Fact_acc}")
    # print(f"Average Locality_Fact_contain_acc: {avg_Locality_Fact_contain_acc}")
    # print(f"Average Portability_Fact_acc: {avg_Portability_Fact_acc}")
    # print(f"Average Portability_Fact_contain_acc: {avg_Portability_Fact_contain_acc}")