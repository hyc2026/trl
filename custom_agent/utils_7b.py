import re
import os
import gc
import time
import json
import torch
import zipfile
import threading
from deepspeed.accelerator import get_accelerator
from custom_agent.agent_dataset import prompts, AgentDataset
from custom_agent.search_tools import search_paper_by_google, laplace, log

select_prompt = "You are an elite researcher in the field of AI, conducting research on {user_query}. Evaluate whether the following paper fully satisfies the detailed requirements of the user query and provide your reasoning. Ensure that your decision and reasoning are consistent.\n\nSearched Paper:\nTitle: {title}\nAbstract: {abstract}\n\nUser Query: {user_query}\n\nOutput format: Decision: True/False\nReason:... \nDecision:"

# regular expressions
search_user_query_template = r"User Query:(.*?)assistant\n\["
search_template = r"Search\](.*?)\["
expand_user_query_template = r"research on `(.*?)`\."
expand_template = r"Expand\](.*?)\["

def keep_letters(s):
    letters = [c for c in s if c.isalpha()]
    result = ''.join(letters)
    return result.lower()

def search_paper_by_title(title, paper_db):
    title_key = keep_letters(title)
    if title_key in paper_db.namelist():
        with paper_db.open(title_key) as f:
            return json.loads(f.read().decode("utf-8"))
    else:
        return None

def get_expand_papers(section, paper, paper_db):
    section = keep_letters(section)
    res = []
    for sec in paper["sections"]:
        if keep_letters(sec) == section:
            for title in paper["sections"][sec]:
                p = search_paper_by_title(title, paper_db)
                if p is not None:
                    res.append(p)
    return res   

def gen_value_model_prompt(title, user_query, paper_db):
    paper = search_paper_by_title(title, paper_db)
    if paper is None:
        return None, None
    
    value_model_prompt = [
        {"role": "user", "content": prompts["select_section"].format(
            user_query=user_query,
            title=paper["title"],
            abstract=paper["abstract"],
            sections=json.dumps(list(paper["sections"].keys()))
        ).strip()},
        {"role": "assistant", "content": "["} # use the value of token-`[` to approximate the value of the paper
    ]
    return value_model_prompt, paper

def response_handler(
        num, 
        response, 
        all_papers, 
        all_scores, 
        lock, 
        query_responses, 
        tokenizer, 
        context_length, 
        value_model, 
        args, 
        paper_db,
        typ="search", 
        f_paper=None
    ): 
    scores = []
    if typ == "search":
        user_query_template = search_user_query_template
        query_keys_template = search_template
        cost = args.search_cost
        select_score = args.search_select_score
        max_action = 5
    else:
        user_query_template = expand_user_query_template
        query_keys_template = expand_template
        cost = args.expand_cost
        select_score = args.expand_select_score
        max_action = 5
    
    # parse the model output
    user_query = re.findall(user_query_template, response, flags=re.DOTALL)
    if len(user_query) > 0:
        user_query = user_query[0].strip()
    else:
        user_query = ""
    query_keys = [q.strip() for q in re.findall(query_keys_template, response, flags=re.DOTALL)]
    searched_paper_set = set()

    for idx in range(max(max_action, len(query_keys))):
        score = -cost
        if idx < max_action: # must call value model `max_action` times
            searched_papers, value_model_prompts, select_prompts = [], [], []

            # do search or expand
            if idx < len(query_keys):
                if typ == "search":
                    for _ in range(10):
                        try:
                            with lock:
                                searched_papers = search_paper_by_google(query_keys[idx], args.max_papers)
                            break
                        except:
                            time.sleep(1)
                            pass
                else:
                    searched_papers = get_expand_papers(query_keys[idx], f_paper, paper_db)
                    # if "introduction" in "".join(query_keys[idx].split()).lower() or "relatedwork" in "".join(query_keys[idx].split()).lower():
                    #     score += cost

            if len(searched_papers) > 0:
                # get selector score
                for searched_paper in searched_papers:
                    select_prompts.append(select_prompt.format(title=searched_paper["title"], abstract=searched_paper["abstract"], user_query=user_query))
                for _ in range(3):
                    try:
                        results = laplace.matx_inference("select_agent", {"text": select_prompts})
                        results = [json.loads(x.decode()) for x in results.output_bytes_lists['output']]
                        break
                    except:
                        results = [{"prob": 0} for i in range(len(select_prompts))]
                
                # gen value model prompt
                all_prompts = []
                for searched_paper, result in zip(searched_papers, results):
                    if keep_letters(searched_paper["title"]) not in searched_paper_set:
                        searched_paper_set.add(keep_letters(searched_paper["title"]))
                    else: # repeated searches will not be counted
                        continue
                    if result["prob"] > 0.5:
                        score += select_score
                    value_model_prompt, paper = gen_value_model_prompt(searched_paper["title"], user_query, paper_db)
                    if value_model_prompt is None:
                        continue
                    all_prompts.append([result["prob"], paper, value_model_prompt])
                all_prompts.sort(key=lambda x: x[0], reverse=True)
                for i in all_prompts[:args.max_papers]:
                    value_model_prompts.append(i[2])
                    with lock:
                        all_papers.append([i[0], i[1], i[2][:1]])
            
            valid_value_num = len(value_model_prompts)
            for _ in range(len(value_model_prompts), args.max_papers):
                value_model_prompts.append([{"role": "user", "content": "hello"}])
            
            if args.use_vm:
                reward_scores = None
                with lock:
                    for value_idx in range(0, args.max_papers, args.value_step):
                        input_ids = tokenizer.apply_chat_template(
                            value_model_prompts[value_idx: value_idx + args.value_step],
                            tokenize=True,
                            padding=True,
                            truncation=True,
                            max_length=992,
                            add_generation_prompt=False,
                        ) # [..., 151644, 77091, 198, 58, 151645]
                        add_score = True
                        input_ids = torch.tensor(input_ids, device=query_responses.device)
                        if input_ids.shape[1] > args.value_max_length:
                            input_ids = input_ids[:, :args.value_max_length]
                            start_idx = reward_scores.shape[0] if reward_scores is not None else 0
                            fail_num = 0
                            for j in range(args.value_step):
                                if start_idx + j < valid_value_num:
                                    fail_num += 1
                            valid_value_num -= fail_num
                            add_score = False
                        attention_mask = input_ids != tokenizer.pad_token_id
                        position_ids = attention_mask.cumsum(1) - attention_mask.long()
                        input_ids = torch.masked_fill(input_ids, ~attention_mask, 0)
                        output = value_model.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            return_dict=True,
                            output_hidden_states=True,
                            use_cache=False,
                        )
                        reward_logits = value_model.score(output.hidden_states[-1])
                        reward_logits = reward_logits.squeeze(-1)[:, -2]
                        if add_score:
                            if reward_scores is None:
                                reward_scores = reward_logits
                            else:
                                reward_scores = torch.cat((reward_scores, reward_logits))
                        del input_ids, attention_mask, position_ids, output, reward_logits
                        gc.collect()
                        torch.cuda.empty_cache()
                        get_accelerator().empty_cache()

                    if reward_scores is not None and valid_value_num > 0:
                        score += args.alpha * reward_scores[:valid_value_num].sum().item()

        if idx < len(query_keys):
            scores.append(max(min(score, 5), -args.search_cost))
    
    # score should be apply to each `[`(58) token
    score_tensor = torch.zeros(query_responses.shape[1] - context_length)
    if len(scores) > 0:
        score_idx = len(scores) - 1
        for i in range(-1, -score_tensor.shape[0] - 1, -1):
            if query_responses[num][i] == 58:
                score_tensor[i] = scores[score_idx]
                score_idx -= 1
                if score_idx < 0:
                    break
    with lock:
        all_scores[num] = score_tensor

def rollout(query_responses, tokenizer, context_length, value_model, args, paper_db, papers=None, typ="search", return_new_query=True):
    
    # decode to strs
    query_response_strs = tokenizer.batch_decode(query_responses, skip_special_tokens=True)
    all_papers, all_scores = [], {}
    lock = threading.Lock()

    # parse response, search paper and generate 
    threads = []
    for num, response in enumerate(query_response_strs):
        f_paper = None
        if typ=="expand":
            f_paper = papers[num]
        thread = threading.Thread(
            target=response_handler, 
            args=(
                num, 
                response, 
                all_papers, 
                all_scores, 
                lock, 
                query_responses, 
                tokenizer, 
                context_length, 
                value_model, 
                args, 
                paper_db,
                typ, 
                f_paper
            )
        )
        threads.append(thread)
        thread.start()
    for thread in threads:
        thread.join()
    
    torch.distributed.barrier()
    all_scores_list = []
    for i in range(query_responses.shape[0]):
        if i in all_scores:
            all_scores_list.append(all_scores[i])
        else:
            all_scores_list.append(torch.zeros(query_responses.shape[1] - context_length))
    all_scores = torch.stack(all_scores_list).to(query_responses.device)

    if return_new_query:
        # hard-coded to return 6 items
        all_papers.sort(key=lambda x: x[0], reverse=True)
        next_data = []
        # all_papers_num = len(all_papers)
        if len(all_papers) == 0:
            return None, None, all_scores
        while len(all_papers) < 6:
            all_papers += all_papers
        next_data = all_papers[:6]
        value_model_prompts = [i[2] for i in next_data]
        
        # tokenize一下，处理一下0条的情况
        input_ids = None
        if len(next_data) > 0:
            input_ids = tokenizer.apply_chat_template(
                value_model_prompts,
                tokenize=True,
                padding=True,
                truncation=True,
                max_length=992,
                add_generation_prompt=True,
            )
            input_ids = torch.tensor(input_ids, device=query_responses.device)

        return input_ids, [i[1] for i in next_data], all_scores
    else:
        return None, None, all_scores
    
