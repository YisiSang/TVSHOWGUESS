import os
from pathlib import Path
import sys
import json
import time
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import Dataset

from transformers import LongformerTokenizer



def collate_fn_bert_single_row(batch):
    elem = batch[0]  
    ret_dict = {}
    for key in elem:
        if key not in ['scene_id', 'y']:
            value_batch = [torch.as_tensor(d[key]) for d in batch]
            ret_dict[key] = torch.stack(value_batch, 0)
        else:
            value_batch = [d[key] for d in batch]
            ret_dict[key] = value_batch
    return ret_dict


def get_alias_dicts():
    alias_dict = {}
    alias_dict['FRIENDS'] = {}
    alias_dict['FRIENDS']['rachel'] = ['Rachel', 'RACHEL', 'RACH']
    alias_dict['FRIENDS']['ross'] = ['Ross', 'ROSS']
    alias_dict['FRIENDS']['joey'] = ['Joey', 'JOEY']
    alias_dict['FRIENDS']['monica'] = ['Monica', 'MONICA', 'MNCA']
    alias_dict['FRIENDS']['phoebe'] = ['Phoebe', 'PHOEBE', 'PHOE']
    alias_dict['FRIENDS']['chandler'] = ['Chandler', 'CHANDLER', 'CHAN']

    alias_dict['The_Big_Bang_Theory'] = {}
    alias_dict['The_Big_Bang_Theory']['leonard'] = ['Leonard', 'LEONARD']
    alias_dict['The_Big_Bang_Theory']['sheldon'] = ['Sheldon', 'SHELDON']
    alias_dict['The_Big_Bang_Theory']['penny'] = ['Penny', 'PENNY']
    alias_dict['The_Big_Bang_Theory']['howard'] = ['Howard', 'HOWARD']
    alias_dict['The_Big_Bang_Theory']['raj'] = ['Raj', 'RAJ']
    alias_dict['The_Big_Bang_Theory']['amy'] = ['Amy', 'AMY']

    alias_dict['Frasier'] = {}
    alias_dict['Frasier']['frasier'] = ['Frasier', 'FRASIER']
    alias_dict['Frasier']['daphne'] = ['Daphne', 'DAPHNE']
    alias_dict['Frasier']['niles'] = ['Niles', 'NILES']
    alias_dict['Frasier']['roz'] = ['Roz', 'ROZ']
    alias_dict['Frasier']['martin'] = ['Martin', 'MARTIN']
    alias_dict['Frasier']['bob'] = ['Bob', 'BOB', 'Bulldog', 'BULLDOG']

    alias_dict['Gilmore_Girls'] = {}
    alias_dict['Gilmore_Girls']['lorelai'] = ['Lorelai', 'LORELAI']
    alias_dict['Gilmore_Girls']['rory'] = ['Rory', 'RORY']
    alias_dict['Gilmore_Girls']['sookie'] = ['Sookie', 'SOOKIE']
    alias_dict['Gilmore_Girls']['lane'] = ['Lane', 'LANE']
    alias_dict['Gilmore_Girls']['michel'] = ['Michel', 'MICHEL']
    alias_dict['Gilmore_Girls']['luke'] = ['Luke', 'LUKE']

    alias_dict['The_Office'] = {}
    alias_dict['The_Office']['pam'] = ['PAM', 'Pam']
    alias_dict['The_Office']['jim'] = ['Jim', 'JIM']
    alias_dict['The_Office']['dwight'] = ['Dwight', 'DWIGHT']
    alias_dict['The_Office']['michel'] = ['Michel', 'MICHEL']
    alias_dict['The_Office']['jan'] = ['Jan', 'JAN']
    alias_dict['The_Office']['ryan'] = ['Ryan', 'RYAN']

    revert_alias_dict = {}
    for show, v in alias_dict.items():
        revert_alias_dict[show] = {}
        for name, aliases in v.items():
            for alias in aliases:
                revert_alias_dict[show][alias.lower()] = name
    return alias_dict, revert_alias_dict


def tokenize_data(DATA_DIR, showname, outpath, tokenizer):
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    with open(outpath, 'w', encoding='cp1256') as fileout:
        for s_id in tqdm(range(1, 30), desc='Tokenizing {} scenes'.format(showname)):
            for e_id in range(1, 31):
                episode_name = os.path.join(
                    DATA_DIR, 
                    'split_scenes/{}/{}_{:02}x{:02}.split_scenes.json'.format(showname, showname, s_id, e_id)
                )
                if not os.path.exists(episode_name):
                    continue

                with open(episode_name) as fp:
                    data_item = json.load(fp)

                episode_id = data_item["Episode Number"]
                for scene in data_item['scenes']:
                    title = scene['title'].lower()
                    bert_tokens = []
                    tokens = tokenizer.tokenize(title)
                    for token in tokens:
                        bert_tokens.append(token)
                    title = ' '.join(bert_tokens)
                    lines = scene['lines']
                    chars = scene['participants']

                    new_scene = {}
                    new_scene['episode_id'] = episode_id
                    new_scene['title'] = title
                    new_scene['participants'] = chars

                    new_lines = []
                    for line in lines:
                        char = line[0]
                        text = line[1].lower()
                        bert_tokens = []
                        tokens = tokenizer.tokenize(text)
                        for token in tokens:
                            bert_tokens.append(token)
                        new_lines.append((char, ' '.join(bert_tokens)))
                    new_scene['lines'] = new_lines
                    fileout.write(json.dumps(new_scene) + '\n')


def get_tokenized_character_dialog_examples_json_with_test(
        data_dir, 
        tokenizer, 
        max_seq_len=300, 
        max_sent_num=200, 
        return_joint_sets=False,
    ):
    print('==================Loading Character Examples==================')
    return_joint_sets = return_joint_sets
    
    alias_dict, revert_alias_dict = get_alias_dicts()
    test_season_dict = {}
    test_season_dict['FRIENDS'] = 9
    test_season_dict['The_Big_Bang_Theory'] = 8
    test_season_dict['Frasier'] = 10
    test_season_dict['Gilmore_Girls'] = 5
    test_season_dict['The_Office'] = 8

    char_utterance_num_dict = {}
    train_char_utterance_dict = {}
    dev_char_utterance_dict = {}
    train_scenes = {}
    dev_scenes = {}
    
    for showname in alias_dict.keys():
        char_utterance_num_dict[showname] = {}
        train_char_utterance_dict[showname] = {}
        dev_char_utterance_dict[showname] = {}
        train_scenes[showname] = []
        dev_scenes[showname] = []
        
        for char in alias_dict[showname].keys():
            char_utterance_num_dict[showname][char] = 0

        for char in alias_dict[showname].keys():
            train_char_utterance_dict[showname][char] = []
            dev_char_utterance_dict[showname][char] = []
    
    num_scenes = 0
    num_binary_scenes = 0
    num_lines = 0
    real_max_line_num = 0
    real_max_line_len = 0
    real_max_line_len_src = 0
    real_max_line_len_trg = 0
    total_line_num = 0
    total_line_len = 0
    total_line_len_src = 0
    total_line_len_trg = 0
    num_char_lines = 0
    new_scenes_train = []
    new_scenes_dev = []
    train_utterances = []
    dev_utterances = []
    num_scenes = 0
    
    for showname in revert_alias_dict.keys():
        tok_file = os.path.join(data_dir, 'screenguess_v1.0/{}.tok.json'.format(showname))
        if not os.path.exists(tok_file):
            tokenize_data(data_dir, showname, tok_file, tokenizer)
        
        fp = open(tok_file)
        for line in fp:
            scene = json.loads(line.strip())
            title = scene['title'].split()
            lines = scene['lines']
            chars = scene['participants']
            season_id = int(scene['episode_id'].split('x')[0])

            src_lines = []
            masked_trg_lines = []
            answers = []
            answer_dict = {}
            scene_utterances = []

            for line_num, line in enumerate(lines):
                char = line[0]
                ori_char = char
                text = line[1]
                if char in revert_alias_dict[showname]:
                    char = revert_alias_dict[showname][char]
                    if char not in answers:
                        masked_char = 'P{}'.format(len(answers))
                        answers.append(char)
                        answer_dict[masked_char] = char
                    else:
                        masked_char = 'P{}'.format(answers.index(char))
                else:
                    masked_char = char

                array = text.split()
                for i, token in enumerate(array):
                    if token == ':':
                        break

                if char != 'background':
                    text = ' '.join(array[i+1:]).lstrip().strip()

                t = text.split()
                src_lines.append((line_num, masked_char, tokenizer.tokenize(masked_char) + [':'] + t))

                if ori_char in revert_alias_dict[showname]:
                    char_utterance_num_dict[showname][ori_char] += 1
                    char_utterance = {
                        'title':title, 
                        'masked_char':masked_char, 
                        'char_name':ori_char, 
                        'src_lines':src_lines[:line_num], 
                        'trg_lines':t, 
                        'answers':answer_dict, 
                        'episode_id':scene['episode_id'], 
                        'scene_id':num_scenes
                    }
                    scene_utterances.append({
                        'masked_char':masked_char, 
                        'char_name':ori_char, 
                        'src_lines':src_lines[:line_num], 
                        'trg_lines':t
                    })

                    if season_id < test_season_dict[showname]:
                        train_char_utterance_dict[showname][ori_char].append(char_utterance)
                        train_utterances.append(char_utterance)
                    else:
                        dev_char_utterance_dict[showname][ori_char].append(char_utterance)
                        dev_utterances.append(char_utterance)

                    masked_trg_lines.append((line_num, masked_char, t))
                    if len(t) > real_max_line_len_trg:
                        real_max_line_len_trg = len(t)
                    total_line_len_trg += len(t)

                    src_len = 0
                    for src_line in src_lines[:line_num]:
                        src_len += len(src_line[2])
                    if src_len > real_max_line_len_src:
                        real_max_line_len_src = src_len
                    total_line_len_src += src_len

                    num_char_lines += 1

                if len(t) > real_max_line_len:
                    real_max_line_len = len(t)

                total_line_len += len(t)
                num_lines += 1

            if len(masked_trg_lines) > real_max_line_num:
                real_max_line_num = len(masked_trg_lines)
            total_line_num += len(masked_trg_lines)
            num_scenes += 1

            original_scene = {
                'title': title, 
                'scene_utterances': scene_utterances, 
                'answers': answer_dict, 
                'episode_id': scene['episode_id'], 
                'scene_id': num_scenes
            }

            if len(answer_dict) == 0:
                continue

            if season_id < test_season_dict[showname]:
                train_scenes[showname].append(original_scene)
                new_scenes_train.append(original_scene)
            else:
                if len(answer_dict) <= 1:
                    continue
                dev_scenes[showname].append(original_scene)
                new_scenes_dev.append(original_scene)
        fp.close()
    
    print("[*] Number of scenes: {}".format(num_scenes))
    print("[*] Number of lines: {}".format(num_lines))
    print('[*] Maximum line number per scene: {}\tAverage line number: {:.2f}'.format(
        real_max_line_num, total_line_num / num_scenes))
    print('[*] Maximum line length: {}\tAverage line length: {:.2f}'.format(
        real_max_line_len, total_line_len / num_lines))
    print('[*] Maximum SOURCE line length: {}\tAverage SOURCE line length: {:.2f}'.format(
        real_max_line_len_src, total_line_len_src / num_char_lines))
    print('[*] Maximum TARGET line length: {}\tAverage TARGET line length: {:.2f}'.format(
        real_max_line_len_trg, total_line_len_trg / num_char_lines))
    print('[*] Character Utterances:')
    print(json.dumps(char_utterance_num_dict, indent=4))

    ret_dict = {}
    ret_dict['train_char_utterance_dict'] = train_char_utterance_dict
    ret_dict['dev_char_utterance_dict'] = dev_char_utterance_dict
    ret_dict['train_scenes'] = train_scenes
    ret_dict['dev_scenes'] = dev_scenes
    if return_joint_sets:
        ret_dict['train_utterances_joint'] = train_utterances
        ret_dict['dev_utterances_joint'] = dev_utterances
        ret_dict['train_scenes_joint'] = new_scenes_train
        ret_dict['dev_scenes_joint'] = new_scenes_dev

    return ret_dict


class BartSingleRowSceneClassificationDataset(Dataset):
    """Beer dataset."""

    def __init__(self, 
            data, 
            label_dict, 
            max_seq_len,
            max_sent_num=8, 
            tokenizer=None, 
            stride=False, 
            fill_empty=False, 
            min_num_row=1, 
            max_char_num=6, 
            reverse=False
        ):
        """
        Args:
            data: the acutal data of beer review after indexing
            stoi: string to index
            max_seq_len: max sequence length
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data = data
        self.max_seq_len = max_seq_len
        self.max_sent_num = max_sent_num
        self.max_char_num = max_char_num
        if tokenizer is None:
            self.tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
        else:
            self.tokenizer = tokenizer
        self.CLS_TOKEN = self.tokenizer.encoder[self.tokenizer._cls_token.content]
        self.SEP_TOKEN = self.tokenizer.encoder[self.tokenizer._sep_token.content]
        self.SPLIT_TOKEN = 50261
        self.PAD_TOKEN = self.tokenizer.encoder[self.tokenizer._pad_token.content]
        self.label_dict = label_dict
        self.stride = stride
        self.fill_empty = fill_empty
        self.min_num_row = min_num_row
        self.reverse = reverse

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        scenes = self.data
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        scene   = scenes[idx]
        title   = scene['title']
        answers = scene['answers'].items()
        scene_id = scene['scene_id']
        utterance_tuples = scene['scene_utterances']
            
        xs      = [] # concatenation of masked lines L
        masks   = [] # number of masked lines L
        cs      = [] # masks of chars C*L
        ys      = [] # number of chars C
        
        answer_dict = {}
        answer_id_dict = {}
        for c_id, c_name in answers:
            ys.append(self.label_dict[c_name])
            cs.append([])
            answer_dict[c_id] = self.label_dict[c_name]
            answer_id_dict[c_id] = len(answer_id_dict)
            
        if self.reverse:
            utterance_tuples = utterance_tuples[: :-1]

        if self.fill_empty:
            covered_answers = {}
            trial_output = []
            must_include = {}
            utterance_lengths = []
            for uid, utterance_tuple in enumerate(utterance_tuples):
                t = utterance_tuple['trg_lines']
                masked_char = utterance_tuple['masked_char']

                t = self.tokenizer.tokenize(masked_char) + [':'] + t + [self.SPLIT_TOKEN]
                utterance_lengths.append(len(t))
                
                if masked_char not in covered_answers:
                    covered_answers[masked_char] = 1
                    trial_output.extend(t)
                    must_include[uid] = 1

            if len(trial_output) > self.max_seq_len:
                print('[*] scene {} exceeds the maximum length'.format(scene_id))

            new_utterance_tuples = []
            if self.min_num_row > 1:
                print('[!] Not implemented')
            else:
                total_length = len(trial_output)
                for uid, utterance_tuple in enumerate(utterance_tuples):
                    if uid in must_include:
                        new_utterance_tuples.append(utterance_tuple)
                    else:
                        if total_length < self.max_seq_len:
                            new_utterance_tuples.append(utterance_tuple)
                            total_length += utterance_lengths[uid]
            utterance_tuples = new_utterance_tuples
        
        xs.append(self.CLS_TOKEN)
        masks.append(1.0)
        for c_id in range(len(cs)):
            cs[c_id].append(0.0)
        covered_answers = {}
        for row_id, utterance_tuple in enumerate(utterance_tuples):
            if len(xs) >= self.max_seq_len:
                break
            
            src_lines = utterance_tuple['src_lines']
            t = utterance_tuple['trg_lines']
                
            char = utterance_tuple['char_name']
            c_id = self.label_dict[char]
            masked_char = utterance_tuple['masked_char']
            c_name = answer_dict[masked_char]
        
            assert c_id == c_name, str(c_id) + '\t' + str(c_name)         
            c_id = answer_id_dict[masked_char]
            
            target = self.tokenizer.tokenize(masked_char) + [':'] + t 
            target = target[:self.max_seq_len]
            target = self.tokenizer.convert_tokens_to_ids(target)

            cs[c_id].extend([1.0] * len(target) + [0.0])
            for c_id_ in range(len(cs)):
                if c_id != c_id_:
                    cs[c_id_].extend([0.0] * (len(target) + 1))
            target = target + [self.SPLIT_TOKEN]
            xs.extend(target)
            masks.extend([1.0] * len(target))
            
            if masked_char not in covered_answers:
                covered_answers[masked_char] = 1
            
        # if len(answer_id_dict) != len(covered_answers):
        #     print('[*] scene {} missed {} chars'.format(scene_id, len(answer_id_dict) - len(covered_answers)))

        for c_id in range(len(cs)):
            cs[c_id] = cs[c_id][:self.max_seq_len]
            cs[c_id].append(0.0)
        xs = xs[:self.max_seq_len]
        masks = masks[:self.max_seq_len]
        xs.append(self.SEP_TOKEN)
        masks.append(1.0)
        
        while len(xs) < self.max_seq_len + 1:
            for c_id in range(len(cs)):
                cs[c_id].append(0.0)
            xs.append(self.PAD_TOKEN)
            masks.append(0.0)
            
        assert len(xs) == self.max_seq_len + 1, str(len(xs))
        assert len(masks) == self.max_seq_len + 1, str(len(masks))
        
        while len(cs) < self.max_char_num:
            cs.append([0.0] * (self.max_seq_len + 1))
        
        sample = {
            "xs": np.array(xs, dtype=np.int64), 
            "masks": np.array(masks, dtype=np.float32),
            "c_masks": np.array(cs, dtype=np.float32),
            "y": ys,
            "scene_id": scene_id
        }
        return sample


def get_single_row_scene_classification_datasets_with_test(
        data_dir, 
        cache_dir=None, 
        max_seq_len=300, 
        max_sent_num=8, 
        word_thres=10, 
        reverse=False,
        fill_empty=False, 
        min_num_row=1,
        tokenizer=None
    ):
    """
    Get datasets (train, dev and test).
    """
    start = time.time()
    
    if tokenizer is None:
        tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
    data_dict = get_tokenized_character_dialog_examples_json_with_test(data_dir, tokenizer)
        
    num_train = 0
    num_dev = 0
    num_test = 0
    data_dict['test_scenes'] = {}
    for showname in data_dict['train_scenes'].keys():
        split_pos = len(data_dict['dev_scenes'][showname]) // 2
        data_dict['test_scenes'][showname] = data_dict['dev_scenes'][showname][split_pos:]
        data_dict['dev_scenes'][showname] = data_dict['dev_scenes'][showname][:split_pos]
        
        print('[*] Showname - {}:'.format(showname))
        print('    # of train scenes', len(data_dict['train_scenes'][showname]))
        print('    # of dev scenes', len(data_dict['dev_scenes'][showname]))
        print('    # of test scenes', len(data_dict['test_scenes'][showname]))
        num_train += len(data_dict['train_scenes'][showname])
        num_dev += len(data_dict['dev_scenes'][showname])
        num_test += len(data_dict['test_scenes'][showname])
        
    print('[*] # of total train scenes:', num_train)
    print('[*] # of total dev scenes:', num_dev)
    print('[*] # of total test scenes:', num_test)
    
    end = time.time()
    print('[*] JSON files loading time: {:.2f}s'.format(end - start))

    label_dict = {}
    for showname in data_dict['train_scenes'].keys():
        label_dict[showname] = {}
        for scene in data_dict['train_scenes'][showname] + data_dict['dev_scenes'][showname]:
            answers = scene['answers']
            for k, v in answers.items():
                if v not in label_dict[showname]:
                    label_dict[showname][v] = len(label_dict[showname])

    D_tr_list = {}
    D_dev_list = {}
    D_test_list = {}

    for showname in data_dict['train_scenes'].keys():
        D_tr = BartSingleRowSceneClassificationDataset(data_dict['train_scenes'][showname], label_dict[showname], 
                                              max_seq_len, 
                                              max_sent_num, 
                                              tokenizer=tokenizer,
                                              fill_empty=fill_empty, 
                                              reverse=reverse,
                                              min_num_row=min_num_row)
        D_dev = BartSingleRowSceneClassificationDataset(data_dict['dev_scenes'][showname], label_dict[showname], 
                                               max_seq_len, 
                                               max_sent_num, 
                                               tokenizer=tokenizer,
                                               fill_empty=fill_empty, 
                                               reverse=reverse,
                                               min_num_row=min_num_row)
        D_test = BartSingleRowSceneClassificationDataset(data_dict['test_scenes'][showname], label_dict[showname], 
                                               max_seq_len, 
                                               max_sent_num, 
                                               tokenizer=tokenizer,
                                               fill_empty=fill_empty, 
                                               reverse=reverse,
                                               min_num_row=min_num_row)  
        D_tr_list[showname] = D_tr
        D_dev_list[showname] = D_dev
        D_test_list[showname] = D_test
    
    return D_tr_list, D_dev_list, D_test_list, label_dict


