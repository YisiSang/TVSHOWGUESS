import os, sys, random
import numpy as np 
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from transformers import AutoConfig
from transformers import LongformerModel, LongformerTokenizer
import argparse

from utils import collate_fn_bert_single_row
from utils import get_single_row_scene_classification_datasets_with_test as get_datasets
from models import LongformerSingleRowClassifier

os.environ["CUDA_VISIBLE_DEVICES"]="0"


# set seed
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False




def load_data(args, tokenizer):
    D_tr_list, D_dev_list, D_test_list, label_dict = get_datasets(
                                                        args.data_dir, 
                                                        max_seq_len=args.max_seq_len, 
                                                        max_sent_num=args.max_sent_num, 
                                                        reverse=False,
                                                        fill_empty=False, 
                                                        min_num_row=1,
                                                        tokenizer=tokenizer,
                                                    )
    show_id_map = {key: i for i, key in enumerate(label_dict.keys())}
    D_tr_list_ = {}
    D_dev_list_ = {}
    D_test_list_ = {}
    for showname in D_tr_list.keys():
        D_tr_ = DataLoader(D_tr_list[showname], batch_size=args.batch_size, shuffle=True, 
                        collate_fn=collate_fn_bert_single_row) 
        D_dev_ = DataLoader(D_dev_list[showname], batch_size=args.batch_size, shuffle=False, 
                            collate_fn=collate_fn_bert_single_row) 
        D_test_ = DataLoader(D_test_list[showname], batch_size=args.batch_size, shuffle=False, 
                            collate_fn=collate_fn_bert_single_row) 
        D_tr_list_[showname] = D_tr_
        D_dev_list_[showname] = D_dev_
        D_test_list_[showname] = D_test_
    print('[*] Loaded TV Shows: {}'.format(list(show_id_map.keys())))

    return D_tr_list_, D_dev_list_, D_test_list_, show_id_map


def compute_loss(args, labels, pred_logits, c_masks):
    sup_loss = 0.
    sample_count = 0
    for inst_id in range(pred_logits.size(0)):
        log_pred_probs = torch.log_softmax(pred_logits[inst_id][:len(labels[inst_id])], dim=1)
        y_mask = torch.sum(c_masks[inst_id], dim=1)
        for c_id in range(len(labels[inst_id])):
            loss = -log_pred_probs[c_id][labels[inst_id][c_id]] / args.gradient_accumulation_steps
            if y_mask[c_id] != 0:
                sup_loss += loss
                sample_count += 1
    sup_loss = sup_loss / sample_count
    return sup_loss


def compute_accu(labels, pred_logits):
    acc = 0
    sample_count = 0
    for inst_id in range(len(labels)):
        for c_id in range(len(labels[inst_id])):
            max_score = torch.min(pred_logits[inst_id][c_id]).cpu().item()
            pred = -1
            for choice in labels[inst_id]:
                if pred_logits[inst_id][c_id][choice] > max_score:
                    max_score = pred_logits[inst_id][c_id][choice]
                    pred = choice
            if pred == labels[inst_id][c_id]:
                acc += 1
            sample_count += 1

    acc /= sample_count
    return acc


def evaluate(model, data_dict, split_name, show_id_map):
    model.eval()
    print('========== Evaluation on {} =========='.format(split_name))
    with torch.no_grad():
        num_sample = 0
        correct = 0.
        num_person = 0
        correct_person = 0.

        for showname in data_dict.keys():
            show_num_sample = 0
            show_correct = 0.
            show_num_person = 0
            show_correct_person = 0.
            show_id = show_id_map[showname]
            for i_batch, data in enumerate(data_dict[showname]):
                x = data["xs"].cuda()
                mask = data["masks"].cuda()
                c = data['c_masks'].cuda()
                y = data["y"]
                scene_id = data["scene_id"]

                pred_logits = model(x, mask, c, show_id)
            
                acc = 0
                sample_count = 0
                for inst_id in range(x.size(0)):
                    scene_acc = 0
                    scene_sample_count = 0
                    for c_id in range(len(y[inst_id])):
                        max_score = torch.min(pred_logits[inst_id][c_id]).cpu().item()
                        pred = -1
                        for choice in y[inst_id]:
                            if pred_logits[inst_id][c_id][choice] > max_score:
                                max_score = pred_logits[inst_id][c_id][choice]
                                pred = choice
                        if pred == y[inst_id][c_id]:
                            acc += 1
                            scene_acc += 1
                        sample_count += 1
                        scene_sample_count += 1
                    scene_acc /= scene_sample_count
                    correct += scene_acc
                    show_correct += scene_acc
                    num_sample += 1
                    show_num_sample += 1

                correct_person += acc
                show_correct_person += acc
                num_person += sample_count
                show_num_person += sample_count

            print('[*] SHOW NAME - {}:'.format(showname))
            # print('    Scene-Level {} Acc:  {:.4f}'.format(split_name.capitalize(), show_correct / show_num_sample))
            print('    Person-Level {} Acc: {:.4f}'.format(split_name.capitalize(), show_correct_person / show_num_person))
            # print('    Number of scenes:\t ', show_num_sample)
            # print('    Number of characters:', show_num_person)

        if split_name.lower() == 'dev':
            # print('[*] Average Scene-Level {} Acc:  {:.4f}'.format(split_name.capitalize(), correct/num_sample))
            print('[*] Average Person-Level {} Acc: {:.4f}'.format(split_name.capitalize(), correct_person/num_person))
        else:
            # print('[*] Average Scene-Level {} Acc: {:.4f}'.format(split_name.capitalize(), correct / num_sample))
            print('[*] Average Person-Level {} Acc: {:.4f}'.format(split_name.capitalize(), correct_person / num_person))
        # print('[*] Number of scenes:\t ', num_sample)
        # print('[*] Number of characters:', num_person)
    print('========== END of Evaluation ==========')
    return correct/num_sample, correct_person/num_person


def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--data_dir', type=str,
                        required=True,                    
                        help='The directory where the merged json files locate.')
    parser.add_argument('--splits', nargs='+',
                        default=[],                    
                        help='The directory where the merged json files locate.')
    parser.add_argument('--from_pretrained', type=str,
                        default=None,                    
                        help='The model to load.')
    parser.add_argument('--max_seq_len', type=int,
                        default=2000,                       
                        help='The maximum length of the input sequence.')
    parser.add_argument('--max_sent_num', type=int,
                        default=12,                       
                        help='The maximum number of dialogue utterances/sentences included in the context.')
    parser.add_argument('--attention_window', type=int,
                        default=256,
                        help='The attention window size in Longformer.')
    parser.add_argument('--gradient_accumulation_steps', type=int,
                        default=1,
                        help='Number of updates steps to accumulate before performing a backward/update pass.')
    parser.add_argument('--num_epochs', type=int,
                        default=40,
                        help='The total number of training epochs.')
    parser.add_argument('--batch_size', type=int,
                        default=4,
                        help='The batch size.')
    parser.add_argument('--lr', type=float,
                        default=2e-5,
                        help='Learning rate.')
    parser.add_argument('--output_dir', type=str,
                        default='trained_models/',
                        help='The directory where the trained model will be saved.')
    parser.add_argument('--train', action='store_true',
                        help='To enable training.')
    parser.add_argument('--test', action='store_true',
                        help='To enable evaluation.')
    args = parser.parse_args()

    
    # load tokenizer and data
    tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
    train_data, dev_data, test_data, show_id_map = load_data(args, tokenizer)

    # load model
    config = AutoConfig.from_pretrained('allenai/longformer-base-4096')
    config.attention_window = args.attention_window
    pretrained_longformer = LongformerModel.from_pretrained('allenai/longformer-base-4096', config=config)
    args.num_classes = 6 
    args.num_shows = len(show_id_map)
    model = LongformerSingleRowClassifier(args, bert_model=pretrained_longformer, fine_tuning=True, max_length=512)
    if args.from_pretrained is not None:
        model.load_state_dict(torch.load(args.from_pretrained))
    model = model.to('cuda')


    # optimization
    pred_optimizer = torch.optim.Adam(model.pred_vars() , lr=args.lr)


    # training
    if args.train:
        best_F1s = [0., 0., 0., 0.]
        best_dev = 0.
        best_dev_person = 0.
        show_list = list(show_id_map.keys())
        for i_epoch in range(args.num_epochs):
            print("================")
            print("   epoch: {:02}".format(i_epoch))
            print("================")

            model.train()
            train_accs = {}
            train_losses = {}
            for showname in train_data.keys():
                train_accs[showname] = []
                train_losses[showname] = []
            
            step = 0
            pred_optimizer.zero_grad()
            
            for i_batch in tqdm(range(len(train_data['The_Office']))):
                random.shuffle(show_list)
                for showname in show_list:
                    show_id = show_id_map[showname]
                    data = next(iter(train_data[showname]))
                    
                    x = data["xs"].to('cuda')
                    mask = data["masks"].to('cuda')
                    c = data['c_masks'].to('cuda')
                    y = data["y"]
                    scene_id = data["scene_id"]
                    pred_logits = model(x, mask, c, show_id)
                    
                    sup_loss = compute_loss(args, y, pred_logits, c)
                    train_losses[showname].append(sup_loss.cpu().item())
                    sup_loss.backward()

                    acc = compute_accu(y, pred_logits)
                    train_accs[showname].append(acc)

                    if (step + 1) % args.gradient_accumulation_steps == 0:
                        pred_optimizer.step()
                        pred_optimizer.zero_grad()
                        step = 0
                    else:    
                        step += 1
                
                # print training accuracy
                if (i_batch + 1) % 500 == 0:
                    print('[*] Training Stats:')
                    for showname in train_data.keys():
                        print("    {:<20} => Train batch: {:<5} step: {:<4} loss: {:07.4f}   acc: {:.4f}".format(
                                    showname,
                                    i_batch, 
                                    step, 
                                    np.mean(train_losses[showname]), 
                                    np.mean(train_accs[showname]))
                                )
                    for showname in train_data.keys():
                        train_accs[showname] = []
                        train_losses[showname] = []

                # evaluate on dev set
                if (i_batch + 1) % 600 == 0 or (i_batch+1) == len(train_data['The_Office']):
                    scene_score_mean, person_score_mean = evaluate(model, dev_data, 'dev', show_id_map)
                    ## scene_score > person_score
                    ## But we use person_score as our metrics in the paper
                    if person_score_mean > best_dev_person:
                        best_dev_person = person_score_mean
                        print('[*] Saving the checkpoint with best dev {:.4f}'.format(best_dev_person))
                        torch.save(model.state_dict(), args.output_dir + 'pytorch_model.pt')
                    
    # testing
    if args.test:
        if 'dev' in args.splits:
            evaluate(model, dev_data, 'dev', show_id_map)
        if 'test' in args.splits:
            evaluate(model, test_data, 'test', show_id_map)



if __name__ == '__main__':
    main()