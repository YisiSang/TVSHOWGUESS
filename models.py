import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from transformers import BertModel

import logging



class LockedDropout(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, dropout=0.2):
        if not self.training or not dropout:
            return x
        m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - dropout)
        mask = Variable(m, requires_grad=False) / (1 - dropout)
        mask = mask.expand_as(x)
        return mask * x


class BiAttention(nn.Module):
    def __init__(self, input_size, dropout):
        super().__init__()
        self.dropout_rate = dropout
        self.dropout = LockedDropout()
        self.input_linear = nn.Linear(input_size, 1, bias=False)
        self.memory_linear = nn.Linear(input_size, 1, bias=False)

        self.dot_scale = nn.Parameter(torch.Tensor(input_size).uniform_(1.0 / (input_size ** 0.5)))

    def forward(self, input, memory, mask=None):
        bsz, input_len, memory_len = input.size(0), input.size(1), memory.size(1)

        input = self.dropout(input, self.dropout_rate)
        memory = self.dropout(memory, self.dropout_rate)

        input_dot = self.input_linear(input)
        memory_dot = self.memory_linear(memory).view(bsz, 1, memory_len)
        cross_dot = torch.bmm(input * self.dot_scale, memory.permute(0, 2, 1).contiguous())
        att = input_dot + memory_dot + cross_dot
        
        if mask is not None:
            att = att - 1e30 * (1 - mask[:,None])

        weight_one = F.softmax(att, dim=-1)
        output_one = torch.bmm(weight_one, memory)
        weight_two = F.softmax(att.max(dim=-1)[0], dim=-1).view(bsz, 1, input_len)
        output_two = torch.bmm(weight_two, input)

        return torch.cat([input, output_one, input*output_one, output_two*output_one], dim=-1)


class LongformerSingleRowClassifier(nn.Module):
    def __init__(self, args, max_length, 
                bert_model=None,
                fine_tuning=True, 
                blank_padding=True):
        super().__init__()
        
        self.args           = args
        self.num_shows      = args.num_shows
        self.num_class      = args.num_classes
        self.max_length     = max_length
        self.blank_padding  = blank_padding
        self.hidden_size    = 768
        # self.prefix_dropout = args.prefix_dropout
        # self.dropout        = nn.Dropout(self.prefix_dropout)
        self.attentive_bert = True
        self.fine_tuning    = fine_tuning
        self.attention_fc   = nn.Linear(self.hidden_size, 1, bias=False)
        
        self.pred_fcs = []
        for i in range(self.num_shows):
            pred_fc = nn.Linear(self.hidden_size, args.num_classes)
            setattr(self, 'pred_fc%d'%i, pred_fc)
            self.pred_fcs.append(pred_fc)
        
        if bert_model is None:
            logging.info('Loading BERT pre-trained checkpoint.')
            self.bert = BertModel.from_pretrained("bert-base-uncased")
        else:
            self.bert = bert_model

        self.loss = nn.CrossEntropyLoss()
        

    def pred_vars(self):
        params = list()
        if self.fine_tuning:
            params = list(self.bert.parameters()) + list(self.attention_fc.parameters())
            for i in range(self.num_shows):
                pred_fc = getattr(self, 'pred_fc%d'%i)
                params += list(pred_fc.parameters())

        return params
    

    def forward(self, seq, mask, c_mask, show_id):
        hiddens = self.bert(seq, attention_mask=mask)[0]
        token_att_logits = self.attention_fc(hiddens).squeeze(-1)  
        token_att_logits = token_att_logits + (1.-mask) * -1e9 
        c_logits = token_att_logits.unsqueeze(1) + (1.-c_mask) * -1e9 
        c_token_probs = F.softmax(c_logits, dim=1).unsqueeze(-1) 
        c_hiddens = hiddens.unsqueeze(1) * c_mask.unsqueeze(-1) 
        # classification
        pred_logits = self.pred_fcs[show_id](torch.sum(c_hiddens * c_token_probs * c_mask.unsqueeze(-1), dim=2)) 

        return pred_logits