import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from math import sqrt
# from transformers import BertModel
import math, copy
from bert_serving.client import BertClient
from sklearn.metrics.pairwise import cosine_similarity
bc = BertClient(check_length=False)

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.cuda.empty_cache()

torch.cuda.set_per_process_memory_fraction(0.8)
torch.backends.cudnn.enabled = True

torch.backends.cuda.max_memory_split_size = 1024 * 8

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def pad_sequences(seq, min_len):
    padded_seq = torch.zeros((len(seq), min_len), dtype=torch.long)
    for i, s in enumerate(seq):
        end = min(len(s), min_len)
        padded_seq[i, :end] = torch.tensor(s[:end])
    return padded_seq

# class AttentionLayer(nn.Module):
#     def __init__(self, input_dim):
#         super(AttentionLayer, self).__init__()
#         self.W = nn.Linear(input_dim, 1)
#         self.b = nn.Parameter(torch.Tensor(1))

#     def forward(self, inputs):
#         outputs = []
#         for sample in inputs:
#             u = self.W(sample) + self.b
#             a = F.softmax(u, dim=0)
#             weighted_input = sample * a
#             weighted_input = weighted_input.unsqueeze(0)  # 添加维度
#             outputs.append(weighted_input)  # 将 weighted_input 添加到 outputs
#         outputs = torch.cat(outputs, dim=0)  # 沿着第0维度拼接
#         return outputs 

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)
        
        # 3) "Concat" using x view and apply x final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

class CNN_Text(nn.Module):
    
    def __init__(self, args):
        super(CNN_Text, self).__init__()
        self.args = args
        
        V = args.embed_num
        D = args.embed_dim
        C = args.class_num
        Ci = 1
        Co = args.kernel_num
        Ks = args.kernel_sizes

        self.embed = nn.Embedding(V, D)
        self.embed.weight.data.copy_(args.pretrained_weight)
        # self.convs1 = [nn.Conv2d(Ci, Co, (K, D)) for K in Ks]
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
        self.relu2 = nn.LeakyReLU(negative_slope=0.01)
        self.dropout = nn.Dropout(args.dropout, self.training) 
        self.fc2 = nn.Linear(128, 64).to(device)
        self.fc3 = nn.Linear(64, 1).to(device)
        # self.mattn = MultiHeadedAttention(h=8, d_model=D, dropout=0.1)



    # def conv_and_pool(self, x, conv):
    #     x = F.relu(conv(x)).squeeze(3)  # (N, Co, W)
    #     x = F.max_pool1d(x, x.size(2)).squeeze(2)
    #     return x

    
    # def Cosine_simlarity(self, vec1, vec2):
    #     up = 0.0                                                                                                                                                                  
    #     down = 0.0
    #     down_1 = 0.0
    #     down_2 = 0.0
    #     for i in range(len(vec1)):
    #         up += (vec1[i] * vec2[i])
    #     for i in range(len(vec1)):
    #         down_1 += (vec1[i] * vec1[i])
    #         down_2 += (vec2[i] * vec2[i])
    #     down = sqrt(down_1) * sqrt(down_2)
    #     return float(up/down)

    def forward(self, q1, q2):

        max_len = max(self.args.kernel_sizes) + 1
        min_len = max_len if max_len % 2 == 0 else max_len + 1
        q1 = pad_sequences(q1, min_len)
        q2 = pad_sequences(q2, min_len)
        
        
        q1 = q1.to(device)
        q2 = q2.to(device)

       
        # 添加Transformer


        q1 = self.embed(q1)  # (seq_len, batch_size, D)
        q2 = self.embed(q2)
        
        # q1=self.mattn(q1,q1,q1)
        # q2=self.mattn(q2,q2,q2)

        # print(q1.data.shape)
        
        # q1=bc.encode([q1])
        # q2=bc.encode([q2])
        
        if self.args.static:
            q1 = Variable(q1)
        q1 = q1.unsqueeze(1)  # (N, Ci, W, D)
        
        # step1
        q1 = [self.relu2(conv(q1)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)
        # step2
        q1 = [i.size(2) * F.max_pool1d(i, i.size(2)).squeeze(2) for i in q1]  # [(N, Co), ...]*len(Ks)
        q1 = [self.relu2(i) for i in q1]
        q1 = torch.cat(q1, 1) # 64 * 300
        
        q1 = self.dropout(q1)  # (N, len(Ks)*Co)
        

  
        

        # q2.data = q2.data.weight.data.copy_(torch.from_numpy(pretrained_weight))
        if self.args.static:
            q2 = Variable(q2)
        q2 = q2.unsqueeze(1)  # (N, Ci, W, D)
        
        q2 = [self.relu2(conv(q2)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)
        q2 = [i.size(2) * F.max_pool1d(i, i.size(2)).squeeze(2) for i in q2]  # [(N, Co), ...]*len(Ks)
        q2 = [self.relu2(i) for i in q2]
        q2 = torch.cat(q2, 1)
        q2 = self.dropout(q2)  # (N, len(Ks)*Co)

 
        
        x = torch.cat((q1, q2), dim=1)
        
        x=x.to(device)
        concatenated_dim = q1.size(1) + q2.size(1)
        self.fc1 = nn.Linear(concatenated_dim, 128).to(device)
         
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        
        return torch.sigmoid(x)


        
        
	# #step3
    #     cos_ans = F.cosine_similarity(q1, q2)
    #     cos_ans=torch.sigmoid(cos_ans)
    #     # cos_ans = nn.functional.pairwise_distance(q1, q2, p=2, eps=1e-06)
    #     # cos_ans = F.relu(cos_ans)
    #     # print(cos_ans.data)
    #     return cos_ans
    



