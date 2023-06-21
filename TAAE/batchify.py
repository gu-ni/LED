import torch

def get_batch(x, vocab, device):
    go_x, x_eos = [], []
    max_len = 64
    for s in x:
        # vocab상에서 문장 내 단어의 index
        s_idx = [vocab.word2idx[w] if w in vocab.word2idx else vocab.unk for w in s]
        padding = [vocab.pad] * (max_len - len(s))
        # 맨 앞에 go 토큰 추가, 뒤에 pad 토큰 추가
        go_x.append([vocab.go] + s_idx + padding)
        # 문장 맨 뒤에 eos 토큰 추가하고 나머지 pad 토큰 추가
        x_eos.append(s_idx + [vocab.eos] + padding)
    # transpose해서 반환
    return torch.LongTensor(go_x).t().contiguous().to(device), \
           torch.LongTensor(x_eos).t().contiguous().to(device)  # time * batch

def get_batches(data, vocab, batch_size, device):
    
    # data : 학습 문장 
    batches = []
    
    i = 0
    for i in range(0, len(data), batch_size):
        batches.append(get_batch(data[i:i+batch_size], vocab, device))
    return batches