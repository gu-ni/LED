import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
import math
from TAAE.noise import noisy

def reparameterize(mu, logvar):
    std = torch.exp(0.5*logvar)
    eps = torch.randn_like(std)
    return eps.mul(std).add_(mu)

def log_prob(z, mu, logvar):
    var = torch.exp(logvar)
    logp = - (z-mu)**2 / (2*var) - torch.log(2*np.pi*var) / 2
    return logp.sum(dim=1)

def loss_kl(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / len(mu)


class TextModel(nn.Module):
    """Container module with word embedding and projection layers"""

    def __init__(self, vocab, args, initrange=0.1):
        super().__init__()
        self.vocab = vocab
        self.args = args
        self.embed = nn.Embedding(vocab.size, args.dim_emb)
        self.proj = nn.Linear(args.dim_emb, vocab.size)

        self.embed.weight.data.uniform_(-initrange, initrange)
        self.proj.bias.data.zero_()
        self.proj.weight.data.uniform_(-initrange, initrange)

class PositionalEncoding(nn.Module):
    def __init__(self, args,  maxlen: int = 128):
        super(PositionalEncoding, self).__init__()
        self.args = args

        den = torch.exp(- torch.arange(0, args.dim_emb, 2)* math.log(10000) / args.dim_emb)
        # den = [dim_emb / 2]
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        # pos = [max_len, 1]
        pos_embedding = torch.zeros((maxlen, args.dim_emb))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(args.transformer_dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

# ------------- Transformer Based AutoEncoder --------------
class TAE(TextModel):
    def __init__(self, vocab, args) :
        super().__init__(vocab,args)
        self.args = args
        self.Encoder = TransformerEncoder(
            TransformerEncoderLayer(d_model=args.dim_emb, nhead=args.n_heads, dim_feedforward=args.dim_feedforward, dropout=args.transformer_dropout),
            num_layers=args.nlayers
        )

        self.Decoder = TransformerDecoder(
            TransformerDecoderLayer(d_model=args.dim_emb, nhead=args.n_heads, dim_feedforward=args.dim_feedforward, dropout=args.transformer_dropout),
            num_layers=args.nlayers
        )

        self.drop = nn.Dropout(args.transformer_dropout)
        self.position_encoding = PositionalEncoding(args)
        self.h2mu = nn.Linear(args.dim_emb, args.dim_z)
        self.h2logvar = nn.Linear(args.dim_emb, args.dim_z)
        self.z2emb = nn.Linear(args.dim_z, args.dim_emb)
        self.opt = optim.Adam(self.parameters(), lr=args.lr, betas=(0.5, 0.999))

    def encode(self, input):
        inputs = self.position_encoding(self.embed(input))
        encoded_token = self.Encoder(inputs)
        return self.h2mu(encoded_token), self.h2logvar(encoded_token), inputs

    def decode(self, z, input, trg_mask):
        _input = self.position_encoding(self.embed(input))
        memory = self.drop(self.z2emb(z))
        output = self.Decoder(_input, memory, trg_mask)
        logits = self.proj(output.view(-1, output.size(-1)))
        return logits.view(output.size(0), output.size(1), -1)
    
    def forward(self, input, is_train=False):
        _input = noisy(self.vocab, input, *self.args.noise) if is_train else input
        mu, logvar, _ = self.encode(_input)
        z = reparameterize(mu, logvar)
        trg_mask = nn.Transformer.generate_square_subsequent_mask(input.size(0)).to(input.device)
        logits= self.decode(z, input, trg_mask)
        return mu, logvar, z, logits
    
    def loss_rec(self, logits, targets):
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
            ignore_index=self.vocab.pad, reduction='none').view(targets.size())
        return loss.sum(dim=0)

    def loss(self, losses):
        return losses['rec']

    def autoenc(self, inputs, targets, is_train=False):
        _, _, _, logits = self(inputs, is_train)
        return {'rec': self.loss_rec(logits, targets).mean()}    
        
    def step(self, losses):
        self.opt.zero_grad()
        losses['loss'].backward()
        self.opt.step()
    
    def nll_is(self, inputs, targets, m):
        """compute negative log-likelihood by importance sampling:
           p(x;theta) = E_{q(z|x;phi)}[p(z)p(x|z;theta)/q(z|x;phi)]
        """
        mu, logvar, _ = self.encode(inputs)
        tmp = []
        for _ in range(m):
            z = reparameterize(mu, logvar)
            logits, _ = self.decode(z, inputs)
            v = log_prob(z, torch.zeros_like(z), torch.zeros_like(z)) - \
                self.loss_rec(logits, targets) - log_prob(z, mu, logvar)
            tmp.append(v.unsqueeze(-1))
        ll_is = torch.logsumexp(torch.cat(tmp, 1), 1) - np.log(m)
        return -ll_is
    
    def generate(self, z, max_len, alg):
        assert alg in ['greedy' , 'sample' , 'top5']
        sents = []
        input = torch.zeros(1, len(z), dtype=torch.long, device=z.device).fill_(self.vocab.go)
        hidden = None
        for l in range(max_len):
            sents.append(input)
            logits, hidden = self.decode(z, input, hidden)
            if alg == 'greedy':
                input = logits.argmax(dim=-1)
            elif alg == 'sample':
                input = torch.multinomial(logits.squeeze(dim=0).exp(), num_samples=1).t()
            elif alg == 'top5':
                not_top5_indices=logits.topk(logits.shape[-1]-5,dim=2,largest=False).indices
                logits_exp=logits.exp()
                logits_exp[:,:,not_top5_indices]=0.
                input = torch.multinomial(logits_exp.squeeze(dim=0), num_samples=1).t()
    
    
class AAE(TAE):
    """Adversarial Auto-Encoder"""

    def __init__(self, vocab, args):
        super().__init__(vocab, args)
        # dim_z = 128 -> dim_d = 512 -> 1
        self.D = nn.Sequential(nn.Linear(args.dim_z, args.dim_d), nn.ReLU(),
            nn.Linear(args.dim_d, 1), nn.Sigmoid())
        self.optD = optim.Adam(self.D.parameters(), lr=args.lr, betas=(0.5, 0.999))

    def loss_adv(self, z):
        z = z.permute(1,0,2)
        zn = torch.randn_like(z)
        zeros = torch.zeros(z.size(0), z.size(1), 1, device=z.device)
        ones = torch.ones(z.size(0), z.size(1), 1, device=z.device)
        loss_d = F.binary_cross_entropy(self.D(z.detach()), zeros) + \
            F.binary_cross_entropy(self.D(zn), ones)
        loss_g = F.binary_cross_entropy(self.D(z), ones)
        return loss_d, loss_g

    def loss(self, losses):
        return losses['rec'] + self.args.lambda_adv * losses['adv'] + \
            self.args.lambda_p * losses['|lvar|']

    def autoenc(self, inputs, targets, is_train=False):
        _, logvar, z, logits = self(inputs, is_train)
        loss_d, adv = self.loss_adv(z)
        return {'rec': self.loss_rec(logits, targets).mean(),
                'adv': adv,
                
                '|lvar|': logvar.abs().sum(dim=1).mean(),
                'loss_d': loss_d}

    def step(self, losses):
        super().step(losses)

        self.optD.zero_grad()
        losses['loss_d'].backward()
        self.optD.step()




        

