import torch
from typing import List
import os
from pathlib import Path
from transformers import LongformerModel, LongformerTokenizerFast, BertModel, BertTokenizerFast


def cross_similarity(reps: List[torch.Tensor]):
    """
    Computes the cosine similarity between all pairs of representations in reps
    """
    sims = []
    for i, rep1 in enumerate(reps):
        for j, rep2 in enumerate(reps):
            if i == j:
                continue
            sims.append(torch.cosine_similarity(rep1, rep2))
    return torch.stack(sims)


def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt


def main():
    """
    model_name = 'allenai/longformer-base-4096'
    tokenizer = LongformerTokenizerFast.from_pretrained(model_name)
    model = LongformerModel.from_pretrained(model_name)
    """
    model_name = 'bert-base-uncased'
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)

    job_files = os.listdir('data')
    results = []
    for fn in job_files:
        with open(Path('data') / fn) as f:
            description = f.read()
            tokens = tokenizer(description, return_tensors='pt', padding=True, truncation=True)
            outputs = model(**tokens)
            results.append(outputs.pooler_output)

    print(job_files)
    print(cross_similarity(results))
    print('Done')


if __name__ == '__main__':
    main()
