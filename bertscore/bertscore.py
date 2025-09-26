import sys
from typing import Optional
from collections import Counter, defaultdict
from functools import partial
from itertools import chain
from math import log
from multiprocessing import Pool

import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm.auto import tqdm
from transformers import (AutoModel, AutoTokenizer)

class BERTScore:
    def __init__(self, 
                 model_type: str = "google-bert/bert-base-multilingual-cased", 
                 num_layers: int = 9, 
                 all_layers: bool = False,
                 device: str = "cpu", 
                 use_fast: bool = False, 
                 nthreads: int = 4):
        self.device = device
        self.all_layers = all_layers
        self.model = self.get_model(model_type, num_layers, all_layers).to(device)
        self.tokenizer = self.get_tokenizer(model_type, use_fast=use_fast)
        self.nthreads = nthreads

    @staticmethod
    def sent_encode(tokenizer, sent):
        sent = sent.strip()
        if sent == "":
            return tokenizer.build_inputs_with_special_tokens([])
        else:
            return tokenizer.encode(
                sent,
                add_special_tokens=True,
                max_length=tokenizer.model_max_length,
                truncation=True,
            )

    @staticmethod
    def get_model(model_type: str, num_layers: int, all_layers: bool = False):
        model = AutoModel.from_pretrained(model_type)
        model.eval()

        if not all_layers:
            assert 0 <= num_layers <= len(model.encoder.layer)
            model.encoder.layer = torch.nn.ModuleList(model.encoder.layer[:num_layers])

        return model


    @staticmethod
    def get_tokenizer(model_type, use_fast=False):
        return AutoTokenizer.from_pretrained(model_type, use_fast=use_fast)

    @staticmethod
    def padding(arr, pad_token, dtype=torch.long):
        lens = torch.LongTensor([len(a) for a in arr])
        max_len = int(lens.max().item())
        padded = torch.ones(len(arr), max_len, dtype=dtype) * pad_token
        mask = torch.zeros(len(arr), max_len, dtype=torch.long)
        for i, a in enumerate(arr):
            padded[i, : lens[i]] = torch.tensor(a, dtype=dtype)
            mask[i, : lens[i]] = 1
        return padded, lens, mask

    @staticmethod
    def bert_encode(model, x, attention_mask, all_layers=False):
        model.eval()
        with torch.no_grad():
            out = model(x, attention_mask=attention_mask, output_hidden_states=all_layers)
        if all_layers:
            emb = torch.stack(out[-1], dim=2)
        else:
            emb = out[0]
        return emb

    @staticmethod
    def process(a, tokenizer=None):
        if tokenizer is not None:
            a = BERTScore.sent_encode(tokenizer, a)
        return set(a)

    @staticmethod
    def get_idf_dict(arr, tokenizer, nthreads=4):
        idf_count = Counter()
        num_docs = len(arr)
        process_partial = partial(BERTScore.process, tokenizer=tokenizer)
        if nthreads > 0:
            with Pool(nthreads) as p:
                idf_count.update(chain.from_iterable(p.map(process_partial, arr)))
        else:
            idf_count.update(chain.from_iterable(map(process_partial, arr)))
        idf_dict = defaultdict(lambda: log((num_docs + 1) / (1)))
        idf_dict.update(
            {idx: log((num_docs + 1) / (c + 1)) for (idx, c) in idf_count.items()}
        )
        return idf_dict

    @staticmethod
    def collate_idf(arr, tokenizer, idf_dict, device="cuda:0"):
        arr = [BERTScore.sent_encode(tokenizer, a) for a in arr]
        idf_weights = [[idf_dict[i] for i in a] for a in arr]
        pad_token = tokenizer.pad_token_id
        padded, lens, mask = BERTScore.padding(arr, pad_token, dtype=torch.long)
        padded_idf, _, _ = BERTScore.padding(idf_weights, 0, dtype=torch.float)
        padded = padded.to(device=device)
        mask = mask.to(device=device)
        lens = lens.to(device=device)
        return padded, padded_idf, lens, mask

    @staticmethod
    def get_bert_embedding(all_sens, model, tokenizer, idf_dict,
                           batch_size=-1, device="cuda:0", all_layers=False):
        padded_sens, padded_idf, lens, mask = BERTScore.collate_idf(
            all_sens, tokenizer, idf_dict, device=device
        )
        if batch_size == -1:
            batch_size = len(all_sens)
        embeddings = []
        with torch.no_grad():
            for i in range(0, len(all_sens), batch_size):
                batch_embedding = BERTScore.bert_encode(
                    model,
                    padded_sens[i: i + batch_size],
                    attention_mask=mask[i: i + batch_size],
                    all_layers=all_layers,
                )
                embeddings.append(batch_embedding)
                del batch_embedding
        total_embedding = torch.cat(embeddings, dim=0)
        return total_embedding, mask, padded_idf

    @staticmethod
    def greedy_cos_idf(ref_embedding, ref_masks, ref_idf,
                       hyp_embedding, hyp_masks, hyp_idf, all_layers=False):
        ref_embedding.div_(torch.norm(ref_embedding, dim=-1).unsqueeze(-1))
        hyp_embedding.div_(torch.norm(hyp_embedding, dim=-1).unsqueeze(-1))
        if all_layers:
            B, _, L, D = hyp_embedding.size()
            hyp_embedding = (
                hyp_embedding.transpose(1, 2)
                .transpose(0, 1)
                .contiguous()
                .view(L * B, hyp_embedding.size(1), D)
            )
            ref_embedding = (
                ref_embedding.transpose(1, 2)
                .transpose(0, 1)
                .contiguous()
                .view(L * B, ref_embedding.size(1), D)
            )
        batch_size = ref_embedding.size(0)
        sim = torch.bmm(hyp_embedding, ref_embedding.transpose(1, 2))
        masks = torch.bmm(hyp_masks.unsqueeze(2).float(),
                          ref_masks.unsqueeze(1).float())
        if all_layers:
            masks = masks.unsqueeze(0).expand(L, -1, -1, -1).contiguous().view_as(sim)
        else:
            masks = masks.expand(batch_size, -1, -1).contiguous().view_as(sim)
        masks = masks.float().to(sim.device)
        sim = sim * masks
        word_precision = sim.max(dim=2)[0]
        word_recall = sim.max(dim=1)[0]
        hyp_idf.div_(hyp_idf.sum(dim=1, keepdim=True))
        ref_idf.div_(ref_idf.sum(dim=1, keepdim=True))
        precision_scale = hyp_idf.to(word_precision.device)
        recall_scale = ref_idf.to(word_recall.device)
        if all_layers:
            precision_scale = (
                precision_scale.unsqueeze(0)
                .expand(L, B, -1)
                .contiguous()
                .view_as(word_precision)
            )
            recall_scale = (
                recall_scale.unsqueeze(0)
                .expand(L, B, -1)
                .contiguous()
                .view_as(word_recall)
            )
        P = (word_precision * precision_scale).sum(dim=1)
        R = (word_recall * recall_scale).sum(dim=1)
        F = 2 * P * R / (P + R)
        hyp_zero_mask = hyp_masks.sum(dim=1).eq(2)
        ref_zero_mask = ref_masks.sum(dim=1).eq(2)
        if all_layers:
            P = P.view(L, B)
            R = R.view(L, B)
            F = F.view(L, B)
        if torch.any(hyp_zero_mask):
            print(
                "Warning: Empty candidate sentence detected; setting raw BERTscores to 0.",
                file=sys.stderr,
            )
            P = P.masked_fill(hyp_zero_mask, 0.0)
            R = R.masked_fill(hyp_zero_mask, 0.0)
        if torch.any(ref_zero_mask):
            print(
                "Warning: Empty reference sentence detected; setting raw BERTScores to 0.",
                file=sys.stderr,
            )
            P = P.masked_fill(ref_zero_mask, 0.0)
            R = R.masked_fill(ref_zero_mask, 0.0)
        F = F.masked_fill(torch.isnan(F), 0.0)
        return P, R, F

    @staticmethod
    def bert_cos_score_idf(model, gts_list, hyps, tokenizer, idf_dict,
                           verbose=False, batch_size=64,
                           device="cuda:0", all_layers=False):
        preds = []

        def dedup_and_sort(l):
            return sorted(list(set(l)), key=lambda x: len(x.split(" ")), reverse=True)

        sentences = dedup_and_sort(gts_list + hyps)
        embs = []
        iter_range = range(0, len(sentences), batch_size)
        if verbose:
            print("computing bert embedding.")
            iter_range = tqdm(iter_range)
        stats_dict = dict()
        for batch_start in iter_range:
            sen_batch = sentences[batch_start: batch_start + batch_size]
            embs, masks, padded_idf = BERTScore.get_bert_embedding(
                sen_batch, model, tokenizer, idf_dict,
                device=device, all_layers=all_layers
            )
            embs = embs.cpu()
            masks = masks.cpu()
            padded_idf = padded_idf.cpu()
            for i, sen in enumerate(sen_batch):
                sequence_len = masks[i].sum().item()
                emb = embs[i, :sequence_len]
                idf = padded_idf[i, :sequence_len]
                stats_dict[sen] = (emb, idf)

        def pad_batch_stats(sen_batch, stats_dict, device):
            stats = [stats_dict[s] for s in sen_batch]
            emb, idf = zip(*stats)
            emb = [e.to(device) for e in emb]
            idf = [i.to(device) for i in idf]
            lens = [e.size(0) for e in emb]
            emb_pad = pad_sequence(emb, batch_first=True, padding_value=2.0)
            idf_pad = pad_sequence(idf, batch_first=True)

            def length_to_mask(lens):
                lens = torch.tensor(lens, dtype=torch.long)
                max_len = int(max(lens))
                base = torch.arange(max_len, dtype=torch.long).expand(len(lens), max_len)
                return base < lens.unsqueeze(1)

            pad_mask = length_to_mask(lens).to(device)
            return emb_pad, pad_mask, idf_pad

        device = next(model.parameters()).device
        iter_range = range(0, len(gts_list), batch_size)
        if verbose:
            print("computing greedy matching.")
            iter_range = tqdm(iter_range)
        with torch.no_grad():
            for batch_start in iter_range:
                batch_gts_list = gts_list[batch_start: batch_start + batch_size]
                batch_hyps = hyps[batch_start: batch_start + batch_size]
                ref_stats = pad_batch_stats(batch_gts_list, stats_dict, device)
                hyp_stats = pad_batch_stats(batch_hyps, stats_dict, device)
                P, R, F1 = BERTScore.greedy_cos_idf(*ref_stats, *hyp_stats, all_layers)
                preds.append(torch.stack((P, R, F1), dim=-1).cpu())
        preds = torch.cat(preds, dim=1 if all_layers else 0)
        return preds

    def compute_score(self, cands, refs, batch_size=64, idf=False):
        """
        BERTScore for dataset.

        Args:
            cands (list[str]): candidate sentences
            refs (list[str] or list[list[str]]): reference sentences
                If list[list[str]], each candidate has multiple refs.
            batch_size (int): batch size for encoding
            idf (bool): whether to use IDF weighting

        Returns:
            (P, R, F): tensors of shape (N,)
        """
        assert len(cands) == len(refs), "Mismatched number of candidates and references"

        # flatten if multiple refs per cand
        ref_group_boundaries = None
        if not isinstance(refs[0], str):
            ref_group_boundaries = []
            ori_cands, ori_refs = cands, refs
            cands, refs = [], []
            count = 0
            for cand, ref_group in zip(ori_cands, ori_refs):
                cands += [cand] * len(ref_group)
                refs += ref_group
                ref_group_boundaries.append((count, count + len(ref_group)))
                count += len(ref_group)

        # IDF
        if not idf:
            idf_dict = defaultdict(lambda: 1.0)
            idf_dict[self.tokenizer.sep_token_id] = 0
            idf_dict[self.tokenizer.cls_token_id] = 0
        else:
            idf_dict = self.get_idf_dict(refs, self.tokenizer, nthreads=self.nthreads)

        # compute
        all_preds = self.bert_cos_score_idf(
            self.model,
            refs,
            cands,
            self.tokenizer,
            idf_dict,
            batch_size=batch_size,
            device=self.device,
            all_layers=self.all_layers,
        ).cpu()

        # handle multi-ref → take max over refs
        if ref_group_boundaries is not None:
            max_preds = []
            for beg, end in ref_group_boundaries:
                max_preds.append(all_preds[beg:end].max(dim=0)[0])
            all_preds = torch.stack(max_preds, dim=0)

        return all_preds[..., 0], all_preds[..., 1], all_preds[..., 2]  # P, R, F


if __name__ == "__main__":
    scorer = BERTScore()
    cands = ["Cậu ấy đang ngồi trên một chiếc xe ô tô.", "Một con mèo nằm."]
    refs = [
        ["Anh ấy ngồi trên xe.", "Có một người ngồi trên xe ô tô."],
        ["Một con mèo đang ngủ."]
    ]
    score = scorer.compute_score(cands, refs)
    print(score)