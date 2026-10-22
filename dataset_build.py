#!/usr/bin/env python3

import os, re, json, argparse, html, ast, random, time
from typing import List, Tuple, Dict, Any
from collections import defaultdict, deque
import numpy as np
import pandas as pd
from tqdm import tqdm


# graph helpers
def _build_undirected(edges):
    UG = defaultdict(set)
    for (u,v,_,_,_) in edges:
        UG[u].add(v); UG[v].add(u)
    return UG


def _connected_components(nodes, edges):
    UG = _build_undirected(edges)
    seen=set()
    comps=[]
    for n in nodes:
        if n in seen: continue
        q=deque([n]); seen.add(n); comp=[]
        while q:
            x=q.popleft(); comp.append(x)
            for y in UG.get(x,[]):
                if y not in seen:
                    seen.add(y); q.append(y)
        comps.append(comp)
    return comps


def _topo_order(sub_nodes, edges):
    # kahn topo; if cycle, fallback to sorted
    indeg = {n:0 for n in sub_nodes}
    out = defaultdict(list)
    for (u,v,_,_,_) in edges:
        if u in indeg and v in indeg:
            out[u].append(v)
            indeg[v]+=1
    q=deque([n for n,d in indeg.items() if d==0])
    order=[]
    while q:
        x=q.popleft()
        order.append(x)
        for y in out.get(x,[]):
            indeg[y]-=1
            if indeg[y]==0:
                q.append(y)
    if len(order)!=len(sub_nodes):
        return sorted(sub_nodes)
    return order


def components_to_paragraph_rows(nodes_final, edges_mapped, openai_model, smooth_temp, seed=42):
    # build paragraph variants per connected component
    comps = _connected_components(nodes_final, edges_mapped)
    rng = random.Random(seed)
    rows=[]
    for cid, comp in enumerate(comps):
        sub_nodes = sorted(comp)
        order = _topo_order(sub_nodes, edges_mapped)

        all_edges=set()
        for (u,v,lab,_,_) in edges_mapped:
            if u in comp and v in comp:
                all_edges.add((u,v))

        plain_raw, role_raw, rr_raw = build_paragraph_variants(order, seed=seed, offset=cid)
        plain_s = gpt_smooth_paragraph_safe(plain_raw, model=openai_model, temperature=smooth_temp)
        role_s  = gpt_smooth_paragraph_safe(role_raw,  model=openai_model, temperature=smooth_temp)
        rr_s    = gpt_smooth_paragraph_safe(rr_raw,    model=openai_model, temperature=smooth_temp)

        indeg = defaultdict(int); outdeg = defaultdict(int)
        for (u,v) in all_edges:
            outdeg[u]+=1; indeg[v]+=1

        roles = roles_for_seq(order)

        graph_json = {
            "nodes": [{"id": n, "role": roles[n], "indegree": int(indeg.get(n,0)), "outdegree": int(outdeg.get(n,0))} for n in sorted(comp)],
            "edges": [{"source": u, "target": v} for (u,v) in sorted(all_edges)]
        }

        rows.append({
            "paragraph_id": cid,
            "merged_paragraph_count": 1,
            "nodes_seq_json": json.dumps(list(order), ensure_ascii=False),  # for reference (not a path)
            "graph_json": json.dumps(graph_json, ensure_ascii=False),
            "paragraph_plain_raw": plain_raw, "paragraph_plain_smoothed": plain_s,
            "paragraph_role_raw":  role_raw,  "paragraph_role_smoothed":  role_s,
            "paragraph_rr_raw":    rr_raw,    "paragraph_rr_smoothed":    rr_s,
        })
    return rows


# regex and normalization
ARG0_RE = re.compile(r"<ARG0>(.*?)</ARG0>", re.I | re.S)
ARG1_RE = re.compile(r"<ARG1>(.*?)</ARG1>", re.I | re.S)
SIGX_RE = re.compile(r"<SIGX>(.*?)</SIGX>", re.I | re.S)
SIG_TAG_RE = re.compile(r"</?sig\d+>", re.I)

PUNCT_RE   = re.compile(r"[^\w\s\-'/]")
SP_RE      = re.compile(r"\s+")
DASH_RE    = re.compile(r"[-‐-‒–—―]+")
QUOTE_RE   = re.compile(r"[“”‘’]")
HTML_TAG_RE= re.compile(r"<[^>]+>")


def normalize_node_text(s: str) -> str:
    if s is None: return ""
    s = str(s)
    s = html.unescape(s)
    s = QUOTE_RE.sub("'", s)
    s = DASH_RE.sub("-", s)
    s = HTML_TAG_RE.sub(" ", s)
    s = PUNCT_RE.sub(" ", s)
    s = SP_RE.sub(" ", s).strip()
    return s


def _parse_multi_marked(val):
    # parse list-like strings safely
    if val is None: return []
    if isinstance(val, (list,tuple)): return list(val)
    s = str(val).strip()
    if not s: return []
    try:
        if s.startswith("[") and s.endswith("]"):
            x = ast.literal_eval(s)
            if isinstance(x, list): return x
    except Exception:
        pass
    return [s]


def _safe_list(x):
    if x is None: return []
    if isinstance(x, list): return x
    return [x]


def _segment_texts(t: str) -> list[str]:
    if not t: return []
    t = str(t)
    t = SP_RE.sub(" ", t).strip()
    if not t: return []
    # naive sentence split
    parts = re.split(r"(?<=[.!?])\s+", t)
    parts = [p.strip() for p in parts if p.strip()]
    return parts


def extract_pairs_from_row(row, text_col):
    # parse one row into (nodes, edges)
    text = row.get(text_col, "")
    texts = _safe_list(text) if isinstance(text, list) else [text]
    nodes=set()
    edges=set()

    for t in texts:
        if not t: continue
        for seg in _segment_texts(t):
            m0 = ARG0_RE.search(seg)
            m1 = ARG1_RE.search(seg)
            if m0 and m1:
                u = normalize_node_text(m0.group(1))
                v = normalize_node_text(m1.group(1))
                if u and v:
                    nodes.add(u); nodes.add(v)
                    edges.add((u,v))
            else:
                # fallback sig parsing
                sx = SIGX_RE.search(seg)
                if sx:
                    sxx = SIG_TAG_RE.sub(" ", sx.group(1))
                    sxx = SP_RE.sub(" ", sxx).strip()
                    toks = [normalize_node_text(z) for z in re.split(r"\s*[,;]\s*", sxx) if z.strip()]
                    if len(toks) >= 2:
                        u, v = toks[0], toks[1]
                        if u and v:
                            nodes.add(u); nodes.add(v)
                            edges.add((u,v))

    return sorted(nodes), sorted(edges)


def build_nodes_edges(df: pd.DataFrame, text_col: str):
    all_nodes=set()
    all_edges=set()
    for _, row in tqdm(df.iterrows(), total=len(df), desc="extract"):
        nodes, edges = extract_pairs_from_row(row, text_col)
        for n in nodes: all_nodes.add(n)
        for (u,v) in edges: all_edges.add((u,v))
    return sorted(all_nodes), sorted(all_edges)


def sbert_embed(nodes: List[str], model_name: str) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(model_name)
    X = m.encode(nodes, normalize_embeddings=True, show_progress_bar=True)
    return np.asarray(X, dtype=np.float32)


def apply_abtt(embs: np.ndarray, abtt_dim: int) -> np.ndarray:
    # remove top principal components
    if abtt_dim <= 0: return embs
    X = embs - embs.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    V = Vt[:abtt_dim]
    X2 = X - X @ V.T @ V
    X2 = X2 / (np.linalg.norm(X2, axis=1, keepdims=True) + 1e-12)
    return X2.astype(np.float32)


def faiss_topk_search(X: np.ndarray, k: int):
    import faiss
    d = X.shape[1]
    idx = faiss.IndexFlatIP(d)
    idx.add(X)
    sim, nn = idx.search(X, k)
    return sim, nn


class UnionFind:
    def __init__(self, xs):
        self.parent={x:x for x in xs}
        self.rank={x:0 for x in xs}

    def find(self, x):
        while self.parent[x]!=x:
            self.parent[x]=self.parent[self.parent[x]]
            x=self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra==rb: return
        if self.rank[ra]<self.rank[rb]:
            self.parent[ra]=rb
        elif self.rank[ra]>self.rank[rb]:
            self.parent[rb]=ra
        else:
            self.parent[rb]=ra
            self.rank[ra]+=1


def dedup_nodes(nodes: List[str], embs: np.ndarray, thr: float, keep_rule="shorter"):
    # near-duplicate merge by cosine threshold
    uf = UnionFind(range(len(nodes)))
    sim, nn = faiss_topk_search(embs, k=min(50, len(nodes)))
    for i in range(len(nodes)):
        for jpos in range(1, nn.shape[1]):
            j = int(nn[i, jpos])
            if j<0 or j>=len(nodes): continue
            if sim[i, jpos] >= thr:
                uf.union(i, j)

    groups=defaultdict(list)
    for i in range(len(nodes)):
        groups[uf.find(i)].append(i)

    rep = {}
    for g, idxs in groups.items():
        cands = [(nodes[i], i) for i in idxs]
        if keep_rule=="shorter":
            cands.sort(key=lambda x:(len(x[0]), x[0]))
        else:
            cands.sort(key=lambda x:(x[0], len(x[0])))
        rep_idx = cands[0][1]
        for i in idxs:
            rep[i]=rep_idx

    new_nodes=[]
    old2new={}
    rep2new={}
    for i in range(len(nodes)):
        ri = rep[i]
        if ri not in rep2new:
            rep2new[ri]=len(new_nodes)
            new_nodes.append(nodes[ri])
        old2new[i]=rep2new[ri]

    return new_nodes, old2new


def remap_edges(edges: List[Tuple[str,str]], nodes: List[str], old2new: Dict[int,int]):
    node2idx = {n:i for i,n in enumerate(nodes)}
    out=set()
    for (u,v) in edges:
        if u not in node2idx or v not in node2idx: 
            continue
        iu, iv = node2idx[u], node2idx[v]
        nu, nv = old2new[iu], old2new[iv]
        if nu==nv: 
            continue
        out.add((nu, nv))
    inv_nodes = None
    return sorted(out)


# paragraph building
ROLE_ORDER = ["source","cause","mediator","outcome","confounder","intermediate"]
VERBS = ["causes", "influences", "affects", "impacts", "drives", "changes"]


def roles_for_seq(seq: List[str]) -> Dict[str,str]:
    if not seq: return {}
    r={}
    if len(seq)==1:
        r[seq[0]]="intermediate"
        return r
    r[seq[0]]="source"
    r[seq[-1]]="outcome"
    for n in seq[1:-1]:
        r[n]="mediator"
    return r


def neighbors(seq: List[str]) -> Dict[str, Tuple[str,str]]:
    nb={}
    for i,n in enumerate(seq):
        left = seq[i-1] if i>0 else ""
        right= seq[i+1] if i<len(seq)-1 else ""
        nb[n]=(left,right)
    return nb


def edges_from_nodes(seq: List[str]):
    if not seq or len(seq)<2: return []
    return [(seq[i], seq[i+1]) for i in range(len(seq)-1)]


def role_tag(node: str, role: str) -> str:
    return f"{node} ({role})"


def relation_tag(u: str, v: str) -> str:
    return f"{u} -> {v}"


def smooth_text_basic(s: str) -> str:
    if not s: return ""
    s = SP_RE.sub(" ", s).strip()
    # ensure spacing around punctuation
    s = re.sub(r"\s+([,.!?;:])", r"\1", s)
    s = re.sub(r"([,.!?;:])([A-Za-z0-9])", r"\1 \2", s)
    return s


def gpt_smooth_paragraph_safe(s: str, model: str, temperature: float=0.0):
    # optional gpt smoothing
    if not s: return ""
    api_key = os.environ.get("OPENAI_API_KEY","")
    if not api_key:
        return smooth_text_basic(s)

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        prompt = (
            "rewrite the paragraph for fluency without changing meaning.\n"
            "keep it concise. keep factual content.\n\n"
            f"paragraph:\n{s}\n"
        )
        rsp = client.chat.completions.create(
            model=model,
            messages=[{"role":"user","content":prompt}],
            temperature=temperature,
        )
        out = rsp.choices[0].message.content or ""
        out = out.strip()
        return out if out else smooth_text_basic(s)
    except Exception:
        return smooth_text_basic(s)


def build_paragraph_variants(seq: List[str], seed=42, offset=0):
    """build plain/role/role+rel paragraphs from a seq."""
    if not seq or len(seq)<2:
        return "", "", ""
    r = roles_for_seq(seq)
    nb = neighbors(seq)
    es = edges_from_nodes(seq)

    # plain
    plain_sents = [f"{u} {VERBS[(offset+i)%len(VERBS)]} {v}." for i,(u,v) in enumerate(es)]
    # role
    role_sents = [f"{role_tag(u, r.get(u,'intermediate'))} {VERBS[(offset+i)%len(VERBS)]} {role_tag(v, r.get(v,'intermediate'))}."
                  for i,(u,v) in enumerate(es)]
    # role+rel
    rr_sents = [f"{role_tag(u, r.get(u,'intermediate'))} {VERBS[(offset+i)%len(VERBS)]} {role_tag(v, r.get(v,'intermediate'))} ({relation_tag(u,v)})."
                for i,(u,v) in enumerate(es)]

    return " ".join(plain_sents), " ".join(role_sents), " ".join(rr_sents)


# cli
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--text_col", default="text")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--sbert_model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--abtt_dim", type=int, default=0)
    ap.add_argument("--dedup_thr", type=float, default=0.92)
    ap.add_argument("--openai_model", default="gpt-4o-mini")
    ap.add_argument("--smooth_temp", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    nodes, edges = build_nodes_edges(df, args.text_col)

    if len(nodes)==0:
        pd.DataFrame([]).to_csv(args.out_csv, index=False)
        return

    X = sbert_embed(nodes, args.sbert_model)
    X = apply_abtt(X, args.abtt_dim)

    nodes2, old2new = dedup_nodes(nodes, X, args.dedup_thr)
    edges2 = remap_edges(edges, nodes, old2new)

    # rebuild node labels for edges2
    edges_mapped=[]
    for (iu,iv) in edges2:
        u = nodes2[iu]
        v = nodes2[iv]
        edges_mapped.append((u,v,"",0,0))

    rows = components_to_paragraph_rows(nodes2, edges_mapped, args.openai_model, args.smooth_temp, seed=args.seed)
    out = pd.DataFrame(rows)
    out.to_csv(args.out_csv, index=False)


if __name__ == "__main__":
    main()
