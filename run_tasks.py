#!/usr/bin/env python3

import os, re, json, argparse, time, random, asyncio
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import pandas as pd
from tqdm import tqdm


def ensure_dir(p): 
    os.makedirs(p, exist_ok=True); 
    return p


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows: 
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line: 
                yield json.loads(line)


def normalize_node(s):
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def parse_graph_text(s: str) -> Tuple[List[str], List[Tuple[str,str]]]:
    # parse edges from text like "a -> b"
    nodes=set()
    edges=set()
    if not s:
        return [], []

    lines = [x.strip() for x in str(s).splitlines() if x.strip()]
    for e in lines:
        u=v=None
        if "->" in e:
            parts = e.split("->", 1)
            u, v = parts[0].strip(), parts[1].strip()
        elif "=>" in e:
            parts = e.split("=>", 1)
            u, v = parts[0].strip(), parts[1].strip()
        elif "-" in e and ">" in e:
            parts = re.split(r"\s*(?:[-=]>|=>|->|causes?|leads to|results in)\s*", e.strip(), maxsplit=1, flags=re.IGNORECASE)
            if len(parts) == 2:
                u, v = parts[0], parts[1]

        u = normalize_node(str(u))
        v = normalize_node(str(v))
        if u and v:
            nodes.add(u); nodes.add(v)
            edges.add((u, v))

    return sorted(nodes), sorted(edges)


# transcript parsing
def strip_transcript(s: str) -> str:
    """extract the last assistant reply from a mixed transcript."""
    if not s:
        return ""
    t = str(s).strip()
    if not t:
        return ""

    # keep the last assistant segment if present
    parts = re.split(r"(?im)^\s*(assistant|user|system)\s*:\s*", t)
    if len(parts) <= 1:
        return t

    last_role = None
    last_text = ""
    for i in range(1, len(parts), 2):
        role = parts[i].strip().lower()
        text = parts[i+1] if i+1 < len(parts) else ""
        if role == "assistant":
            last_role = role
            last_text = text

    return last_text.strip() if last_role == "assistant" else t


def safe_get(d, k, default=None):
    try:
        return d.get(k, default)
    except Exception:
        return default


# model clients
class BaseClient:
    def __init__(self, provider: str, model: str, **kwargs):
        self.provider = provider
        self.model = model
        self.kwargs = kwargs

    async def acomplete(self, prompt: str, **kwargs) -> str:
        raise NotImplementedError

    def complete(self, prompt: str, **kwargs) -> str:
        return asyncio.run(self.acomplete(prompt, **kwargs))


class HFClient(BaseClient):
    def __init__(self, provider: str, model: str, **kwargs):
        super().__init__(provider, model, **kwargs)
        self._init_done = False
        self.tokenizer = None
        self.model_obj = None

    def _init(self):
        if self._init_done:
            return
        from transformers import AutoTokenizer, AutoModelForCausalLM
        dtype = os.environ.get("HF_DTYPE", "float16").lower()
        torch_dtype = None
        try:
            import torch
            if dtype == "bfloat16":
                torch_dtype = torch.bfloat16
            elif dtype == "float16":
                torch_dtype = torch.float16
            else:
                torch_dtype = None
        except Exception:
            torch_dtype = None

        self.tokenizer = AutoTokenizer.from_pretrained(self.model)
        self.model_obj = AutoModelForCausalLM.from_pretrained(
            self.model,
            device_map="auto",
            torch_dtype=torch_dtype,
        )
        self._init_done = True

    async def acomplete(self, prompt: str, **kwargs) -> str:
        self._init()
        import torch
        max_new_tokens = int(os.environ.get("HF_MAX_NEW_TOKENS", "256"))
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model_obj.device)
        with torch.no_grad():
            out = self.model_obj.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        return strip_transcript(text) if text else ""


class HFLlama3InstructClient(HFClient):
    pass


class GroqClient(BaseClient):
    async def acomplete(self, prompt: str, **kwargs) -> str:
        from groq import Groq
        api_key = os.environ.get("GROQ_API_KEY", "")
        client = Groq(api_key=api_key)
        rsp = client.chat.completions.create(
            model=self.model,
            messages=[{"role":"user","content":prompt}],
            temperature=kwargs.get("temperature", 0.0),
        )
        return (rsp.choices[0].message.content or "").strip()


class LocalHFChatClient(BaseClient):
    def __init__(self, provider: str, model: str, **kwargs):
        super().__init__(provider, model, **kwargs)
        self._init_done = False
        self.tokenizer = None
        self.model_obj = None

    def _init(self):
        if self._init_done:
            return
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        gpu_id = int(os.environ.get("LOCAL_GPU_ID", "0"))
        dtype = os.environ.get("LOCAL_DTYPE", "float16").lower()
        torch_dtype = torch.float16 if dtype == "float16" else torch.bfloat16
        attn = os.environ.get("LOCAL_ATTN", "sdpa").lower()

        self.tokenizer = AutoTokenizer.from_pretrained(self.model, use_fast=True)
        self.model_obj = AutoModelForCausalLM.from_pretrained(
            self.model,
            torch_dtype=torch_dtype,
            device_map={"": gpu_id},
            attn_implementation=attn,
        )
        self._init_done = True

    async def acomplete(self, prompt: str, **kwargs) -> str:
        self._init()
        import torch
        max_new_tokens = int(os.environ.get("LOCAL_MAX_NEW_TOKENS", "256"))
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model_obj.device)
        with torch.no_grad():
            out = self.model_obj.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        return strip_transcript(text) if text else ""


class LocalLlama3InstructClient(BaseClient):
    """
    Local Transformers runner for Llama-3 Instruct on GPU (CUDA), FP16.
    - deterministic: do_sample=False (no temperature/top_p/top_k)
    Env:
      LOCAL_MAX_NEW_TOKENS=256
      LOCAL_DTYPE=float16|bfloat16 (default float16)
      LOCAL_GPU_ID=0
      LOCAL_ATTN=sdpa|eager (default sdpa)
    """
    def __init__(self, provider: str, model: str, **kwargs):
        super().__init__(provider, model, **kwargs)
        self.inner = LocalHFChatClient(provider, model, **kwargs)

    async def acomplete(self, prompt: str, **kwargs) -> str:
        return await self.inner.acomplete(prompt, **kwargs)


class GeminiClient(BaseClient):
    async def acomplete(self, prompt: str, **kwargs) -> str:
        import google.generativeai as genai
        api_key = os.environ.get("GEMINI_API_KEY", "")
        genai.configure(api_key=api_key)
        m = genai.GenerativeModel(self.model)
        rsp = m.generate_content(prompt)
        return (getattr(rsp, "text", "") or "").strip()


class OpenAIClient(BaseClient):
    async def acomplete(self, prompt: str, **kwargs) -> str:
        from openai import OpenAI
        api_key = os.environ.get("OPENAI_API_KEY", "")
        client = OpenAI(api_key=api_key)
        rsp = client.chat.completions.create(
            model=self.model,
            messages=[{"role":"user","content":prompt}],
            temperature=kwargs.get("temperature", 0.0),
        )
        return (rsp.choices[0].message.content or "").strip()


class XAIGrokClient(BaseClient):
    async def acomplete(self, prompt: str, **kwargs) -> str:
        from openai import OpenAI
        api_key = os.environ.get("XAI_API_KEY", "")
        base_url = os.environ.get("XAI_BASE_URL", "https://api.x.ai/v1")
        client = OpenAI(api_key=api_key, base_url=base_url)
        rsp = client.chat.completions.create(
            model=self.model,
            messages=[{"role":"user","content":prompt}],
            temperature=kwargs.get("temperature", 0.0),
        )
        return (rsp.choices[0].message.content or "").strip()


class AnthropicClient(BaseClient):
    async def acomplete(self, prompt: str, **kwargs) -> str:
        import anthropic
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        client = anthropic.Anthropic(api_key=api_key)
        rsp = client.messages.create(
            model=self.model,
            max_tokens=kwargs.get("max_tokens", 512),
            temperature=kwargs.get("temperature", 0.0),
            messages=[{"role":"user","content":prompt}],
        )
        out = ""
        try:
            out = rsp.content[0].text
        except Exception:
            out = ""
        return (out or "").strip()


class GoogleClient(BaseClient):
    async def acomplete(self, prompt: str, **kwargs) -> str:
        import google.generativeai as genai
        api_key = os.environ.get("GOOGLE_API_KEY", "")
        genai.configure(api_key=api_key)
        m = genai.GenerativeModel(self.model)
        rsp = m.generate_content(prompt)
        return (getattr(rsp, "text", "") or "").strip()


def _is_llama3_instruct_repo(repo_id: str) -> bool:
    s = (repo_id or "").lower()
    return ("llama-3" in s or "llama3" in s) and ("instruct" in s or "chat" in s)


def get_client(provider: str, model: str) -> BaseClient:
    p = (provider or "").lower()
    if p in ("openai",):
        return OpenAIClient(provider, model)
    if p in ("anthropic","claude"):
        return AnthropicClient(provider, model)
    if p in ("google","gemini"):
        return GeminiClient(provider, model)
    if p in ("xai","grok"):
        return XAIGrokClient(provider, model)
    if p in ("groq",):
        return GroqClient(provider, model)
    if p in ("hf","huggingface"):
        if _is_llama3_instruct_repo(model):
            return HFLlama3InstructClient(provider, model)
        return HFClient(provider, model)
    if p in ("local", "localhf"):
        if _is_llama3_instruct_repo(model):
            return LocalLlama3InstructClient(provider, model)
        return LocalHFChatClient(provider, model)

    raise ValueError(f"unknown provider: {provider}")


def apply_prefix_and_fewshot(prompt: str, prefix_path: str=None, fewshot_path: str=None, fewshot_path_override: str=None) -> str:
    # optional prefix/fewshot injection
    out = prompt or ""
    if prefix_path:
        with open(prefix_path, "r", encoding="utf-8") as f:
            pre = f.read().strip()
        if pre:
            out = pre + "\n\n" + out

    fp = fewshot_path_override or fewshot_path
    if fp:
        with open(fp, "r", encoding="utf-8") as f:
            fs = f.read().strip()
        if fs:
            out = fs + "\n\n" + out
    return out


@dataclass
class ArgsNS:
    task: str
    input_path: str
    output_dir: str
    provider: str
    model: str
    temperature: float = 0.0
    max_examples: int = -1
    prefix_path: str = None
    fewshot_path: str = None
    fewshot_path_override: str = None


def run_t2g(client, texts, gold_graphs, args_ns: ArgsNS):
    rows=[]
    for i, text in enumerate(tqdm(texts, desc="t2g")):
        prompt = f"convert text to a causal graph.\n\ntext:\n{text}\n\noutput:\n"
        prompt = apply_prefix_and_fewshot(prompt, args_ns.prefix_path, args_ns.fewshot_path, args_ns.fewshot_path_override)
        pred = client.complete(prompt, temperature=args_ns.temperature)
        nodes, edges = parse_graph_text(pred)
        rows.append({
            "id": i,
            "text": text,
            "pred_raw": pred,
            "pred_nodes": nodes,
            "pred_edges": list(edges),
            "gold": gold_graphs[i] if gold_graphs is not None else None
        })
    return rows


def run_g2t(client, graphs, gold_texts, args_ns: ArgsNS):
    rows=[]
    for i, g in enumerate(tqdm(graphs, desc="g2t")):
        prompt = f"convert causal graph to text.\n\ngraph:\n{g}\n\noutput:\n"
        prompt = apply_prefix_and_fewshot(prompt, args_ns.prefix_path, args_ns.fewshot_path, args_ns.fewshot_path_override)
        pred = client.complete(prompt, temperature=args_ns.temperature)
        rows.append({
            "id": i,
            "graph": g,
            "pred_raw": pred,
            "gold": gold_texts[i] if gold_texts is not None else None
        })
    return rows


def load_input(path: str):
    if path.endswith(".jsonl"):
        return list(read_jsonl(path))
    if path.endswith(".json"):
        return json.load(open(path, "r", encoding="utf-8"))
    if path.endswith(".csv"):
        return pd.read_csv(path).to_dict("records")
    raise ValueError(f"unsupported input: {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["t2g","g2t"])
    ap.add_argument("--input", required=True)
    ap.add_argument("--output_dir", default="runs")
    ap.add_argument("--provider", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max_examples", type=int, default=-1)
    ap.add_argument("--prefix_path", default=None)
    ap.add_argument("--fewshot_path", default=None)
    ap.add_argument("--fewshot_path_override", default=None)
    args = ap.parse_args()

    args_ns = ArgsNS(
        task=args.task,
        input_path=args.input,
        output_dir=args.output_dir,
        provider=args.provider,
        model=args.model,
        temperature=args.temperature,
        max_examples=args.max_examples,
        prefix_path=args.prefix_path,
        fewshot_path=args.fewshot_path,
        fewshot_path_override=args.fewshot_path_override,
    )

    ensure_dir(args_ns.output_dir)

    data = load_input(args_ns.input_path)
    if args_ns.max_examples and args_ns.max_examples > 0:
        data = data[:args_ns.max_examples]

    client = get_client(args_ns.provider, args_ns.model)

    ts = time.strftime("%Y%m%d-%H%M%S")
    out_jsonl = os.path.join(args_ns.output_dir, f"{args_ns.task}-{args_ns.provider}-{args_ns.model.replace('/','_')}-{ts}.jsonl")

    if args_ns.task == "t2g":
        texts = [safe_get(r, "text", "") for r in data]
        gold_graphs = [safe_get(r, "graph", None) for r in data] if data and isinstance(data[0], dict) else None
        rows = run_t2g(client, texts, gold_graphs, args_ns)
        write_jsonl(out_jsonl, rows)
    else:
        graphs = [safe_get(r, "graph", "") for r in data]
        gold_texts = [safe_get(r, "text", None) for r in data] if data and isinstance(data[0], dict) else None
        rows = run_g2t(client, graphs, gold_texts, args_ns)
        write_jsonl(out_jsonl, rows)


if __name__ == "__main__":
    main()
