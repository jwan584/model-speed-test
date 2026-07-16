from __future__ import annotations
import json, shutil
from dataclasses import dataclass
from pathlib import Path
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

@dataclass
class Task:
    task_id: str; sector: str; occupation: str; prompt: str; reference_files: list[str]

class GDPvalDataset:
    def __init__(self, cache_dir: str | Path):
        self.cache=Path(cache_dir); self.cache.mkdir(parents=True,exist_ok=True)
    def load(self) -> list[Task]:
        path=hf_hub_download("openai/gdpval","data/train-00000-of-00001.parquet",repo_type="dataset",cache_dir=self.cache/"hf")
        rows=pq.read_table(path).to_pylist()
        def values(v):
            if v is None: return []
            if isinstance(v,str):
                try: return json.loads(v)
                except json.JSONDecodeError: return [v]
            return list(v)
        return [Task(str(r["task_id"]),r["sector"],r["occupation"],r["prompt"],values(r["reference_files"])) for r in rows]
    def stage(self, task: Task, destination: Path) -> list[str]:
        destination.mkdir(parents=True,exist_ok=True); staged=[]
        for remote in task.reference_files:
            source=Path(hf_hub_download("openai/gdpval",remote,repo_type="dataset",cache_dir=self.cache/"hf"))
            target=destination/source.name
            if target.exists(): target=destination/f"{source.parent.name}_{source.name}"
            shutil.copy2(source,target); staged.append(target.name)
        return staged
