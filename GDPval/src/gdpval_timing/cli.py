from __future__ import annotations
import argparse, asyncio
from pathlib import Path
import yaml
from gdpval_timing.runner import Harness
from gdpval_timing.sandbox import Sandbox

def main():
    parser=argparse.ArgumentParser(description="Time GDPval tasks across native LLM APIs")
    parser.add_argument("--config",default="config.yaml"); parser.add_argument("--task-id"); parser.add_argument("--offset",type=int); parser.add_argument("--limit",type=int)
    parser.add_argument("--preflight",action="store_true",help="validate the sandbox without loading data or calling an LLM")
    args=parser.parse_args(); config=yaml.safe_load(Path(args.config).read_text())
    if args.preflight:
        import json
        print(json.dumps(asyncio.run(Sandbox.preflight(config["sandbox"])),indent=2))
        return
    if args.task_id: config["run"]["task_ids"]=[args.task_id]
    if args.offset is not None: config["run"]["offset"]=args.offset
    if args.limit is not None: config["run"]["limit"]=args.limit
    asyncio.run(Harness(config).run())

if __name__=="__main__": main()
