from __future__ import annotations
import base64, json, mimetypes
from pathlib import Path
from typing import Any
import httpx
from PIL import Image
from gdpval_timing.models import ToolCall, ToolResult
from gdpval_timing.sandbox import Sandbox

SCHEMAS=[
{"name":"run_shell","description":"Run a shell command in the persistent task sandbox at /workspace. Write final deliverables under /workspace/output.","parameters":{"type":"object","properties":{"command":{"type":"string"}},"required":["command"],"additionalProperties":False}},
{"name":"web_search","description":"Search the public web and return titles, URLs, and snippets.","parameters":{"type":"object","properties":{"query":{"type":"string"},"count":{"type":"integer","minimum":1,"maximum":10}},"required":["query"],"additionalProperties":False}},
{"name":"web_fetch","description":"Fetch a public HTTP(S) URL as text.","parameters":{"type":"object","properties":{"url":{"type":"string"}},"required":["url"],"additionalProperties":False}},
{"name":"view_image","description":"Inspect an image file in the sandbox; the image is returned to vision-capable models.","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"],"additionalProperties":False}},
{"name":"finish","description":"Signal completion only after deliverables exist under /workspace/output.","parameters":{"type":"object","properties":{"summary":{"type":"string"},"deliverables":{"type":"array","items":{"type":"string"}}},"required":["summary","deliverables"],"additionalProperties":False}}]

class Tools:
    def __init__(self,sandbox:Sandbox,workdir:Path,config:dict[str,Any],secrets:dict[str,str]):
        self.sandbox,self.workdir,self.config,self.secrets=sandbox,workdir.resolve(),config,secrets
        self.client=httpx.AsyncClient(timeout=60,follow_redirects=True)
    async def close(self): await self.client.aclose()
    async def execute(self,call:ToolCall)->ToolResult:
        try:
            if call.name=="run_shell": output=await self.sandbox.run(str(call.arguments["command"]))
            elif call.name=="web_fetch":
                r=await self.client.get(call.arguments["url"],headers={"user-agent":"gdpval-timing/0.1"}); r.raise_for_status()
                limit=int(self.config.get("web_fetch",{}).get("max_bytes",50_000)); output=r.text[:limit]
                if len(r.text)>limit: output+=f"\n[TRUNCATED: response had {len(r.text)} characters; showing first {limit}]"
            elif call.name=="web_search":
                key=self.secrets.get("brave")
                if not key: raise RuntimeError("BRAVE_API_KEY is not configured")
                r=await self.client.get("https://api.search.brave.com/res/v1/web/search",params={"q":call.arguments["query"],"count":call.arguments.get("count",10)},headers={"X-Subscription-Token":key}); r.raise_for_status()
                output=json.dumps(r.json().get("web",{}).get("results",[]),ensure_ascii=False)[:100000]
            elif call.name=="view_image":
                rel=str(call.arguments["path"]).removeprefix("/workspace/"); path=(self.workdir/rel).resolve()
                if self.workdir not in path.parents: raise RuntimeError("Path escapes workspace")
                with Image.open(path) as im: info=f"Image {path.name}: {im.width}x{im.height}, mode={im.mode}"
                mime=mimetypes.guess_type(path.name)[0] or "image/png"; data=base64.b64encode(path.read_bytes()).decode()
                return ToolResult(call.id,call.name,info,image_data_url=f"data:{mime};base64,{data}")
            elif call.name=="finish": output=json.dumps(call.arguments)
            else: raise RuntimeError(f"Unknown tool: {call.name}")
            return ToolResult(call.id,call.name,output)
        except Exception as exc: return ToolResult(call.id,call.name,f"{type(exc).__name__}: {exc}",True)
