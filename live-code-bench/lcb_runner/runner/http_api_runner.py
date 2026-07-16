from openai import OpenAI

from lcb_runner.runner.base_runner import BaseRunner


class HTTPAPIRunner(BaseRunner):
    """OpenAI-compatible chat completions runner."""

    def __init__(self, args, model):
        super().__init__(args, model)
        self.client = OpenAI(
            api_key=args.api_key,
            base_url=args.api_url.rstrip("/"),
            timeout=args.openai_timeout,
        )

    def _run_single(self, prompt: list[dict[str, str]]) -> list[str]:
        response = self.client.chat.completions.create(
            model=self.args.model,
            messages=prompt,
            n=1,
            temperature=0,
            max_tokens=self.args.max_tokens,
        )
        return [response.choices[0].message.content or ""]
