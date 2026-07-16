import json
import os
import zlib
import pickle
import base64
import re
from enum import Enum
from datetime import datetime
from dataclasses import dataclass

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from datasets import Dataset, Features, Value, load_dataset
from huggingface_hub import hf_hub_download


class Platform(Enum):
    LEETCODE = "leetcode"
    CODEFORCES = "codeforces"
    ATCODER = "atcoder"


class Difficulty(Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class TestType(Enum):
    STDIN = "stdin"
    FUNCTIONAL = "functional"


@dataclass
class Test:
    input: str
    output: str
    testtype: TestType

    def __post_init__(self):
        self.testtype = TestType(self.testtype)
        # if self.testtype == TestType.FUNCTIONAL:
        #     self.input = json.loads(self.input)
        #     self.output = json.loads(self.output)


@dataclass
class CodeGenerationProblem:
    question_title: str
    question_content: str
    platform: Platform
    question_id: str
    contest_id: str
    contest_date: datetime
    starter_code: str
    difficulty: Difficulty
    public_test_cases: list[Test]
    private_test_cases: list[Test]
    metadata: dict

    def __post_init__(self):
        self.platform = Platform(self.platform)
        self.difficulty = Difficulty(self.difficulty)
        self.contest_date = datetime.fromisoformat(self.contest_date)

        self.public_test_cases = json.loads(self.public_test_cases)  # type: ignore
        self.public_test_cases = [Test(**t) for t in self.public_test_cases]

        try:
            self.private_test_cases = json.loads(self.private_test_cases)  # type: ignore
        except:
            self.private_test_cases = json.loads(
                pickle.loads(
                    zlib.decompress(
                        base64.b64decode(self.private_test_cases.encode("utf-8"))  # type: ignore
                    )
                )
            )  # type: ignore
        self.private_test_cases = [Test(**t) for t in self.private_test_cases]

        self.metadata = json.loads(self.metadata)  # type: ignore

    def insert_output(self, output_list: list[str], code_list: list[str]) -> dict:
        return {
            "question_title": self.question_title,
            "question_content": self.question_content,
            "platform": self.platform.value,
            "question_id": self.question_id,
            "contest_id": self.contest_id,
            "contest_date": self.contest_date.isoformat(),
            "starter_code": self.starter_code,
            "difficulty": self.difficulty.value,
            "output_list": output_list,
            "code_list": code_list,
        }

    def insert_output_evaluation(
        self,
        output_list: list[str],
        code_list: list[str],
        graded_list: list[bool],
        **kwargs,
    ) -> dict:
        output = self.insert_output(output_list, code_list)
        output["graded_list"] = graded_list
        output["pass@1"] = graded_list.count(True) / len(graded_list)
        for k, v in kwargs.items():
            output[k] = v
        return output

    def get_evaluation_sample(self):
        return {
            "input_output": json.dumps(
                {
                    "inputs": [
                        t.input
                        for t in self.public_test_cases + self.private_test_cases
                    ],
                    "outputs": [
                        t.output
                        for t in self.public_test_cases + self.private_test_cases
                    ],
                    "fn_name": self.metadata.get("func_name", None),
                }
            ),
        }


def _release_files(release_version: str) -> list[str]:
    if release_version == "release_latest":
        numbers = range(1, 7)
    elif match := re.fullmatch(r"release_v([1-6])", release_version):
        numbers = range(1, int(match.group(1)) + 1)
    elif match := re.fullmatch(r"v([1-6])", release_version):
        numbers = [int(match.group(1))]
    elif match := re.fullmatch(r"v([1-6])_v([1-6])", release_version):
        start, end = map(int, match.groups())
        if start > end:
            raise ValueError(f"Invalid release version: {release_version}")
        numbers = range(start, end + 1)
    else:
        raise ValueError(f"Invalid release version: {release_version}")
    return ["test.jsonl" if number == 1 else f"test{number}.jsonl" for number in numbers]


def load_code_generation_dataset(
    release_version="release_v1", start_date=None, end_date=None, problem_ids=None
) -> list[CodeGenerationProblem]:
    local_arrow = os.environ.get("LCB_DATASET_ARROW")
    if local_arrow:
        local_release = os.environ.get("LCB_DATASET_ARROW_RELEASE")
        if local_release != release_version:
            raise ValueError(
                "LCB_DATASET_ARROW_RELEASE must exactly match the requested "
                f"release ({release_version})"
            )
        dataset_rows = Dataset.from_file(local_arrow)
        selected_ids = set(problem_ids) if problem_ids is not None else None
        dataset = [
            CodeGenerationProblem(**row)  # type: ignore
            for row in dataset_rows
            if selected_ids is None or row["question_id"] in selected_ids
        ]
    else:
        data_files = [
            hf_hub_download(
                repo_id="livecodebench/code_generation_lite",
                repo_type="dataset",
                filename=filename,
            )
            for filename in _release_files(release_version)
        ]
        features = Features(
            {
                name: Value("string")
                for name in (
                    "question_title", "question_content", "platform", "question_id",
                    "contest_id", "contest_date", "starter_code", "difficulty",
                    "public_test_cases", "private_test_cases", "metadata",
                )
            }
        )
        dataset_rows = load_dataset(
            "json", data_files=data_files, split="train", features=features
        )
        if problem_ids is not None:
            problem_ids = set(problem_ids)
            dataset_rows = dataset_rows.filter(
                lambda row: row["question_id"] in problem_ids
            )
        dataset = [CodeGenerationProblem(**p) for p in dataset_rows]  # type: ignore
    if start_date is not None:
        p_start_date = datetime.strptime(start_date, "%Y-%m-%d")
        dataset = [e for e in dataset if p_start_date <= e.contest_date]

    if end_date is not None:
        p_end_date = datetime.strptime(end_date, "%Y-%m-%d")
        dataset = [e for e in dataset if e.contest_date <= p_end_date]

    print(f"Loaded {len(dataset)} problems")
    return dataset


def load_code_generation_dataset_not_fast(release_version="release_v1") -> list[CodeGenerationProblem]:
    dataset = load_dataset("livecodebench/code_generation", split="test")
    dataset = [CodeGenerationProblem(**p) for p in dataset]  # type: ignore
    print(f"Loaded {len(dataset)} problems")
    return dataset


if __name__ == "__main__":
    dataset = load_code_generation_dataset()
