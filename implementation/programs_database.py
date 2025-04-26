# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A programs database that implements the evolutionary algorithm."""
from __future__ import annotations

import profile
from collections.abc import Mapping, Sequence
import copy
import dataclasses
import time
from typing import Any, Tuple, Mapping

from absl import logging
import numpy as np
import scipy

from implementation import code_manipulation
from implementation import config as config_lib

# RZ: I change the original code "tuple[float, ...]" to "Tuple[float, ...]"
Signature = Tuple[float, ...]

# RZ: the code is also incorrect
# We should use typing.Mapping rather than abc.Mapping
ScoresPerTest = Mapping[Any, float]


def _softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Returns the tempered softmax of 1D finite `logits`."""
    if not np.all(np.isfinite(logits)):
        non_finites = set(logits[~np.isfinite(logits)])
        raise ValueError(f'`logits` contains non-finite value(s): {non_finites}')
    if not np.issubdtype(logits.dtype, np.floating):
        logits = np.array(logits, dtype=np.float32)

    result = scipy.special.softmax(logits / temperature, axis=-1)
    # Ensure that probabilities sum to 1 to prevent error in `np.random.choice`.
    index = np.argmax(result)
    result[index] = 1 - np.sum(result[0:index]) - np.sum(result[index + 1:])
    return result


def _reduce_score(scores_per_test: ScoresPerTest) -> float:
    """Reduces per-test scores into a single score.
    """
    # TODO RZ: change the code to average the score of each test.
    # return scores_per_test[list(scores_per_test.keys())[-1]]
    test_scores = [scores_per_test[k] for k in scores_per_test.keys()]
    return sum(test_scores) / len(test_scores)


def _get_signature(scores_per_test: ScoresPerTest) -> Signature:
    """Represents test scores as a canonical signature."""
    return tuple(scores_per_test[k] for k in sorted(scores_per_test.keys()))


@dataclasses.dataclass(frozen=True)
class Prompt:
    """A prompt produced by the ProgramsDatabase, to be sent to Samplers.

    Attributes:
      code: The prompt, ending with the header of the function to be completed.
      version_generated: The function to be completed is `_v{version_generated}`.
      island_id: Identifier of the island that produced the implementations
         included in the prompt. Used to direct the newly generated implementation
         into the same island.
    """
    code: str
    version_generated: int
    island_id: int


class ProgramsDatabase:
    """A collection of programs, organized as islands."""

    def __init__(
            self,
            config: config_lib.ProgramsDatabaseConfig,
            template: code_manipulation.Program,
            function_to_evolve: str,
    ) -> None:
        self._config: config_lib.ProgramsDatabaseConfig = config
        self._template: code_manipulation.Program = template
        self._function_to_evolve: str = function_to_evolve

        # Initialize empty islands.
        self._islands: list[Island] = []
        for _ in range(config.num_islands):
            self._islands.append(
                Island(template, function_to_evolve, config.functions_per_prompt,
                       config.cluster_sampling_temperature_init,
                       config.cluster_sampling_temperature_period))
        self._best_score_per_island: list[float] = (
                [-float('inf')] * config.num_islands)
        self._best_program_per_island: list[code_manipulation.Function | None] = (
                [None] * config.num_islands)
        self._best_scores_per_test_per_island: list[ScoresPerTest | None] = (
                [None] * config.num_islands)

        self._last_reset_time: float = time.time()

    def get_prompt(self) -> Prompt:
        """Returns a prompt containing implementations from one chosen island."""
        island_id = np.random.randint(len(self._islands))
        code, version_generated = self._islands[island_id].get_prompt()
        return Prompt(code, version_generated, island_id)

    def _register_program_in_island(
            self,
            program: code_manipulation.Function,
            island_id: int,
            scores_per_test: ScoresPerTest,
            **kwargs  # RZ: add this for profiling
    ) -> None:
        """Registers `program` in the specified island."""
        self._islands[island_id].register_program(program, scores_per_test)
        score = _reduce_score(scores_per_test)
        if score > self._best_score_per_island[island_id]:
            self._best_program_per_island[island_id] = program
            self._best_scores_per_test_per_island[island_id] = scores_per_test
            self._best_score_per_island[island_id] = score
            logging.info('Best score of island %d increased to %s', island_id, score)

        # --- 在这里加入检查和醒目打印 ---
        best_fit_score_baseline = -212.0000000001 # 设置一个略高于 BF 的阈值，避免浮点数精度问题
        current_best_score_on_island = self._best_score_per_island[island_id]

        if score > self._best_score_per_island[island_id]:
            self._best_program_per_island[island_id] = program
            self._best_scores_per_test_per_island[island_id] = scores_per_test
            self._best_score_per_island[island_id] = score
            logging.info('Best score of island %d increased to %s', island_id, score)

            # --- 关键的检查和打印逻辑 ---
            if score > best_fit_score_baseline and current_best_score_on_island <= best_fit_score_baseline:
                 # 只有当新分数 > BF 且之前的最佳分数 <= BF 时才打印（确保只在第一次突破时打印）
                 print("\n" + "*"*20 + " BREAKTHROUGH ALERT " + "*"*20)
                 print(f"Island {island_id}: Found new best score {score:.6f}, surpassing Best Fit ({best_fit_score_baseline:.6f})!")
                 print(f"Sample Order: {kwargs.get('global_sample_nums', 'N/A')}")
                 print("See registered function details above/below this message.")
                 print("*"*62 + "\n")
            # --- 结束关键逻辑 ---

        # ======== RZ: profiling ========
        profiler: profile.Profiler = kwargs.get('profiler', None)
        if profiler:
            global_sample_nums = kwargs.get('global_sample_nums', None)
            sample_time = kwargs.get('sample_time', None)
            evaluate_time = kwargs.get('evaluate_time', None)
            program.score = score
            program.global_sample_nums = global_sample_nums
            program.sample_time = sample_time
            program.evaluate_time = evaluate_time
            profiler.register_function(program) # Profiler 会打印函数的详细信息

    def register_program(
            self,
            program: code_manipulation.Function,
            island_id: int | None,
            scores_per_test: ScoresPerTest,
            **kwargs  # RZ: add this for profiling
    ) -> None:
        """Registers `program` in the database."""
        # In an asynchronous implementation we should consider the possibility of
        # registering a program on an island that had been reset after the prompt
        # was generated. Leaving that out here for simplicity.
        if island_id is None:
            # This is a program added at the beginning, so adding it to all islands.
            for island_id in range(len(self._islands)):
                self._register_program_in_island(program, island_id, scores_per_test, **kwargs)
        else:
            self._register_program_in_island(program, island_id, scores_per_test, **kwargs)

        # Check whether it is time to reset an island.
        if time.time() - self._last_reset_time > self._config.reset_period:
            self._last_reset_time = time.time()
            self.reset_islands()

    def reset_islands(self) -> None:
        """Resets the weaker half of islands."""
        # We sort best scores after adding minor noise to break ties.
        indices_sorted_by_score: np.ndarray = np.argsort(
            self._best_score_per_island +
            np.random.randn(len(self._best_score_per_island)) * 1e-6)
        num_islands_to_reset = self._config.num_islands // 2
        reset_islands_ids = indices_sorted_by_score[:num_islands_to_reset]
        keep_islands_ids = indices_sorted_by_score[num_islands_to_reset:]
        for island_id in reset_islands_ids:
            self._islands[island_id] = Island(
                self._template,
                self._function_to_evolve,
                self._config.functions_per_prompt,
                self._config.cluster_sampling_temperature_init,
                self._config.cluster_sampling_temperature_period)
            self._best_score_per_island[island_id] = -float('inf')
            founder_island_id = np.random.choice(keep_islands_ids)
            founder = self._best_program_per_island[founder_island_id]
            founder_scores = self._best_scores_per_test_per_island[founder_island_id]
            self._register_program_in_island(founder, island_id, founder_scores)


class Island:
    """A sub-population of the programs database."""

    def __init__(
            self,
            template: code_manipulation.Program,
            function_to_evolve: str,
            functions_per_prompt: int,
            cluster_sampling_temperature_init: float,
            cluster_sampling_temperature_period: int,
    ) -> None:
        self._template: code_manipulation.Program = template
        self._function_to_evolve: str = function_to_evolve
        self._functions_per_prompt: int = functions_per_prompt
        self._cluster_sampling_temperature_init = cluster_sampling_temperature_init
        self._cluster_sampling_temperature_period = (
            cluster_sampling_temperature_period)

        self._clusters: dict[Signature, Cluster] = {}
        self._num_programs: int = 0

    def register_program(
            self,
            program: code_manipulation.Function,
            scores_per_test: ScoresPerTest,
    ) -> None:
        """Stores a program on this island, in its appropriate cluster."""
        signature = _get_signature(scores_per_test)
        if signature not in self._clusters:
            score = _reduce_score(scores_per_test)
            self._clusters[signature] = Cluster(score, program)
        else:
            self._clusters[signature].register_program(program)
        self._num_programs += 1

    def get_prompt(self) -> tuple[str, int]:
        """Constructs a prompt containing functions from this island."""
        signatures = list(self._clusters.keys())
        cluster_scores = np.array(
            [self._clusters[signature].score for signature in signatures])

        # Convert scores to probabilities using softmax with temperature schedule.
        period = self._cluster_sampling_temperature_period
        temperature = self._cluster_sampling_temperature_init * (
                1 - (self._num_programs % period) / period)
        probabilities = _softmax(cluster_scores, temperature)

        # At the beginning of an experiment when we have few clusters, place fewer
        # programs into the prompt.
        functions_per_prompt = min(len(self._clusters), self._functions_per_prompt)

        idx = np.random.choice(
            len(signatures), size=functions_per_prompt, p=probabilities)
        chosen_signatures = [signatures[i] for i in idx]
        implementations = []
        scores = []
        for signature in chosen_signatures:
            cluster = self._clusters[signature]
            implementations.append(cluster.sample_program())
            scores.append(cluster.score)

        indices = np.argsort(scores)
        sorted_implementations = [implementations[i] for i in indices]
        version_generated = len(sorted_implementations) + 1
        return self._generate_prompt(sorted_implementations), version_generated

    def _generate_prompt(
            self,
            implementations: Sequence[code_manipulation.Function]) -> str:
        """创建一个包含适当结构和指导的提示。"""
        implementations = copy.deepcopy(implementations)  # 我们将修改这些实现

        # 格式化要包含在提示中的函数的名称和文档字符串
        versioned_functions: list[code_manipulation.Function] = []
        for i, implementation in enumerate(implementations):
            new_function_name = f'{self._function_to_evolve}_v{i}'
            implementation.name = new_function_name
            # 更新`_v0`之后的所有后续函数的文档字符串
            if i >= 1:
                implementation.docstring = (
                    f'Improved version of `{self._function_to_evolve}_v{i - 1}`.')
            # 如果函数是递归的，将对自身的调用替换为其新名称
            implementation = code_manipulation.rename_function_calls(
                str(implementation), self._function_to_evolve, new_function_name)
            versioned_functions.append(
                code_manipulation.text_to_function(implementation))

        # 添加一个额外的"演示版本" (仅适用于priority函数)
        # 检查是否是处理priority函数
        if self._function_to_evolve == "priority" and len(implementations) > 0:
            try:
                # 使用小数版本号，避免与真实版本冲突
                demo_version = len(implementations) - 0.5
                demo_name = f'{self._function_to_evolve}_v{demo_version}'
                
                # 创建演示函数
                demo_function = code_manipulation.Function(
                    name=demo_name,
                    params=implementations[-1].args,  # 使用相同参数
                    body="""    # DEMONSTRATION: This shows correct structure with proper error handling
        try:
            # 1. Input validation and conversion
            if not isinstance(bins, np.ndarray):
                bins = np.array(bins, dtype=float)
                
            # 2. Calculate scores using a strategy better than Best-Fit
            # This is just an example - create your own smart strategy
            remaining_space = bins - item
            bin_capacity = np.max(bins) + item  # Estimate original capacity
            fullness_ratio = 1 - (remaining_space / bin_capacity)
            
            # Apply weight to remaining space based on bin fullness
            # Higher score = better choice
            scores = fullness_ratio * 10 - remaining_space
            
            # 3. Handle edge cases and ensure clean return
            # Replace any NaN/inf with very negative value
            return np.nan_to_num(scores, nan=-1e9, posinf=-1e9, neginf=-1e9)
        except Exception as e:
            # Always include this safe fallback
            return np.full_like(bins, -1e9) if isinstance(bins, np.ndarray) else np.array([], dtype=float)
        """,
                    docstring="Demonstration of correct code structure with advanced bin packing strategy.",
                    return_type=implementations[-1].return_type if hasattr(implementations[-1], 'return_type') else None,
                )
                
                # 插入演示函数到版本列表 (在最后一个函数之前)
                versioned_functions.append(demo_function)
            except Exception as e:
                # 如果添加演示代码失败，记录错误但继续
                logging.warning(f"Failed to add demonstration code: {e}")
                # 不要中断正常流程，继续执行

        # 创建由LLM生成的函数的头部
        next_version = len(implementations)
        new_function_name = f'{self._function_to_evolve}_v{next_version}'
        
        # 针对priority函数添加特殊的文档字符串和指导
        if self._function_to_evolve == "priority":
            header_docstring = (
                'Improved version of '
                f'`{self._function_to_evolve}_v{next_version - 1}`.\n\n'
                'REQUIREMENTS:\n'
                '1. Must return a numpy array with SAME SHAPE as `bins`\n'
                '2. Always wrap code in try-except as shown in examples\n'
                '3. Never return None\n'
                '4. Handle non-numpy input properly\n\n'
                'USE THIS STRUCTURE:\n'
                'def priority(item, bins):\n'
                '    try:\n'
                '        # YOUR IMPROVED ALGORITHM HERE\n'
                '        return scores # Must be numpy array with same shape as bins\n'
                '    except Exception as e:\n'
                '        return np.full_like(bins, -1e9) if isinstance(bins, np.ndarray) else np.array([], dtype=float)'
            )
        else:
            # 对于非priority函数，使用默认文档字符串
            header_docstring = (
                'Improved version of '
                f'`{self._function_to_evolve}_v{next_version - 1}`.'
            )
        
        header = dataclasses.replace(
            implementations[-1],
            name=new_function_name,
            body='',
            docstring=header_docstring,
        )
        versioned_functions.append(header)

        # 用这里构造的列表替换模板中的函数
        prompt = dataclasses.replace(self._template, functions=versioned_functions)
        return str(prompt)


class Cluster:
    """A cluster of programs on the same island and with the same Signature."""

    def __init__(self, score: float, implementation: code_manipulation.Function):
        self._score = score
        self._programs: list[code_manipulation.Function] = [implementation]
        self._lengths: list[int] = [len(str(implementation))]

    @property
    def score(self) -> float:
        """Reduced score of the signature that this cluster represents."""
        return self._score

    def register_program(self, program: code_manipulation.Function) -> None:
        """Adds `program` to the cluster."""
        self._programs.append(program)
        self._lengths.append(len(str(program)))

    def sample_program(self) -> code_manipulation.Function:
        """Samples a program, giving higher probability to shorther programs."""
        normalized_lengths = (np.array(self._lengths) - min(self._lengths)) / (
                max(self._lengths) + 1e-6)
        probabilities = _softmax(-normalized_lengths, temperature=1.0)
        return np.random.choice(self._programs, p=probabilities)
