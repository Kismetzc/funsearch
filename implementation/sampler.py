# Copyright 2023 DeepMind Technologies Limited
# Copyright 2025 Google (Original Modifications for V5.0)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Class for sampling new programs."""
from __future__ import annotations
from abc import ABC, abstractmethod
import logging # <-- Added Import
import time
from typing import Collection, Sequence, Optional # <-- Added Optional

import numpy as np

# Assuming these modules are correctly located relative to this file
try:
    from implementation import evaluator
    from implementation import programs_database
    # Import the actual base LLM class if it's defined elsewhere
    # from llm_implementations_placeholder import BaseLLM as LLM
except ImportError as e:
    logging.error(f"Import error in sampler.py: {e}")
    # Define dummy classes if imports fail, to allow parsing
    class LLM(ABC):
         def __init__(self, *args, **kwargs): pass
         @abstractmethod
         def draw_samples(self, prompt: str) -> Collection[str]: pass
         def _construct_prompt(self, code: str, island_id: Optional[int]) -> str: return code # Dummy method for AdvancedLLM check
    class ProgramsDatabase: pass
    class Evaluator:
        def analyse(self, *args, **kwargs): pass


# --- LLM Base Class Definition (Keep as is or import actual BaseLLM) ---
class LLM(ABC):
    """Language model that predicts continuation of provided source code.

    RZ: The sampled function code must be trimmed! Especially using instruct-based LLM.
    (RZ's comments remain relevant)
    """

    # RZ: Note (April 2025): Consider if samples_per_prompt should still be here
    # or purely managed by the specific LLM implementation's config/init.
    # Keeping it here for compatibility with the original abstract class idea,
    # but concrete implementations might handle this internally.
    def __init__(self, samples_per_prompt: int = 1) -> None:
        # Defaulting samples_per_prompt to 1 if not provided by subclass/instance creation
        self._samples_per_prompt = samples_per_prompt
        logging.info(f"LLM Base Initialized with samples_per_prompt={self._samples_per_prompt}")

    @property
    def samples_per_prompt(self) -> int:
        """Returns the number of samples the LLM is configured to generate per prompt."""
        return self._samples_per_prompt

    def _draw_sample(self, prompt: str) -> str:
        """Returns a predicted continuation of `prompt`."""
        # This method might become less relevant if draw_samples handles batching directly.
        raise NotImplementedError('Must provide a language model implementation for _draw_sample.')

    @abstractmethod
    def draw_samples(self, prompt: str, temperature_override: Optional[float] = None) -> Collection[str]:
        """
        Returns multiple predicted continuations of `prompt`.

        Args:
            prompt: The input prompt string.
            temperature_override: Optional temperature to use for this specific call,
                                  overriding the LLM's default if provided.

        Returns:
            A collection of sampled string continuations.
        """
        # Default implementation remains based on _draw_sample for subclasses that only override that.
        # However, implementations interacting with APIs often override draw_samples directly for efficiency.
        logging.debug(f"Using default draw_samples based on _draw_sample {self.samples_per_prompt} times.")
        # Note: This default implementation doesn't use temperature_override
        return [self._draw_sample(prompt) for _ in range(self.samples_per_prompt)]

    # Add a dummy _construct_prompt for type checking if needed, especially for AdvancedLLM.
    # Actual AdvancedLLM should implement this properly.
    def _construct_prompt(self, code: str, island_id: Optional[int]) -> str:
        """Placeholder for prompt construction, relevant for AdvancedLLM."""
        # BaselineLLM would just return code, AdvancedLLM would add context.
        logging.debug(f"LLM Base _construct_prompt called (island_id: {island_id}). Returning original code.")
        return code

# --- Modified Sampler Class ---
class Sampler:
    """
    Node that samples program continuations using a provided LLM instance
    and sends them for analysis. Manages its own sample processing budget.
    """
    # --- V5.0: Removed class variable _global_samples_nums ---

    # --- V5.0: Modified __init__ method ---
    def __init__(
        self,
        database: programs_database.ProgramsDatabase,
        evaluators: Sequence[evaluator.Evaluator],
        llm_instance: LLM, # <-- MODIFIED: Receive LLM instance
        max_sample_nums: Optional[int] = None, # Max samples for THIS instance
        # <-- REMOVED: llm_class parameter
        # <-- REMOVED: samples_per_prompt parameter (managed by llm_instance)
    ):
        """
        Initializes the Sampler.

        Args:
            database: ProgramsDatabase instance.
            evaluators: List of Evaluator instances.
            llm_instance: An *already initialized* LLM object instance.
            max_sample_nums: Maximum number of samples this specific Sampler
                             instance should process in its lifetime. If None,
                             runs indefinitely until interrupted.
        """
        self._database = database
        self._evaluators = evaluators
        self._llm = llm_instance # <-- MODIFIED: Use passed instance directly
        self._max_sample_nums = max_sample_nums
        self._instance_samples_processed = 0 # <-- ADDED: Instance-level counter

        # Log the type of LLM received for clarity
        logging.info(f"Sampler initialized with LLM instance: {self._llm.__class__.__name__}")
        # Log the sample budget for this instance
        if self._max_sample_nums is not None:
            logging.info(f"Sampler instance sample budget: {self._max_sample_nums}")
        else:
            logging.info("Sampler instance has no sample budget limit.")

    # --- V5.0: Modified sample method ---
    def sample(self, **kwargs):
        """
        Continuously gets prompts, samples programs using the provided LLM,
        and sends them for analysis, respecting the instance's sample limit.
        """
        logging.info(f"Sampler instance starting sampling loop (budget: {self._max_sample_nums})...")
        profiler = kwargs.get('profiler', None) # Get profiler if passed

        while True:
            # Check if instance budget is reached *before* getting a new prompt
            if self._max_sample_nums is not None and self._instance_samples_processed >= self._max_sample_nums:
                logging.info(f"Sampler instance reached sample limit ({self._instance_samples_processed}/{self._max_sample_nums}). Stopping.")
                break

            try:
                # 1. Get a prompt specification from the database
                prompt_object = self._database.get_prompt()
                if prompt_object is None: # Handle case where DB might be exhausted or uninitialized
                   logging.warning("Database returned None prompt. Waiting and retrying...")
                   time.sleep(10) # Wait before retrying
                   continue

                # 2. Construct the actual prompt string (AdvancedLLM might add context here)
                # Assuming LLM instance has a method to potentially add context based on island_id
                # This relies on the specific LLM implementation (e.g., AdvancedLLM having _construct_prompt)
                final_prompt_for_llm = self._llm._construct_prompt(prompt_object.code, prompt_object.island_id)

                # 3. Draw samples from the LLM
                logging.debug(f"Drawing samples for island {prompt_object.island_id}, version {prompt_object.version_generated}")
                reset_time = time.time()
                # Pass temperature override if provided in kwargs, allowing phase-specific temps
                temperature_override = kwargs.get('temperature_override', None)
                samples = self._llm.draw_samples(final_prompt_for_llm, temperature_override=temperature_override)
                num_samples_drawn = len(samples)

                # Calculate average time per sample *drawn* in this batch
                sample_time_avg = (time.time() - reset_time) / num_samples_drawn if num_samples_drawn > 0 else 0
                logging.debug(f"Drew {num_samples_drawn} samples in {time.time() - reset_time:.2f}s (avg {sample_time_avg:.2f}s/sample)")

                # 4. Distribute samples to evaluators
                samples_distributed_this_iteration = 0
                for sample_idx, sample_code in enumerate(samples):
                    # Check budget *before* processing each sample within the batch
                    current_iteration_target_sample_num = self._instance_samples_processed + 1
                    if self._max_sample_nums is not None and current_iteration_target_sample_num > self._max_sample_nums:
                        logging.info(f"Sampler instance reached sample limit ({self._max_sample_nums}) "
                                     f"while processing batch. Stopping distribution.")
                        break # Stop processing this batch

                    logging.debug(f"[Sampler] Distributing sample {sample_idx+1}/{num_samples_drawn}, "
                                  f"instance count: {current_iteration_target_sample_num}")

                    # Choose an evaluator (randomly)
                    chosen_evaluator: evaluator.Evaluator = np.random.choice(self._evaluators)

                    # Send for analysis
                    chosen_evaluator.analyse(
                        sample=sample_code,
                        island_id=prompt_object.island_id,
                        version_generated=prompt_object.version_generated,
                        profiler=profiler, # Pass profiler
                        # Pass instance sample number for logging/tracking if needed
                        instance_sample_num=current_iteration_target_sample_num,
                        sample_time=sample_time_avg, # Pass average time for this batch
                        **kwargs # Pass other potential args like temperature used
                    )

                    # IMPORTANT: Increment instance counter only *after* successful analyse call submission
                    self._instance_samples_processed += 1
                    samples_distributed_this_iteration += 1

                # Check if the inner loop was broken due to budget limit
                if self._max_sample_nums is not None and self._instance_samples_processed >= self._max_sample_nums:
                    logging.info(f"Sampler instance finished processing batch and reached limit ({self._instance_samples_processed}/{self._max_sample_nums}). Stopping.")
                    break # Exit the main while loop

            except KeyboardInterrupt:
                logging.warning("Keyboard interrupt detected in sampler loop. Stopping.")
                break # Exit the loop cleanly on Ctrl+C
            except Exception as e:
                # Catch other exceptions, log them, and continue (or break if preferred)
                logging.error(f"Error in sampler loop: {e}", exc_info=True)
                # Optional: Add a delay before continuing to prevent rapid-fire errors
                time.sleep(5)
                # Consider if you want to `continue` or `break` on general errors
                continue

        logging.info(f"Sampler instance finished sampling loop. Total samples processed by this instance: {self._instance_samples_processed}")

    # --- V5.0: Removed global counter methods ---
    # def _get_global_sample_nums(self) -> int: ...
    # def set_global_sample_nums(self, num): ...
    # def _global_sample_nums_plus_one(self): ...