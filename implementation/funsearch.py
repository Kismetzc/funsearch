# Copyright 2023 DeepMind Technologies Limited
# Copyright 2025 Google (Modifications for V5.0 based on user discussion)
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

"""A single-threaded implementation of the FunSearch pipeline."""
from __future__ import annotations

import logging
import os
import time
from typing import Any, Tuple, Sequence, Optional, Dict, Mapping  # Added Optional, Dict
from collections.abc import Mapping 

# FunSearch implementation modules - adjust paths if needed
try:
    from implementation import code_manipulation
    from implementation import config as config_lib
    from implementation import evaluator
    from implementation import programs_database
    from implementation import sampler # Sampler class needed
    from implementation import profile # Profiler class
except ImportError as e:
    logging.error(f"Error importing FunSearch implementation modules: {e}")
    raise # Critical imports failed

# Placeholder: Import your actual LLM base and implementation classes
# Adjust the path 'llm_implementations' or similar as necessary
# Assuming you have BaseLLM, BaselineLLM, AdvancedLLM defined elsewhere
try:
    # ===> IMPORTANT: Replace this with your actual import path <===
    # from llm_implementations import BaseLLM, BaselineLLM, AdvancedLLM
    # Using the LLM base class defined in sampler.py for now if others not found
    from implementation.sampler import LLM as BaseLLM
    # Assuming BaselineLLM and AdvancedLLM might inherit from BaseLLM
    # Define dummy classes if needed for script parsing without actual implementations
    class BaselineLLM(BaseLLM): pass
    class AdvancedLLM(BaseLLM): pass

except ImportError as e:
    logging.warning(f"Could not import LLM implementations. Using dummy classes. Ensure implementations are available.")
    class BaseLLM: pass
    class BaselineLLM(BaseLLM):
        def __init__(self, *args, **kwargs): logging.info("Dummy BaselineLLM initialized.")
        def draw_samples(self, *args, **kwargs): return []
    class AdvancedLLM(BaseLLM):
        def __init__(self, *args, **kwargs): logging.info("Dummy AdvancedLLM initialized.")
        def draw_samples(self, *args, **kwargs): return []

# Assume NumPy is available for evaluator selection randomness
import numpy as np


def _extract_function_names(specification: str) -> Tuple[str, str]:
    """Returns the name of the function to evolve and of the function to run.

    RZ: The so-called specification refers to the boilerplate code template for a task.
    The template MUST have two important functions decorated with '@funsearch.run', '@funsearch.evolve' respectively.
    The function labeled with '@funsearch.run' is going to evaluate the generated code (like fitness evaluation).
    The function labeled with '@funsearch.evolve' is the function to be searched (like 'greedy' in cap-set).
    This function (_extract_function_names) makes sure that these decorators appears in the specification.
    """
    run_functions = list(code_manipulation.yield_decorated(specification, 'funsearch', 'run'))
    if len(run_functions) != 1:
        raise ValueError('Expected 1 function decorated with `@funsearch.run`.')
    evolve_functions = list(code_manipulation.yield_decorated(specification, 'funsearch', 'evolve'))
    if len(evolve_functions) != 1:
        raise ValueError('Expected 1 function decorated with `@funsearch.evolve`.')
    return evolve_functions[0], run_functions[0]


# --- V5.0 MODIFIED main function ---
def main(
    specification: str,
    inputs: Sequence[Any], # Usually the dataset dict e.g., {'dataset_key': data}
    config: config_lib.Config, # Contains database, evaluator, sampler counts etc.
    max_samples_in_this_run: Optional[int], # Max samples budget for *this* execution/phase
    class_config: config_lib.ClassConfig, # Specifies LLM and Sandbox classes
    # --- V5.0 Added parameters for LLM instantiation ---
    llm_api_key: str,
    llm_base_url: str,
    llm_model_name: str,
    llm_base_temperature: float, # Base temperature for LLM
    # --- Optional parameters ---
    log_dir: Optional[str] = None, # For profiler logs
    resume_log_dir: Optional[str] = None, # For resuming (needs specific implementation)
    **kwargs # For additional args like temperature_override per phase
) -> Optional[programs_database.ProgramsDatabase]: # Return the database instance
    """
    Launches or continues a FunSearch experiment phase.

    Args:
        specification: The specification code defining functions to evolve and run.
        inputs: The dataset or inputs required for evaluation.
        config: Configuration object for the experiment phase.
        max_samples_in_this_run: Max samples this invocation should process.
        class_config: Configuration specifying which LLM and Sandbox classes to use.
        llm_api_key: API key for the LLM.
        llm_base_url: Base URL for the LLM API.
        llm_model_name: Name of the LLM model to use.
        llm_base_temperature: Default temperature for the LLM sampling.
        log_dir: Optional directory to save profiling information.
        resume_log_dir: Optional directory to resume from (requires DB load logic).
        **kwargs: Additional arguments (e.g., temperature_override for samplers).

    Returns:
        The ProgramsDatabase instance containing the results of the run, or None on failure.
    """
    logging.info("--- FunSearch Main Execution Start ---")
    logging.info(f"LLM Class from config: {class_config.llm_class.__name__}")
    logging.info(f"LLM Model: {llm_model_name}, Base URL: {llm_base_url}")
    logging.info(f"LLM Base Temperature: {llm_base_temperature}")
    logging.info(f"Sandbox Class: {class_config.sandbox_class.__name__}")
    logging.info(f"Config used: {config}")
    logging.info(f"Max Samples for this run: {max_samples_in_this_run}")
    logging.info(f"Log Directory: {log_dir}")
    logging.info(f"Resume From Directory: {resume_log_dir}")

    # 1. Parse specification and get template
    try:
        function_to_evolve, function_to_run = _extract_function_names(specification)
        template = code_manipulation.text_to_program(specification)
    except ValueError as e:
        logging.error(f"Failed to parse specification or find decorated functions: {e}")
        return None

    # 2. Handle Database Instantiation and Resuming
    database: Optional[programs_database.ProgramsDatabase] = None
    # TODO: Implement robust database loading logic if resume_log_dir is provided.
    if resume_log_dir and os.path.isdir(resume_log_dir):
        logging.warning(f"Database resuming from {resume_log_dir} is requested but "
                        f"loading logic is not fully implemented. Creating a new database.")
        database = programs_database.ProgramsDatabase(config.programs_database, template, function_to_evolve)
    else:
        if resume_log_dir:
             logging.warning(f"Resume directory {resume_log_dir} not found. Creating new database.")
        logging.info("Creating new ProgramsDatabase instance.")
        database = programs_database.ProgramsDatabase(config.programs_database, template, function_to_evolve)

    if database is None:
        logging.error("Failed to obtain a ProgramsDatabase instance.")
        return None
    logging.info(f"ProgramsDatabase ready (ID: {id(database)}).")


    # 3. Instantiate Profiler (if log directory is provided)
    profiler = None
    if log_dir:
        try:
            profiler = profile.Profiler(log_dir)
            logging.info(f"Profiler initialized for log directory: {log_dir}")
        except Exception as e_prof:
            logging.error(f"Failed to initialize Profiler in '{log_dir}': {e_prof}")

    # 4. --- V5.0 [CORE MODIFICATION]: Create LLM instance ---
    llm_instance: Optional[BaseLLM] = None
    try:
        # Prepare arguments for the LLM constructor
        llm_init_args = {
            'samples_per_prompt': config.samples_per_prompt,
            'api_key': llm_api_key,
            'base_url': llm_base_url,
            'model': llm_model_name,
            'base_temperature': llm_base_temperature
            # Add other necessary LLM base parameters here from config if needed
        }
        TargetLLMClass = class_config.llm_class

        # ** CRUCIAL: Inject database if it's an AdvancedLLM **
        if TargetLLMClass.__name__ == 'AdvancedLLM':
            llm_init_args['database'] = database
            logging.info(f"Preparing to instantiate AdvancedLLM with database reference (ID: {id(database)})")
        else:
            logging.info(f"Preparing to instantiate {TargetLLMClass.__name__}")

        # Instantiate the LLM
        logging.info(f"Using LLM init args: {llm_init_args}")
        llm_instance = TargetLLMClass(**llm_init_args)
        logging.info(f"Successfully instantiated LLM: {llm_instance.__class__.__name__}")

    except Exception as e_llm_init:
        logging.error(f"Failed to instantiate LLM class {class_config.llm_class.__name__}: {e_llm_init}", exc_info=True)
        return database # Return current DB state if LLM fails

    if llm_instance is None:
         logging.error("LLM instance could not be created. Aborting run.")
         return database


    # 5. Instantiate Evaluators
    evaluators = []
    try:
        # Determine the dataset key - assuming 'inputs' is a dict like {'OR3': dataset_data}
        # If multiple datasets are possible, config might need to specify which key to use
        # For simplicity, let's assume 'inputs' has one key or we use the first one.
        if not isinstance(inputs, Mapping):
             raise TypeError(f"Expected 'inputs' to be a dictionary-like mapping, but got {type(inputs)}")
        if not inputs:
            raise ValueError("'inputs' dictionary cannot be empty.")

        # Use the first key found in the inputs dictionary if only one dataset is expected
        dataset_key_to_use = next(iter(inputs.keys()))
        logging.info(f"Using dataset key '{dataset_key_to_use}' for evaluators.")
        dataset_for_eval = inputs[dataset_key_to_use]

        for i in range(config.num_evaluators):
            evaluators.append(evaluator.Evaluator(
                database=database,
                template=template,
                function_to_evolve=function_to_evolve,
                function_to_run=function_to_run,
                inputs=dataset_for_eval, # Pass the actual dataset content
                timeout_seconds=config.evaluate_timeout_seconds,
                sandbox_class=class_config.sandbox_class,
                # evaluator_id=i # Optional ID
            ))
        logging.info(f"Instantiated {len(evaluators)} evaluator(s).")
    except Exception as e_eval_init:
        logging.error(f"Failed to instantiate Evaluator: {e_eval_init}", exc_info=True)
        return database

    # 6. Analyze initial seed implementation if database is new
    is_new_db = True # Default assumption
    try:
        # Check based on scores (more reliable than checking _clusters)
        if hasattr(database, '_best_score_per_island') and database._best_score_per_island:
            is_new_db = all(score == -float('inf') for score in database._best_score_per_island)
        else:
             logging.warning("Could not determine if DB is new based on scores. Assuming new.")
    except Exception as e_check_new:
        logging.warning(f"Error checking if DB is new: {e_check_new}. Assuming new.")

    if is_new_db:
        logging.info("Database appears new. Analyzing initial seed program...")
        try:
            initial_program_body = template.get_function(function_to_evolve).body
            if evaluators:
                 # Pass the first dataset key for evaluation context
                 first_dataset_key = next(iter(inputs.keys()))
                 evaluators[0].analyse(
                     sample=initial_program_body,
                     island_id=None,
                     version_generated=None,
                     profiler=profiler,
                     # Pass necessary context like the test_input key for the sandbox
                     test_input=first_dataset_key,
                     **kwargs
                 )
                 logging.info("Initial seed analysis submitted.")
            else:
                 logging.error("No evaluators available to analyze initial seed.")
        except Exception as e_initial:
            logging.error(f"Failed to analyze initial seed program: {e_initial}", exc_info=True)


    # 7. --- V5.0 [CORE MODIFICATION]: Create Sampler instance(s), passing LLM instance ---
    samplers = []
    if llm_instance:
        try:
            # Use the modified Sampler.__init__ that takes llm_instance
            for i in range(config.num_samplers):
                samplers.append(sampler.Sampler(
                    database=database,
                    evaluators=evaluators,
                    llm_instance=llm_instance, # <-- Pass the created LLM instance
                    max_sample_nums=max_samples_in_this_run # <-- Pass sample budget
                    # sampler_id = i # Optional ID
                ))
            logging.info(f"Instantiated {len(samplers)} sampler(s) using the shared LLM instance.")
        except Exception as e_sampler_init:
            logging.error(f"Failed to instantiate Sampler: {e_sampler_init}", exc_info=True)
            return database # Samplers are critical
    else:
         # Should have been caught before, but safety check
         logging.error("LLM instance is None, cannot create Samplers.")
         return database

    # 8. Run Sampling Loop(s)
    try:
        if samplers:
            logging.info(f"Starting {len(samplers)} sampler loop(s)...")
            # Note: Original code runs samplers sequentially.
            # For parallel execution, threading/multiprocessing would be needed here.
            for s in samplers:
                # Pass profiler and any kwargs (like temperature_override)
                s.sample(profiler=profiler, **kwargs)
            logging.info("Sampler loop(s) finished.")
        else:
            logging.warning("No samplers were instantiated. Skipping sampling loop.")
    except KeyboardInterrupt:
        logging.warning("Keyboard interrupt received during sampling loop(s). Exiting main.")
    except Exception as e_sample_loop:
        logging.error(f"Error during sampler loop: {e_sample_loop}", exc_info=True)

    # 9. --- V5.0: Return the database instance ---
    logging.info("--- FunSearch Main Execution End ---")
    # Optional: Save the database state here if desired, e.g., database.save(log_dir)
    return database

# Note: The original funsearch.py did not have a __main__ block.
# This main function is designed to be called by another script (like the bin_packing ones or notebooks).