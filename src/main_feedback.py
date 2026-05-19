
import os
import json
import argparse
import numpy as np
from tqdm import tqdm
from models import *
from utils import load_data_by_name
from prompting import BasePrompter, PARSE_MAP, get_stop_words
from pathlib import Path
import atexit
import torch
from datasets import Dataset


def parse_arguments():
    parser = argparse.ArgumentParser(description='Test LLMs on different datasets with various prompt styles.')

    # dataset args
    parser.add_argument('--dataset', type=str, help='Dataset name, e.g., "gsm8k" or "spider"')
    parser.add_argument('--num_examples', type=int, default=-1, help='max number of examples')
    parser.add_argument('--start_idx', type=int, default=0,
                        help='Index of the first example to process (inclusive). Use to resume after a crash.')
    parser.add_argument('--end_idx', type=int, default=-1,
                        help='Index of the last example to process (exclusive). -1 means process until the end.')

    # results args
    parser.add_argument("--log_dir", type=str, default="logging", help="Directory to save logs")
    parser.add_argument('--overwrite_results', type=bool, default=False, help='overwrite results file')
    parser.add_argument('--write_file', type=bool, default=False, help='save results in file')
    parser.add_argument('--save_tmp_files', type=bool, default=False, help='save tmp files created during evaluation')

    # input output format args
    parser.add_argument('--cot_grammar', type=str, default='text', help='Grammar of the CoT steps')
    parser.add_argument('--out_grammar', type=str, default='text', help='Grammar of the output')

    # CoT args
    parser.add_argument('--do_cot', type=bool, default=False, help='Whether to do CoT prompting')
    parser.add_argument('--cot_model', type=str, help='Model name')
    parser.add_argument('--cot_device', type=str, default='cuda:1', help='Device for CoT Model')
    parser.add_argument('--cot_grammar_mode', type=str, default='original', help='Generation mode during CoT')
    parser.add_argument('--num_shots', type=int, default=8, help='number of few shot for CoT')
    parser.add_argument('--modify_system_prompt', type=bool, default=False, help='modify system prompt')

    # Output extraction LLM args
    parser.add_argument('--llm_parser', type=bool, default=False, help='Whether to use LLM to parse output')
    parser.add_argument('--regex_parser', type=bool, default=False, help='Whether to use regex to parse output')
    parser.add_argument('--reprompt_reasoning', type=bool, default=False, help='Whether to use LLM to parse output')
    parser.add_argument('--llm_parser_model', type=str, default=None, help='Model name for parser')
    parser.add_argument('--llm_parser_device', type=str, default='cuda', help='Device for Out Model')
    parser.add_argument('--llm_parser_grammar_mode', type=str, default='original', help='Generation mode during output extraction')

    # general generation args
    parser.add_argument('--temperature', type=float, default=0.0, help='sampling temperature')
    parser.add_argument('--max_tokens', type=int, default=600, help='max number of tokens to generate')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size of generation')
    parser.add_argument('--num_return_sequences', type=int, default=1, help='Number of return sequences')

    # IterGen generation args
    parser.add_argument('--max_itergen_iter', type=int, default=80, help='max number of iterations for IterGen')
    parser.add_argument('--backwards_limit', type=int, default=20, help='max number of backwards steps for IterGen')
    parser.add_argument('--recurrence_penalty', type=float, default=0.3, help='recurrence penalty for IterGen')

    # Adaptive args
    parser.add_argument('--start_symbol', type=str, default=None)
    parser.add_argument('--end_symbol', type=str, default=None)
    parser.add_argument('--start_in_grammar', type=bool, default=True)
    parser.add_argument('--end_in_grammar', type=bool, default=True)

    # Distributed inference args
    parser.add_argument('--enable_dist', type=bool, default=False, help='enable distributed inference')
    parser.add_argument('--num_gpus', type=int, default=1, help='number of gpus')
    parser.add_argument('--num_workers_per_gpu', type=int, default=1, help='number of workers per gpu')

    # Refinement args
    parser.add_argument('--max_refinement_retries', type=int, default=0,
                        help='Number of refinement attempts with Prover9 feedback (0 = disabled)')

    return parser.parse_args()


def set_default(obj):
    if isinstance(obj, set):
        return list(obj)
    raise TypeError


def create_result_file(args):
    parsing_type = "parsing=none"
    if args.reprompt_reasoning:
        parsing_type = 'parsing=llm+reasoning'
    elif args.llm_parser:
        parsing_type = 'parsing=llm'
    elif args.regex_parser:
        parsing_type = 'parsing=regex'

    model_identifier = f"cot-model={args.cot_model.split('/')[-1]}"
    grammar_mode = f'cot-grammar-mode={args.cot_grammar_mode}'
    if args.llm_parser:
        model_identifier += f"_llm-parser={args.llm_parser_model.split('/')[-1]}"
        grammar_mode += f"_llm-parser-grammar-mode={args.llm_parser_grammar_mode}"

    result_file = (
        f"{args.log_dir}/{args.dataset}/{model_identifier}/cot={args.do_cot}/"
        f"{parsing_type}/{args.cot_grammar}-{args.out_grammar}/{grammar_mode}/"
        f"{args.num_shots}-shot_{args.num_return_sequences}_samples_{args.modify_system_prompt}"
        f"_refinement={args.max_refinement_retries}.jsonl"
    )
    Path(result_file).parent.mkdir(parents=True, exist_ok=True)
    return result_file


def build_refinement_prompt(original_prompt, logic_program, error_message):
    """
    Builds a correction prompt using the chat history format.

    original_prompt : list of chat messages [{'role': ..., 'content': ...}]
    logic_program   : the FOL formula that failed to compile
    error_message   : the error returned by Prover9

    We append:
      - an assistant turn with the bad formula (what the LLM generated)
      - a user turn with the correction instruction

    The instruction explicitly asks the LLM to start with 'Predicates:'
    so that CRANE detects the start_symbol and activates constrained mode
    immediately.

    The guidance is tailored to the type of error to give the LLM
    actionable feedback rather than a generic error message.
    """
    error_message = error_message or "Unknown error"

    if "multiple arities" in error_message:
        # e.g. "Student/1, Student/0" — predicate used as both constant and predicate
        guidance = (
            f"Prover9 error: {error_message}\n"
            f"This means a predicate name is used both as a zero-argument constant "
            f"and as a predicate with arguments (e.g. Student vs Student(x)). "
            f"Fix this by ensuring every predicate name is used consistently "
            f"with the same number of arguments throughout Predicates, Premises, "
            f"and Conclusion. "
            f"In particular, avoid using predicate names as standalone constants."
        )
    elif "Answer not recognized" in error_message:
        # Prover9 compiled but could not reach True/False/Uncertain
        guidance = (
            f"Prover9 compiled the formula but could not determine an answer. "
            f"This usually means the logical encoding does not faithfully represent "
            f"the problem. Please check:\n"
            f"- The Conclusion correctly encodes the question being asked.\n"
            f"- The Premises faithfully represent all the problem statements.\n"
            f"- There are no undefined predicates used in Premises or Conclusion.\n"
            f"- Predicates are declared with the correct arity in the Predicates section."
        )
    elif "Failed to compile" in error_message:
        # Prover9 could not parse the formula at all
        guidance = (
            f"Prover9 failed to compile the formula. "
            f"This is likely a syntax error. Please check:\n"
            f"- All predicates used in Premises and Conclusion are declared "
            f"in the Predicates section.\n"
            f"- The formula strictly follows the Prover9 grammar "
            f"(only use: {{and}}, {{or}}, {{xor}}, {{not}}, {{implies}}, "
            f"{{iff}}, {{forall}}, {{exists}}).\n"
            f"- No extra text, keywords, or symbols appear outside the "
            f"Predicates / Premises / Conclusion sections."
        )
    else:
        guidance = (
            f"Prover9 returned this error:\n{error_message}\n\n"
            f"Please correct the formula accordingly."
        )

    return original_prompt + [
        {
            "role": "assistant",
            "content": logic_program
        },
        {
            "role": "user",
            "content": (
                f"{guidance}\n\n"
                f"Please provide the corrected FOL formula. "
                f"Start your answer directly with 'Predicates:' "
                f"and use the exact same format as before. "
                f"Do not include any explanation or additional text."
            )
        }
    ]



def process_batch(batch, llm, parse_fn, results, result_file, args, pbar, chat_mode):

    # ── 1. Initial generation ───────────────────────────────────────────────
    new_batch = llm(batch)

    # ── 2. Parsing + Prover9 evaluation ────────────────────────────────────
    parsed_results = []
    for i in range(0, len(new_batch), args.batch_size):
        parsed_results.extend(
            parse_fn.parse_answer(
                new_batch[i:i + args.batch_size],
                modify_system_prompt=args.modify_system_prompt,
                chat_mode=chat_mode
            )
        )

    # ── INIT refinement tracking ───────────────────────────────────────────
    refinement_history = {i: [] for i in range(len(new_batch))}
    refinement_attempts = {i: 0 for i in range(len(new_batch))}

    # ── 3. Refinement loop ──────────────────────────────────────────────────
    for attempt in range(args.max_refinement_retries):

        failed_indices = [
            i for i, r in enumerate(parsed_results)
            if not r['compiles'] or r.get('random', False)
        ]

        if not failed_indices:
            print(f"[Refinement] Attempt {attempt + 1}: all examples OK, stopping.")
            break

        print(f"[Refinement] Attempt {attempt + 1}/{args.max_refinement_retries} "
              f"— {len(failed_indices)} example(s) to fix: {failed_indices}")

        retry_batch = []

        for i in failed_indices:
            original = new_batch[i]
            error_msg = parsed_results[i].get('error_message', 'Unknown error')
            logic_program = parsed_results[i].get('logic_program', '')

            # Track attempt
            refinement_attempts[i] += 1
            refinement_history[i].append({
                "attempt": attempt + 1,
                "error_message": error_msg,
                "logic_program_before": logic_program
            })

            correction_prompt = build_refinement_prompt(
                original['prompt'],
                logic_program,
                error_msg
            )

            retry_batch.append({**original, 'prompt': correction_prompt})

        # ── Re-run LLM ──────────────────────────────────────────────────────
        retried_batch = llm(retry_batch)

        # ── Re-parse ────────────────────────────────────────────────────────
        retried_parsed = []
        for i in range(0, len(retried_batch), args.batch_size):
            retried_parsed.extend(
                parse_fn.parse_answer(
                    retried_batch[i:i + args.batch_size],
                    modify_system_prompt=args.modify_system_prompt,
                    chat_mode=chat_mode
                )
            )

        # ── Update results ───────────────────────────────────────────────────
        improved = 0

        for j, i in enumerate(failed_indices):
            new_res = retried_parsed[j]
            old_res = parsed_results[i]

            # Log AFTER result
            refinement_history[i][-1].update({
                "logic_program_after": new_res.get("logic_program", ""),
                "compiles": new_res.get("compiles", False),
                "correct": new_res.get("correct", False)
            })

            # Smarter update rule
            if new_res['compiles'] and (
                not old_res['compiles'] or new_res.get('correct', False)
            ):
                parsed_results[i] = new_res
                improved += 1
                print(f"[Refinement] idx={old_res['idx']} ✓ improved")
            else:
                print(f"[Refinement] idx={old_res['idx']} ✗ no improvement")

        print(f"[Refinement] Attempt {attempt + 1}: {improved}/{len(failed_indices)} improved")

        # early stopping
        #if improved == 0:
            #print("[Refinement] No improvement, stopping early.")
           # break

    # ── 4. Final recording ─────────────────────────────────────────────────
    for i, res in enumerate(parsed_results):

        # Attach refinement info
        res["refinement_attempts"] = refinement_attempts[i]
        res["refinement_history"] = refinement_history[i]
        res["was_refined"] = refinement_attempts[i] > 0

        results.append(res['correct'])

        if args.write_file:
            with open(result_file, 'a') as fout:
                fout.write(json.dumps(res, default=set_default) + '\n')

        if i % args.num_return_sequences == args.num_return_sequences - 1:
            pbar.update(1)
            pbar.set_description(f"acc={np.mean(results):.4f}")

def process_dataset_interactive(args):
    if args.llm_parser_model is None:
        args.llm_parser_model = args.cot_model

    if args.reprompt_reasoning:
        args.llm_parser = True

    llm = BaseLM(
        args.cot_model, args.cot_grammar_mode,
        grammar=args.cot_grammar,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        device=args.cot_device,
        stop_words=get_stop_words(args.dataset, args.do_cot, args.cot_grammar),
        task=args.dataset,
        recurrence_penalty=args.recurrence_penalty,
        max_itergen_iter=args.max_itergen_iter,
        backwards_limit=args.backwards_limit,
        num_return_sequences=args.num_return_sequences,
        start_symbol=args.start_symbol,
        start_in_grammar=args.start_in_grammar,
        end_symbol=args.end_symbol,
        end_in_grammar=args.end_in_grammar
    )

    dataset = load_data_by_name(args.dataset, do_cot=args.do_cot)
    if args.num_examples > 0:
        dataset = dataset.take(args.num_examples)

    results = []
    processed_ids = set()
    result_file = create_result_file(args)
    print(result_file)

    if os.path.exists(result_file) and args.overwrite_results:
        os.remove(result_file)

    if os.path.exists(result_file):
        with open(result_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                processed_ids.add(data['idx'])
                results.append(data['correct'])

    llm_parser = None

    if args.llm_parser:
        if (
            (args.cot_model != args.llm_parser_model)
            or (args.llm_parser_grammar_mode != args.cot_grammar_mode)
            or (args.cot_grammar != args.out_grammar and args.llm_parser_grammar_mode != 'original')
        ):
            llm_parser = BaseLM(
                args.llm_parser_model, args.llm_parser_grammar_mode,
                grammar=args.out_grammar,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                device=args.llm_parser_device,
                stop_words=get_stop_words(args.dataset, args.do_cot, args.out_grammar),
                task=args.dataset,
                recurrence_penalty=args.recurrence_penalty,
                max_itergen_iter=args.max_itergen_iter,
                backwards_limit=args.backwards_limit,
                num_return_sequences=args.num_return_sequences,
                start_symbol=args.start_symbol,
                start_in_grammar=args.start_in_grammar,
                end_symbol=args.end_symbol,
                end_in_grammar=args.end_in_grammar
            )
        else:
            llm_parser = llm

    if args.cot_grammar != args.out_grammar:
        instruct_type = f'{args.cot_grammar}-{args.out_grammar}'
    else:
        instruct_type = args.cot_grammar

    prompt_fn = BasePrompter(
        dataset=args.dataset,
        do_cot=args.do_cot,
        instruct_type=instruct_type,
        num_shots=args.num_shots
    )

    parse_fn = PARSE_MAP[args.dataset][args.out_grammar](
        dataset=args.dataset,
        do_cot=args.do_cot,
        instruct_type=args.out_grammar,
        reprompt_reasoning=args.reprompt_reasoning,
        lm_parser=llm_parser,
        regex_parser=args.regex_parser,
        num_return_sequences=args.num_return_sequences,
        start_symbol=args.start_symbol,
        end_symbol=args.end_symbol
    )

    if not args.save_tmp_files:
        atexit.register(parse_fn.remove_tmp_paths)

    chat_mode = (
        'instruct' in args.cot_model
        or 'Instruct' in args.cot_model
        or 'it' in args.cot_model
        or 'chat' in args.cot_model
    )

    # Compute effective end index
    end_idx = args.end_idx if args.end_idx >= 0 else len(dataset)
    total_to_process = end_idx - args.start_idx
    print(f"[Batch] Processing examples {args.start_idx} to {end_idx - 1} "
          f"({total_to_process} examples)")

    batch = []
    # NOTE: do NOT clear results here — we keep results loaded from file
    # for accurate running accuracy display
    with tqdm(total=total_to_process, dynamic_ncols=True) as pbar:
        for idx, row in enumerate(dataset):

            # Skip examples before start_idx
            if idx < args.start_idx:
                continue

            # Stop after end_idx
            if idx >= end_idx:
                break

            # Skip already processed examples (resume after crash)
            if idx in processed_ids and not args.overwrite_results:
                pbar.update(1)
                continue

            prompt = prompt_fn.prompt(
                row,
                modify_system_prompt=args.modify_system_prompt,
                chat_mode=chat_mode
            )
            batch.append({**row, 'prompt': prompt})

            if len(batch) == args.batch_size:
                process_batch(batch, llm, parse_fn, results, result_file, args, pbar, chat_mode)
                print(f"accuracy {np.mean(results):.4f} "
                      f"(idx {idx - args.batch_size + 1} to {idx})")
                batch = []

        if batch:
            process_batch(batch, llm, parse_fn, results, result_file, args, pbar, chat_mode)

    accuracy = np.mean(results) if results else 0
    print(f"Final accuracy: {accuracy:.4f} "
          f"(examples {args.start_idx} to {end_idx - 1})")


def process_dataset_parallel(args):
    import ray
    if not ray.is_initialized():
        ray.init()

    if args.llm_parser_model is None:
        args.llm_parser_model = args.cot_model

    if args.reprompt_reasoning:
        args.llm_parser = True

    if not args.llm_parser:
        assert (
            args.llm_parser_model == args.cot_model
            and args.llm_parser_grammar_mode == 'original'
        ), 'Warning: LLM parser is disabled but LLM parser args are set'

    dataset = load_data_by_name(args.dataset, do_cot=args.do_cot)
    if args.num_examples > 0:
        dataset = dataset.take(args.num_examples)

    results = []
    processed_ids = set()
    result_file = create_result_file(args)
    print(result_file)

    if os.path.exists(result_file) and args.overwrite_results:
        os.remove(result_file)

    if os.path.exists(result_file):
        with open(result_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                processed_ids.add(data['idx'])
                results.append(data['correct'])

    if args.cot_grammar != args.out_grammar:
        instruct_type = f'{args.cot_grammar}-{args.out_grammar}'
    else:
        instruct_type = args.cot_grammar

    chat_mode = (
        'instruct' in args.cot_model
        or 'Instruct' in args.cot_model
        or 'it' in args.cot_model
        or 'chat' in args.cot_model
    )

    @ray.remote(num_gpus=1 / args.num_workers_per_gpu)
    class PromptActor:
        def __init__(self):
            self.prompt_fn = BasePrompter(
                dataset=args.dataset,
                do_cot=args.do_cot,
                instruct_type=instruct_type,
                num_shots=args.num_shots
            )
            self.llm = BaseLM(
                args.cot_model, args.cot_grammar_mode,
                grammar=args.cot_grammar,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                device='cuda',
                stop_words=get_stop_words(args.dataset, args.do_cot),
                task=args.dataset,
                recurrence_penalty=args.recurrence_penalty,
                max_itergen_iter=args.max_itergen_iter,
                backwards_limit=args.backwards_limit,
                num_return_sequences=args.num_return_sequences,
                start_symbol=args.start_symbol,
                start_in_grammar=args.start_in_grammar,
                end_symbol=args.end_symbol,
                end_in_grammar=args.end_in_grammar
            )

        def process_item(self, batch):
            batch = [
                {key: value[i] for key, value in batch.items()}
                for i in range(len(next(iter(batch.values()))))
            ]
            for i in range(len(batch)):
                batch[i]['prompt'] = self.prompt_fn.prompt(
                    batch[i],
                    modify_system_prompt=args.modify_system_prompt,
                    chat_mode=chat_mode
                )
            return self.llm(batch)

    ray_dataset = ray.data.from_huggingface(dataset)
    actors = [PromptActor.remote() for _ in range(args.num_gpus * args.num_workers_per_gpu)]

    futures = []
    for batch in ray_dataset.iter_batches(batch_size=args.batch_size):
        actor_id = len(futures) % len(actors)
        futures.append(actors[actor_id].process_item.remote(batch))

    initial_responses = ray.get(futures)

    del actors
    torch.cuda.empty_cache()

    new_dataset = []
    for initial_batched_response in initial_responses:
        new_dataset.extend(initial_batched_response)

    new_dataset = Dataset.from_list(new_dataset)
    ray_dataset = ray.data.from_huggingface(new_dataset)

    @ray.remote(num_gpus=1 / args.num_workers_per_gpu)
    class ParseActor:
        def __init__(self):
            llm_parser = None
            if args.llm_parser:
                if (
                    (args.cot_model != args.llm_parser_model)
                    or (args.llm_parser_grammar_mode != args.cot_grammar_mode)
                    or (args.cot_grammar != args.out_grammar and args.llm_parser_grammar_mode != 'original')
                ):
                    llm_parser = BaseLM(
                        args.llm_parser_model, args.llm_parser_grammar_mode,
                        grammar=args.out_grammar,
                        max_tokens=args.max_tokens,
                        temperature=args.temperature,
                        device='cuda',
                        stop_words=get_stop_words(args.dataset, args.do_cot),
                        task=args.dataset,
                        recurrence_penalty=args.recurrence_penalty,
                        max_itergen_iter=args.max_itergen_iter,
                        backwards_limit=args.backwards_limit,
                        num_return_sequences=args.num_return_sequences,
                        start_symbol=args.start_symbol,
                        start_in_grammar=args.start_in_grammar,
                        end_symbol=args.end_symbol,
                        end_in_grammar=args.end_in_grammar
                    )
                else:
                    llm_parser = BaseLM(
                        args.cot_model, args.cot_grammar_mode,
                        grammar=args.cot_grammar,
                        max_tokens=args.max_tokens,
                        temperature=args.temperature,
                        device='cuda',
                        stop_words=get_stop_words(args.dataset, args.do_cot),
                        task=args.dataset,
                        recurrence_penalty=args.recurrence_penalty,
                        max_itergen_iter=args.max_itergen_iter,
                        backwards_limit=args.backwards_limit,
                        num_return_sequences=args.num_return_sequences,
                        start_symbol=args.start_symbol,
                        start_in_grammar=args.start_in_grammar,
                        end_symbol=args.end_symbol,
                        end_in_grammar=args.end_in_grammar
                    )

            self.parse_fn = PARSE_MAP[args.dataset][args.out_grammar](
                dataset=args.dataset,
                do_cot=args.do_cot,
                instruct_type=args.out_grammar,
                reprompt_reasoning=args.reprompt_reasoning,
                lm_parser=llm_parser,
                regex_parser=args.regex_parser,
                num_return_sequences=args.num_return_sequences,
                start_symbol=args.start_symbol,
                end_symbol=args.end_symbol
            )

            if not args.save_tmp_files:
                atexit.register(self.parse_fn.remove_tmp_paths)

        def process_item(self, batch):
            batch = [
                {key: value[i] for key, value in batch.items()}
                for i in range(len(next(iter(batch.values()))))
            ]
            return self.parse_fn.parse_answer(
                batch,
                modify_system_prompt=args.modify_system_prompt,
                chat_mode=chat_mode
            )

    actors = [ParseActor.remote() for _ in range(args.num_gpus * args.num_workers_per_gpu)]

    futures = []
    for batch in ray_dataset.iter_batches(batch_size=args.batch_size):
        actor_id = len(futures) % len(actors)
        futures.append(actors[actor_id].process_item.remote(batch))

    results = ray.get(futures)

    if args.write_file:
        for result in results:
            for res in result:
                with open(result_file, 'a') as fout:
                    if args.dataset == 'fol':
                        res['idx'] = int(str(res['idx']))
                    fout.write(json.dumps(res, default=set_default) + '\n')


def main():
    args = parse_arguments()

    if args.start_symbol == "tc":
        args.start_symbol = "```"
        args.end_symbol = "```"
    elif args.start_symbol == "tcs":
        args.start_symbol = "`"
        args.end_symbol = "`"

    if args.start_symbol is not None:
        args.start_symbol = args.start_symbol.replace("\\n", "\n")

    if args.enable_dist:
        process_dataset_parallel(args)
    else:
        process_dataset_interactive(args)


if __name__ == "__main__":
    main()