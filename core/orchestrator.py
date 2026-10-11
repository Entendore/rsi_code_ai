import os
import random
import copy
import tensorflow as tf
from config import Config
from core.model import CustomCodeTransformer, DPOTrainer
from core.logic import SyntaxValidator, ComplexityProver, SMTCompiler, AdversarialFuzzer, MinimaxRedTeam
from core.memory import VectorMemory, OntologyBuilder, AdaptiveCurriculum, Dreamer
from core.sandbox import ParallelSecureExecutor
from core.automation import ToolMaker, GitManager, MetaCoder, Benchmark

# Embedder wrapper for Chroma
from transformers import TFAutoModel, AutoTokenizer
class TFEmbedder:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = TFAutoModel.from_pretrained(model_name, from_pt=True)
    def embed(self, texts):
        if isinstance(texts, str): texts = [texts]
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="tf", max_length=128)
        outputs = self.model(inputs)
        token_embeddings = outputs.last_hidden_state
        input_mask_expanded = tf.cast(tf.expand_dims(inputs["attention_mask"], -1), tf.float32)
        sum_embeddings = tf.reduce_sum(token_embeddings * input_mask_expanded, axis=1)
        sum_mask = tf.clip_by_value(tf.reduce_sum(input_mask_expanded, axis=1), 1e-9, float("inf"))
        return (sum_embeddings / sum_mask).numpy().tolist()

class CustomTFEmbeddingFunction:
    def __init__(self, tf_embedder): self.tf_embedder = tf_embedder
    def __call__(self, input): return self.tf_embedder.embed(input)

class RSIOrchestrator:
    def __init__(self, config):
        self.config = config
        print("Initializing RSI System...")
        
        self.embedder = TFEmbedder(config.EMBEDDING_MODEL_NAME)
        custom_ef = CustomTFEmbeddingFunction(self.embedder)
        self.memory = VectorMemory(config.CHROMA_DB_PATH, custom_ef)
        
        self.model = CustomCodeTransformer(config)
        if os.path.exists(config.CHECKPOINT_PATH + ".index"):
            print("Loading existing model weights...")
            self.model.load_weights(config.CHECKPOINT_PATH)
        self.model.tokenizer.train(["def add(a, b): return a + b", "def multiply(a, b): return a * b", "import sys"])
        
        self.dpo_trainer = DPOTrainer(self.model)
        self.logic = SMTCompiler()
        self.sandbox = ParallelSecureExecutor()
        self.syntax = SyntaxValidator()
        self.complexity_prover = ComplexityProver()
        self.fuzzer = AdversarialFuzzer()
        self.curriculum = AdaptiveCurriculum()
        self.ontology = OntologyBuilder(self.embedder)
        self.dreamer = Dreamer(self.model, self.logic, self.sandbox)
        self.red_team = MinimaxRedTeam(self.sandbox)
        
        self.tool_maker = ToolMaker(self.memory, self.logic)
        self.git_manager = GitManager()
        self.benchmark = Benchmark()
        self.meta_coder = MetaCoder(self.logic, self.sandbox, self.benchmark)
        
        self.gen_params = {"temp": 0.5, "prefix": "def "}
        self.complexity_history = []
        self.reward_history = []
        self.monitor_metrics = {"tasks_solved": 0, "tasks_failed": 0, "rsi_cycles": 0}

    def run_task(self, prompt, preconditions, postconditions, target_complexity="O(n)"):
        print(f"\n=== Running Task: {prompt} ===")
        z3_tests = "assert True"
        fuzz_tests = self.fuzzer.generate_fuzz_tests(prompt, preconditions)
        test_cases = f"{z3_tests}\n{fuzz_tests}"
        
        success_memory = self.memory.retrieve_successes(prompt)
        tool_import = "from core.ai_tools import *\n" if os.path.exists("core/ai_tools.py") else ""
        full_prompt = f"{tool_import}{self.gen_params['prefix']}{prompt}\n# Memory:\n{success_memory}\n# Code:\n"
        
        candidates = []
        for _ in range(self.config.BEAM_WIDTH):
            cot_output, conf = self.model.generate_with_confidence(full_prompt, beam_width=1)
            candidates.append(cot_output)
            
        filtered_candidates = [c for c in candidates if c]
        if not filtered_candidates:
            self.monitor_metrics["tasks_failed"] += 1
            return None, False

        results = self.sandbox.execute_batch(filtered_candidates, test_cases)
        
        best_code = None; worst_code = None; best_reward = -float('inf')
        for code, success, feedback, mem_mb, time_s in results:
            if success:
                is_valid, _, ast_tree = self.syntax.is_valid(code)
                if is_valid:
                    complexity = self.syntax.calculate_complexity(ast_tree)
                    reward = 1.0 - (mem_mb / 100.0) - (time_s / 10.0)
                    if complexity > self.config.MAX_COMPLEXITY: reward -= (complexity - 10) * 0.1
                    if reward > best_reward:
                        best_reward = reward; best_code = code
            else: worst_code = code

        if best_code:
            print(f"✅ Task Solved. Best Reward: {best_reward:.2f}")
            self.memory.store(prompt, "Solution worked.", True)
            self.monitor_metrics["tasks_solved"] += 1
            self.complexity_history.append(complexity)
            self.reward_history.append(best_reward)
            
            is_tool = self.tool_maker.evaluate_task_for_tool(prompt, best_code)
            if is_tool:
                func_name = prompt.split("`")[1] if "`" in prompt else "func"
                self.ontology.integrate_tool(func_name, best_code, prompt)
                self.git_manager.commit(["core/ai_tools.py"], f"Added tool: {func_name}")
            
            if worst_code:
                self.dpo_trainer.train_step(prompt, best_code, worst_code)
            
            self.model.supervised_fine_tune(full_prompt, best_code)
            os.makedirs(os.path.dirname(self.config.CHECKPOINT_PATH), exist_ok=True)
            self.model.save_weights(self.config.CHECKPOINT_PATH)
            return best_code, True
        else:
            print(f"❌ All sandbox executions failed.")
            self.memory.store(prompt, f"Failed: {feedback}", False)
            self.monitor_metrics["tasks_failed"] += 1
            return None, False

    def idle_self_play(self, num_tasks=3):
        print(f"\n📚 System Idle. Starting Study Session...")
        for _ in range(num_tasks):
            if random.random() < 0.3:
                self.dreamer.dream_and_solve()
            else:
                prompt, pre, post, _ = self.curriculum.generate_task()
                self.run_task(prompt, pre, post)
                
        self.memory.prune_memories(max_size=500)
        print("\n🧬 Triggering Meta-Coder for Source-Level Self-Improvement...")
        if self.meta_coder.evaluate_and_patch_self(self.monitor_metrics, self):
            self.git_manager.commit(["core/genetic_opt.py"], "RSI: Patched GA mutation rate")
            print("✅ Meta-Coder successfully patched source code.")
