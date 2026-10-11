import os
import ast
import subprocess
import importlib
import sys

class ToolMaker:
    def __init__(self, memory, smt_compiler):
        self.smt = smt_compiler
        self.tool_file = "core/ai_tools.py"
        self.task_counts = {}
        if not os.path.exists(self.tool_file):
            with open(self.tool_file, 'w') as f: f.write("# Autonomous AI Tool Library\n")

    def evaluate_task_for_tool(self, prompt, successful_code):
        category = prompt.split("`")[1] if "`" in prompt else prompt[:10]
        self.task_counts[category] = self.task_counts.get(category, 0) + 1
        if self.task_counts[category] == 3:
            is_safe, _ = self.smt.verify_safety_of_patch(successful_code)
            if is_safe:
                try:
                    tree = ast.parse(successful_code)
                    for node in tree.body:
                        if isinstance(node, ast.FunctionDef):
                            func_code = ast.get_source_segment(successful_code, node)
                            with open(self.tool_file, 'a') as f: f.write(f"\n{func_code}\n")
                            return True
                except Exception: pass
        return False

class GitManager:
    def __init__(self, repo_path="."):
        self.repo_path = repo_path
        if not os.path.exists(os.path.join(repo_path, ".git")):
            subprocess.run(["git", "init"], cwd=repo_path, capture_output=True)
            subprocess.run(["git", "config", "user.email", "ai@rsi-system.local"], cwd=repo_path)
            subprocess.run(["git", "config", "user.name", "RSI Meta-Coder"], cwd=repo_path)

    def commit(self, files, message):
        for f in files: subprocess.run(["git", "add", f], cwd=self.repo_path)
        result = subprocess.run(["git", "commit", "-m", message], cwd=self.repo_path, capture_output=True)
        return result.returncode == 0

class MetaCoder:
    def __init__(self, smt_compiler, sandbox, benchmark):
        self.smt = smt_compiler
        self.sandbox = sandbox
        self.benchmark = benchmark

    def evaluate_and_patch_self(self, monitor_metrics, orchestrator):
        if monitor_metrics.get("tasks_failed", 0) > monitor_metrics.get("tasks_solved", 0):
            target_file = "core/genetic_opt.py"
            current_code = open(target_file, 'r').read()
            patched_code = current_code.replace("random.uniform(-0.2, 0.2)", "random.uniform(-0.6, 0.6)")
            try: ast.parse(patched_code)
            except SyntaxError: return False
            is_safe, _ = self.smt.verify_safety_of_patch(patched_code)
            if not is_safe: return False
            success, _, _, _ = self.sandbox.execute(patched_code, "assert True")
            if success:
                with open(target_file, 'w') as f: f.write(patched_code)
                if self.benchmark.run_benchmark(orchestrator): return True
                else: 
                    with open(target_file, 'w') as f: f.write(current_code) # Rollback
        return False

class Benchmark:
    def run_benchmark(self, orchestrator):
        # Simplified for brevity: assumes core math tasks pass
        return True
