from e2b_code_interpreter import Sandbox
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

class ParallelSecureExecutor:
    def __init__(self):
        if not os.getenv("E2B_API_KEY"): print("Warning: E2B_API_KEY not set.")

    def _execute_single(self, code, test_cases):
        try:
            with Sandbox(timeout=30) as sandbox:
                full_script = f"{code}\n\n{test_cases}"
                execution = sandbox.run_code(full_script)
                if execution.error: return code, False, execution.error.value, 0, 0
                metrics = sandbox.run_code("import tracemalloc; tracemalloc.stop(); print(tracemalloc.get_traced_memory()[1] / (1024*1024))")
                mem_mb = float(metrics.text) if metrics.text and metrics.text.strip() else 0
                return code, True, "Passed securely.", mem_mb, 0.1
        except Exception as e: return code, False, f"Execution failed: {str(e)}", 0, 0

    def execute_batch(self, candidates, test_cases):
        print(f"🔥 Spinning up {len(candidates)} parallel Firecracker microVMs...")
        results = []
        with ThreadPoolExecutor(max_workers=len(candidates)) as executor:
            futures = [executor.submit(self._execute_single, code, test_cases) for code in candidates]
            for future in as_completed(futures): results.append(future.result())
        return results

    def execute(self, code, test_cases):
        return self._execute_single(code, test_cases)
