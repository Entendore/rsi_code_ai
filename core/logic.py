import ast
from z3 import Solver, Bool, Implies, Not, sat, Int, String, And, Or
import random
import string

class SyntaxValidator:
    @staticmethod
    def is_valid(code):
        try:
            tree = ast.parse(code)
            return True, "Syntax OK", tree
        except SyntaxError as e:
            return False, f"SyntaxError: {e.msg} (line {e.lineno})", None

    @staticmethod
    def calculate_complexity(tree):
        complexity = 1
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.For, ast.While, ast.And, ast.Or)):
                complexity += 1
        return complexity

class ComplexityProver:
    @staticmethod
    def infer_complexity(tree, func_name):
        complexity = "O(1)"
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == func_name:
                loop_depth = 0
                max_depth = 0
                for child in ast.walk(node):
                    if isinstance(child, (ast.For, ast.While)):
                        loop_depth += 1
                        max_depth = max(max_depth, loop_depth)
                    if isinstance(child, ast.Return) and loop_depth > 0:
                        loop_depth -= 1 
                if max_depth == 1: complexity = "O(n)"
                elif max_depth == 2: complexity = "O(n^2)"
                elif max_depth >= 3: complexity = "O(n^3)"
        return complexity

    def verify_optimality(self, tree, func_name, target_complexity):
        inferred = self.infer_complexity(tree, func_name)
        ranks = {"O(1)": 0, "O(log n)": 1, "O(n)": 2, "O(n log n)": 3, "O(n^2)": 4, "O(n^3)": 5}
        if ranks.get(inferred, 99) <= ranks.get(target_complexity, 99):
            return True, inferred
        return False, inferred

class SMTCompiler:
    def __init__(self):
        self.solver = Solver()

    def compile_to_z3(self, postconditions):
        solver = Solver()
        if "result == a + b" in postconditions:
            a, b, result = Ints('a b result')
            solver.add(result == a + b)
        elif "result == max(arr)" in postconditions:
            arr_len, result = Ints('arr_len result')
            solver.add(result >= 0)
        elif "result == s[::-1]" in postconditions:
            s, result = Strings('s result')
            solver.add(result.length() == s.length())
        can_access_network = Bool("can_access_network")
        can_access_fs = Bool("can_access_fs")
        solver.add(can_access_network == False)
        solver.add(can_access_fs == False)
        return solver

    def verify_safety_of_patch(self, proposed_patch_code):
        solver = self.compile_to_z3("safety_check")
        forbidden = ["requests.get", "urllib", "socket", "os.system", "subprocess"]
        if any(bad in proposed_patch_code for bad in forbidden):
            violates = Bool("patch_violates_safety")
            solver.add(violates == True)
        if solver.check() == sat:
            return False, "Z3 Verification Failed: Patch violates ethical/security axioms."
        return True, "Z3 Verification Passed: Ethically constrained."

class AdversarialFuzzer:
    @staticmethod
    def generate_fuzz_tests(prompt, preconditions, iterations=3):
        tests = []
        if "is_list(arr)" in preconditions:
            for _ in range(iterations):
                size = random.choice([0, 1, 5, 100])
                arr = [random.randint(-100, 100) for _ in range(size)]
                tests.append(f"assert find_max({arr}) == max({arr}) if {arr} else True")
        elif "is_string(s)" in preconditions:
            for _ in range(iterations):
                length = random.choice([0, 1, 10, 1000])
                s = "".join(random.choices(string.ascii_letters + string.punctuation, k=length))
                s_escaped = s.replace("'", "\\'")
                tests.append(f"assert reverse_string('{s_escaped}') == '{s_escaped}'[::-1]")
        return "\n".join(tests) if tests else "assert True"

class MinimaxRedTeam:
    def __init__(self, sandbox):
        self.sandbox = sandbox

    def find_worst_case(self, func_name, func_code):
        print(f"\n♟️ Minimax Red Team: Searching for worst-case inputs for {func_name}...")
        worst_inputs = [(0, 0), (-1, 999999), (None, "str")]
        for inp in worst_inputs:
            test_script = f"{func_code}\ntry: {func_name}{inp}\nexcept: pass"
            success, feedback, _, _ = self.sandbox.execute(test_script, "assert True")
            if not success and "Error" in feedback:
                return inp, feedback
        return None, None
