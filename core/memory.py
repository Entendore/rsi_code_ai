import chromadb
import random
import numpy as np
from sklearn.cluster import AgglomerativeClustering

class VectorMemory:
    def __init__(self, db_path, embedding_function):
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(name="code_lessons", embedding_function=embedding_function)

    def retrieve_successes(self, query, n_results=3):
        results = self.collection.query(query_texts=[query], n_results=n_results, where={"success": True})
        if results['documents'] and results['documents'][0]:
            return "\n".join([f"- Past Success: {m}" for m in results['documents'][0]])
        return "No successful vector memories found."

    def store(self, prompt, lesson, success):
        doc = f"Task: {prompt} | Lesson: {lesson}"
        existing = self.collection.query(query_texts=[doc], n_results=1)
        if existing['distances'] and existing['distances'][0] and existing['distances'][0][0] < 0.1: return
        self.collection.add(documents=[doc], metadatas=[{"success": success}], ids=[f"lesson_{self.collection.count() + 1}"])
        
    def prune_memories(self, max_size=500):
        count = self.collection.count()
        if count > max_size:
            to_delete = self.collection.get(where={"success": False}, limit=count - max_size)
            if to_delete['ids']: self.collection.delete(ids=to_delete['ids'])

class OntologyBuilder:
    def __init__(self, embedder):
        self.embedder = embedder
        self.concepts = {}
        
    def integrate_tool(self, func_name, func_code, prompt):
        embedding = self.embedder.embed([prompt])[0]
        best_concept = f"Concept_{len(self.concepts) + 1}"
        self.concepts[best_concept] = [embedding]
        return best_concept

    def get_concept_counts(self):
        return {k: len(v) for k, v in self.concepts.items()}

class AdaptiveCurriculum:
    def __init__(self):
        self.tasks = {
            "easy": [{"prompt": "Write a function `add(a, b)` that returns the sum.", "pre": "is_numeric", "post": "result == a + b"}],
            "medium": [{"prompt": "Write a function `find_max(arr)` that returns the maximum value.", "pre": "is_list(arr)", "post": "result == max(arr)"}],
            "hard": [{"prompt": "Write a function `binary_search(arr, t)` that returns index.", "pre": "is_sorted(arr)", "post": "arr[result] == t"}]
        }
    def generate_task(self, difficulty=None):
        if difficulty is None: difficulty = random.choice(list(self.tasks.keys()))
        task = random.choice(self.tasks[difficulty])
        return task["prompt"], task["pre"], task["post"], difficulty

class Dreamer:
    def __init__(self, model, logic, sandbox):
        self.model = model
        self.logic = logic
        self.sandbox = sandbox

    def dream_and_solve(self):
        print("\n💭 Dreamer: Imagining a novel problem space...")
        prompt = "Write a function `remove_duplicates(arr)` that returns a list with no duplicates."
        post = "len(result) == len(set(result))"
        test_cases = f"assert {post.replace('result', 'val')}"
        code = self.model.generate_with_confidence(prompt, beam_width=1)[0]
        success, _, _, _ = self.sandbox.execute(code, test_cases)
        if success:
            print("  - ✅ Dream successful. Learning new concept.")
            self.model.supervised_fine_tune(prompt, code)
            return True
        return False
