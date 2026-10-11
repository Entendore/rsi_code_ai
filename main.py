from config import Config
from core.orchestrator import RSIOrchestrator

def main():
    config = Config()
    ai_system = RSIOrchestrator(config)
    
    print("\n🚀 Starting Recursively Self-Improving Code AI")
    print("🧠 Vector DB: ChromaDB")
    print("📦 Sandbox: E2B Firecracker microVM")
    print("🤖 Code Model: Custom TF Transformer (DPO + EWC)")
    print("🛡️  Logic: Z3 Formal Verification + Minimax Red Team")
    
    # Run an initial study session
    ai_system.idle_self_play(num_tasks=2)

if __name__ == "__main__":
    main()
