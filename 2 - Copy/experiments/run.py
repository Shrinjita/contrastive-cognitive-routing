from experiments.naive import naive
from experiments.ccr import run
from core.deepseek_mcp import DeepSeekMCP

mcp = DeepSeekMCP()

questions = [
    "Should the company enter the Chinese market?",
    "Should we cut R&D spending?",
    "Is this acquisition overvalued?"
]

for q in questions:
    print("Q:", q)
    print("Naive:", naive(mcp, q))
    print("CCR:", run(q))
    print()
