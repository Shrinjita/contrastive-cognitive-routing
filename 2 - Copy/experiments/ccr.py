from core.ccr_router import route
from core.deepseek_mcp import DeepSeekMCP

mcp = DeepSeekMCP()

def run(question):
    return route(mcp, question)
