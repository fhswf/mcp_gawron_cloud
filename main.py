from fastmcp import FastMCP
from typing import List
import random

mcp = FastMCP("Hello World", stateless_http=True)

@mcp.tool
def roll_dice(n_dice: int) -> list[int]:
    """Roll `n_dice` 6-sided dice and return the results."""
    return [random.randint(1, 6) for _ in range(n_dice)]

# entry point - not needed but suggested
if __name__ == '__main__':
    mcp.run(transport="http",
        host="0.0.0.0",           # Bind to all interfaces
        port=8000,                # Custom port
        log_level="DEBUG")