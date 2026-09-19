from fastmcp import FastMCP
mcp = FastMCP(name="test")

@mcp.tool()
def hello(name: str) -> str:
    return f"Hello {name}"

if __name__ == "__main__":
    mcp.run(transport="stdio")
