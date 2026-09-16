try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    from fastmcp import FastMCP

mcp = FastMCP("arithmetic")


def _as_number(x):
    # Accept ints / floats or numeric strings; raise clean errors otherwise
    if isinstance(x, (int, float)):
        return float(x)
    
    if(isinstance(x, str)):
        return float(x.strip())
    
    raise TypeError("Expected a number (int/float or numeric string)")


@mcp.tool()
async def add(a: float, b:float) -> float:
    """Return a + b"""
    return _as_number(a) + _as_number(b)


@mcp.tool()
async def subtract(a: float, b: float) -> float:
    """Return a - b"""
    return _as_number(a) - _as_number(b)


@mcp.tool()
async def multiply(a: float, b: float) -> float:
    """Return a * b"""
    return _as_number(a) * _as_number(b)


@mcp.tool()
async def divide(a: float, b: float) -> float:
    """Return a / b. Raises on division by zero."""
    a = _as_number(a)
    b = _as_number(b)
    if b == 0:
        raise ZeroDivisionError("Division by zero")
    return a / b


@mcp.tool()
async def power(a: float, b: float) -> float:
    """Return a ** b"""
    return _as_number(a) ** _as_number(b)


@mcp.tool()
async def modulus(a: float, b: float) -> float:
    """Return a % b. Raises on modulus by zero."""
    a = _as_number(a)
    b = _as_number(b)
    if b == 0:
        raise ZeroDivisionError("Modulus by zero")
    return a % b


def main():
    """Start the arithmetic MCP server."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()