import typer
from typing import Optional
from rich.console import Console

app = typer.Typer()
console = Console()

@app.command()
def hello(name: Optional[str] = typer.Argument(None)):
    """Simple greeting command"""
    if name:
        console.print(f"Hello [bold green]{name}[/bold green]!")
    else:
        console.print("Hello [bold blue]World[/bold blue]!")

if __name__ == "__main__":
    app()
