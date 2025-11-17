
from rich.console import Console
from rich.theme import Theme

# Custom theme for better visibility
custom_theme = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "bold red",
    "success": "bold green",
    "step": "bold magenta"
})

console = Console(theme=custom_theme)
