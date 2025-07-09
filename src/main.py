import typer
from rich.console import Console
from rich.markdown import Markdown
from agent import AgenticRAG

app = typer.Typer()
console = Console()

@app.command()
def chat(
    model_name: str = typer.Option("gpt-3.5-turbo", help="Name of the LLM model to use"),
    temperature: float = typer.Option(0.7, help="Temperature for LLM responses"),
    documents_path: str = typer.Option("./data/documents", help="Path to documents for RAG"),
    db_path: str = typer.Option("./data/database.db", help="Path to SQLite database")
):
    """Start an interactive chat session with the Agentic RAG system"""
    
    # Initialize the agent
    agent = AgenticRAG(
        model_name=model_name,
        temperature=temperature,
        documents_path=documents_path,
        db_path=db_path
    )
    
    console.print(Markdown("# Agentic RAG Chatbot"))
    console.print("Type 'exit' to end the conversation\n")
    
    while True:
        # Get user input
        user_input = typer.prompt("You")
        
        if user_input.lower() == 'exit':
            break
        
        # Get agent response
        try:
            response = agent.chat(user_input)
            console.print(Markdown(f"\nAssistant: {response}\n"))
        except Exception as e:
            console.print(f"\n[red]Error:[/red] {str(e)}\n")

if __name__ == "__main__":
    app()
