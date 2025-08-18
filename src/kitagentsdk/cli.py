# src/kitagentsdk/cli.py
import argparse
import os
import stat
from pathlib import Path

def main():
    """Main entry point for the kitagentcli command-line tool."""
    parser = argparse.ArgumentParser(description="Kit Agent Developer CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- 'new' command ---
    new_parser = subparsers.add_parser("new", help="Create a new Kit agent project.")
    new_parser.add_argument("name", help="The name of the new agent project.")
    
    args = parser.parse_args()

    if args.command == "new":
        create_new_agent(args.name)

def create_new_agent(name: str):
    """Scaffolds a new agent project directory."""
    project_path = Path(name)
    if project_path.exists():
        print(f"Error: Directory '{name}' already exists.")
        return

    print(f"Creating new Kit agent project '{name}'...")
    
    # Project Structure
    project_path.mkdir()
    
    # Template Data
    template_dir = Path(__file__).parent / "templates"
    templates = {
        "main.py.template": "main.py",
        "manifest.json.template": "manifest.json",
        "requirements.txt.template": "requirements.txt",
        "README.md.template": "README.md",
        ".gitignore.template": ".gitignore",
    }
    
    for template_file, dest_file in templates.items():
        template_path = template_dir / template_file
        with open(template_path, 'r') as f:
            content = f.read()
        
        # Replace placeholders
        content = content.replace("{{AGENT_NAME}}", name)
        
        dest_path = project_path / dest_file
        with open(dest_path, 'w') as f:
            f.write(content)
            
    print("\n✅ Project created successfully!")
    print("\nNext steps:")
    print(f"1. cd {name}")
    print("2. (Optional) Create and activate a Python virtual environment.")
    print("3. pip install -r requirements.txt")
    print("4. Implement your agent's logic in main.py")