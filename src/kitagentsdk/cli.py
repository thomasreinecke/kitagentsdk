# src/kitagentsdk/cli.py
import argparse
import re
import stat
from pathlib import Path

from . import __version__


def _to_agent_class_name(agent_name: str) -> str:
    """
    Convert an arbitrary project name (e.g. 'my-agent') into a valid Python class name (e.g. 'MyAgent').
    """
    parts = re.findall(r"[A-Za-z0-9]+", agent_name)
    if not parts:
        return "Agent"

    class_name = "".join(p[:1].upper() + p[1:] for p in parts if p)
    if not class_name:
        return "Agent"

    if not (class_name[0].isalpha() or class_name[0] == "_"):
        class_name = f"Agent{class_name}"

    if not class_name.isidentifier():
        class_name = re.sub(r"[^A-Za-z0-9_]", "", class_name)
        if not class_name or not class_name.isidentifier():
            return "Agent"

    return class_name


def main():
    """Main entry point for the kitagentcli command-line tool."""
    parser = argparse.ArgumentParser(description="Kit Agent Developer CLI")
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
        help="Print the installed kitagentsdk version and exit.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    new_parser = subparsers.add_parser("new", help="Create a new Kit agent project.")
    new_parser.add_argument("name", help="The name of the new agent project directory (e.g. my-agent).")

    args = parser.parse_args()

    if args.command == "new":
        create_new_agent(args.name)


def create_new_agent(name: str):
    """Scaffolds a new agent project directory."""
    project_path = Path(name)
    if project_path.exists():
        print(f"Error: Directory '{name}' already exists.")
        return

    agent_class_name = _to_agent_class_name(name)

    print(f"Creating new Kit agent project '{name}'...")

    project_path.mkdir(parents=True, exist_ok=False)

    template_dir = Path(__file__).parent / "templates"
    templates = {
        "main.py.template": "main.py",
        "manifest.json.template": "manifest.json",
        "requirements.txt.template": "requirements.txt",
        "README.md.template": "README.md",
        ".gitignore.template": ".gitignore",
        "Makefile.template": "Makefile",
    }

    for template_file, dest_file in templates.items():
        template_path = template_dir / template_file
        if not template_path.exists():
            raise FileNotFoundError(
                f"Missing template '{template_file}' in '{template_dir}'. "
                f"Ensure kitagentsdk templates are packaged correctly."
            )

        content = template_path.read_text(encoding="utf-8")
        content = content.replace("{{AGENT_NAME}}", name)
        content = content.replace("{{AGENT_CLASS_NAME}}", agent_class_name)

        dest_path = project_path / dest_file
        dest_path.write_text(content, encoding="utf-8")

        if dest_file == "main.py":
            current_mode = dest_path.stat().st_mode
            dest_path.chmod(current_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    print("\n✅ Project created successfully!")
    print("\nNext steps:")
    print(f"1. cd {name}")
    print("2. make install")
    print("3. make config-train   (edit config-train.json as needed)")
    print("4. make train          (or: make test)")
