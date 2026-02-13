#!/usr/bin/env python3
"""
Git Commit Message Generator
Analyzes staged changes and suggests conventional commit messages.

Usage:
    python git-commit-gen.py           # Analyze and suggest commits
    python git-commit-gen.py --ai      # Use AI for smarter suggestions
    python git-commit-gen.py --apply   # Auto-apply the best suggestion
"""

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class ChangeStats:
    """Statistics about the staged changes."""
    files_added: int = 0
    files_modified: int = 0
    files_deleted: int = 0
    lines_added: int = 0
    lines_deleted: int = 0
    file_types: dict = None
    scope_hints: List[str] = None
    
    def __post_init__(self):
        if self.file_types is None:
            self.file_types = {}
        if self.scope_hints is None:
            self.scope_hints = []


@dataclass
class CommitSuggestion:
    """A suggested commit message with metadata."""
    type_: str
    scope: str
    description: str
    body: str = ""
    confidence: float = 0.0
    reason: str = ""
    
    def full_message(self) -> str:
        """Generate the full conventional commit message."""
        scope_part = f"{self.scope}: " if self.scope else ""
        message = f"{self.type_}({scope_part}{self.description}"
        if self.body:
            message += f"\n\n{self.body}"
        return message
    
    def short_message(self) -> str:
        """Generate just the header line."""
        scope_part = f"{self.scope}: " if self.scope else ""
        return f"{self.type_}({scope_part}{self.description}"


class GitAnalyzer:
    """Analyzes git staged changes."""
    
    CONVENTIONAL_TYPES = {
        "feat": "A new feature",
        "fix": "A bug fix",
        "docs": "Documentation only changes",
        "style": "Code style changes (formatting, semicolons, etc)",
        "refactor": "Code refactoring without behavior change",
        "perf": "Performance improvements",
        "test": "Adding or correcting tests",
        "chore": "Build process, dependencies, or auxiliary tools",
        "ci": "CI/CD configuration changes",
        "build": "Build system changes",
        "revert": "Reverting a previous commit"
    }
    
    FILE_TYPE_PATTERNS = {
        "docs": [r"\.(md|rst|txt)$", r"README", r"CHANGELOG", r"LICENSE"],
        "test": [r"test", r"spec", r"_test\.(py|js|ts|go|rs)$"],
        "config": [r"\.(json|yaml|yml|toml|ini|cfg)$", r"config", r"setup"],
        "style": [r"\.(css|scss|sass|less|styl)$"],
        "frontend": [r"\.(html|jsx|tsx|vue|svelte)$"],
        "backend": [r"\.(py|rb|php|java|go|rs)$"],
        "scripts": [r"\.(sh|bash|ps1)$"],
        "ci": [r"\.(github|gitlab|circleci)", r"\.(yml|yaml)$"],
    }
    
    KEYWORD_PATTERNS = {
        "feat": [r"add", r"implement", r"create", r"introduce", r"new", r"feature"],
        "fix": [r"fix", r"bug", r"repair", r"resolve", r"correct", r"patch"],
        "docs": [r"doc", r"comment", r"readme", r"guide", r"manual"],
        "refactor": [r"refactor", r"restructure", r"rewrite", r"cleanup", r"clean up", r"simplify"],
        "perf": [r"performance", r"optimize", r"speed", r"fast", r"improve", r"faster"],
        "test": [r"test", r"spec", r"assert", r"coverage"],
        "chore": [r"update", r"bump", r"upgrade", r"dependency", r"package", r"version"],
        "ci": [r"ci", r"pipeline", r"workflow", r"github.?action", r"gitlab.?ci"],
        "style": [r"format", r"lint", r"style", r"whitespace", r"indent"],
        "revert": [r"revert", r"undo", r"rollback"],
    }
    
    def __init__(self):
        self.stats = ChangeStats()
    
    def run_git_command(self, args: List[str]) -> Tuple[str, str, int]:
        """Execute a git command and return stdout, stderr, returncode."""
        try:
            result = subprocess.run(
                ["git"] + args,
                capture_output=True,
                text=True,
                cwd=os.getcwd()
            )
            return result.stdout, result.stderr, result.returncode
        except FileNotFoundError:
            print("Error: Git is not installed or not in PATH")
            sys.exit(1)
    
    def check_git_repo(self) -> bool:
        """Check if we're in a git repository."""
        _, _, code = self.run_git_command(["rev-parse", "--git-dir"])
        return code == 0
    
    def get_staged_files(self) -> List[str]:
        """Get list of staged files."""
        stdout, _, _ = self.run_git_command([
            "diff", "--staged", "--name-only", "--diff-filter=ACMRT"
        ])
        return [f for f in stdout.strip().split("\n") if f]
    
    def get_staged_diff(self) -> str:
        """Get the full diff of staged changes."""
        stdout, _, _ = self.run_git_command(["diff", "--staged"])
        return stdout
    
    def get_staged_diff_stat(self) -> str:
        """Get diff statistics."""
        stdout, _, _ = self.run_git_command(["diff", "--staged", "--stat"])
        return stdout
    
    def analyze_changes(self) -> ChangeStats:
        """Analyze staged changes and return statistics."""
        self.stats = ChangeStats()
        
        # Get file statistics
        stdout, _, _ = self.run_git_command([
            "diff", "--staged", "--numstat"
        ])
        
        for line in stdout.strip().split("\n"):
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) >= 3:
                try:
                    added = int(parts[0]) if parts[0] != "-" else 0
                    deleted = int(parts[1]) if parts[1] != "-" else 0
                    filename = parts[2]
                    
                    self.stats.lines_added += added
                    self.stats.lines_deleted += deleted
                    
                    # Track file type
                    ext = os.path.splitext(filename)[1].lower()
                    if ext:
                        self.stats.file_types[ext] = self.stats.file_types.get(ext, 0) + 1
                    
                    # Extract scope hints from path
                    parts_path = filename.split("/")
                    if len(parts_path) > 1:
                        self.stats.scope_hints.extend(parts_path[:-1])
                        
                except ValueError:
                    pass
        
        # Get file status (added/modified/deleted)
        stdout, _, _ = self.run_git_command([
            "diff", "--staged", "--name-status"
        ])
        
        for line in stdout.strip().split("\n"):
            if not line.strip():
                continue
            parts = line.split("\t")
            if parts:
                status = parts[0][0]  # First character
                if status == "A":
                    self.stats.files_added += 1
                elif status == "M":
                    self.stats.files_modified += 1
                elif status == "D":
                    self.stats.files_deleted += 1
        
        # Remove duplicate scope hints
        self.stats.scope_hints = list(set(self.stats.scope_hints))
        
        return self.stats
    
    def classify_file(self, filename: str) -> List[str]:
        """Classify a file based on its name and path."""
        categories = []
        filename_lower = filename.lower()
        
        for category, patterns in self.FILE_TYPE_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, filename_lower, re.I):
                    categories.append(category)
                    break
        
        return categories
    
    def extract_keywords_from_diff(self, diff: str) -> dict:
        """Extract keywords from diff content."""
        keyword_scores = {k: 0 for k in self.CONVENTIONAL_TYPES.keys()}
        
        # Look at added lines (lines starting with +)
        added_lines = [line[1:].lower() for line in diff.split("\n") 
                      if line.startswith("+") and not line.startswith("+++")]
        
        content = " ".join(added_lines)
        
        for commit_type, patterns in self.KEYWORD_PATTERNS.items():
            for pattern in patterns:
                matches = len(re.findall(pattern, content, re.I))
                keyword_scores[commit_type] += matches
        
        return keyword_scores
    
    def determine_scope(self, files: List[str]) -> Optional[str]:
        """Determine the most likely scope from file paths."""
        if not files:
            return None
        
        # Find common directory
        dir_parts = [f.split("/") for f in files if "/" in f]
        if not dir_parts:
            return None
        
        # Get the first directory component as scope
        first_dirs = [p[0] for p in dir_parts if p]
        if first_dirs:
            from collections import Counter
            most_common = Counter(first_dirs).most_common(1)[0][0]
            # Filter out common non-scope directories
            if most_common not in ["src", "lib", "app", "packages", "tests"]:
                return most_common
            # If it's a common dir, try the second level
            if len(dir_parts[0]) > 1:
                return dir_parts[0][1] if dir_parts[0][1] not in ["components", "utils", "helpers"] else None
        
        return None


class CommitMessageGenerator:
    """Generates commit message suggestions based on analysis."""
    
    def __init__(self, analyzer: GitAnalyzer):
        self.analyzer = analyzer
    
    def generate_suggestions(self, diff: str, files: List[str], stats: ChangeStats) -> List[CommitSuggestion]:
        """Generate a list of commit message suggestions."""
        suggestions = []
        
        # Extract keywords
        keyword_scores = self.analyzer.extract_keywords_from_diff(diff)
        
        # Determine scope
        scope = self.analyzer.determine_scope(files)
        
        # Analyze file types
        file_categories = {}
        for f in files:
            cats = self.analyzer.classify_file(f)
            for cat in cats:
                file_categories[cat] = file_categories.get(cat, 0) + 1
        
        # Generate suggestions based on analysis
        
        # 1. Primary suggestion based on keyword analysis
        if keyword_scores:
            sorted_keywords = sorted(keyword_scores.items(), key=lambda x: x[1], reverse=True)
            if sorted_keywords[0][1] > 0:
                primary_type = sorted_keywords[0][0]
                desc = self._generate_description(primary_type, files, diff, stats)
                suggestions.append(CommitSuggestion(
                    type_=primary_type,
                    scope=scope or "",
                    description=desc,
                    confidence=min(0.7 + sorted_keywords[0][1] * 0.1, 0.95),
                    reason=f"Detected keywords related to '{primary_type}'"
                ))
        
        # 2. Suggestion based on file types
        if file_categories:
            sorted_cats = sorted(file_categories.items(), key=lambda x: x[1], reverse=True)
            cat_type_map = {
                "docs": "docs",
                "test": "test",
                "style": "style",
                "config": "chore",
                "ci": "ci",
                "scripts": "chore"
            }
            
            for cat, count in sorted_cats[:2]:
                if cat in cat_type_map and cat_type_map[cat] not in [s.type_ for s in suggestions]:
                    commit_type = cat_type_map[cat]
                    desc = self._generate_description(commit_type, files, diff, stats)
                    suggestions.append(CommitSuggestion(
                        type_=commit_type,
                        scope=scope or "",
                        description=desc,
                        confidence=0.6,
                        reason=f"Changes primarily in {cat} files"
                    ))
        
        # 3. Suggestion based on change statistics
        if stats.files_added > 0 and stats.files_modified == 0 and not any(s.type_ == "feat" for s in suggestions):
            # All new files - likely a feature
            desc = self._generate_description("feat", files, diff, stats)
            suggestions.append(CommitSuggestion(
                type_="feat",
                scope=scope or "",
                description=desc,
                confidence=0.5,
                reason="All new files suggest a new feature"
            ))
        
        if stats.files_deleted > stats.files_added + stats.files_modified:
            # Mostly deletions
            desc = self._generate_description("refactor", files, diff, stats)
            suggestions.append(CommitSuggestion(
                type_="refactor",
                scope=scope or "",
                description=desc,
                confidence=0.5,
                reason="Significant deletions suggest cleanup or removal"
            ))
        
        # 4. Fallback suggestion
        if not suggestions:
            desc = self._generate_description("chore", files, diff, stats)
            suggestions.append(CommitSuggestion(
                type_="chore",
                scope=scope or "",
                description=desc,
                confidence=0.4,
                reason="Default suggestion based on file changes"
            ))
        
        # Sort by confidence
        suggestions.sort(key=lambda x: x.confidence, reverse=True)
        
        # Limit to top 5
        return suggestions[:5]
    
    def _generate_description(self, commit_type: str, files: List[str], diff: str, stats: ChangeStats) -> str:
        """Generate a descriptive message based on the commit type and changes."""
        
        # Get file names without paths
        filenames = [os.path.basename(f) for f in files[:3]]
        file_str = ", ".join(filenames) if len(files) <= 3 else f"{filenames[0]} and {len(files)-1} others"
        
        # Try to extract a meaningful description from the diff
        added_lines = [line[1:].strip() for line in diff.split("\n") 
                      if line.startswith("+") and not line.startswith("+++") and len(line) > 10]
        
        # Look for function/class definitions
        patterns = [
            r"def\s+(\w+)",
            r"class\s+(\w+)",
            r"function\s+(\w+)",
            r"const\s+(\w+)\s*=",
            r"export\s+(?:default\s+)?(?:class|function)\s+(\w+)"
        ]
        
        extracted_names = []
        for line in added_lines[:20]:  # Check first 20 meaningful lines
            for pattern in patterns:
                match = re.search(pattern, line)
                if match:
                    extracted_names.append(match.group(1))
        
        if extracted_names:
            name = extracted_names[0]
            descriptions = {
                "feat": f"add {name} functionality",
                "fix": f"fix issue in {name}",
                "docs": f"update documentation for {name}",
                "style": f"format {name}",
                "refactor": f"refactor {name}",
                "perf": f"optimize {name} performance",
                "test": f"add tests for {name}",
                "chore": f"update {name}",
                "ci": f"update CI configuration for {name}",
                "build": f"update build for {name}",
                "revert": f"revert changes to {name}"
            }
            return descriptions.get(commit_type, f"update {name}")
        
        # Fallback descriptions based on file changes
        if len(files) == 1:
            base = os.path.splitext(os.path.basename(files[0]))[0]
            descriptions = {
                "feat": f"add {base}",
                "fix": f"fix {base}",
                "docs": f"update {base} documentation",
                "style": f"format {base}",
                "refactor": f"refactor {base}",
                "perf": f"optimize {base}",
                "test": f"add {base} tests",
                "chore": f"update {base}",
                "ci": f"update CI configuration",
                "build": f"update build configuration",
                "revert": f"revert {base} changes"
            }
            return descriptions.get(commit_type, f"update {base}")
        
        # Multiple files
        if stats.files_added > 0 and stats.files_modified == 0:
            return "add new files" if commit_type == "feat" else f"{commit_type} multiple files"
        elif stats.files_modified > 0 and stats.files_added == 0:
            return "update existing files" if commit_type in ["fix", "chore"] else f"{commit_type} multiple files"
        else:
            return f"{commit_type} across multiple files"
    
    def generate_body(self, stats: ChangeStats, files: List[str]) -> str:
        """Generate a commit body with details about the changes."""
        lines = []
        
        if stats.files_added:
            lines.append(f"- Add {stats.files_added} new file(s)")
        if stats.files_modified:
            lines.append(f"- Modify {stats.files_modified} file(s)")
        if stats.files_deleted:
            lines.append(f"- Delete {stats.files_deleted} file(s)")
        
        lines.append(f"- Total: +{stats.lines_added}/-{stats.lines_deleted} lines")
        
        if len(files) <= 10:
            lines.append("")
            lines.append("Files changed:")
            for f in files:
                lines.append(f"  - {f}")
        
        return "\n".join(lines)


class AISuggester:
    """Optional AI-powered commit message suggestions."""
    
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
        self.provider = "openai" if os.getenv("OPENAI_API_KEY") else "anthropic" if os.getenv("ANTHROPIC_API_KEY") else None
    
    def is_available(self) -> bool:
        return self.provider is not None
    
    def suggest(self, diff: str, files: List[str], stats: ChangeStats) -> Optional[CommitSuggestion]:
        """Get AI-powered suggestion."""
        if not self.is_available():
            return None
        
        # Truncate diff if too long
        max_diff_length = 4000
        truncated_diff = diff[:max_diff_length]
        if len(diff) > max_diff_length:
            truncated_diff += "\n...[truncated]"
        
        prompt = f"""Analyze this git diff and suggest a conventional commit message.

Files changed: {', '.join(files[:10])}{'...' if len(files) > 10 else ''}
Stats: {stats.files_added} added, {stats.files_modified} modified, {stats.files_deleted} deleted (+{stats.lines_added}/-{stats.lines_deleted} lines)

Diff:
```diff
{truncated_diff}
```

Provide your response in this exact format:
type: <conventional commit type (feat/fix/docs/style/refactor/perf/test/chore/ci/build/revert)>
scope: <optional scope, or leave empty>
description: <short description in imperative mood, lowercase, no period>

Just the three lines, nothing else."""

        try:
            if self.provider == "openai":
                return self._call_openai(prompt, stats)
            else:
                return self._call_anthropic(prompt, stats)
        except Exception as e:
            print(f"Warning: AI suggestion failed: {e}")
            return None
    
    def _call_openai(self, prompt: str, stats: ChangeStats) -> Optional[CommitSuggestion]:
        import urllib.request
        import json
        
        data = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3
        }
        
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=json.dumps(data).encode(),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
        )
        
        with urllib.request.urlopen(req, timeout=30) as response:
            result = json.loads(response.read().decode())
            content = result["choices"][0]["message"]["content"]
            return self._parse_ai_response(content, stats)
    
    def _call_anthropic(self, prompt: str, stats: ChangeStats) -> Optional[CommitSuggestion]:
        import urllib.request
        import json
        
        data = {
            "model": "claude-3-haiku-20240307",
            "max_tokens": 150,
            "messages": [{"role": "user", "content": prompt}]
        }
        
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=json.dumps(data).encode(),
            headers={
                "x-api-key": self.api_key,
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01"
            }
        )
        
        with urllib.request.urlopen(req, timeout=30) as response:
            result = json.loads(response.read().decode())
            content = result["content"][0]["text"]
            return self._parse_ai_response(content, stats)
    
    def _parse_ai_response(self, content: str, stats: ChangeStats) -> Optional[CommitSuggestion]:
        """Parse AI response into a CommitSuggestion."""
        type_match = re.search(r"type:\s*(\w+)", content, re.I)
        scope_match = re.search(r"scope:\s*(\S*)", content, re.I)
        desc_match = re.search(r"description:\s*(.+?)(?:\n|$)", content, re.I)
        
        if type_match and desc_match:
            return CommitSuggestion(
                type_=type_match.group(1).lower(),
                scope=scope_match.group(1) if scope_match else "",
                description=desc_match.group(1).strip().rstrip("."),
                body="",
                confidence=0.85,
                reason="AI-generated based on diff analysis"
            )
        return None


def print_colored(text: str, color: str = "", bold: bool = False):
    """Print colored text if terminal supports it."""
    colors = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "magenta": "\033[95m",
        "cyan": "\033[96m",
        "white": "\033[97m",
        "gray": "\033[90m"
    }
    
    if bold:
        text = f"\033[1m{text}\033[0m"
    elif color in colors:
        text = f"{colors[color]}{text}\033[0m"
    
    print(text)


def main():
    parser = argparse.ArgumentParser(
        description="Generate conventional commit messages from staged changes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                    # Show suggestions interactively
  %(prog)s --ai               # Include AI-powered suggestion
  %(prog)s --apply            # Auto-apply best suggestion
  %(prog)s --stats            # Show change statistics only

Environment Variables:
  OPENAI_API_KEY              # For AI suggestions via OpenAI
  ANTHROPIC_API_KEY           # For AI suggestions via Anthropic
        """
    )
    parser.add_argument("--ai", action="store_true", help="Use AI for smarter suggestions")
    parser.add_argument("--apply", action="store_true", help="Automatically apply the best suggestion")
    parser.add_argument("--stats", action="store_true", help="Show statistics only")

    parser.add_argument("--no-color", action="store_true", help="Disable colored output")
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = GitAnalyzer()
    
    # Check git repository
    if not analyzer.check_git_repo():
        print_colored("Error: Not a git repository", "red")
        sys.exit(1)
    
    # Check for staged changes
    staged_files = analyzer.get_staged_files()
    if not staged_files:
        print_colored("No staged changes found.", "yellow")
        print("Run 'git add <files>' to stage changes first.")
        sys.exit(0)
    
    # Analyze changes
    stats = analyzer.analyze_changes()
    diff = analyzer.get_staged_diff()
    
    if args.stats:
        print_colored("Change Statistics:", "cyan", bold=True)
        print(f"  Files: {stats.files_added} added, {stats.files_modified} modified, {stats.files_deleted} deleted")
        print(f"  Lines: +{stats.lines_added} / -{stats.lines_deleted}")
        print(f"  File types: {', '.join(f'{ext}({count})' for ext, count in list(stats.file_types.items())[:5])}")
        sys.exit(0)
    
    # Generate suggestions
    generator = CommitMessageGenerator(analyzer)
    suggestions = generator.generate_suggestions(diff, staged_files, stats)
    
    # Add AI suggestion if requested and available
    if args.ai:
        ai = AISuggester()
        if ai.is_available():
            print_colored("Querying AI for suggestion...", "gray")
            ai_suggestion = ai.suggest(diff, staged_files, stats)
            if ai_suggestion:
                suggestions.insert(0, ai_suggestion)
        else:
            print_colored("Note: AI not available. Set OPENAI_API_KEY or ANTHROPIC_API_KEY for AI suggestions.", "yellow")
    
    if args.apply:
        # Apply best suggestion
        best = suggestions[0]
        message = best.full_message()
        stdout, stderr, code = analyzer.run_git_command(["commit", "-m", message])
        if code == 0:
            print_colored(f"✓ Committed with message:", "green")
            print(f"  {best.short_message()}")
        else:
            print_colored(f"Error: {stderr}", "red")
            sys.exit(1)
    else:
        # Interactive selection
        print_colored("📋 Suggested Commit Messages:", "cyan", bold=True)
        print()
        
        for i, suggestion in enumerate(suggestions[:5], 1):
            conf_color = "green" if suggestion.confidence > 0.7 else "yellow" if suggestion.confidence > 0.5 else "gray"
            conf_pct = int(suggestion.confidence * 100)
            
            print(f"  {i}. ", end="")
            print_colored(suggestion.short_message(), "white", bold=True)
            print(f"     Confidence: ", end="")
            print_colored(f"{conf_pct}%", conf_color)
            print(f"     Reason: {suggestion.reason}")
            print()
        
        print("  c. ", end="")
        print_colored("Custom message", "magenta")
        print("  q. ", end="")
        print_colored("Quit without committing", "red")
        print()
        
        try:
            choice = input("Select option (1-5/c/q): ").strip().lower()
            
            if choice == "q":
                print("Aborted.")
                sys.exit(0)
            elif choice == "c":
                custom = input("Enter your commit message: ").strip()
                if custom:
                    stdout, stderr, code = analyzer.run_git_command(["commit", "-m", custom])
                    if code == 0:
                        print_colored("✓ Committed successfully!", "green")
                    else:
                        print_colored(f"Error: {stderr}", "red")
            elif choice.isdigit() and 1 <= int(choice) <= len(suggestions):
                selected = suggestions[int(choice) - 1]
                
                # Ask if they want to edit
                edit = input(f"Edit message? [y/N]: ").strip().lower()
                if edit == "y":
                    edited = input(f"Message [{selected.short_message()}]: ").strip()
                    if edited:
                        message = edited
                    else:
                        message = selected.full_message()
                else:
                    message = selected.full_message()
                
                stdout, stderr, code = analyzer.run_git_command(["commit", "-m", message])
                if code == 0:
                    print_colored("✓ Committed successfully!", "green")
                    print(f"  {selected.short_message()}")
                else:
                    print_colored(f"Error: {stderr}", "red")
            else:
                print_colored("Invalid selection.", "red")
                
        except KeyboardInterrupt:
            print("\nAborted.")
            sys.exit(0)


if __name__ == "__main__":
    main()
