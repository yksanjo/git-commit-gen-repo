# Git Commit Message Generator

A smart CLI tool that analyzes your staged git changes and suggests conventional commit messages. Uses heuristics to detect the type of changes (feat, fix, docs, refactor, etc.) and generates appropriate commit messages.

## Features

- 🔍 **Automatic Analysis**: Scans staged changes for keywords, file patterns, and change statistics
- 🎯 **Conventional Commits**: Follows the [Conventional Commits](https://www.conventionalcommits.org/) specification
- 🤖 **AI Integration**: Optional AI-powered suggestions via OpenAI or Anthropic
- 📊 **Change Statistics**: View detailed stats about your staged changes
- ⚡ **Interactive Selection**: Choose from multiple suggestions or write your own
- 🎨 **Colored Output**: Easy-to-read terminal output

## Installation

### Option 1: Direct Usage

```bash
# Make executable
chmod +x ~/git-commit-gen.py

# Run directly
python3 ~/git-commit-gen.py
```

### Option 2: Install to PATH

```bash
# Copy to a directory in your PATH
cp ~/git-commit-gen.py /usr/local/bin/git-commit-gen
chmod +x /usr/local/bin/git-commit-gen

# Or create an alias in your shell profile
echo 'alias git-commit-gen="python3 ~/git-commit-gen.py"' >> ~/.bashrc
```

### Option 3: Git Alias (Recommended)

Add to your git config for the ultimate convenience:

```bash
# Add alias
git config --global alias.suggest '!python3 ~/git-commit-gen.py'
git config --global alias.cg '!python3 ~/git-commit-gen.py'

# Usage
git suggest
git cg --ai
git cg --apply
```

## Usage

### Basic Usage

```bash
# Stage your changes
git add src/feature.py

# Run the generator
python3 git-commit-gen.py

# Select from suggestions interactively
```

### Command Options

```bash
python3 git-commit-gen.py [OPTIONS]

Options:
  --ai          Include AI-powered suggestion (requires API key)
  --apply       Automatically apply the best suggestion
  --stats       Show change statistics only
  --no-color    Disable colored output
  -h, --help    Show help message
```

### Examples

```bash
# Interactive mode - select from suggestions
python3 git-commit-gen.py

# Include AI suggestion
python3 git-commit-gen.py --ai

# Auto-commit with best suggestion
python3 git-commit-gen.py --apply

# View change statistics
python3 git-commit-gen.py --stats
```

## How It Works

The analyzer looks at:

1. **Keywords in Code**: Detects patterns like `add`, `fix`, `refactor`, `test` in your changes
2. **File Types**: Identifies documentation, tests, configuration, style files
3. **Change Statistics**: New files suggest `feat`, deletions suggest `refactor` or `chore`
4. **File Paths**: Extracts scope from directory structure
5. **Code Patterns**: Recognizes function/class definitions being added

### Supported Commit Types

| Type | Description | Triggers |
|------|-------------|----------|
| `feat` | New feature | "add", "implement", new files |
| `fix` | Bug fix | "fix", "bug", "repair", "resolve" |
| `docs` | Documentation | `.md`, `README`, comments |
| `style` | Code style | `.css`, formatting, linting |
| `refactor` | Refactoring | "refactor", "cleanup", deletions |
| `perf` | Performance | "optimize", "speed", "fast" |
| `test` | Tests | `test_`, `spec`, `_test.` |
| `chore` | Maintenance | configs, dependencies, build |
| `ci` | CI/CD | `.github`, `.yml` workflow files |
| `build` | Build system | build configs, scripts |
| `revert` | Revert | "revert", "undo", "rollback" |

## AI Integration

Set your API key for smarter, context-aware suggestions:

```bash
# OpenAI
export OPENAI_API_KEY="sk-..."

# Or Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."

# Run with AI
python3 git-commit-gen.py --ai
```

The AI analyzes the actual diff content to generate highly relevant commit messages.

## Sample Output

```
📋 Suggested Commit Messages:

  1. feat(api): add user authentication endpoint
     Confidence: 85%
     Reason: Detected keywords related to 'feat'

  2. feat: add new files
     Confidence: 60%
     Reason: All new files suggest a new feature

  3. chore: update api configuration
     Confidence: 50%
     Reason: Changes primarily in config files

  c. Custom message
  q. Quit without committing

Select option (1-5/c/q): 
```

## Integration with IDEs

### VS Code

Add to `tasks.json`:

```json
{
  "label": "Suggest Commit",
  "type": "shell",
  "command": "python3",
  "args": ["${env:HOME}/git-commit-gen.py"],
  "group": "build"
}
```

### Vim/Neovim

Add to `.vimrc`:

```vim
command! Gsuggest :!python3 ~/git-commit-gen.py
```

## Tips

1. **Stage related changes**: The tool works best when you stage logically related changes
2. **Review suggestions**: Always review AI-generated suggestions before committing
3. **Use scopes**: The tool auto-detects scopes from your directory structure
4. **Combine with hooks**: Use with `prepare-commit-msg` hook for automatic suggestions

## Git Hook Integration

Create `.git/hooks/prepare-commit-msg`:

```bash
#!/bin/bash
COMMIT_MSG_FILE=$1

# Only suggest if no message provided
if [ -z "$(head -n1 $COMMIT_MSG_FILE)" ]; then
    python3 ~/git-commit-gen.py --apply
fi
```

Make it executable:
```bash
chmod +x .git/hooks/prepare-commit-msg
```

## Requirements

- Python 3.7+
- Git
- (Optional) OpenAI or Anthropic API key for AI suggestions

## License

MIT - Feel free to use and modify!
