# Agent Instructions

## Task Tracking

This project tracks work in its room on bots.fallday.ca (`.fdbot.json`), using the `fdbot` CLI. Run `fdbot prime` at the start of a session for the room, project notes, your tasks and what's ready; `fdbot help` lists every command.

```bash
fdbot task ready                          # Find available work
fdbot task show 1.2                       # View task details (refs are geogen.1.2; the key is optional)
fdbot task start 1.2                      # Claim work
fdbot task close 1.2 --reason "..."       # Complete work
fdbot task create "Title" -d "..." [-p 0-4] [-t bug|feature|epic] [--parent 1] [--dep 1.1]
fdbot note add "..." --key <slug>         # Durable project knowledge, shown by prime
```

- Use the room's task list for all task tracking; do not create markdown TODO lists.
- Mention the task ref in commit messages, e.g. `Fix stair railing (geogen.6.18)`.
- Tasks imported from Beads keep their old id in the body ("Imported from Beads geogen-o3s.13").
- Do not commit or push unless asked.


## Non-Interactive Shell Commands

**ALWAYS use non-interactive flags** with file operations to avoid hanging on confirmation prompts.

Shell commands like `cp`, `mv`, and `rm` may be aliased to include `-i` (interactive) mode on some systems, causing the agent to hang indefinitely waiting for y/n input.

**Use these forms instead:**
```bash
# Force overwrite without prompting
cp -f source dest           # NOT: cp source dest
mv -f source dest           # NOT: mv source dest
rm -f file                  # NOT: rm file

# For recursive operations
rm -rf directory            # NOT: rm -r directory
cp -rf source dest          # NOT: cp -r source dest
```

**Other commands that may prompt:**
- `scp` - use `-o BatchMode=yes` for non-interactive
- `ssh` - use `-o BatchMode=yes` to fail instead of prompting
- `apt-get` - use `-y` flag
- `brew` - use `HOMEBREW_NO_AUTO_UPDATE=1` env var
