"""`gpu doctor` / `/doctor` (phase 8a; spec "Slash commands" and "Docs, in three layers").

`probe` (what doctor looks at, injectable), `checks` (one function per check), `runner`
(parallel, one deadline), `model` (Report / CheckResult / DriftItem, the `--json` shape),
`render` (rich output for the CLI and the shell), `catalog_fix` (`--update-catalog`),
`cli` (Typer command + shell entry). See CLAUDE.md "Notifications and doctor (phase 8a)".
"""
