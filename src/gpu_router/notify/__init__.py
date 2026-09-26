"""macOS notifications for job events (phase 8a; spec UX principle 5).

`settings` (config.yaml `notifications:`), `format` (which events notify, their text),
`backends` (osascript / terminal-notifier / null), `service` (the daemon's Notifier on the
EventBus), `cli` (`gpu notify [status|test]`). See CLAUDE.md "Notifications and doctor".
"""
