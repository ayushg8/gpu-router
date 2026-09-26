#!/bin/bash
# Reference Claude Code status line for the gpu-router tests ("aligned columns").
#
# The gpu rows (gpu_router.statusline.fast) copy this script's tokens, grid and meter, and
# tests/unit/statusline/test_user_style.py renders it to check they line up. It is a trimmed,
# portable copy of a real ~/.claude/statusline.sh: two rows, no caches, no background work.
#
# Three columns on a fixed grid, so the same figure sits in the same place on every render.
# One accent (the model name); colour only when a number crosses 70% or 90%; no emoji.
set -f

input=$(cat)
[ -z "$input" ] && { printf 'Claude'; exit 0; }

# ── tokens ──────────────────────────────────────────────
accent=$'\033[38;5;75m'   # the model name, and nothing else
fg=$'\033[38;5;252m'      # values
dim=$'\033[38;5;243m'     # labels, separators, anything secondary
warn=$'\033[38;5;179m'    # 70%+
crit=$'\033[38;5;174m'    # 90%+
off=$'\033[0m'
COL=30                    # column width; the grid everything lands on

state_color() {
    if   [ "${1:-0}" -ge 90 ] 2>/dev/null; then printf '%s' "$crit"
    elif [ "${1:-0}" -ge 70 ] 2>/dev/null; then printf '%s' "$warn"
    else printf '%s' "$fg"
    fi
}
# epoch -> formatted local time: BSD date (-r) first, then GNU date (-d @)
fmt() { date -r "$1" "$2" 2>/dev/null || date -d "@$1" "$2" 2>/dev/null; }
# A reset inside the next day reads as a time; further out it needs the weekday to mean anything.
when()  {
    [ -n "${1:-}" ] || return
    local now; now=$(date +%s)
    if [ $(( $1 - now )) -lt 79200 ]; then fmt "$1" '+%-I%p' | tr 'AMP' 'amp'
    else fmt "$1" '+%a %-I%p' | tr 'AMP' 'amp'
    fi
}

meter() {  # 10 cells; filled carries the state colour, empty stays dim
    local pct=${1:-0} cells=10 i filled
    filled=$(( (pct * cells + 99) / 100 ))
    [ "$filled" -gt "$cells" ] && filled=$cells
    printf '%s' "$(state_color "$pct")"
    for ((i = 0; i < filled; i++)); do printf '█'; done
    printf '%s' "$dim"
    for ((i = filled; i < cells; i++)); do printf '░'; done
    printf '%s' "$off"
}

cell() {  # $1 = coloured text, $2 = its visible width -> padded to the grid
    local pad=$(( COL - $2 ))
    [ "$pad" -lt 1 ] && pad=1
    printf '%s%*s' "$1" "$pad" ''
}

project() {  # the project being worked on, not whichever subfolder the shell sits in
    local d="${1%/}" root="" m
    while [ -n "$d" ] && [ "$d" != "/" ] && [ "$d" != "$HOME" ]; do
        for m in .git package.json pyproject.toml Cargo.toml go.mod deno.json .projectroot; do
            if [ -e "$d/$m" ]; then root="$d"; break 2; fi
        done
        d="${d%/*}"
    done
    local p="${root:-${1%/}}"
    [ "$p" = "$HOME" ] && { printf 'no project'; return; }
    case "$p" in "$HOME"/*) p="${p#"$HOME"/}" ;; esac
    local parts; IFS='/' read -ra parts <<< "$p"
    local n=${#parts[@]}
    if [ "$n" -le 2 ]; then printf '%s' "$p"; else printf '…/%s/%s' "${parts[n-2]}" "${parts[n-1]}"; fi
}

# ── one parse, one process ──────────────────────────────
# Joined on unit separators, not tabs: `read` collapses empty tab fields, which shifts every
# value after a missing one into the wrong variable.
IFS=$'\037' read -r model cwd effort five_pct five_tiny five_reset week_pct week_tiny week_reset fast < <(
    jq -r '[
        (.model.display_name // "Claude"),
        (.workspace.project_dir // .workspace.current_dir // .cwd // ""),
        (.effort.level // ""),
        (.rate_limits.five_hour.used_percentage // "" | if . == "" then "" else (.|round|tostring) end),
        (.rate_limits.five_hour.used_percentage // "" | if . != "" and . > 0 and . < 1 then "1" else "" end),
        (.rate_limits.five_hour.resets_at // "" | tostring),
        (.rate_limits.seven_day.used_percentage // "" | if . == "" then "" else (.|round|tostring) end),
        (.rate_limits.seven_day.used_percentage // "" | if . != "" and . > 0 and . < 1 then "1" else "" end),
        (.rate_limits.seven_day.resets_at // "" | tostring),
        (if .fast_mode then "fast" else "" end)
    ] | join("\u001f")' <<<"$input" 2>/dev/null
)

window=""
case "$model" in
    *"(1M context)"*) model="${model% (1M context)}"; window="1M" ;;
esac

# ── row 1: what this session is ─────────────────────────
row1="${accent}${model}${off}"
w1=${#model}
if [ -n "$window" ]; then row1+=" ${dim}${window}${off}"; w1=$(( w1 + 1 + ${#window} )); fi
if [ -n "$effort" ]; then
    row1+="   ${dim}${effort}${off}"
    w1=$(( w1 + 3 + ${#effort} ))
fi
if [ -n "$fast" ]; then row1+="   ${dim}fast${off}"; w1=$(( w1 + 7 )); fi

if [ -n "$cwd" ]; then
    place=$(project "$cwd")
    pad=$(( COL - w1 )); [ "$pad" -lt 2 ] && pad=2
    row1+="$(printf '%*s' "$pad" '')${fg}${place}${off}"
    if branch=$(git -C "$cwd" symbolic-ref --short HEAD 2>/dev/null); then
        [ -n "$(git -C "$cwd" --no-optional-locks status --porcelain 2>/dev/null | head -1)" ] && branch+="*"
        row1+="  ${dim}${branch}${off}"
    fi
fi

# ── row 2: the budgets, on the grid ─────────────────────
row2=""
if [ -n "$five_pct" ]; then
    # A window that just reset reads as broken at "0%", so a real but sub-1% figure says so.
    five_show="$five_pct"; [ -n "$five_tiny" ] && five_show="<1"
    week_show="$week_pct"; [ -n "$week_tiny" ] && week_show="<1"
    s="${dim}session${off} $(meter "$five_pct") $(state_color "$five_pct")${five_show}%${off}"
    sw=$(( 8 + 10 + 1 + ${#five_show} + 1 ))
    r=$(when "$five_reset")
    if [ -n "$r" ]; then s+=" ${dim}↻${r}${off}"; sw=$(( sw + 2 + ${#r} )); fi
    row2+=$(cell "$s" "$sw")
    if [ -n "$week_pct" ]; then
        row2+="${dim}week${off} $(meter "$week_pct") $(state_color "$week_pct")${week_show}%${off}"
        wr=$(when "$week_reset")
        [ -n "$wr" ] && row2+=" ${dim}↻${wr}${off}"
    fi
fi

# ── out ─────────────────────────────────────────────────
printf '%b' "$row1"
[ -n "$row2" ] && printf '\n%b' "$row2"
exit 0
