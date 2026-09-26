#!/bin/bash
# gpu-router status line wrapper (installed by `gpu statusline install`; undo with
# `gpu statusline uninstall`).
#
# Claude Code runs this instead of your own status line command. It runs your original
# command with the same stdin JSON, prints its output unchanged, then appends the rows of
# `gpu status --line`: 0-2 rows, only while GPU work is active. If gpu is missing or
# broken you get exactly your own line.
#
# Where things come from (first match wins):
#   your command   $GPU_STATUSLINE_ORIGINAL, else <this dir>/original-command
#   gpu            $GPU_ROUTER_BIN, else <this dir>/gpu-bin, else `gpu` on PATH
#   gpu data dir   $GPU_ROUTER_HOME, else <this dir>/gpu-home
#
# Same bytes as src/gpu_router/statusline/gpu-statusline.sh (a test keeps them equal).

IFS= read -r -d '' input   # all of stdin, byte for byte (no trailing-newline trimming)

here="${BASH_SOURCE[0]%/*}"
[ "$here" = "${BASH_SOURCE[0]}" ] && here=.

if [ -n "${GPU_STATUSLINE_ORIGINAL+set}" ]; then
    orig="$GPU_STATUSLINE_ORIGINAL"
elif [ -f "$here/original-command" ]; then
    orig=$(<"$here/original-command")
else
    orig=""
fi
# never run ourselves again (a statusLine that points back at this wrapper)
if [ -n "${GPU_STATUSLINE_ACTIVE:-}" ] || [[ "$orig" == *gpu-statusline.sh* ]]; then
    orig=""
fi
export GPU_STATUSLINE_ACTIVE=1

gpu="${GPU_ROUTER_BIN:-}"
[ -z "$gpu" ] && [ -f "$here/gpu-bin" ] && gpu=$(<"$here/gpu-bin")
[ -z "$gpu" ] && gpu=$(command -v gpu 2>/dev/null)
if [ -z "${GPU_ROUTER_HOME:-}" ] && [ -f "$here/gpu-home" ]; then
    GPU_ROUTER_HOME=$(<"$here/gpu-home")
    export GPU_ROUTER_HOME
fi

# The gpu rows start first and render while your command runs (they only read a file).
# The first line on fd 3 is gpu's pid (sh execs into gpu), so a stuck gpu can be stopped.
have_gpu=""
if [ -n "$gpu" ] && [ -x "$gpu" ]; then
    have_gpu=1
    exec 3< <(exec 2>/dev/null; printf '%s' "$input" |
        /bin/sh -c 'echo $$; exec "$0" status --line --stdin' "$gpu")
fi

mine=""
if [ -n "$orig" ]; then
    mine=$(printf '%s' "$input" | /bin/sh -c "$orig" 3<&-)  # fd 3 is ours, not theirs
fi
printf '%s' "$mine"  # your lines never wait for gpu

rows=""
if [ -n "$have_gpu" ]; then
    # at most ~1 s for the rows (bash 3.2 takes whole seconds); a slow or stuck gpu costs
    # only its own rows: a timeout leaves them empty (bash 4+ returns >128 with a partial
    # read, dropped too) and the gpu process is stopped
    gpid=""
    IFS= read -r -t 1 gpid <&3
    IFS= read -r -t 1 -d '' rows <&3
    [ $? -gt 128 ] && rows=""
    [ -z "$rows" ] && [ -n "$gpid" ] && kill "$gpid" 2>/dev/null
    exec 3<&-
fi
while [ "${rows%$'\n'}" != "$rows" ]; do rows="${rows%$'\n'}"; done

if [ -n "$rows" ]; then
    [ -n "$mine" ] && printf '\n'
    printf '%s' "$rows"
fi
exit 0
