#!/bin/bash
# thor-bench wrapper — automatically sets LD_PRELOAD for nocudaextend.so
# to prevent libnvcuextend.so .init constructor crash (driver bug 595.58.03)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NOCUDAEXTEND="$SCRIPT_DIR/nocudaextend.so"

if [ -f "$NOCUDAEXTEND" ]; then
    if [ -z "$LD_PRELOAD" ]; then
        export LD_PRELOAD="$NOCUDAEXTEND"
    elif ! echo "$LD_PRELOAD" | grep -q "nocudaextend"; then
        export LD_PRELOAD="$NOCUDAEXTEND:$LD_PRELOAD"
    fi
    EXEC_ARGS=("$SCRIPT_DIR/thor_bench" "$@")
else
    EXEC_ARGS=("$SCRIPT_DIR/thor_bench" "$@")
fi

exec "${EXEC_ARGS[@]}"
