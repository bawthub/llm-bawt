#!/bin/sh
# TASK-1062: shared host toolchain; run with sudo only after host authorization.
# Retain distribution packages (and their dependants); never overwrite /usr/bin.
set -eu
[ "$(id -u)" = 0 ] || { printf '%s\n' 'Run as root after host authorization.' >&2; exit 1; }
version=24.21.0
prefix=/opt/node-v${version}
case "$(uname -m)" in
    x86_64) arch=x64; sha=fd8e59d5a511510f6a298afb548f18c7d2b1be404d8b4a27d94fbe49f56cb2d6 ;;
    aarch64) arch=arm64; sha=6ad1325edbdb5649c379b75a237147a666c95d4f9ae8d340fef2d1575d289ad2 ;;
    *) printf '%s\n' 'Unsupported architecture.' >&2; exit 1 ;;
esac
if [ ! -d "$prefix" ]; then
    work=$(mktemp -d)
    trap 'rm -rf "$work"' EXIT HUP INT TERM
    curl -fsSL "https://nodejs.org/dist/v${version}/node-v${version}-linux-${arch}.tar.xz" -o "$work/node.tar.xz"
    printf '%s  %s/node.tar.xz\n' "$sha" "$work" | sha256sum -c -
    mkdir "$work/runtime"
    tar -xJf "$work/node.tar.xz" -C "$work/runtime" --strip-components=1 --no-same-owner
    test "$("$work/runtime/bin/node" --version)" = "v$version"
    mv "$work/runtime" "$prefix"
fi
test "$("$prefix/bin/node" --version)" = "v$version"
export PATH="$prefix/bin:$PATH"
# Do not treat the invoking user's home .npmrc as a root project config.
cd /
npm install --global --prefix "$prefix" corepack@0.36.0
corepack enable --install-directory "$prefix/bin"
# Capture every replaced local command before updating the shared default.
backup=/var/lib/llm-bawt/runtime-backups/TASK-1062-$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$backup" /usr/local/bin
for command in node npm npx corepack pnpm pnpx; do
    target=/usr/local/bin/$command
    if [ -e "$target" ] || [ -L "$target" ]; then
        cp -a "$target" "$backup/$command"
    else
        printf '%s\n' "$command" >> "$backup/previously-absent"
    fi
    test -e "$prefix/bin/$command"
    ln -sfn "$prefix/bin/$command" "$target"
done
printf 'Node %s installed; command rollback records: %s\n' "$version" "$backup"
node --version
npm --version
corepack --version
