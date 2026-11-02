#!/usr/bin/env bash
# Known-good ("stable-*") states of IsaacLab-rocm and its component repositories (components.toml).
# Uses git only; never touches the GPU. See docs/UPDATING.md.
#
#   scripts/stable.sh status [--fetch]       per repo: branch, HEAD, nearest stable tag, dirty/unpushed
#   scripts/stable.sh checkout <tag> [--top] [--yes]
#                                            detach every component at <tag> (refuses if a repo is dirty);
#                                            --top also detaches this repository
#   scripts/stable.sh branches [--top] [--yes]
#                                            switch the components back to their branches (components.toml)
#   scripts/stable.sh tag <stable-name> [-m MSG] [--push] [--yes]
#                                            write requirements/lock-<date>.txt, update components.toml and
#                                            the README setup lines, commit them, and create the same
#                                            annotated tag in every repository
#   scripts/stable.sh push <stable-name> [--yes]
#                                            push the tag and the current branches to the remote `tfoehr`
#
# Isaac Lab is not forked: its tag is local only, and `checkout` falls back to the `base` commit
# recorded in components.toml at <tag> when the Isaac Lab checkout has no such tag.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PUSH_REMOTE=tfoehr
TOP_NAME=IsaacLab-rocm
TOP_BRANCH=main
# Packages installed by hand in the validated venv; the lock file lists them as comments.
HAND_INSTALLED=(py-spy)

die() { echo "stable.sh: $*" >&2; exit 1; }
usage() { sed -n '2,19s/^# \{0,1\}//p' "${BASH_SOURCE[0]}"; }

# --- components.toml --------------------------------------------------------------------------
# Prints "name|path|branch|tag|base" per section of a components.toml read from stdin.
parse_components() {
    awk '
        function flush() { if (sec != "") print sec "|" path "|" branch "|" tag "|" base }
        /^\[/ { flush(); sec = $0; gsub(/[][ \t]/, "", sec); path = branch = tag = base = ""; next }
        /^[a-z_]+[ \t]*=[ \t]*"/ {
            k = $0; sub(/[ \t]*=.*/, "", k); split($0, a, "\"")
            if (k == "path") path = a[2]; else if (k == "branch") branch = a[2]
            else if (k == "tag") tag = a[2]; else if (k == "base") base = a[2]
        }
        END { flush() }'
}

NAMES=() PATHS=() BRANCHES=() TAGS=() BASES=()
load_components() {
    [ -f "$root/components.toml" ] || die "no components.toml in $root"
    local n p b t base
    while IFS='|' read -r n p b t base; do
        NAMES+=("$n"); PATHS+=("$p"); BRANCHES+=("$b"); TAGS+=("$t"); BASES+=("$base")
    done < <(parse_components <"$root/components.toml")
    [ ${#NAMES[@]} -gt 0 ] || die "components.toml lists no components"
    for p in "${PATHS[@]}"; do
        git -C "$root/$p" rev-parse --git-dir >/dev/null 2>&1 || die "$p is not a git checkout (see README Setup)"
    done
}

# --- git helpers ------------------------------------------------------------------------------
g() { git -C "$1" "${@:2}"; }
cur_branch() { g "$1" symbolic-ref --quiet --short HEAD 2>/dev/null || true; }
tracked_changes() { g "$1" status --porcelain --untracked-files=no | wc -l; }
untracked_files() { g "$1" status --porcelain | grep -c '^??' || true; }
has_ref() { g "$1" rev-parse --verify --quiet "$2^{commit}" >/dev/null; }
has_remote() { g "$1" remote | grep -qx "$2"; }

# "stable-X" (at the tag), "stable-X +N" (N commits after it), or "-".
stable_position() {
    local d="$1" t n
    t="$(g "$d" describe --tags --abbrev=0 --match 'stable-*' HEAD 2>/dev/null)" || { echo "-"; return; }
    n="$(g "$d" rev-list --count "$t..HEAD")"
    if [ "$n" -eq 0 ]; then echo "$t"; else echo "$t +$n"; fi
}

confirm() {  # confirm "question"; honours --yes
    [ "${ASSUME_YES:-0}" = 1 ] && return 0
    [ -t 0 ] || die "not a terminal; pass --yes to confirm: $1"
    local a; read -r -p "$1 [y/N] " a
    [[ "$a" =~ ^[Yy]([Ee][Ss])?$ ]]
}

# repos: "label|dir|branch" for the top-level repo (optional) and every component.
repo_list() {  # repo_list [with_top]
    [ "${1:-}" = top ] && echo "$TOP_NAME|$root|$TOP_BRANCH"
    local i
    for i in "${!NAMES[@]}"; do echo "${NAMES[$i]}|$root/${PATHS[$i]}|${BRANCHES[$i]}"; done
}

require_clean() {  # require_clean dir label
    local n; n="$(tracked_changes "$1")"
    [ "$n" -eq 0 ] || die "$2 has $n uncommitted change(s) to tracked files; commit or stash them first"
}

# --- status -----------------------------------------------------------------------------------
cmd_status() {
    local fetch=0
    while [ $# -gt 0 ]; do
        case "$1" in --fetch) fetch=1 ;; *) die "status: unknown argument '$1'" ;; esac; shift
    done
    load_components
    local line label d br head pos dirty untr push tags_missing i base
    printf '%-14s %-22s %-10s %-24s %s\n' REPO BRANCH HEAD STABLE STATE
    while IFS='|' read -r label d br; do
        [ $fetch = 1 ] && { g "$d" fetch --all --quiet || echo "$label: fetch failed" >&2; }
        local b; b="$(cur_branch "$d")"
        head="$(g "$d" rev-parse --short=9 HEAD)"
        pos="$(stable_position "$d")"
        local state=()
        dirty="$(tracked_changes "$d")"; untr="$(untracked_files "$d")"
        [ "$dirty" -gt 0 ] && state+=("DIRTY:$dirty")
        [ "$untr" -gt 0 ] && state+=("untracked:$untr")
        [ -n "$br" ] && [ "$b" != "$br" ] && state+=("expected-branch:$br")
        if [ "$label" = isaaclab ]; then
            for i in "${!NAMES[@]}"; do [ "${NAMES[$i]}" = isaaclab ] && base="${BASES[$i]}"; done
            if [ -n "${base:-}" ] && has_ref "$d" "$base"; then
                if [ "$(g "$d" rev-parse HEAD)" = "$(g "$d" rev-parse "$base^{commit}")" ]; then
                    state+=("=components.toml-base")
                else state+=("differs-from-components.toml-base:$base"); fi
            fi
        fi
        if has_remote "$d" "$PUSH_REMOTE"; then
            if [ -n "$b" ] && has_ref "$d" "refs/remotes/$PUSH_REMOTE/$b"; then
                local ahead behind
                ahead="$(g "$d" rev-list --count "$PUSH_REMOTE/$b..HEAD")"
                behind="$(g "$d" rev-list --count "HEAD..$PUSH_REMOTE/$b")"
                [ "$ahead" -gt 0 ] && state+=("unpushed:$ahead")
                [ "$behind" -gt 0 ] && state+=("behind-$PUSH_REMOTE:$behind")
            elif [ -n "$b" ]; then
                state+=("no-$PUSH_REMOTE/$b")
            fi
            if [ $fetch = 1 ]; then
                tags_missing="$(comm -23 <(g "$d" tag -l 'stable-*' | sort) \
                    <(g "$d" ls-remote --tags --refs "$PUSH_REMOTE" 'stable-*' 2>/dev/null | sed 's#.*refs/tags/##' | sort) | paste -sd, -)"
                [ -n "$tags_missing" ] && state+=("tags-not-on-$PUSH_REMOTE:$tags_missing")
            fi
        fi
        [ ${#state[@]} -eq 0 ] && state=(clean)
        printf '%-14s %-22s %-10s %-24s %s\n' "$label" "${b:-(detached)}" "$head" "$pos" "${state[*]}"
    done < <(repo_list top)
    [ $fetch = 1 ] || echo "(unpushed/behind as of the last fetch; --fetch updates remotes and checks pushed tags)"
    # Pinned ROCm stack in the venv (read from package metadata; does not import anything).
    if [ -x "$root/.venv/bin/python" ] && command -v uv >/dev/null; then
        echo "venv: $(uv pip freeze --python "$root/.venv/bin/python" 2>/dev/null \
            | grep -E '^(torch|rocm|mujoco)==' | paste -sd' ' -)"
    fi
}

# --- checkout / branches ----------------------------------------------------------------------
# Commit-ish for component i at tag $1: its own tag, else (Isaac Lab) the base in components.toml@tag.
target_for() {
    local i="$1" tag="$2" d="$root/${PATHS[$1]}" base
    if has_ref "$d" "refs/tags/$tag"; then echo "refs/tags/$tag"; return; fi
    if [ "${NAMES[$i]}" = isaaclab ]; then
        base="$(git -C "$root" show "refs/tags/$tag:components.toml" 2>/dev/null \
            | parse_components | awk -F'|' '$1 == "isaaclab" { print $5 }')"
        [ -n "$base" ] || base="${BASES[$i]}"
        if [ -n "$base" ] && has_ref "$d" "$base"; then echo "$base"; return; fi
    fi
    return 1
}

lock_name() {  # lock file for a tag: stable-YYYY-MM-DD -> lock-YYYY-MM-DD.txt, else lock-<tag>.txt
    if [[ "$1" =~ ^stable-([0-9]{4}-[0-9]{2}-[0-9]{2})$ ]]; then echo "lock-${BASH_REMATCH[1]}.txt"
    else echo "lock-$1.txt"; fi
}

cmd_checkout() {
    local tag="" top=0
    while [ $# -gt 0 ]; do
        case "$1" in
            --top) top=1 ;; --yes|-y) ASSUME_YES=1 ;;
            -*) die "checkout: unknown option '$1'" ;;
            *) [ -z "$tag" ] || die "checkout: one tag only"; tag="$1" ;;
        esac; shift
    done
    [ -n "$tag" ] || die "checkout: missing <tag> (e.g. stable-2026-10-30)"
    load_components
    local i d tgt targets=() warp_rebuild=0
    for i in "${!NAMES[@]}"; do
        d="$root/${PATHS[$i]}"
        require_clean "$d" "${PATHS[$i]}"
        tgt="$(target_for "$i" "$tag")" || die "${PATHS[$i]} has no tag $tag (git -C ${PATHS[$i]} fetch $PUSH_REMOTE --tags?)"
        targets+=("$tgt")
    done
    if [ $top = 1 ]; then
        require_clean "$root" "$TOP_NAME"
        has_ref "$root" "refs/tags/$tag" || die "$TOP_NAME has no tag $tag"
    fi
    echo "Detach at $tag:"
    for i in "${!NAMES[@]}"; do
        d="$root/${PATHS[$i]}"
        printf '  %-28s %s -> %s (%s)\n' "${PATHS[$i]}" "$(g "$d" rev-parse --short=9 HEAD)" \
            "$(g "$d" rev-parse --short=9 "${targets[$i]}^{commit}")" "${targets[$i]#refs/tags/}"
    done
    [ $top = 1 ] && printf '  %-28s %s -> %s\n' "$TOP_NAME" "$(g "$root" rev-parse --short=9 HEAD)" \
        "$(g "$root" rev-parse --short=9 "refs/tags/$tag^{commit}")"
    confirm "Proceed?" || die "aborted"
    for i in "${!NAMES[@]}"; do
        d="$root/${PATHS[$i]}"
        if [ "${NAMES[$i]}" = warp ] && ! g "$d" diff --quiet HEAD "${targets[$i]}" -- warp/native build_lib.py; then
            warp_rebuild=1
        fi
        g "$d" -c advice.detachedHead=false checkout --quiet --detach "${targets[$i]}"
    done
    [ $top = 1 ] && g "$root" -c advice.detachedHead=false checkout --quiet --detach "refs/tags/$tag"
    echo "Done. Next (one GPU job at a time, nothing else running on the GPU):"
    if [ $warp_rebuild = 1 ]; then
        echo "  scripts/build_warp.sh            # Warp native code changed; also clears the kernel cache"
    else
        echo "  scripts/build_warp.sh            # only needed if the ROCm SDK changed (Warp native code is unchanged)"
    fi
    local lock; lock="requirements/$(lock_name "$tag")"
    if [ -f "$root/$lock" ]; then echo "  scripts/install.sh --constraint $lock"
    else echo "  scripts/install.sh               # no $lock in this checkout"; fi
    echo "  scripts/healthcheck.sh"
    if [ $top = 1 ]; then echo "Back to the working branches: git checkout $TOP_BRANCH && scripts/stable.sh branches"
    else echo "Back to the working branches: scripts/stable.sh branches"; fi
}

cmd_branches() {
    local top=0
    while [ $# -gt 0 ]; do
        case "$1" in --top) top=1 ;; --yes|-y) ASSUME_YES=1 ;; *) die "branches: unknown argument '$1'" ;; esac; shift
    done
    load_components
    local label d br
    while IFS='|' read -r label d br; do
        [ -n "$br" ] || { echo "  $label: no branch in components.toml, left as is"; continue; }
        require_clean "$d" "$label"
        has_ref "$d" "refs/heads/$br" || die "$label has no local branch $br"
    done < <(repo_list $([ $top = 1 ] && echo top))
    confirm "Switch the repositories back to their branches?" || die "aborted"
    while IFS='|' read -r label d br; do
        [ -n "$br" ] || continue
        g "$d" checkout --quiet "$br"
        printf '  %-14s %s %s\n' "$label" "$br" "$(g "$d" rev-parse --short=9 HEAD)"
    done < <(repo_list $([ $top = 1 ] && echo top))
    echo "Rebuild and reinstall as after a checkout (scripts/build_warp.sh, scripts/install.sh)."
}

# --- tag --------------------------------------------------------------------------------------
write_lock() {  # write_lock file tag date
    local out="$1" tag="$2" date="$3" py="$root/.venv/bin/python" i hashes=() hw cpu gfx osname pyver kernel
    [ -x "$py" ] || die "no venv at $root/.venv"
    for i in "${!NAMES[@]}"; do
        local label="${PATHS[$i]##*/}"
        hashes+=("$label $(g "$root/${PATHS[$i]}" rev-parse --short=12 HEAD)")
    done
    pyver="$("$py" -c 'import platform; print(platform.python_version())')"
    kernel="$(uname -r)"; kernel="${kernel%%-*}"
    osname="$( (. /etc/os-release 2>/dev/null && echo "${NAME:-Linux}") || echo Linux)"; osname="${osname% Linux}"
    cpu="$(awk -F': ' '/^model name/ { print $2; exit }' /proc/cpuinfo)"
    cpu="${cpu#AMD }"; cpu="${cpu/ w\/ / / }"; cpu="${cpu/RYZEN/Ryzen}"
    gfx="$(grep -oE 'v2/gfx[0-9a-zA-Z-]+' "$root/scripts/install.sh" | head -1 | cut -d/ -f2)"
    hw="$cpu${gfx:+ ($gfx)}"
    local comps; comps="$(printf '%s, ' "${hashes[@]}")"; comps="${comps%, }"
    local freeze; freeze="$(uv pip freeze --python "$py")"
    {
        echo "# Exact package set of the validated IsaacLab-rocm venv ($date), from"
        echo "#   uv pip freeze --python .venv/bin/python"
        echo "# Python $pyver, Linux $kernel ($osname), $hw."
        echo "# Components (tag $tag): ${hashes[0]}, ${hashes[1]},"
        echo "# ${comps#"${hashes[0]}, ${hashes[1]}, "}. Includes the asset-importer packages (requirements/importers.txt)."
        echo "#"
        echo "# A fresh install that follows README.md reproduces every non-comment line below exactly."
        echo "#"
        echo "# The editable installs of the local checkouts (third_party/*, isaaclab/*) are listed as"
        echo "# comments: they point at local paths and are installed by scripts/build_warp.sh and"
        echo "# scripts/install.sh. Use this file as a constraints file (see README \"Reproducing the"
        echo "# validated environment\"):"
        echo "#   scripts/install.sh --constraint requirements/${out##*/}"
        local line pkg h
        while IFS= read -r line; do
            if [[ "$line" == "-e file://"* ]]; then
                line="${line#-e file://}"; echo "# editable (local checkout): ${line#"$root"/}"; continue
            fi
            pkg="${line%%[=@ ]*}"
            for h in "${HAND_INSTALLED[@]}"; do
                if [ "${pkg,,}" = "$h" ]; then
                    line="# $line (installed by hand in the validated venv; the setup scripts do not install it)"
                fi
            done
            echo "$line"
        done <<<"$freeze"
    } >"$out"
}

update_components_toml() {  # tag isaaclab_commit
    local f="$root/components.toml" tmp
    tmp="$(mktemp "$f.XXXXXX")"
    awk -v tag="$1" -v il="$2" '
        /^\[/ { sec = $0; gsub(/[][ \t]/, "", sec) }
        /^tag[ \t]*=[ \t]*"/ { sub(/"[^"]*"/, "\"" tag "\"") }
        sec == "isaaclab" && il != "" && /^base[ \t]*=[ \t]*"/ { sub(/"[^"]*"/, "\"" il "\"") }
        { print }' "$f" >"$tmp"
    mv "$tmp" "$f"
}

update_readme() {  # tag isaaclab_commit lockfile: the setup lines that name the stable state
    local f="$root/README.md"
    [ -f "$f" ] || return 0
    sed -i -E \
        -e "s/(git clone -b )stable-[A-Za-z0-9._-]+/\1$1/" \
        -e "s/(git -C isaaclab checkout )[0-9a-f]{7,40}/\1$2/" \
        -e "s/(upstream Isaac Lab at commit \`)[0-9a-f]{7,40}\`/\1$2\`/" \
        -e "s/^The \`stable-[A-Za-z0-9._-]+\` tags mark/The \`$1\` tags mark/" \
        -e "s/(which match the \`)stable-[A-Za-z0-9._-]+(\` tags)/\1$1\2/" \
        -e "s#requirements/lock-[A-Za-z0-9._-]+\.txt#requirements/$3#g" \
        "$f"
}

cmd_tag() {
    local tag="" msg="" push=0
    while [ $# -gt 0 ]; do
        case "$1" in
            --push) push=1 ;; --yes|-y) ASSUME_YES=1 ;;
            -m) [ $# -ge 2 ] || die "tag: -m needs a message"; msg="$2"; shift ;;
            -*) die "tag: unknown option '$1'" ;;
            *) [ -z "$tag" ] || die "tag: one name only"; tag="$1" ;;
        esac; shift
    done
    [ -n "$tag" ] || die "tag: missing <name> (e.g. stable-$(date +%F))"
    [[ "$tag" =~ ^stable-[A-Za-z0-9._-]+$ ]] || die "tag: the name must look like stable-YYYY-MM-DD (letters, digits, . _ -)"
    git check-ref-format "refs/tags/$tag" || die "tag: invalid tag name '$tag'"
    command -v uv >/dev/null || die "uv not found"
    load_components
    local label d br date lock il
    [ -n "$msg" ] || msg="Known-good IsaacLab-rocm state ($tag)"
    if [[ "$tag" =~ ^stable-([0-9]{4}-[0-9]{2}-[0-9]{2})$ ]]; then date="${BASH_REMATCH[1]}"; else date="$(date +%F)"; fi
    lock="$(lock_name "$tag")"
    # Checks first; nothing is changed until every repository passes them.
    while IFS='|' read -r label d br; do
        require_clean "$d" "$label"
        has_ref "$d" "refs/tags/$tag" && die "$label already has a tag $tag"
        local b; b="$(cur_branch "$d")"
        if [ -n "$br" ] && [ "$b" != "$br" ]; then
            echo "warning: $label is on '${b:-(detached)}', components.toml says '$br'" >&2
        fi
    done < <(repo_list top)
    [ -e "$root/requirements/$lock" ] && die "requirements/$lock already exists"
    for label in "${NAMES[@]}"; do [ "$label" = isaaclab ] && il="$(g "$root/isaaclab" rev-parse --short=9 HEAD)"; done
    write_lock "$root/requirements/$lock" "$tag" "$date"
    update_components_toml "$tag" "${il:-}"
    update_readme "$tag" "${il:-}" "$lock"
    echo "Tag $tag (message: $msg)"
    while IFS='|' read -r label d br; do
        printf '  %-14s %-22s %s\n' "$label" "$(cur_branch "$d")" "$(g "$d" rev-parse --short=9 HEAD)"
    done < <(repo_list)
    echo "  $TOP_NAME: new commit with requirements/$lock, components.toml, README.md:"
    g "$root" --no-pager diff --stat -- components.toml README.md
    g "$root" --no-pager diff -U0 -- components.toml README.md | grep -E '^[-+][^-+]' || true
    if ! confirm "Commit these files and create the tags?"; then
        g "$root" checkout --quiet -- components.toml README.md
        rm -f "$root/requirements/$lock"
        die "aborted; components.toml/README.md restored and requirements/$lock removed"
    fi
    while IFS='|' read -r label d br; do
        g "$d" tag -a "$tag" -m "$msg"
        echo "  tagged $label"
    done < <(repo_list)
    g "$root" add -- "requirements/$lock" components.toml README.md
    g "$root" commit --quiet -m "Mark the $date state as $tag and add its lock file" \
        -- "requirements/$lock" components.toml README.md
    g "$root" tag -a "$tag" -m "$msg"
    echo "  committed and tagged $TOP_NAME ($(g "$root" rev-parse --short=9 HEAD))"
    if [ $push = 1 ]; then cmd_push "$tag"; else echo "Not pushed. Later: scripts/stable.sh push $tag"; fi
}

cmd_push() {
    local tag=""
    while [ $# -gt 0 ]; do
        case "$1" in --yes|-y) ASSUME_YES=1 ;; -*) die "push: unknown option '$1'" ;; *) tag="$1" ;; esac; shift
    done
    [ -n "$tag" ] || die "push: missing <tag>"
    [ ${#NAMES[@]} -gt 0 ] || load_components
    local label d br b plan=()
    while IFS='|' read -r label d br; do
        has_ref "$d" "refs/tags/$tag" || { echo "  $label: no tag $tag, skipped"; continue; }
        has_remote "$d" "$PUSH_REMOTE" || { echo "  $label: no remote $PUSH_REMOTE, skipped (local tag only)"; continue; }
        b="$(cur_branch "$d")"
        plan+=("$label|$d|$b")
        echo "  $label: push ${b:+branch $b and }tag $tag to $PUSH_REMOTE ($(g "$d" remote get-url "$PUSH_REMOTE"))"
    done < <(repo_list top)
    [ ${#plan[@]} -gt 0 ] || die "nothing to push"
    confirm "Push?" || die "aborted"
    local p
    for p in "${plan[@]}"; do
        IFS='|' read -r label d b <<<"$p"
        if [ -n "$b" ]; then g "$d" push "$PUSH_REMOTE" "refs/heads/$b:refs/heads/$b" "refs/tags/$tag"
        else g "$d" push "$PUSH_REMOTE" "refs/tags/$tag"; fi
    done
}

main() {
    [ $# -gt 0 ] || { usage; exit 2; }
    local cmd="$1"; shift
    case "$cmd" in
        status) cmd_status "$@" ;;
        checkout) cmd_checkout "$@" ;;
        branches) cmd_branches "$@" ;;
        tag) cmd_tag "$@" ;;
        push) cmd_push "$@" ;;
        -h|--help|help) usage ;;
        *) usage >&2; die "unknown command '$cmd'" ;;
    esac
}

# The whole script is parsed before main runs, so `checkout --top` can replace this file safely.
main "$@"
exit
