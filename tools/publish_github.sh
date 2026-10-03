#!/usr/bin/env bash
# Publish to GitHub: https://github.com/ALEXalesha/AiFor3DPrinter
#
# NOT YET: Alexey checks the repository first (see README). Nothing calls this script automatically.
#
#   bash tools/publish_github.sh            # code only
#   bash tools/publish_github.sh vX.Y.Z     # code and tag
#   DRY_RUN=1 bash tools/publish_github.sh  # build and check github-main, no push
#
# The source of truth is Gitea. GitHub gets a copy of main that is rebuilt every time
# (hence --force: nothing but this script writes to main on GitHub). The working
# history on Gitea is not touched.
#
# Personal data is replaced in the published copy, in commit authors and in file
# contents across the whole history:
#   - the author's e-mail       -> the GitHub account's noreply address;
#   - the Gitea server address  -> gitea.local;
#   - the full name / surname   -> the nick ALEXalesha (plus git config publish.privateExtra);
#   - the local projects folder -> C:\Projects.
# None of these strings is written here: the e-mail comes from git config user.email,
# the address from remote origin (host only), the name from git config publish.privateName
# and publish.privateExtra (local settings, never in the history), the folder from the
# location of this repository.
#
# Tags: a release on GitHub is bound to a tag. Each GitHub tag that does not lead into
# the fresh branch is moved to its commit with the same author time and subject, so no
# old (unscrubbed) commit stays reachable. A new tag is pushed straight to GitHub,
# without a local tag (a local one would point at a rewritten commit and get mixed up
# with the Gitea tags).
set -euo pipefail

export PATH="$PATH:/c/Program Files/GitHub CLI"
REPO=ALEXalesha/AiFor3DPrinter
PUBLIC_EMAIL=203467574+ALEXalesha@users.noreply.github.com
PUBLIC_NAME=ALEXalesha
TAG="${1:-}"

cd "$(dirname "$0")/.."

PRIVATE_EMAIL="$(git config user.email)"
PRIVATE_NAME="$(git config --get publish.privateName || true)"
PRIVATE_EXTRA="$(git config --get publish.privateExtra || true)"
LAN_GITEA="$(git remote get-url origin | sed -E 's#^[a-z]+://([^/:]+).*#\1#')"
if [ -z "$PRIVATE_NAME" ]; then
  echo "git config publish.privateName is not set - the full name cannot be replaced" >&2
  exit 1
fi
if [ -n "$(git status --porcelain)" ]; then
  echo "commit first: filter-branch silently refuses to run on a dirty tree" >&2
  exit 1
fi

# --- the list of private strings and their public replacements -------------------
WIN_PARENT="$(cd .. && pwd -W)"                    # e.g. X:/Folder (forward slashes)
PAIRS=()                                           # "private<TAB>public"
add() { [ -n "$1" ] && [ ${#1} -ge 4 ] && PAIRS+=("$1"$'\t'"$2"); return 0; }
add "$PRIVATE_EMAIL" "$PUBLIC_EMAIL"
add "$LAN_GITEA" "gitea.local"
add "$PRIVATE_NAME" "$PUBLIC_NAME"
add "${PRIVATE_NAME##* }" "$PUBLIC_NAME"           # surname alone
IFS='|' read -r -a EXTRA <<< "$PRIVATE_EXTRA"
for x in "${EXTRA[@]:-}"; do add "$x" "$PUBLIC_NAME"; done
add "$WIN_PARENT" "C:/Projects"
add "${WIN_PARENT//\//\\}" 'C:\Projects'
add "$(cd .. && pwd)" "/c/Projects"
REL_PARENT="${WIN_PARENT#*:/}"                     # the folder path without the drive
add "$REL_PARENT" "Projects"
add "${REL_PARENT//\//\\}" "Projects"
add "${REL_PARENT//\//\\\\}" "Projects"            # as written inside Python strings

esc() { printf '%s' "$1" | sed 's/[]\/.[\*^$+?(){}|]/\\&/g'; }
SEDFILE="$(mktemp)"; GREPFILE="$(mktemp)"
trap 'rm -f "$SEDFILE" "$GREPFILE"' EXIT          # temp files of this script only
for pair in "${PAIRS[@]}"; do
  priv="${pair%%$'\t'*}"; pub="${pair#*$'\t'}"
  printf 's/%s/%s/gI\n' "$(esc "$priv")" "$(printf '%s' "$pub" | sed 's/[\/&]/\\&/g')" >> "$SEDFILE"
  printf '%s\n' "$priv" >> "$GREPFILE"
done
echo "private strings to replace: ${#PAIRS[@]}"

git remote get-url github >/dev/null 2>&1 || git remote add github "https://github.com/$REPO.git"

# --- authors ---------------------------------------------------------------------
git branch -f github-main main
FILTER_BRANCH_SQUELCH_WARNING=1 git filter-branch -f --env-filter "
  if [ \"\$GIT_AUTHOR_EMAIL\" = '$PRIVATE_EMAIL' ]; then export GIT_AUTHOR_EMAIL='$PUBLIC_EMAIL'; fi
  if [ \"\$GIT_COMMITTER_EMAIL\" = '$PRIVATE_EMAIL' ]; then export GIT_COMMITTER_EMAIL='$PUBLIC_EMAIL'; fi
  if [ \"\$GIT_AUTHOR_NAME\" = '$PRIVATE_NAME' ]; then export GIT_AUTHOR_NAME='$PUBLIC_NAME'; fi
  if [ \"\$GIT_COMMITTER_NAME\" = '$PRIVATE_NAME' ]; then export GIT_COMMITTER_NAME='$PUBLIC_NAME'; fi
" github-main >/dev/null 2>&1
git update-ref -d refs/original/refs/heads/github-main 2>/dev/null || true

# --- file contents across the history --------------------------------------------
# `|| true`: git grep returns 1 when nothing matches, and with pipefail the script
# would stop here silently.
DIRTY=$(git grep -I -i -l -F -f "$GREPFILE" $(git rev-list github-main) -- 2>/dev/null \
        | sed 's/^[^:]*://' | sort -u | tr '\n' ' ' || true)
if [ -n "$DIRTY" ]; then
  echo "private data in files: $DIRTY- rewriting contents"
  FILTER_BRANCH_SQUELCH_WARNING=1 git filter-branch -f --index-filter "
    for f in $DIRTY; do
      mode=\$(git ls-tree \$GIT_COMMIT -- \"\$f\" | awk '{print \$1}')
      [ -n \"\$mode\" ] || continue
      blob=\$(git cat-file blob \$GIT_COMMIT:\"\$f\" | sed -E -f '$SEDFILE' | git hash-object -w --stdin)
      git update-index --cacheinfo \$mode,\$blob,\"\$f\"
    done
  " github-main >/dev/null 2>&1
  git update-ref -d refs/original/refs/heads/github-main 2>/dev/null || true
fi

# --- verify before pushing (authors, messages, files) -------------------------------
if git log github-main --format='%ae%n%ce%n%an%n%cn%n%B' | grep -q -i -F -f "$GREPFILE"; then
  echo "private data left in commit authors or messages - push cancelled" >&2
  exit 1
fi
if git grep -I -q -i -F -f "$GREPFILE" $(git rev-list github-main) -- 2>/dev/null; then
  echo "private data left in the files of the published history - push cancelled" >&2
  exit 1
fi

# --- tags on GitHub that do not lead into the fresh branch --------------------------
MOVES=""
git for-each-ref --format='%(refname)' refs/github-tags | xargs -r -n1 git update-ref -d
git fetch -q github '+refs/tags/*:refs/github-tags/*' 2>/dev/null || true
for ref in $(git for-each-ref --format='%(refname)' refs/github-tags); do
  name=${ref#refs/github-tags/}
  old=$(git rev-parse "$ref^{commit}")
  git merge-base --is-ancestor "$old" github-main && continue
  key=$(git log -1 --format='%at %s' "$old")
  # awk reads to the end, no early exit: an early exit gives git log SIGPIPE (141),
  # and with pipefail the script would stop here silently.
  new=$(git log github-main --format='%H %at %s' \
        | awk -v k="$key" '{h=$1; $1=""; if (!f && substr($0,2)==k) {print h; f=1}}')
  if [ -z "$new" ]; then
    echo "tag $name: no commit with the same time and subject in the published branch - left alone" >&2
    continue
  fi
  echo "tag $name: $(git rev-parse --short "$old") -> $(git rev-parse --short "$new")"
  MOVES="$MOVES +$new:refs/tags/$name"
done
git for-each-ref --format='%(refname)' refs/github-tags | xargs -r -n1 git update-ref -d

echo "branch github-main: $(git rev-list --count github-main) commits, $(git log -1 --format=%h github-main)"
if [ "${DRY_RUN:-}" = 1 ]; then
  echo "DRY_RUN=1: checks passed, nothing pushed"
  exit 0
fi
gh auth setup-git
git push github github-main:main --force
if [ -n "$MOVES" ]; then
  git push github $MOVES
fi

if [ -n "$TAG" ]; then
  git push github "+github-main:refs/tags/$TAG"
  echo "tag $TAG pushed"
fi
