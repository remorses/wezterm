# WezTerm Fork — Release CI

This is a fork of [wezterm/wezterm](https://github.com/wezterm/wezterm) with CI configured to publish binary releases on GitHub.

The upstream repo guards release uploads with `if: github.repository == 'wezterm/wezterm'`, preventing any fork from publishing. This fork replaces that guard with `if: true` and removes external publishing steps (homebrew tap, gemfury, winget, flathub) that require upstream-only secrets.

## Changes from upstream

- `if: github.repository == 'wezterm/wezterm'` → `if: true` on all build/upload jobs
- Removed homebrew tap push (`wez/homebrew-wezterm`, needs `GH_PAT`)
- Removed linuxbrew tap push (`wez/homebrew-wezterm-linuxbrew`, needs `GH_PAT`)
- Removed winget manifest push (`wez/winget-pkgs`, needs `GH_PAT`)
- Removed flathub PR (`flathub/org.wezfurlong.wezterm`, needs `GH_PAT`)
- Removed gemfury upload (needs `FURY_TOKEN`)
- Removed macOS code signing env vars (needs Apple Developer secrets)
- Disabled cron schedules on continuous workflows to save CI minutes
- Added `workflow_dispatch` trigger to all tag workflows for manual dispatch

## Triggering a release

### Option 1: Push a tag (triggers all 11 platforms at once)

Tags matching `20*` trigger all `*_tag.yml` workflows simultaneously.

```bash
# From main branch
TAG="$(date -u +%Y%m%d-%H%M%S)-$(git rev-parse --short=8 HEAD)"
git tag "$TAG"
git push origin "$TAG"
```

This builds and uploads binaries for: macOS (universal), Windows (.exe + .zip), Ubuntu 20.04/22.04/24.04, Debian 11/12, CentOS 9, Fedora 39/40/41.

The tag workflows auto-create a GitHub pre-release via `ci/create-release.sh` if one doesn't exist, then upload artifacts to it.

### Option 2: Manual dispatch (selective platforms)

Trigger individual platform builds without creating a tag:

```bash
# Single platform
gh workflow run macos_tag --repo remorses/wezterm

# All platforms
for wf in centos9_tag debian11_tag debian12_tag fedora39_tag fedora40_tag fedora41_tag macos_tag ubuntu20.04_tag ubuntu22.04_tag ubuntu24.04_tag windows_tag; do
  gh workflow run "$wf" --repo remorses/wezterm
done
```

### Option 3: Build from a feature branch

Tags are commit-based, not branch-based. To build from any branch:

```bash
# Tag a specific branch
git checkout my-feature-branch
TAG="$(date -u +%Y%m%d-%H%M%S)-$(git rev-parse --short=8 HEAD)"
git tag "$TAG"
git push origin "$TAG"
```

Or use `workflow_dispatch` with a branch ref:

```bash
gh workflow run macos_tag --repo remorses/wezterm --ref my-feature-branch
```

Note: `workflow_dispatch` runs use `ci/tag-name.sh` to generate the release name from the HEAD commit date + hash. Different commits produce different release names, so builds from different branches won't collide.

## macOS code signing

Builds are currently **unsigned**. Users must bypass Gatekeeper:

```bash
xattr -cr /Applications/WezTerm.app
```

To re-enable signing, add these secrets to the repo (Settings → Secrets → Actions):

| Secret | Source |
|---|---|
| `MACOS_TEAM_ID` | Apple Developer account → Membership |
| `MACOS_CERT` | Base64-encoded .p12 exported from Keychain Access |
| `MACOS_CERT_PW` | Base64-encoded password used during .p12 export |
| `MACOS_APPLEID` | Apple ID email |
| `MACOS_APP_PW` | App-specific password from appleid.apple.com |

Then restore the env block in `gen_macos_tag.yml` and `gen_macos_continuous.yml`:

```yaml
- name: "Package"
  env:
    MACOS_APPLEID: ${{ secrets.MACOS_APPLEID }}
    MACOS_APP_PW: ${{ secrets.MACOS_APP_PW }}
    MACOS_CERT: ${{ secrets.MACOS_CERT }}
    MACOS_CERT_PW: ${{ secrets.MACOS_CERT_PW }}
    MACOS_TEAM_ID: ${{ secrets.MACOS_TEAM_ID }}
  run: "bash ci/deploy.sh"
```

All signing/notarization logic is already in `ci/deploy.sh` behind `if [ -n "$MACOS_TEAM_ID" ]` guards.

## Syncing with upstream

```bash
git remote add upstream https://github.com/wezterm/wezterm.git
git fetch upstream
git merge upstream/main
```

After merging, the workflow changes in `.github/workflows/` may conflict. Our changes are minimal (guard removal + step deletion) so conflicts should be easy to resolve.

## CI minutes

A full release across all 11 platforms takes roughly 300-500 GitHub Actions minutes. The free tier provides 2000 min/month. Cron schedules are disabled to avoid burning minutes on nightly builds — only manual dispatch and tag pushes trigger builds.
