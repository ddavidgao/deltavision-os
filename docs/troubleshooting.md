# Troubleshooting

Known issues we've hit, with fixes. Add entries here when you discover a
new one — future you will thank past you.

---

## Install & import

### `import deltavision_os` raises ModuleNotFoundError

Happens on old wheels (0.1.0a0 and earlier). Upgrade:

```bash
pip install --upgrade deltavision-os
```

If it still fails: you're in a cwd with a local `deltavision_os/` directory
that's shadowing the install. `cd /tmp` and retry.

### `ImportError: cannot import name 'X' from 'vision'`

Your cwd has a `vision/` directory (or `agent/`, `capture/`, `model/`,
`observation/`, `results/`) that's unrelated to DeltaVision but happens to
use one of its internal module names. Older versions of this library kept
modules at site-packages root; `>=0.1.0a1` nests everything under
`deltavision_os/` so this shouldn't happen. Upgrade, or `cd` away from the
offending directory.

---

## `OSNativePlatform` (native desktop capture)

### `pyautogui.FailSafeException` or `Xlib.error.DisplayConnectionError`

`--platform os` and `OSNativePlatform.setup()` require a real X display
(Linux) or Quartz (macOS) or Desktop Window Manager (Windows). **Will not
work in headless CI.** Options:

- Run on a real desktop session.
- Run on a machine with `xvfb` (Linux) and launch via `xvfb-run`.
- Use `OSWorldPlatform` instead — the VM provides its own display.

### pyautogui hotkeys (Cmd+Space, Ctrl+Up, etc.) silently don't register

Happens intermittently on macOS when the invoking process lacks Accessibility
permission. Grant Accessibility to Terminal / iTerm / Claude Code / whatever
Python is running from, in **System Settings → Privacy & Security →
Accessibility**. The intermittency is because the grant is per-process-identity
and can silently revoke after a restart. Verify with a single-shot test:

```python
import pyautogui
pyautogui.hotkey('cmd', 'space')  # Spotlight should open
```

If it doesn't, fix the permission before running any ablation that relies
on keyboard shortcuts triggering real UI changes.

---

## `OSWorldPlatform` (OSWorld VM integration)

### Docker Desktop won't start over SSH (`com.docker.service` Stopped / Manual)

Docker Desktop on Windows requires an interactive session for its credstore
helper (`docker-credential-desktop.exe`). From a headless SSH session, the
helper returns *"A specified logon session does not exist"* and all docker
commands fail. **Use WSL2 native docker instead:**

```bash
wsl -d Ubuntu-24.04 -u root -- service docker start
wsl -d Ubuntu-24.04 -- bash -c "docker ps"
echo '{}' > ~/.docker/config.json  # clear the credsStore pointer
```

Then construct `DesktopEnv(provider_name="docker", ...)` from inside WSL.
Don't try to bridge from Windows-side docker to WSL-side docker — just run
everything in WSL.

### VM zip gets corrupted by partial download → `BadZipFile: Bad magic number`

OSWorld's `_download_vm` opens the zip in append mode (`ab`). If the download
is interrupted and resumed, it can concatenate partial content into the same
file and end up with a corrupted ~24 GB zip. Fix:

```bash
rm -f docker_vm_data/Ubuntu.qcow2.zip
# retry the DesktopEnv() construction — it will redownload cleanly
```

### `require_a11y_tree=True` hangs for 10 minutes on LibreOffice Calc

Known upstream bug: [OSWorld#185](https://github.com/xlang-ai/OSWorld/issues/185).
AT-SPI walks every rendered cell. Workarounds:

- Skip Calc tasks for now (in `run_osworld.py` filter by
  `category != "libreoffice_calc"`).
- Wrap `controller.get_accessibility_tree()` with a 3–5 s timeout; on
  timeout set `a11y_fallback: "timeout"` and keep going. The a11y hybrid
  in [`observation/a11y.py`](../deltavision_os/observation/a11y.py) already
  fails open with a `status` field for the model to see.

### Chrome tasks fail with *"Playwright Sync API inside the asyncio loop"*

Also upstream. `_chrome_open_tabs_setup` uses sync Playwright inside async
code. These tasks score 0 through no fault of your agent. Filter them out
until fixed upstream — there are 7 such tasks in `test_small.json`.

---

## Model backends

### Local `llama-server` dies silently under `Start-Process -WindowStyle Hidden`

Starting llama-server (llama.cpp) via `powershell Start-Process -WindowStyle
Hidden` on Windows results in the child process dying during tensor load
with no error in the log. Process detachment / stdout handle issue.

**Fix: run via SSH foreground.** From a Mac/Linux shell:

```bash
ssh david-computer 'cd C:\Users\david\llama.cpp && llama-server.exe \
    -m model.gguf --mmproj mmproj.gguf --host 0.0.0.0 --port 8080 -ngl 99' \
    > /tmp/llama.log 2>&1 &
```

The SSH session keeps the process's parent alive. Kill with
`ssh david-computer 'taskkill /F /IM llama-server.exe'`.

### `BadRequestError` partway through an OSWorld run

Usually token-budget overrun. llama-server's default context (`--ctx-size
4096` or 8192) fills up fast with full-frame VLM observations + multi-step
conversations. Bump to 16k:

```
llama-server.exe ... --ctx-size 16384
```

If the BadRequests recur at 16k, check whether the agent is sending delta
observations (the point of DeltaVision) or whether something is forcing
full frames every step.

### UI-TARS-1.5-7B at Q4 quantization terminates early on delta observations

Documented finding. Action-tuned VLMs trained on full screenshots appear to
interpret a diff-heatmap + crop payload as "task complete" and emit
`done=True` after 1–2 steps. Two workarounds:

1. Use `--force-full-frame` with UI-TARS; lose DeltaVision's token savings
   but recover trajectory length.
2. Use a general-reasoning VLM (Claude Sonnet, Qwen2.5-VL) and enable the
   a11y hybrid — general VLMs handle delta crops as intended when given
   structured text about what elements are visible.

---

## Reproducibility / benchmark DB

### `results/deltavision.db` doesn't exist after `pip install`

The SQLite DB is created on first benchmark run, in the cwd's `results/`
directory. If you want it pre-created:

```python
from deltavision_os.results.store import ResultStore
db = ResultStore()  # creates ./results/deltavision.db and the schema
db.close()
```

### Per-run artifact dir missing after crash mid-run

Artifact directories are created BEFORE the DB row saves, on the
`save_run` path. If your runner crashes between capture and DB-save, you
may have an orphan `benchmarks/runs/<bench>_<backend>/run_?/` with data
but no corresponding DB row. Safe to delete, or backfill via
`benchmarks/backfill_runs.py` if you point it at the right JSON.

---

## Long-running benchmark has gone silent

stdout is block-buffered when not attached to a terminal (e.g. when
redirected via `tee` or `>` or backgrounded). A 30-minute OSWorld run
looks dead for 29 of those 30 minutes even though it's healthy.

Always prefix benchmark commands with `PYTHONUNBUFFERED=1`:

```bash
PYTHONUNBUFFERED=1 python benchmarks/run_osworld.py ...
```

Or equivalently, pass `-u`:

```bash
python -u benchmarks/run_osworld.py ...
```

---

## `sqlite3: command not found` in fresh WSL Ubuntu

The sqlite3 CLI isn't in Ubuntu 24.04's base install. Two options:

```bash
# Install it
sudo apt install sqlite3

# OR use Python's stdlib (no install needed) — see docs/benchmarks.md
# for a ready-made query snippet.
```

---

## OSWorld VM triggers an 11 GB re-download when you run from a fresh cwd

OSWorld's docker provider resolves `./docker_vm_data/` relative to the
**current working directory** at env-construction time. If your existing
VM image lives in `/mnt/c/Users/david/OSWorld/docker_vm_data/` but you run
a benchmark script from `/tmp/my_clone/`, OSWorld doesn't find the image
and starts a fresh download.

As of commit after 2026-04-19, `benchmarks/run_osworld.py` `chdir`s into
`--oswo-repo` before constructing DesktopEnv so the existing image is
reused. If you're writing your own runner that uses OSWorld directly:

```python
import os
os.chdir(str(oswo_repo_path))  # BEFORE DesktopEnv(...)
env = DesktopEnv(provider_name="docker", ...)
```

---

## Default `--max-tasks 3` gives you 0/3 successes

OSWorld's `test_small.json` happens to have Chrome tasks first, which hit
the upstream Playwright-asyncio bug and fail at setup (pre-agent-loop).
A naive `--max-tasks 3` thus produces guaranteed 0/3 with no signal.

As of commit after 2026-04-19, `run_osworld.py` skips `chrome` and
`libreoffice_calc` by default (both have known-broken upstream code — see
the OSWorld-integration section above). You can:

```bash
# Pick specific categories (recommended for the comprehensive test):
python benchmarks/run_osworld.py --categories gimp,libreoffice_writer,vs_code --max-tasks 3 ...

# Or disable the skip list (paper runs, once upstream fixes land):
python benchmarks/run_osworld.py --no-skip-default-broken ...
```

---

## "Everything is broken, where do I start?"

1. Does `python -c "import deltavision_os; print(deltavision_os.__version__)"`
   succeed? If no, packaging problem — see top of this doc.
2. Does the [quickstart smoke test](quickstart.md#3-a-runnable-smoke-test)
   run from `/tmp`? If no, display/permission problem.
3. Does a single OSWorld task round-trip work (one `env.reset` +
   `env.step("WAIT")` + `env.evaluate()`)? If no, VM setup problem — see
   the OSWorld section above.
4. Does `llama-server --help` work and does a single image + prompt return
   a JSON response? If no, model backend problem.

Each layer built on the last. Verify each before composing.
