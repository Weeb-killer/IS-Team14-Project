# SecBERT HTTP Gateway APP

## Model weights download

`secbert-concept.zip` : https://drive.google.com/file/d/1RPF_cLTZ-FXRVC8Kq_3wlR1E5uOTQJBA/view?usp=drive_link

`secbert-frozen.zip` : https://drive.google.com/file/d/1fLKMIr7g10elq0ODzX6Czw9OHj48fgFf/view?usp=drive_link

## 1. Prepare the project and model files

Use Python 3.11 or newer. Keep the training repository beside this APP so its `http_attack_agent` package can be imported. Place your trained SecBERT checkpoint and matching concept probes in this layout:

```text
IS-Team14-Project/
  src/http_attack_agent/

IS-Team14-Project-APP/
  manage.py
  artifacts/
    secbert-frozen/
      metadata.json
      model.pt
      embedding_head.pt
      backbone_config/config.json
      tokenizer/
        tokenizer.json
        tokenizer_config.json
    secbert-concepts/
      bank/
        concept_bank.npz
        probe_report.json
    crs-4.10.0/                 # Required when enabling live CRS
      crs-setup.conf.example
      rules/
        REQUEST-*.conf
        *.data
```

`SECBERT_CHECKPOINT` points to `secbert-frozen`. `SECBERT_PROBE_BANK` points to `secbert-concepts/bank`. `OWASP_CRS` points to `crs-4.10.0`. Adjust the example checkout paths below if your projects are stored elsewhere.

## 2. Windows setup: SecBERT and concept probes

Run these commands in PowerShell from the APP directory:

```powershell
cd you_local_path\IS-Team14-Project-APP
python -m venv .venv
python.exe -m pip install -r requirements-model.txt
```

Edit `env.ps1` and set:

```powershell
$env:TRAINING_REPO = 'you_local_path\IS-Team14-Project'
$env:SECBERT_CHECKPOINT = Join-Path $PSScriptRoot 'artifacts\secbert-frozen'
$env:SECBERT_PROBE_BANK = Join-Path $PSScriptRoot 'artifacts\secbert-concepts\bank'
$env:SECBERT_DEVICE = 'cpu'
$env:GATEWAY_MODE = 'observe'
$env:GATEWAY_UPSTREAM = 'http://127.0.0.1:9000'
$env:OWASP_CRS = ''
```

Load the configuration, verify the model, and initialize or upgrade the database:

```powershell
. .\env.ps1
python.exe manage.py check_model
python.exe manage.py migrate
```

Open three PowerShell terminals in the APP directory. Run `. .\env.ps1` in each terminal, then run one service per terminal:

| Service | Command |
| --- | --- |
| Test backend | `.\.venv\Scripts\python.exe -m http.server 9000 --bind 127.0.0.1` |
| Gateway | `.\.venv\Scripts\python.exe manage.py serve_gateway` |
| Dashboard | `.\.venv\Scripts\python.exe manage.py serve_dashboard --settings=secbert_app.dashboard_settings` |

Python does not load `env.ps1` automatically. Load it in every new terminal before starting a service. To enable live CRS with the current adapter, use the WSL setup below.

## 3. WSL setup: SecBERT, concept probes, and live CRS

Stop any Windows instances using ports 8080, 8081, or 9000 before starting the WSL services.

### Install Ubuntu and ModSecurity

If Ubuntu 24.04 is not installed, run this in an administrator PowerShell terminal and follow the restart and user-creation prompts:

```powershell
wsl --install -d Ubuntu-24.04
```

Enter Ubuntu from PowerShell:

```powershell
wsl -d Ubuntu-24.04
```

Run the remaining commands in Ubuntu. [WSL installation instructions](https://learn.microsoft.com/en-us/windows/wsl/install).

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip git curl ca-certificates libmodsecurity3t64

cd /mnt/c/code/pythoncode/IS-Team14-Project-APP
python3 -m venv "$HOME/.venvs/secbert-gateway"
source "$HOME/.venvs/secbert-gateway/bin/activate"
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-model.txt
```

Use a separate Linux virtual environment. Ubuntu 24.04 provides the engine as [`libmodsecurity3t64`](https://packages.ubuntu.com/noble/amd64/libmodsecurity3t64). The commands above install the CPU build of [PyTorch](https://pytorch.org/get-started/locally/).

### Download the CRS rules

From the APP directory, download the rule version used by the experiment. Skip this command if the complete directory already exists:

```bash
git clone --depth 1 --branch v4.10.0 \
  https://github.com/coreruleset/coreruleset.git \
  artifacts/crs-4.10.0
```

Keep `crs-setup.conf.example`, all request rules, and their data files together. The adapter loads this directory directly; no Nginx or Apache configuration is needed.

### Create the Ubuntu configuration

Create or update `$HOME/secbert-env.sh` with the following configuration:

```bash
cat > "$HOME/secbert-env.sh" <<'EOF'
source "$HOME/.venvs/secbert-gateway/bin/activate"
cd /mnt/c/code/pythoncode/IS-Team14-Project-APP

export TRAINING_REPO=/mnt/c/code/pythoncode/IS-Team14-Project
export PYTHONPATH="$TRAINING_REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export SECBERT_CHECKPOINT="$PWD/artifacts/secbert-frozen"
export SECBERT_PROBE_BANK="$PWD/artifacts/secbert-concepts/bank"
export SECBERT_DEVICE=cpu
export OWASP_CRS="$PWD/artifacts/crs-4.10.0"
export CRS_PARANOIA=1
export GATEWAY_MODE=observe
export GATEWAY_UPSTREAM=http://127.0.0.1:9000

mkdir -p "$HOME/.local/share/secbert-gateway"
export GATEWAY_DATABASE="$HOME/.local/share/secbert-gateway/requests.sqlite3"
EOF

source "$HOME/secbert-env.sh"
```

This configuration uses a separate WSL database. Existing Windows request history stays in the Windows database.

### Verify and start the services

```bash
python -m http_attack_agent.concept_analysis check-waf --crs "$OWASP_CRS"
python manage.py check_model
python manage.py migrate
```

The CRS check should return `"status": "pass"`; the model check should print `SecBERT OK`.

Open three Ubuntu terminals. Run `source "$HOME/secbert-env.sh"` in each, then run one service per terminal:

| Service | Command |
| --- | --- |
| Test backend | `python -m http.server 9000 --bind 127.0.0.1` |
| Gateway | `python manage.py serve_gateway` |
| Dashboard | `python manage.py serve_dashboard --settings=secbert_app.dashboard_settings` |

## 4. Use the app

- Send business requests to `http://127.0.0.1:8080`.
- Open the dashboard at [http://127.0.0.1:8081](http://127.0.0.1:8081).
- Set `GATEWAY_UPSTREAM` to your backend's HTTP(S) origin when replacing the test backend.
- Use `GATEWAY_MODE=observe` to inspect and forward requests. Open a completed request's details, select **Mark as attack** or **Mark as non-attack**, and click **Save review**. Select **Awaiting review** to clear a label.
- Use `GATEWAY_MODE=enforce` for automatic model-based blocking. Restart the services after changing configuration.
- Raw detection JSON files are created under `evidence_cache` after new requests. Use **Clear JSON cache** on the dashboard to remove those copies. Request history and human reviews remain available.

Send test requests from Ubuntu; use `curl.exe` instead of `curl` in PowerShell:

```bash
curl -i "http://127.0.0.1:8080/README.md"
curl -i "http://127.0.0.1:8080/?id=1%20OR%201=1--"
```

The first request loads the model and can take longer. Refresh the dashboard and open **Details** to inspect the results. In observe mode, a request can return HTTP 200 while detection results reach an attack threshold.

After pulling code updates, load the environment configuration, run `python manage.py migrate` using the configured virtual environment, and restart the gateway and dashboard.
