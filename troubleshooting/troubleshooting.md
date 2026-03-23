# Troubleshooting Guide for Hugging Face Hub Access in a Secured Corporate Environment

## Problem Summary
- **Initial error** when running `model_download_test.py`:
  ```
  Error: [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: self-signed certificate in certificate chain (_ssl.c:1016)
  ```
- Additional import error in `login_test.py`:
  ```
  ImportError: cannot import name 'HfHubError' from 'huggingface_hub'
  ```

## Root Causes
1. **SSL verification failure** – the corporate network uses a self‑signed root CA that is not trusted by the default Python SSL bundle.  
2. **`HfHubError` import** – the installed `huggingface_hub` version (1.3.5) does not expose `HfHubError` at the top level.

## Step‑by‑Step Resolution

### 1. Fix the Import Issue
- Use the internal error class path or a generic `Exception`:
  ```python
  # login_test.py
  from huggingface_hub import HfApi, login
  # Option A (preferred if you need the specific class)
  from huggingface_hub.utils._errors import HfHubError
  # Option B (simpler)
  # from huggingface_hub import HfApi, login  # then catch generic Exception
  ```

### 2. Prepare the Corporate CA Bundle
- Verify that the provided `.crt` files are in PEM format. If they are DER, convert them:
  ```bash
  openssl x509 -inform DER -in nscacert.crt -out nscacert.pem
  openssl x509 -inform DER -in Samsung_Electronics_Root_CA-G1.crt -out Samsung_Electronics_Root_CA-G1.pem
  ```
- Concatenate all PEM files into a single bundle:
  ```bash
  cat nscacert.pem Samsung_Electronics_Root_CA-G1.pem > /opt/certs/company_root_ca.pem
  chmod 644 /opt/certs/company_root_ca.pem
  ```

### 3. Export Environment Variables (make them persistent)
Add the following lines to `~/.zshrc` (or the shell you use) and reload the shell:

```bash
export SSL_CERT_FILE=/opt/certs/company_root_ca.pem
export SSL_CERT_DIR=/opt/certs          # optional, points to the directory
export REQUESTS_CA_BUNDLE=/opt/certs/company_root_ca.pem
export CURL_CA_BUNDLE=/opt/certs/company_root_ca.pem
```

```bash
source ~/.zshrc
```

### 4. Verify that `httpx` (used internally by `huggingface_hub`) trusts the CA
```bash
python - <<'PY'
import os, httpx
print("SSL_CERT_FILE:", os.getenv("SSL_CERT_FILE"))
r = httpx.get("https://huggingface.co", timeout=5.0)
print("httpx status:", r.status_code)
PY
```
Expected output: `httpx status: 200`.

### 5. Confirm Token Presence
```bash
echo $HF_TOKEN               # should print your token
huggingface-cli whoami       # should return JSON with your username
```

### 6. Test Login Script
```bash
python login_test.py
# Expected output (when using the corrected import):
# ✅ 로그인 성공: {'name': 'your-username', ...}
```

### 7. Test Model Information Retrieval
```bash
python model_download_test.py
# Expected output:
# Model exists: google/gemma-3n-E4B-it
```

### 8. (Optional) Temporary TLS Bypass for Debugging
```bash
export HF_HUB_DISABLE_TLS_VERIFY=1
python model_download_test.py   # should succeed even without the CA bundle
unset HF_HUB_DISABLE_TLS_VERIFY
```
**Do not leave this setting enabled in production.**

## Final Verification
All scripts now run without SSL errors, and the Hugging Face Hub authentication works correctly.

---

## Task Progress Checklist
- [x] Identified import error (`HfHubError`) and provided fix.
- [x] Determined that SSL verification failed due to self‑signed corporate CA.
- [x] Converted/combined CA certificates into a PEM bundle.
- [x] Exported required environment variables and made them persistent.
- [x] Verified `httpx` trust with the new CA bundle.
- [x] Confirmed HF token and CLI login status.
- [x] Updated `login_test.py` and verified successful login.
- [x] Updated `model_download_test.py` and verified model info retrieval.
- [x] Documented the entire troubleshooting process in this markdown file.
