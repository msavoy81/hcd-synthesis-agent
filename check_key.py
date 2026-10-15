"""Diagnose the API key without ever printing it."""
import os, re, sys
path = os.environ.get("ANTHROPIC_API_KEY_FILE", "")
text = open(path, errors="ignore").read()
m = re.search(r"sk-[A-Za-z0-9_\-]{20,}", text)
if not m:
    sys.exit("No key found in the file.")
k = m.group(0)
print("Key found. Starts with:", k[:13], "| length:", len(k))
import anthropic
try:
    r = anthropic.Anthropic(api_key=k).models.list(limit=5)
    print("SUCCESS. The key works. Models available:", [x.id for x in r.data])
except Exception as e:
    print("FAILED:", type(e).__name__, "|", str(e)[:300].replace(k, "[hidden]"))
