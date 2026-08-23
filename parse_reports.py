#!/usr/bin/env python3
"""Parse bandit, vulture, radon, mypy results and print a clean summary."""
import json, os, subprocess, sys

BASE = os.path.dirname(os.path.abspath(__file__))
TARGET = os.path.join(BASE, 'final_package', '07_13_methodology_validation')

sep = lambda title: print(f"\n{'='*68}\n  {title}\n{'='*68}")

# ── 1. BANDIT ────────────────────────────────────────────────────────────────
sep("BANDIT: Security Vulnerability Scan")
try:
    r = json.load(open(os.path.join(BASE, 'bandit_report.json'), encoding='utf-8'))
    results = r.get('results', [])
    metrics = r.get('metrics', {})
    if not results:
        print("  [✓] No security issues found!")
    for issue in results:
        fname = os.path.basename(issue['filename'])
        print(f"  [{issue['issue_severity']:6}] {fname}:{issue['line_number']} — {issue['issue_text']}")
    print(f"\n  Summary: {len(results)} issues | SEVERITY: HIGH={sum(1 for x in results if x['issue_severity']=='HIGH')}, MEDIUM={sum(1 for x in results if x['issue_severity']=='MEDIUM')}, LOW={sum(1 for x in results if x['issue_severity']=='LOW')}")
except Exception as e:
    print(f"  [!] {e}")

# ── 2. VULTURE ───────────────────────────────────────────────────────────────
sep("VULTURE: Dead Code Finder")
try:
    res = subprocess.run(
        ['vulture', TARGET, '--min-confidence', '80'],
        capture_output=True, text=True, cwd=BASE, check=False
    )
    lines = [l for l in res.stdout.strip().splitlines() if l.strip()]
    if not lines:
        print("  [✓] No dead code found!")
    for l in lines[:30]:
        parts = l.split(':')
        if len(parts) >= 3:
            fname = os.path.basename(parts[0])
            print(f"  {fname}:{':'.join(parts[1:]).strip()}")
    if len(lines) > 30:
        print(f"  ... and {len(lines)-30} more")
    print(f"\n  Summary: {len(lines)} dead code items found")
except Exception as e:
    print(f"  [!] {e}")

# ── 3. RADON CC ──────────────────────────────────────────────────────────────
sep("RADON: Cyclomatic Complexity (functions rated A-F)")
try:
    res = subprocess.run(
        ['radon', 'cc', TARGET, '-s', '-n', 'B'],
        capture_output=True, text=True, cwd=BASE, check=False
    )
    lines = [l for l in res.stdout.strip().splitlines() if l.strip()]
    if not lines:
        print("  [✓] All functions have complexity grade A (excellent)!")
    for l in lines[:40]:
        print(f"  {l}")
    print(f"\n  Grade scale: A=1-5 ✅  B=6-10 🟡  C=11-15 🟠  D=16-20 🔴  F=21+ 💀")
except Exception as e:
    print(f"  [!] {e}")

# ── 4. RADON MI ──────────────────────────────────────────────────────────────
sep("RADON: Maintainability Index (A=easy to maintain, C=hard)")
try:
    res = subprocess.run(
        ['radon', 'mi', TARGET, '-s'],
        capture_output=True, text=True, cwd=BASE, check=False
    )
    for l in res.stdout.strip().splitlines()[:25]:
        print(f"  {l}")
except Exception as e:
    print(f"  [!] {e}")

# ── 5. MYPY ──────────────────────────────────────────────────────────────────
sep("MYPY: Static Type Checker")
try:
    res = subprocess.run(
        ['mypy', TARGET, '--ignore-missing-imports', '--no-error-summary',
         '--disable-error-code', 'import-untyped'],
        capture_output=True, text=True, cwd=BASE, check=False
    )
    lines = [l for l in res.stdout.strip().splitlines() if 'error:' in l or 'warning:' in l]
    if not lines:
        print("  [✓] No type errors found!")
    for l in lines[:30]:
        parts = l.split(':')
        if len(parts) >= 3:
            fname = os.path.basename(parts[0])
            print(f"  {fname}:{':'.join(parts[1:]).strip()}")
    print(f"\n  Summary: {len(lines)} type issues found")
except Exception as e:
    print(f"  [!] {e}")

print("\n" + "="*68)
print("  ALL SCANS COMPLETE")
print("="*68)
