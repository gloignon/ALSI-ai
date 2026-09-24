#!/usr/bin/env python3
"""
Scrape the Alector corpus from corpusalector.huma-num.fr

Downloads all 79 text pairs (source + simplified) with metadata.
Based on the approach from github.com/thiborose/alector_corpus, rewritten
for Playwright (instead of Selenium).

Outputs:
  - out/alector_v2/texts/  : {idx:03d}_source.txt and {idx:03d}_target.txt
  - out/alector_v2/alector_metadata.csv

Requires: playwright
  pip install playwright && python -m playwright install chromium

Usage:
  python R/artefact_builders/scrape_alector.py --user USERNAME --password PASSWORD
"""

import argparse
import csv
import re
import time
from pathlib import Path
from playwright.sync_api import sync_playwright

BASE_URL = "https://corpusalector.huma-num.fr/"
LOGIN_URL = "https://corpusalector.huma-num.fr/faces/register.xhtml"
OUT_DIR = Path("out/alector_v2/texts")
META_FILE = Path("out/alector_v2/alector_metadata.csv")

N_PAGES = 8
ITEMS_PER_PAGE = 10
ITEMS_LAST_PAGE = 9


def login(page, username, password):
    """Log in via the register.xhtml page (has the login form directly)."""
    page.goto(LOGIN_URL, wait_until="networkidle", timeout=30000)
    time.sleep(1)

    # Fill login form (IDs from the reference scraper)
    page.fill("#input_connection\\:j_idt24", username)
    page.fill("#input_connection\\:j_idt27", password)
    time.sleep(0.3)

    # Click "Se connecter"
    page.locator("#connection\\:j_idt32").click()
    page.wait_for_load_state("networkidle")
    time.sleep(2)

    # Verify
    if "Déconnexion" in page.inner_text("body"):
        print("Login successful.")
        return True
    else:
        print("Login failed!")
        return False


def go_to_page(page, page_num):
    """Click the paginator button for the given page number (1-based).

    After browser back(), PrimeFaces resets to page 1. The reference scraper
    clicks the page button twice and waits for ui-state-active — we do the same.
    """
    if page_num <= 1:
        return

    # Click the page number button (match by text content)
    page_buttons = page.locator(".ui-paginator-page").all()
    for btn in page_buttons:
        if btn.inner_text().strip() == str(page_num):
            btn.click()
            break
    page.wait_for_load_state("networkidle")
    time.sleep(0.5)

    # Re-click (the reference scraper does this for reliability)
    page_buttons = page.locator(".ui-paginator-page").all()
    for btn in page_buttons:
        if btn.inner_text().strip() == str(page_num):
            btn.click()
            break
    page.wait_for_load_state("networkidle")
    time.sleep(0.5)

    # Wait until the correct page button has the active class
    page.locator(f".ui-paginator-page.ui-state-active:text-is('{page_num}')").wait_for(
        state="visible", timeout=10000
    )


def scrape_all(page, debug=False):
    """Scrape all 79 text pairs."""
    all_metadata = []
    global_idx = 0

    for page_num in range(1, N_PAGES + 1):
        n_items = ITEMS_LAST_PAGE if page_num == N_PAGES else ITEMS_PER_PAGE
        print(f"\nPage {page_num}/{N_PAGES}: {n_items} texts")

        for j in range(n_items):
            # Navigate to the correct paginator page (back() resets to page 1)
            if page_num > 1:
                go_to_page(page, page_num)

            # Get all title links on this page
            rows = page.locator("td:first-of-type .ui-commandlink").all()
            title = rows[j].inner_text().strip()
            print(f"  [{global_idx:02d}] {title}...", end=" ", flush=True)

            # Click into detail page
            rows[j].click()
            page.wait_for_load_state("networkidle")
            time.sleep(1)

            if debug and global_idx < 3:
                page.screenshot(path=str(OUT_DIR / f"debug_{global_idx:03d}.png"))

            # Extract texts (IDs from reference scraper)
            source_text = ""
            source_el = page.locator("#form\\:corpusOrig")
            if source_el.count() > 0:
                source_text = source_el.inner_text().strip()

            target_text = ""
            target_el = page.locator("#form\\:corpusSimp")
            if target_el.count() > 0:
                target_text = target_el.inner_text().strip()

            # Extract author from panel header "Original (Author Name)"
            author = ""
            orig_header = page.locator("#form\\:panelOrig .ui-panel-title")
            if orig_header.count() > 0:
                header_text = orig_header.inner_text().strip()
                m = re.search(r"Original\s*\((.+?)\)", header_text)
                if m:
                    author = m.group(1).strip()

            # Extract genre badges
            genre_parts = []
            badges = page.locator(".badge, .label").all()
            for b in badges:
                try:
                    t = b.inner_text().strip()
                    if t and t not in ("Original", "Simplifié"):
                        genre_parts.append(t)
                except:
                    pass
            genre = "|".join(genre_parts)

            # Save texts
            if source_text:
                (OUT_DIR / f"{global_idx:03d}_source.txt").write_text(
                    source_text, encoding="utf-8"
                )
            if target_text:
                (OUT_DIR / f"{global_idx:03d}_target.txt").write_text(
                    target_text, encoding="utf-8"
                )

            status = f"src={len(source_text)} tgt={len(target_text)}"
            if not source_text or not target_text:
                status += " MISSING!"
            print(status)

            all_metadata.append({
                "index": global_idx,
                "title": title,
                "author": author,
                "genre": genre,
                "source_chars": len(source_text),
                "target_chars": len(target_text),
            })
            global_idx += 1

            # Go back to list (resets to page 1 — we re-navigate at top of loop)
            page.go_back(wait_until="networkidle")
            time.sleep(1)

    return all_metadata


def main():
    parser = argparse.ArgumentParser(description="Scrape Alector corpus")
    parser.add_argument("--user", required=True, help="Username")
    parser.add_argument("--password", required=True, help="Password")
    parser.add_argument("--debug", action="store_true", help="Save screenshots")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    META_FILE.parent.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()

        # Login
        if not login(page, args.user, args.password):
            browser.close()
            return

        # Navigate to main page
        page.goto(BASE_URL, wait_until="networkidle", timeout=30000)
        time.sleep(1)

        # Scrape
        metadata = scrape_all(page, debug=args.debug)
        browser.close()

    # Write metadata CSV
    if metadata:
        with open(META_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=metadata[0].keys())
            writer.writeheader()
            writer.writerows(metadata)
        print(f"\nMetadata: {META_FILE}")

    print(f"Done. {len(metadata)} texts processed.")


if __name__ == "__main__":
    main()
