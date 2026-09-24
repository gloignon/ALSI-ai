#!/usr/bin/env python3
"""Recon: check detail page metadata and navigation."""

import time
from playwright.sync_api import sync_playwright

BASE_URL = "https://corpusalector.huma-num.fr/"

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    page.goto(BASE_URL, wait_until="networkidle", timeout=30000)
    time.sleep(2)

    # Login
    page.locator("#j_idt14\\:j_idt16").click()
    page.wait_for_load_state("networkidle")
    time.sleep(2)
    page.fill("#input_connection\\:j_idt24", "YOUR_USERNAME")
    page.fill("#input_connection\\:j_idt27", "YOUR_PASSWORD")
    time.sleep(0.5)
    page.locator("#connection\\:j_idt32").click()
    page.wait_for_load_state("networkidle")
    time.sleep(3)
    print("Logged in!")

    # Click first title
    first_title = page.locator("a[id*='j_idt64:j_idt65']").first
    first_title.click()
    page.wait_for_load_state("networkidle")
    time.sleep(2)

    # Get metadata from the detail page
    print("\n=== Detail page metadata ===")

    # Check panel headers
    headers = page.locator(".ui-panel-title, .panel-title, .panel-heading, h3, h4").all()
    for h in headers:
        try:
            print(f"  Header: {h.inner_text().strip()[:200]}")
        except:
            pass

    # Get the "Original (Author)" header to extract author
    orig_col = page.locator("#form\\:columnCorpusOriginal")
    if orig_col.count() > 0:
        full_text = orig_col.inner_text()
        first_line = full_text.split("\n")[0].strip()
        print(f"  Original header: {first_line}")

    # Get genre/type info
    print("\n=== Looking for genre/type metadata ===")
    # Check the panel header area
    panel_header = page.locator("#form\\:j_idt27 .panel-heading, #form\\:j_idt27 .panel-title").all()
    for ph in panel_header:
        try:
            print(f"  Panel header: {ph.inner_text()[:200]}")
        except:
            pass

    # Check for badges, labels, tags
    badges = page.locator(".badge, .label, .tag, span.ui-badge").all()
    for b in badges:
        try:
            t = b.inner_text().strip()
            if t:
                print(f"  Badge/label: {t}")
        except:
            pass

    # Try to find the title and type displayed on detail page
    # The full body text might contain this info
    body = page.inner_text("body")
    lines = body.split("\n")
    for line in lines[:30]:
        line = line.strip()
        if line and len(line) < 100:
            print(f"  Line: {line}")

    # Now go back to list
    print("\n=== Testing navigation back ===")
    page.go_back(wait_until="networkidle")
    time.sleep(2)
    print(f"Back URL: {page.url}")

    # Check how many title links on page
    titles = page.locator("a[id*='j_idt64:j_idt65']").all()
    print(f"Title links on page: {len(titles)}")

    # Check pagination
    print("\n=== Pagination ===")
    paginators = page.locator(".ui-paginator, .pagination").all()
    print(f"Paginator elements: {len(paginators)}")
    for pg in paginators:
        try:
            print(f"  {pg.inner_text()[:200]}")
        except:
            pass

    # Also check for "rows per page" selector
    rpp = page.locator("select[id*='rpp'], .ui-paginator-rpp-options").all()
    print(f"Rows per page selectors: {len(rpp)}")

    # Count items on list page
    # From original recon, items are in j_idt62:{i}:j_idt64 forms
    items = page.locator("a[id*='j_idt64:j_idt65']").all()
    print(f"\nItems on current page: {len(items)}")
    for item in items:
        try:
            print(f"  {item.inner_text().strip()}")
        except:
            pass

    browser.close()
