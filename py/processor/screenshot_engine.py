import os

def capture_article_screenshot(url, filename, y_offset=0):
    """
    Captures a website screenshot using Playwright (Local Renderer).
    """
    os.makedirs("output/screenshots", exist_ok=True)
    final_output_path = f"output/screenshots/{filename}.png"

    try:
        from playwright.sync_api import sync_playwright
        
        print(f"[Screenshot] Launching Playwright to capture {url}...")
        with sync_playwright() as p:
            # Launch browser
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={'width': 1280, 'height': 800})
            
            # Navigate to URL
            page.goto(url, wait_until="networkidle", timeout=60000)
            
            # Apply vertical offset if specified
            if y_offset > 0:
                print(f"[Screenshot] Scrolling to y-offset: {y_offset}")
                page.evaluate(f"window.scrollTo(0, {y_offset})")
                # Wait a bit for lazy-loaded content to stabilize if any
                page.wait_for_timeout(500)
            
            # Take screenshot
            page.screenshot(path=final_output_path)
            browser.close()
            
        if os.path.exists(final_output_path):
            print(f"[Screenshot] Saved local Playwright render to {final_output_path}")
            return final_output_path
            
    except Exception as e:
        print(f"[Screenshot] Playwright Error: {e}")

    return None
