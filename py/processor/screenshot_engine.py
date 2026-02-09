from PIL import Image, ImageDraw, ImageFont
import os
import requests
from bs4 import BeautifulSoup
import textwrap
import io
import wikipedia

def capture_article_screenshot(url, filename, y_offset=0):
    """
    Captures a website screenshot using thum.io API.
    """
    os.makedirs("output/screenshots", exist_ok=True)
    final_output_path = f"output/screenshots/{filename}.png"
    
    try:
        # thum.io simple URL-based screenshot
        # We ignore y_offset as requested for now
        # API URL example: https://image.thum.io/get/width/1000/crop/800/https://google.com
        thum_url = f"https://image.thum.io/get/width/1000/crop/800/{url}"
        
        print(f"[Screenshot] Requesting thum.io for {url}")
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(thum_url, headers=headers, timeout=20)
        
        if response.status_code == 200:
            with open(final_output_path, 'wb') as f:
                f.write(response.content)
            print(f"[Screenshot] Saved to {final_output_path}")
            return final_output_path
        else:
            print(f"[Screenshot] thum.io failed with status {response.status_code}")
            return None

    except Exception as e:
        print(f"Failed to capture screenshot with thum.io: {e}")
        return None
