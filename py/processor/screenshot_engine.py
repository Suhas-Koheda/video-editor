import asyncio
import os
import requests

def capture_article_screenshot(url, filename):
    """
    Captures a Wikipedia 'Knowledge Card' screenshot.
    Uses a high-quality rendering API to ensure it works without complex local drivers.
    """
    os.makedirs("output/screenshots", exist_ok=True)
    output_path = f"output/screenshots/{filename}.png"
    
    try:
        width = 1024
        height = 768
        
        thum_url = f"https://image.thum.io/get/width/{width}/crop/800/noanimate/{url}"
        
        response = requests.get(thum_url, stream=True, timeout=30)
        if response.status_code == 200:
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            return output_path
        else:
            raise Exception("API capture failed")
            
    except Exception as e:
        print(f"Digital Screenshot Failed, using fallback: {e}")
        return None
