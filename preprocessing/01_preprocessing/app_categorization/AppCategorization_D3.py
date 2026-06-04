#!/usr/bin/env python3
"""
D-4 App Categorization Script
=============================

This script categorizes apps from both KeyEvent.csv and AppUsage.csv files using web scraping.
It performs a 3-stage categorization process:
1. apkcombo.com
2. play.google.com  
3. apkpure.net

Author: Generated from notebook
Date: 2024
"""

import os
import sys
import shutil
import numpy as np
import pandas as pd
import json
from glob import glob
from tqdm import tqdm
from time import sleep
import warnings
warnings.filterwarnings("ignore")

# Web scraping imports
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.webdriver.common.action_chains import ActionChains
from bs4 import BeautifulSoup
import requests

# =============================================================================
# CONFIGURATION
# =============================================================================

# D-4 paths
PATH_DATA = "<DATA_ROOT_D3>"
PATH_SENSOR = os.path.join(PATH_DATA, 'newdata')

# File paths
UNCATEGORIZED_APPS_PATH = os.path.join(PATH_DATA, "uncategorized_apps.csv")
OUTPUT_CSV_PATH = os.path.join(PATH_DATA, "app_category_1.csv")

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def setup_browser(headless=True):
    """Setup Firefox browser with appropriate options."""
    print("Setting up Firefox browser...")
    
    # Check for Firefox binary
    FIREFOX_BIN = shutil.which("firefox")
    if not FIREFOX_BIN:
        # Try alternative Firefox paths
        firefox_paths = [
            "/usr/bin/firefox",
            "/usr/local/bin/firefox",
            "/snap/bin/firefox",
            "/opt/firefox/firefox",
            "/Applications/Firefox.app/Contents/MacOS/firefox"  # macOS
        ]
        
        for path in firefox_paths:
            if os.path.exists(path):
                FIREFOX_BIN = path
                print(f"Found Firefox at: {FIREFOX_BIN}")
                break
    
    if not FIREFOX_BIN:
        print("WARNING: Firefox binary not found in PATH or common locations")
        print("Available browsers in PATH:")
        for browser in ["firefox", "google-chrome", "chromium-browser", "chrome"]:
            path = shutil.which(browser)
            if path:
                print(f"  {browser}: {path}")
    
    # Check for geckodriver
    GECKO_BIN = shutil.which("geckodriver")
    if not GECKO_BIN:
        # Try alternative geckodriver paths
        gecko_paths = [
            "/usr/local/bin/geckodriver",
            "/usr/bin/geckodriver",
            "./geckodriver"
        ]
        
        for path in gecko_paths:
            if os.path.exists(path):
                GECKO_BIN = path
                print(f"Found geckodriver at: {GECKO_BIN}")
                break
    
    if not GECKO_BIN:
        print("ERROR: geckodriver not found!")
        print("Please install geckodriver or add it to PATH")
        return None
    
    print(f"Using Firefox: {FIREFOX_BIN}")
    print(f"Using geckodriver: {GECKO_BIN}")
    
    try:
        opts = FirefoxOptions()
        if headless:
            opts.add_argument("--headless")
            print("Running in headless mode")
        
        if FIREFOX_BIN:
            opts.binary_location = FIREFOX_BIN
            print(f"Set Firefox binary location: {FIREFOX_BIN}")
        
        # Add additional options for better compatibility
        opts.add_argument("--no-sandbox")
        opts.add_argument("--disable-dev-shm-usage")
        opts.add_argument("--disable-gpu")
        opts.add_argument("--window-size=1920,1080")
        
        service = Service(executable_path=GECKO_BIN, log_output="geckodriver.log")
        
        print("Creating Firefox driver...")
        driver = webdriver.Firefox(service=service, options=opts)
        print("Firefox driver created successfully!")
        
        return driver
        
    except Exception as e:
        print(f"Error creating Firefox driver: {e}")
        print("Trying alternative setup...")
        
        try:
            # Try without specifying binary location
            opts = FirefoxOptions()
            if headless:
                opts.add_argument("--headless")
            
            opts.add_argument("--no-sandbox")
            opts.add_argument("--disable-dev-shm-usage")
            
            service = Service(executable_path=GECKO_BIN, log_output="geckodriver.log")
            driver = webdriver.Firefox(service=service, options=opts)
            print("Firefox driver created successfully (alternative setup)!")
            return driver
            
        except Exception as e2:
            print(f"Alternative setup also failed: {e2}")
            print("\nTroubleshooting tips:")
            print("1. Install Firefox: sudo apt-get install firefox")
            print("2. Install geckodriver: wget https://github.com/mozilla/geckodriver/releases/latest/download/geckodriver-v0.33.0-linux64.tar.gz")
            print("3. Extract and move to PATH: sudo mv geckodriver /usr/local/bin/")
            print("4. Make executable: sudo chmod +x /usr/local/bin/geckodriver")
            return None

def setup_chrome_browser(headless=True):
    """Setup Chrome/Chromium browser as a fallback option."""
    print("Setting up Chrome/Chromium browser as fallback...")
    
    # Check for Chrome/Chromium
    chrome_paths = [
        shutil.which("google-chrome"),
        shutil.which("chromium-browser"),
        shutil.which("chrome"),
        "/usr/bin/google-chrome",
        "/usr/bin/chromium-browser",
        "/usr/bin/chromium"
    ]
    
    chrome_bin = None
    for path in chrome_paths:
        if path and os.path.exists(path):
            chrome_bin = path
            print(f"Found Chrome/Chromium at: {chrome_bin}")
            break
    
    if not chrome_bin:
        print("Chrome/Chromium not found")
        return None
    
    # Check for chromedriver
    chromedriver_paths = [
        shutil.which("chromedriver"),
        "/usr/local/bin/chromedriver",
        "/usr/bin/chromedriver",
        "./chromedriver"
    ]
    
    chromedriver_bin = None
    for path in chromedriver_paths:
        if path and os.path.exists(path):
            chromedriver_bin = path
            print(f"Found chromedriver at: {chromedriver_bin}")
            break
    
    if not chromedriver_bin:
        print("chromedriver not found")
        return None
    
    try:
        from selenium.webdriver.chrome.options import Options as ChromeOptions
        from selenium.webdriver.chrome.service import Service as ChromeService
        
        opts = ChromeOptions()
        if headless:
            opts.add_argument("--headless")
        
        opts.add_argument("--no-sandbox")
        opts.add_argument("--disable-dev-shm-usage")
        opts.add_argument("--disable-gpu")
        opts.add_argument("--window-size=1920,1080")
        opts.add_argument("--disable-web-security")
        opts.add_argument("--allow-running-insecure-content")
        
        service = ChromeService(executable_path=chromedriver_bin)
        
        print("Creating Chrome driver...")
        driver = webdriver.Chrome(service=service, options=opts)
        print("Chrome driver created successfully!")
        
        return driver
        
    except ImportError:
        print("Chrome WebDriver not available (selenium version too old)")
        return None
    except Exception as e:
        print(f"Error creating Chrome driver: {e}")
        return None

def install_dependencies():
    """Provide instructions and commands to install missing dependencies."""
    print("\n" + "=" * 70)
    print("INSTALLING MISSING DEPENDENCIES")
    print("=" * 70)
    
    print("Please run the following commands to install missing dependencies:")
    print()
    
    print("1. Install Firefox:")
    print("   sudo apt-get update")
    print("   sudo apt-get install firefox")
    print()
    
    print("2. Install geckodriver:")
    print("   wget https://github.com/mozilla/geckodriver/releases/latest/download/geckodriver-v0.33.0-linux64.tar.gz")
    print("   tar -xzf geckodriver-v0.33.0-linux64.tar.gz")
    print("   sudo mv geckodriver /usr/local/bin/")
    print("   sudo chmod +x /usr/local/bin/geckodriver")
    print()
    
    print("3. Install Python packages:")
    print("   pip install selenium beautifulsoup4 requests")
    print()
    
    print("4. Alternative: Install Chrome/Chromium:")
    print("   sudo apt-get install google-chrome-stable")
    print("   wget https://chromedriver.storage.googleapis.com/LATEST_RELEASE")
    print("   wget https://chromedriver.storage.googleapis.com/$(cat LATEST_RELEASE)/chromedriver_linux64.zip")
    print("   unzip chromedriver_linux64.zip")
    print("   sudo mv chromedriver /usr/local/bin/")
    print("   sudo chmod +x /usr/local/bin/chromedriver")
    print()
    
    print("After installation, run the script again.")
    print("=" * 70)

def collect_app_packages():
    """Collect all unique app package names from KeyEvent.csv files."""
    print("Collecting app packages from KeyEvent.csv files...")
    
    uids = [uid for uid in os.listdir(PATH_SENSOR) 
            if os.path.isdir(os.path.join(PATH_SENSOR, uid))]
    
    app_list = set()
    for uid in uids:
        keyevent_paths = sorted(glob(os.path.join(PATH_SENSOR, uid, 'KeyEvent.csv')))
        dfs = [pd.read_csv(path, index_col=False, header=0) for path in keyevent_paths]
        
        if dfs:
            df = pd.concat(dfs, ignore_index=True)
            if 'packageName' in df.columns:
                app_list.update(df['packageName'].dropna().unique().tolist())
    
    print(f"# Total Apps from KeyEvent: {len(app_list)}")
    return app_list

def collect_appusage_packages():
    """Collect all unique app package names from AppUsageEvent.csv files."""
    print("Collecting app packages from AppUsageEvent.csv files...")
    
    uids = [uid for uid in os.listdir(PATH_SENSOR) 
            if os.path.isdir(os.path.join(PATH_SENSOR, uid))]
    
    app_list = set()
    for uid in uids:
        appusage_paths = sorted(glob(os.path.join(PATH_SENSOR, uid, 'AppUsageEvent.csv')))
        dfs = [pd.read_csv(path, index_col=False, header=0) for path in appusage_paths]
        
        if dfs:
            df = pd.concat(dfs, ignore_index=True)
            if 'packageName' in df.columns:
                app_list.update(df['packageName'].dropna().unique().tolist())
    
    print(f"# Total Apps from AppUsageEvent: {len(app_list)}")
    return app_list

def collect_all_app_packages():
    """Collect all unique app package names from both KeyEvent.csv and AppUsageEvent.csv files."""
    print("Collecting app packages from both KeyEvent.csv and AppUsageEvent.csv files...")
    
    keyevent_apps = collect_app_packages()
    appusage_apps = collect_appusage_packages()
    
    # Combine both sets
    all_apps = keyevent_apps.union(appusage_apps)
    all_apps = sorted(all_apps)
    
    print(f"# Total Unique Apps (combined): {len(all_apps)}")
    print(f"# Apps from KeyEvent only: {len(keyevent_apps - appusage_apps)}")
    print(f"# Apps from AppUsageEvent only: {len(appusage_apps - keyevent_apps)}")
    print(f"# Apps in both: {len(keyevent_apps.intersection(appusage_apps))}")
    
    return all_apps

def save_uncategorized_apps(app_list):
    """Save the list of uncategorized apps to CSV."""
    print("Saving uncategorized apps list...")
    
    if os.path.exists(UNCATEGORIZED_APPS_PATH):
        os.remove(UNCATEGORIZED_APPS_PATH)
    
    df_uncat = pd.DataFrame(app_list, columns=['packageName'])
    df_uncat.to_csv(UNCATEGORIZED_APPS_PATH, index=False)
    
    print(df_uncat.head(), "\n", df_uncat.columns, "\n", df_uncat.shape, "\n")

def get_processing_summary():
    """Generate a summary of the processing results."""
    print("\n" + "=" * 70)
    print("PROCESSING SUMMARY")
    print("=" * 70)
    
    # Count files processed
    uids = [uid for uid in os.listdir(PATH_SENSOR) 
            if os.path.isdir(os.path.join(PATH_SENSOR, uid))]
    
    keyevent_count = 0
    appusage_count = 0
    
    for uid in uids:
        keyevent_paths = glob(os.path.join(PATH_SENSOR, uid, 'KeyEvent.csv'))
        appusage_paths = glob(os.path.join(PATH_SENSOR, uid, 'AppUsageEvent.csv'))
        
        keyevent_count += len(keyevent_paths)
        appusage_count += len(appusage_paths)
    
    print(f"Users processed: {len(uids)}")
    print(f"KeyEvent.csv files found: {keyevent_count}")
    print(f"AppUsageEvent.csv files found: {appusage_count}")
    
    # Check categorization results
    if os.path.exists(OUTPUT_CSV_PATH):
        apps_df = pd.read_csv(OUTPUT_CSV_PATH)
        total_apps = len(apps_df)
        categorized_apps = apps_df['category'].notna().sum()
        uncategorized_apps = total_apps - categorized_apps
        
        print(f"\nApp Categorization Results:")
        print(f"Total unique apps: {total_apps}")
        print(f"Successfully categorized: {categorized_apps}")
        print(f"Remaining uncategorized: {uncategorized_apps}")
        print(f"Categorization success rate: {(categorized_apps/total_apps)*100:.1f}%")
        
        # Show some examples of categorized apps
        if categorized_apps > 0:
            print(f"\nSample categorized apps:")
            sample_categorized = apps_df[apps_df['category'].notna()].head(5)
            for _, row in sample_categorized.iterrows():
                print(f"  {row['packageName']} -> {row['category']} (via {row['source']})")
        
        # Show some examples of uncategorized apps
        if uncategorized_apps > 0:
            print(f"\nSample uncategorized apps:")
            sample_uncategorized = apps_df[apps_df['category'].isna()].head(5)
            for _, row in sample_uncategorized.iterrows():
                print(f"  {row['packageName']} -> UNCATEGORIZED")

def validate_data_files():
    """Validate that the required data files exist and have the expected structure."""
    print("Validating data files...")
    
    uids = [uid for uid in os.listdir(PATH_SENSOR) 
            if os.path.isdir(os.path.join(PATH_SENSOR, uid))]
    
    validation_results = {
        'total_users': len(uids),
        'users_with_keyevent': 0,
        'users_with_appusage': 0,
        'users_with_both': 0,
        'users_with_neither': 0
    }
    
    for uid in uids:
        has_keyevent = os.path.exists(os.path.join(PATH_SENSOR, uid, 'KeyEvent.csv'))
        has_appusage = os.path.exists(os.path.join(PATH_SENSOR, uid, 'AppUsageEvent.csv'))
        
        if has_keyevent and has_appusage:
            validation_results['users_with_both'] += 1
        elif has_keyevent:
            validation_results['users_with_keyevent'] += 1
        elif has_appusage:
            validation_results['users_with_appusage'] += 1
        else:
            validation_results['users_with_neither'] += 1
    
    print(f"Validation Results:")
    print(f"  Total users: {validation_results['total_users']}")
    print(f"  Users with KeyEvent.csv only: {validation_results['users_with_keyevent']}")
    print(f"  Users with AppUsageEvent.csv only: {validation_results['users_with_appusage']}")
    print(f"  Users with both: {validation_results['users_with_both']}")
    print(f"  Users with neither: {validation_results['users_with_neither']}")
    
    return validation_results

# =============================================================================
# WEB SCRAPING FUNCTIONS
# =============================================================================

def extract_app_category_stage1(browser, path_data):
    """Stage 1: Extract app categories from apkcombo.com"""
    print("Stage 1: Extracting categories from apkcombo.com...")
    
    search_platform = "https://apkcombo.com"
    df_rows = []
    wait = WebDriverWait(browser, 20)
    
    app_list = pd.read_csv(os.path.join(path_data, "uncategorized_apps.csv"))["packageName"].astype(str).str.lower().tolist()
    
    for packageName in tqdm(app_list):
        row = [packageName, None, None, search_platform]
        try:
            browser.get(search_platform)
            q = wait.until(EC.presence_of_element_located((By.NAME, "q")))
            q.clear()
            q.send_keys(packageName)
            wait.until(EC.element_to_be_clickable((By.CLASS_NAME, "button-search"))).click()
            wait.until(EC.presence_of_all_elements_located((By.TAG_NAME, "body")))
            
            soup = BeautifulSoup(browser.page_source, "html.parser")
            
            # Extract app name
            t = soup.select_one("div.app_name h1 a")
            if t and t.text:
                row[1] = t.text.strip().upper()
            
            # Extract category
            label = soup.find("div", string=lambda s: s and "Category" in s)
            if label:
                nxt = label.find_next_sibling("div")
                if nxt and nxt.text:
                    row[2] = nxt.text.strip().upper()
                    
        except Exception as e:
            print(f"[WARN] {packageName}: {e}")
        
        df_rows.append(row)
    
    return pd.DataFrame(df_rows, columns=["packageName", "appName", "category", "source"])

def extract_app_category_stage2(apps):
    """Stage 2: Extract app categories from play.google.com for uncategorized apps"""
    print("Stage 2: Extracting categories from play.google.com...")
    
    dics = []
    opts = FirefoxOptions()
    opts.add_argument("--headless")
    
    capabilities = DesiredCapabilities.FIREFOX
    capabilities["marionette"] = True
    capabilities["acceptInsecureCerts"] = True
    opts.set_preference("intl.accept_languages", "en-US, en")
    
    browser = webdriver.Firefox(options=opts, capabilities=capabilities)
    
    for idx, row in tqdm(apps.iterrows(), total=apps.shape[0]):
        package_id = row['packageName']
        if pd.notna(row['category']):
            continue
            
        row_data = [package_id, None, None, "https://play.google.com"]
        search_platform = f'https://play.google.com/store/apps/details?id={package_id}&hl=en'
        
        try:
            browser.get(search_platform)
            sleep(3)
            soup = BeautifulSoup(browser.page_source, 'html.parser')
            
            # Extract app name
            app_name_tag = soup.find('span', {'itemprop': 'name'})
            if app_name_tag:
                row_data[1] = app_name_tag.text.strip()
            
            # Try to extract category using multiple methods
            category_text = None
            
            # Method 1: JSON-LD script
            json_ld_script = soup.find('script', {'type': 'application/ld+json'})
            if json_ld_script:
                try:
                    json_data = json.loads(json_ld_script.string)
                    if "applicationCategory" in json_data:
                        category_text = json_data["applicationCategory"].strip().upper()
                except json.JSONDecodeError:
                    pass
            
            # Method 2: Genre div
            if not category_text:
                genre_div = soup.find('div', {'itemprop': 'genre'})
                if genre_div:
                    genre_span = genre_div.find('span', {'aria-hidden': 'true'})
                    if genre_span:
                        category_text = genre_span.text.strip().upper()
            
            # Method 3: Category links
            if not category_text:
                category_a_tags = soup.find_all('a', href=True)
                for a_tag in category_a_tags:
                    if "/store/apps/category/" in a_tag['href']:
                        category_text = a_tag.text.strip().upper()
                        break
            
            if category_text:
                row_data[2] = category_text
                
        except Exception as e:
            print(f"Error processing {package_id}: {e}")
        
        dics.append(row_data)
        sleep(2)
    
    browser.quit()
    return pd.DataFrame(dics, columns=['packageName', 'appName', 'category', 'source'])

def extract_app_category_stage3(app):
    """Stage 3: Extract app categories from apkpure.net for remaining uncategorized apps"""
    print("Stage 3: Extracting categories from apkpure.net...")
    
    for idx, row in tqdm(app.iterrows(), total=app.shape[0]):
        package_id = row['packageName']
        if pd.notna(row.get('category')):
            continue
        
        try:
            search_url = f'https://apkpure.net/search?q={package_id}'
            search_response = requests.get(search_url)
            search_soup = BeautifulSoup(search_response.text, 'html.parser')
            
            url_tag = search_soup.find('a', class_='first-info')
            if url_tag:
                app_url = url_tag.get('href')
                if not app_url.startswith('http'):
                    app_url = 'https://apkpure.net' + app_url
                
                response = requests.get(app_url)
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Extract category information
                information_box = soup.find('div', class_='information-box')
                if not information_box:
                    continue
                
                apk_info = information_box.find('div', class_='apk-info')
                if not apk_info:
                    continue
                
                row_div = apk_info.find('div', class_='row')
                if not row_div:
                    continue
                
                info_divs = row_div.find_all('div', class_='info')
                for info_div in info_divs:
                    title_div = info_div.find('div', class_='title')
                    title_text = title_div.get_text(strip=True) if title_div else None
                    
                    if title_text == 'Category':
                        category_tag = info_div.find(class_='additional-info')
                        if category_tag and category_tag.get_text(strip=True):
                            category = category_tag.get_text(strip=True)
                            app.at[idx, 'category'] = category.upper()
                            break
                
                # Extract app name
                title_tag = soup.find('title')
                if title_tag:
                    app_name = title_tag.text.strip()
                    app.at[idx, 'appName'] = app_name
                
                app.at[idx, 'source'] = 'apkpure.net'
                
        except Exception as e:
            print(f"Error processing {package_id}: {e}")
    
    return app

def export_categorization_results():
    """Export the categorization results in different formats for analysis."""
    print("\n" + "=" * 70)
    print("EXPORTING CATEGORIZATION RESULTS")
    print("=" * 70)
    
    if not os.path.exists(OUTPUT_CSV_PATH):
        print("No categorization results found. Run the categorization first.")
        return
    
    apps_df = pd.read_csv(OUTPUT_CSV_PATH)
    
    # Export main results
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Export categorized apps only
    categorized_df = apps_df[apps_df['category'].notna()].copy()
    if len(categorized_df) > 0:
        categorized_path = os.path.join(PATH_DATA, f"categorized_apps_{timestamp}.csv")
        categorized_df.to_csv(categorized_path, index=False)
        print(f"Exported {len(categorized_df)} categorized apps to: {categorized_path}")
    
    # 2. Export uncategorized apps only
    uncategorized_df = apps_df[apps_df['category'].isna()].copy()
    if len(uncategorized_df) > 0:
        uncategorized_path = os.path.join(PATH_DATA, f"uncategorized_apps_{timestamp}.csv")
        uncategorized_df.to_csv(uncategorized_path, index=False)
        print(f"Exported {len(uncategorized_df)} uncategorized apps to: {uncategorized_path}")
    
    # 3. Export summary statistics
    summary_stats = {
        'total_apps': len(apps_df),
        'categorized_apps': len(categorized_df),
        'uncategorized_apps': len(uncategorized_df),
        'success_rate': (len(categorized_df) / len(apps_df)) * 100 if len(apps_df) > 0 else 0,
        'timestamp': timestamp
    }
    
    # Count apps by source
    source_counts = apps_df['source'].value_counts().to_dict()
    summary_stats.update({f'source_{k.lower().replace("https://", "").replace(".", "_")}': v 
                         for k, v in source_counts.items()})
    
    summary_path = os.path.join(PATH_DATA, f"categorization_summary_{timestamp}.json")
    with open(summary_path, 'w') as f:
        json.dump(summary_stats, f, indent=2)
    print(f"Exported summary statistics to: {summary_path}")
    
    # 4. Export category distribution
    if len(categorized_df) > 0:
        category_dist = categorized_df['category'].value_counts()
        category_path = os.path.join(PATH_DATA, f"category_distribution_{timestamp}.csv")
        category_dist.to_csv(category_path)
        print(f"Exported category distribution to: {category_path}")
        print(f"Top 5 categories: {category_dist.head().to_dict()}")

def safe_categorize_apps():
    """Safely run the categorization process with error handling."""
    try:
        print("Starting safe categorization process...")
        
        # Stage 1: apkcombo.com
        print("=" * 50)
        print("STAGE 1: apkcombo.com")
        print("=" * 50)
        
        if os.path.exists(OUTPUT_CSV_PATH):
            os.remove(OUTPUT_CSV_PATH)
        
        # Try Firefox first
        browser = setup_browser(headless=True)
        if browser is None:
            print("Firefox setup failed, trying Chrome/Chromium...")
            browser = setup_chrome_browser(headless=True)
        
        if browser is None:
            print("Both Firefox and Chrome setup failed!")
            print("Please install the required dependencies.")
            install_dependencies()
            return None
        
        try:
            df = extract_app_category_stage1(browser, PATH_DATA)
        except Exception as e:
            print(f"Error in Stage 1: {e}")
            return None
        finally:
            try:
                browser.quit()
            except:
                pass
        
        df.drop(columns=[c for c in df.columns if "Unnamed" in c], inplace=True, errors="ignore")
        df.to_csv(OUTPUT_CSV_PATH, index=False)
        print("Stage 1 results - Missing categories:", df.isna().sum())
        
        # Stage 2: play.google.com
        print("\n" + "=" * 50)
        print("STAGE 2: play.google.com")
        print("=" * 50)
        
        try:
            df = pd.read_csv(OUTPUT_CSV_PATH)
            apps_to_categorize = df[df['category'].isna()]
            if len(apps_to_categorize) > 0:
                res = extract_app_category_stage2(apps_to_categorize)
                
                df.set_index('packageName', inplace=True)
                res.set_index('packageName', inplace=True)
                df.update(res)
                df.reset_index(inplace=True)
                df.to_csv(OUTPUT_CSV_PATH, index=False)
                print("Stage 2 results - Missing categories:", df.isna().sum())
            else:
                print("No apps need Stage 2 categorization.")
        except Exception as e:
            print(f"Error in Stage 2: {e}")
        
        # Stage 3: apkpure.net
        print("\n" + "=" * 50)
        print("STAGE 3: apkpure.net")
        print("=" * 50)
        
        try:
            df = pd.read_csv(OUTPUT_CSV_PATH, index_col=False, header=0)
            df = extract_app_category_stage3(df)
            df.drop(columns=[col for col in df.columns if 'Unnamed' in col], inplace=True)
            df.to_csv(OUTPUT_CSV_PATH, index=False)
            print("Stage 3 results - Missing categories:", df.isna().sum())
        except Exception as e:
            print(f"Error in Stage 3: {e}")
        
        return df
        
    except Exception as e:
        print(f"Critical error in categorization process: {e}")
        return None

# =============================================================================
# MAIN PROCESSING FUNCTIONS
# =============================================================================

def categorize_apps():
    """Main function to categorize apps using all three stages."""
    
    # Stage 1: apkcombo.com
    print("=" * 50)
    print("STAGE 1: apkcombo.com")
    print("=" * 50)
    
    if os.path.exists(OUTPUT_CSV_PATH):
        os.remove(OUTPUT_CSV_PATH)
    
    try:
        browser = setup_browser(headless=True)
        df = extract_app_category_stage1(browser, PATH_DATA)
    finally:
        browser.quit()
    
    df.drop(columns=[c for c in df.columns if "Unnamed" in c], inplace=True, errors="ignore")
    df.to_csv(OUTPUT_CSV_PATH, index=False)
    print("Stage 1 results - Missing categories:", df.isna().sum())
    
    # Stage 2: play.google.com
    print("\n" + "=" * 50)
    print("STAGE 2: play.google.com")
    print("=" * 50)
    
    df = pd.read_csv(OUTPUT_CSV_PATH)
    apps_to_categorize = df[df['category'].isna()]
    res = extract_app_category_stage2(apps_to_categorize)
    
    df.set_index('packageName', inplace=True)
    res.set_index('packageName', inplace=True)
    df.update(res)
    df.reset_index(inplace=True)
    df.to_csv(OUTPUT_CSV_PATH, index=False)
    print("Stage 2 results - Missing categories:", df.isna().sum())
    
    # Stage 3: apkpure.net
    print("\n" + "=" * 50)
    print("STAGE 3: apkpure.net")
    print("=" * 50)
    
    df = pd.read_csv(OUTPUT_CSV_PATH, index_col=False, header=0)
    df = extract_app_category_stage3(df)
    df.drop(columns=[col for col in df.columns if 'Unnamed' in col], inplace=True)
    df.to_csv(OUTPUT_CSV_PATH, index=False)
    print("Stage 3 results - Missing categories:", df.isna().sum())

def update_user_files():
    """Update each user's KeyEvent.csv and AppUsageEvent.csv with the new app categories."""
    print("\n" + "=" * 50)
    print("UPDATING USER FILES")
    print("=" * 50)
    
    apps = pd.read_csv(OUTPUT_CSV_PATH, index_col=0)
    package_category_mapping = dict(zip(apps.index, apps['category']))
    
    uids = [uid for uid in os.listdir(PATH_SENSOR) 
            if os.path.isdir(os.path.join(PATH_SENSOR, uid))]
    
    for uid in tqdm(uids, desc="Updating user files"):
        # Update KeyEvent.csv files
        keyevent_paths = glob(os.path.join(PATH_SENSOR, uid, 'KeyEvent.csv'))
        for keyevent_path in keyevent_paths:
            try:
                df = pd.read_csv(keyevent_path, index_col=0)
                if 'packageName' in df.columns:
                    df['category'] = df['packageName'].map(package_category_mapping)
                    df.to_csv(keyevent_path)
                    print(f"Updated {keyevent_path}")
            except Exception as e:
                print(f"Error updating {keyevent_path}: {e}")
        
        # Update AppUsageEvent.csv files
        appusage_paths = glob(os.path.join(PATH_SENSOR, uid, 'AppUsageEvent.csv'))
        for appusage_path in appusage_paths:
            try:
                df = pd.read_csv(appusage_path, index_col=0)
                if 'packageName' in df.columns:
                    df['category'] = df['packageName'].map(package_category_mapping)
                    df.to_csv(appusage_path)
                    print(f"Updated {appusage_path}")
            except Exception as e:
                print(f"Error updating {appusage_path}: {e}")

def check_system_requirements():
    """Check if the system meets the requirements for web scraping."""
    print("\n" + "=" * 70)
    print("SYSTEM REQUIREMENTS CHECK")
    print("=" * 70)
    
    requirements_met = True
    
    # Check Python packages
    print("Checking Python packages...")
    try:
        import selenium
        print(f"✓ Selenium: {selenium.__version__}")
    except ImportError:
        print("✗ Selenium: Not installed")
        requirements_met = False
    
    try:
        import bs4
        print(f"✓ BeautifulSoup4: {bs4.__version__}")
    except ImportError:
        print("✗ BeautifulSoup4: Not installed")
        requirements_met = False
    
    try:
        import requests
        print(f"✓ Requests: {requests.__version__}")
    except ImportError:
        print("✗ Requests: Not installed")
        requirements_met = False
    
    # Check system binaries
    print("\nChecking system binaries...")
    
    # Firefox
    firefox_found = False
    firefox_paths = [
        shutil.which("firefox"),
        "/usr/bin/firefox",
        "/usr/local/bin/firefox",
        "/snap/bin/firefox",
        "/opt/firefox/firefox"
    ]
    
    for path in firefox_paths:
        if path and os.path.exists(path):
            print(f"✓ Firefox: {path}")
            firefox_found = True
            break
    
    if not firefox_found:
        print("✗ Firefox: Not found")
        requirements_met = False
    
    # Geckodriver
    gecko_found = False
    gecko_paths = [
        shutil.which("geckodriver"),
        "/usr/local/bin/geckodriver",
        "/usr/bin/geckodriver",
        "./geckodriver"
    ]
    
    for path in gecko_paths:
        if path and os.path.exists(path):
            print(f"✓ Geckodriver: {path}")
            gecko_found = True
            break
    
    if not gecko_found:
        print("✗ Geckodriver: Not found")
        requirements_met = False
    
    # Check file permissions
    print("\nChecking file permissions...")
    try:
        # Test if we can write to the output directory
        test_file = os.path.join(PATH_DATA, "test_write.tmp")
        with open(test_file, 'w') as f:
            f.write("test")
        os.remove(test_file)
        print("✓ Write permissions: OK")
    except Exception as e:
        print(f"✗ Write permissions: Failed - {e}")
        requirements_met = False
    
    # Check internet connectivity
    print("\nChecking internet connectivity...")
    try:
        response = requests.get("https://www.google.com", timeout=5)
        if response.status_code == 200:
            print("✓ Internet connectivity: OK")
        else:
            print(f"✗ Internet connectivity: HTTP {response.status_code}")
            requirements_met = False
    except Exception as e:
        print(f"✗ Internet connectivity: Failed - {e}")
        requirements_met = False
    
    print("\n" + "=" * 70)
    if requirements_met:
        print("✓ ALL REQUIREMENTS MET - Ready to proceed with categorization")
    else:
        print("✗ SOME REQUIREMENTS NOT MET - Please fix the issues above")
        print("\nInstallation commands:")
        print("sudo apt-get update")
        print("sudo apt-get install firefox")
        print("wget https://github.com/mozilla/geckodriver/releases/latest/download/geckodriver-v0.33.0-linux64.tar.gz")
        print("tar -xzf geckodriver-v0.33.0-linux64.tar.gz")
        print("sudo mv geckodriver /usr/local/bin/")
        print("sudo chmod +x /usr/local/bin/geckodriver")
        print("pip install selenium beautifulsoup4 requests")
    
    print("=" * 70)
    return requirements_met

def test_mode():
    """Run in test mode without web scraping - useful for debugging data collection."""
    print("\n" + "=" * 70)
    print("TEST MODE - Data Collection Only (No Web Scraping)")
    print("=" * 70)
    
    # Check system requirements (skip browser checks)
    print("Checking basic system requirements...")
    
    try:
        import pandas as pd
        print(f"✓ Pandas: {pd.__version__}")
    except ImportError:
        print("✗ Pandas: Not installed")
        return False
    
    try:
        import numpy as np
        print(f"✓ NumPy: {np.__version__}")
    except ImportError:
        print("✗ NumPy: Not installed")
        return False
    
    # Validate data files
    validation_results = validate_data_files()
    
    # Collect app packages
    app_list = collect_all_app_packages()
    
    # Save uncategorized apps
    save_uncategorized_apps(app_list)
    
    # Create a dummy categorization file for testing
    print("\nCreating dummy categorization file for testing...")
    dummy_df = pd.DataFrame({
        'packageName': app_list,
        'appName': [f"Test App {i}" for i in range(len(app_list))],
        'category': ['TEST_CATEGORY'] * len(app_list),
        'source': ['TEST_MODE'] * len(app_list)
    })
    
    test_output_path = os.path.join(PATH_DATA, "test_categorization.csv")
    dummy_df.to_csv(test_output_path, index=False)
    print(f"Created test categorization file: {test_output_path}")
    
    # Test file updates (dry run)
    print("\nTesting file update process (dry run)...")
    uids = [uid for uid in os.listdir(PATH_SENSOR) 
            if os.path.isdir(os.path.join(PATH_SENSOR, uid))]
    
    for uid in uids[:3]:  # Test with first 3 users only
        print(f"Testing user: {uid}")
        
        # Test KeyEvent.csv
        keyevent_path = os.path.join(PATH_SENSOR, uid, 'KeyEvent.csv')
        if os.path.exists(keyevent_path):
            try:
                df = pd.read_csv(keyevent_path, index_col=0)
                if 'packageName' in df.columns:
                    print(f"  ✓ KeyEvent.csv: {len(df)} rows, packageName column exists")
                else:
                    print(f"  ✗ KeyEvent.csv: packageName column missing")
            except Exception as e:
                print(f"  ✗ KeyEvent.csv: Error reading - {e}")
        
        # Test AppUsageEvent.csv
        appusage_path = os.path.join(PATH_SENSOR, uid, 'AppUsageEvent.csv')
        if os.path.exists(appusage_path):
            try:
                df = pd.read_csv(appusage_path, index_col=0)
                if 'packageName' in df.columns:
                    print(f"  ✓ AppUsageEvent.csv: {len(df)} rows, packageName column exists")
                else:
                    print(f"  ✗ AppUsageEvent.csv: packageName column missing")
            except Exception as e:
                print(f"  ✗ AppUsageEvent.csv: Error reading - {e}")
    
    print("\n" + "=" * 70)
    print("TEST MODE COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print(f"Total unique apps found: {len(app_list)}")
    print(f"Test categorization file created: {test_output_path}")
    print("\nTo run full categorization, install dependencies and run without --test flag")
    
    return True

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function."""
    import sys
    
    # Check for test mode
    test_mode_flag = '--test' in sys.argv
    
    if test_mode_flag:
        print("D-4 App Categorization Script - TEST MODE")
        print("=" * 70)
        test_mode()
        return
    
    print("D-4 App Categorization Script - Enhanced for KeyEvent and AppUsageEvent")
    print("=" * 70)
    
    # Step 0: Check system requirements
    if not check_system_requirements():
        print("\nSystem requirements not met. Please fix the issues above and try again.")
        print("\nTo run in test mode (data collection only), use: python AppCategorization.py --test")
        return
    
    # Step 1: Validate data files
    validation_results = validate_data_files()
    
    # Step 2: Collect app packages from both sources
    app_list = collect_all_app_packages()
    
    # Step 3: Save uncategorized apps
    save_uncategorized_apps(app_list)
    
    # Step 4: Categorize apps (3-stage process with error handling)
    print("\nStarting app categorization process...")
    result_df = safe_categorize_apps()
    
    if result_df is None:
        print("Categorization failed. Check the error messages above.")
        return
    
    # Step 5: Update user files (both KeyEvent.csv and AppUsageEvent.csv)
    update_user_files()
    
    # Step 6: Export results in various formats
    export_categorization_results()
    
    # Step 7: Generate processing summary
    get_processing_summary()
    
    print("\n" + "=" * 70)
    print("APP CATEGORIZATION COMPLETE FOR BOTH KEYEVENT AND APPUSAGEEVENT!")
    print("=" * 70)
    print(f"Total unique apps processed: {len(app_list)}")
    print("Files updated:")
    print("- KeyEvent.csv files: Added 'category' column")
    print("- AppUsageEvent.csv files: Added 'category' column")
    print(f"\nData coverage:")
    print(f"- Users with KeyEvent data: {validation_results['users_with_keyevent'] + validation_results['users_with_both']}")
    print(f"- Users with AppUsageEvent data: {validation_results['users_with_appusage'] + validation_results['users_with_both']}")
    print(f"- Users with both data types: {validation_results['users_with_both']}")
    
    print(f"\nResults exported to:")
    print(f"- Main results: {OUTPUT_CSV_PATH}")
    print(f"- Additional exports: {PATH_DATA}/categorized_apps_*.csv")
    print(f"- Summary statistics: {PATH_DATA}/categorization_summary_*.json")
    print(f"- Category distribution: {PATH_DATA}/category_distribution_*.csv")

if __name__ == "__main__":
    main()