#!/usr/bin/env python3
"""
D-2 App Categorization Script
=============================

This script categorizes apps from AppUsageEvent.csv files using web scraping.
It performs a 3-stage categorization process:
1. apkcombo.com
2. play.google.com  
3. apkpure.net

Author: Generated from notebook
Date: 2025.7.30
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

# D-2 paths
PATH_DATA = "<DATA_ROOT_D2>"
PATH_SENSOR = os.path.join(PATH_DATA, 'newdata')

# File paths
UNCATEGORIZED_APPS_PATH = os.path.join(PATH_DATA, "uncategorized_apps.csv")
OUTPUT_CSV_PATH = os.path.join(PATH_DATA, "app_category_1.csv")

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def setup_browser(headless=True):
    """Setup Firefox browser with appropriate options."""
    FIREFOX_BIN = shutil.which("firefox")
    GECKO_BIN = shutil.which("geckodriver")
    
    opts = FirefoxOptions()
    if headless:
        opts.add_argument("--headless")
    if FIREFOX_BIN:
        opts.binary_location = FIREFOX_BIN
    
    service = Service(executable_path=GECKO_BIN, log_output="geckodriver.log")
    return webdriver.Firefox(service=service, options=opts)

def collect_app_packages():
    """Collect all unique app package names from AppUsageEvent.csv files that need categorization."""
    print("Collecting app packages from all users...")
    
    uids = [uid for uid in os.listdir(PATH_SENSOR) 
            if os.path.isdir(os.path.join(PATH_SENSOR, uid))]
    
    app_list = set()
    categorized_apps = set()
    
    # First pass: collect all apps and check which ones are already categorized
    for uid in uids:
        app_usage_paths = sorted(glob(os.path.join(PATH_SENSOR, uid, 'AppUsageEvent.csv')))
        dfs = [pd.read_csv(path, index_col=False, header=0) for path in app_usage_paths]
        
        if dfs:
            df = pd.concat(dfs, ignore_index=True)
            if 'packageName' in df.columns and 'category' in df.columns:
                # Check which packages already have categories
                for _, row in df.iterrows():
                    package = row['packageName']
                    category = row.get('category')
                    
                    if pd.notna(package):
                        app_list.add(package)
                        # If category exists and is not NA, mark as categorized
                        if pd.notna(category) and category != 'UNKNOWN':
                            categorized_apps.add(package)
    
    # Only keep apps that are not categorized
    uncategorized_apps = app_list - categorized_apps
    uncategorized_apps = sorted(uncategorized_apps)
    
    print(f"# Total Apps found: {len(app_list)}")
    print(f"# Already categorized: {len(categorized_apps)}")
    print(f"# Apps needing categorization: {len(uncategorized_apps)}")
    
    return uncategorized_apps

def save_uncategorized_apps(app_list):
    """Save the list of uncategorized apps to CSV."""
    print("Saving uncategorized apps list...")
    
    if os.path.exists(UNCATEGORIZED_APPS_PATH):
        os.remove(UNCATEGORIZED_APPS_PATH)
    
    df_uncat = pd.DataFrame(app_list, columns=['packageName'])
    df_uncat.to_csv(UNCATEGORIZED_APPS_PATH, index=False)
    
    print(df_uncat.head(), "\n", df_uncat.columns, "\n", df_uncat.shape, "\n")

def merge_with_existing_categories(new_df):
    """Merge new categories with existing ones, preserving existing data."""
    if os.path.exists(OUTPUT_CSV_PATH):
        existing_df = pd.read_csv(OUTPUT_CSV_PATH)
        # Combine existing and new data
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        # Remove duplicates, keeping the first occurrence (existing data)
        combined_df = combined_df.drop_duplicates(subset=['packageName'], keep='first')
        return combined_df
    else:
        return new_df

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

# =============================================================================
# MAIN PROCESSING FUNCTIONS
# =============================================================================

def categorize_apps():
    """Main function to categorize apps using all three stages."""
    
    # Stage 1: apkcombo.com
    print("=" * 50)
    print("STAGE 1: apkcombo.com")
    print("=" * 50)
    
    try:
        browser = setup_browser(headless=True)
        df = extract_app_category_stage1(browser, PATH_DATA)
    finally:
        browser.quit()
    
    df.drop(columns=[c for c in df.columns if "Unnamed" in c], inplace=True, errors="ignore")
    
    # Merge with existing categories
    df = merge_with_existing_categories(df)
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
    """Update each user's AppUsageEvent.csv with the new app categories."""
    print("\n" + "=" * 50)
    print("UPDATING USER FILES")
    print("=" * 50)
    
    apps = pd.read_csv(OUTPUT_CSV_PATH, index_col=0)
    package_category_mapping = dict(zip(apps.index, apps['category']))
    
    uids = [uid for uid in os.listdir(PATH_SENSOR) 
            if os.path.isdir(os.path.join(PATH_SENSOR, uid))]
    
    for uid in tqdm(uids, desc="Updating user files"):
        df = pd.read_csv(os.path.join(PATH_SENSOR, uid, 'AppUsageEvent.csv'), index_col=0)
        
        # Only update categories that are NaN or missing, preserve existing non-null categories
        mask = df['category'].isna() | (df['category'] == 'UNKNOWN')
        df.loc[mask, 'category'] = df.loc[mask, 'packageName'].map(package_category_mapping)
        
        df.to_csv(os.path.join(PATH_SENSOR, uid, 'AppUsageEvent.csv'))

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function."""
    print("D-2 App Categorization Script")
    print("=" * 50)
    
    # Step 1: Collect app packages (only uncategorized ones)
    app_list = collect_app_packages()
    
    # If no apps need categorization, exit early
    if len(app_list) == 0:
        print("No apps need categorization. All apps are already categorized!")
        return
    
    # Step 2: Save uncategorized apps
    save_uncategorized_apps(app_list)
    
    # Step 3: Categorize apps (3-stage process)
    categorize_apps()
    
    # Step 4: Update user files
    update_user_files()
    
    print("\n" + "=" * 50)
    print("APP CATEGORIZATION COMPLETE!")
    print("=" * 50)

if __name__ == "__main__":
    main()