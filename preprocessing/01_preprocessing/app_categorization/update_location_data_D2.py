#!/usr/bin/env python3
"""
Script to update Location.csv files in PATH_SENSOR with correct coordinates from origin_location folder.

This script:
1. Reads Location.csv files from origin_location folder
2. Updates corresponding Location.csv files in PATH_SENSOR (newdata folder)
3. Preserves all other data while correcting the longitude coordinates
4. Creates backups of original files before updating
"""

import os
import pandas as pd
import shutil
from pathlib import Path
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('location_update.log'),
        logging.StreamHandler()
    ]
)

# Paths
PATH_DATA = "<DATA_ROOT_D2>"
PATH_SENSOR = os.path.join(PATH_DATA, 'newdata')
ORIGIN_LOCATION = os.path.join(PATH_DATA, 'origin_location')
BACKUP_DIR = os.path.join(PATH_DATA, 'newdata_backup')

def create_backup_directory():
    """Create backup directory if it doesn't exist."""
    if not os.path.exists(BACKUP_DIR):
        os.makedirs(BACKUP_DIR)
        logging.info(f"Created backup directory: {BACKUP_DIR}")

def backup_original_file(user_id):
    """Create a backup of the original Location.csv file."""
    original_file = os.path.join(PATH_SENSOR, user_id, 'Location.csv')
    backup_file = os.path.join(BACKUP_DIR, f'{user_id}_Location_backup.csv')
    
    if os.path.exists(original_file):
        shutil.copy2(original_file, backup_file)
        logging.info(f"Created backup: {backup_file}")
        return True
    else:
        logging.warning(f"Original file not found: {original_file}")
        return False

def update_location_file(user_id):
    """
    Update Location.csv file for a specific user.
    
    Args:
        user_id (str): User ID (e.g., 'P126')
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Paths for this user
        origin_file = os.path.join(ORIGIN_LOCATION, user_id, 'Location.csv')
        target_file = os.path.join(PATH_SENSOR, user_id, 'Location.csv')
        
        # Check if origin file exists
        if not os.path.exists(origin_file):
            logging.warning(f"Origin file not found for {user_id}: {origin_file}")
            return False
        
        # Check if target file exists
        if not os.path.exists(target_file):
            logging.warning(f"Target file not found for {user_id}: {target_file}")
            return False
        
        # Read the origin file (correct coordinates)
        logging.info(f"Reading origin file for {user_id}: {origin_file}")
        origin_df = pd.read_csv(origin_file)
        
        # Read the target file (incorrect coordinates)
        logging.info(f"Reading target file for {user_id}: {target_file}")
        target_df = pd.read_csv(target_file)
        
        # Verify the files have the same structure
        if not (origin_df.columns == target_df.columns).all():
            logging.error(f"Column mismatch for {user_id}")
            logging.error(f"Origin columns: {origin_df.columns.tolist()}")
            logging.error(f"Target columns: {target_df.columns.tolist()}")
            return False
        
        # Verify they have the same number of rows
        if len(origin_df) != len(target_df):
            logging.warning(f"Row count mismatch for {user_id}: origin={len(origin_df)}, target={len(target_df)}")
        
        # Create backup
        backup_original_file(user_id)
        
        # Update the target file with correct coordinates
        # We'll replace the entire file with the origin data
        origin_df.to_csv(target_file, index=False)
        
        logging.info(f"Successfully updated {user_id}: {target_file}")
        
        # Log some statistics
        logging.info(f"{user_id} - Rows updated: {len(origin_df)}")
        logging.info(f"{user_id} - Longitude range: {origin_df['longitude'].min():.6f} to {origin_df['longitude'].max():.6f}")
        
        return True
        
    except Exception as e:
        logging.error(f"Error updating {user_id}: {str(e)}")
        return False

def get_user_ids():
    """Get list of user IDs that have Location.csv files in both directories."""
    origin_users = set()
    target_users = set()
    
    # Get users from origin_location
    if os.path.exists(ORIGIN_LOCATION):
        for item in os.listdir(ORIGIN_LOCATION):
            origin_file = os.path.join(ORIGIN_LOCATION, item, 'Location.csv')
            if os.path.exists(origin_file):
                origin_users.add(item)
    
    # Get users from PATH_SENSOR
    if os.path.exists(PATH_SENSOR):
        for item in os.listdir(PATH_SENSOR):
            target_file = os.path.join(PATH_SENSOR, item, 'Location.csv')
            if os.path.exists(target_file):
                target_users.add(item)
    
    # Return intersection
    common_users = origin_users.intersection(target_users)
    logging.info(f"Found {len(common_users)} users with Location.csv files in both directories")
    logging.info(f"Origin users: {len(origin_users)}")
    logging.info(f"Target users: {len(target_users)}")
    
    return sorted(list(common_users))

def main():
    """Main function to update all Location.csv files."""
    logging.info("Starting Location.csv update process")
    logging.info(f"PATH_DATA: {PATH_DATA}")
    logging.info(f"PATH_SENSOR: {PATH_SENSOR}")
    logging.info(f"ORIGIN_LOCATION: {ORIGIN_LOCATION}")
    
    # Check if directories exist
    if not os.path.exists(PATH_SENSOR):
        logging.error(f"PATH_SENSOR does not exist: {PATH_SENSOR}")
        return
    
    if not os.path.exists(ORIGIN_LOCATION):
        logging.error(f"ORIGIN_LOCATION does not exist: {ORIGIN_LOCATION}")
        return
    
    # Create backup directory
    create_backup_directory()
    
    # Get user IDs
    user_ids = get_user_ids()
    
    if not user_ids:
        logging.error("No users found with Location.csv files in both directories")
        return
    
    # Update each user's Location.csv file
    successful_updates = 0
    failed_updates = 0
    
    for user_id in user_ids:
        logging.info(f"Processing {user_id}...")
        if update_location_file(user_id):
            successful_updates += 1
        else:
            failed_updates += 1
    
    # Summary
    logging.info("=" * 50)
    logging.info("UPDATE SUMMARY")
    logging.info("=" * 50)
    logging.info(f"Total users processed: {len(user_ids)}")
    logging.info(f"Successful updates: {successful_updates}")
    logging.info(f"Failed updates: {failed_updates}")
    logging.info(f"Backup directory: {BACKUP_DIR}")
    logging.info("=" * 50)
    
    if successful_updates > 0:
        logging.info("Location.csv files have been updated with correct coordinates!")
        logging.info("Original files have been backed up in the backup directory.")
    else:
        logging.error("No files were successfully updated!")

if __name__ == "__main__":
    main() 