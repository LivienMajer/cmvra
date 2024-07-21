import os
import getpass
import sys
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException
import time
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def setup_ccname():
    user = getpass.getuser()
    # check if k5start is running, exit otherwise
    try:
        pid = open("/tmp/k5pid_" + user).read().strip()
        os.kill(int(pid), 0)
    except:
        sys.stderr.write("Unable to setup KRB5CCNAME!\nk5start not running!\n")
        sys.exit(1)
    try:
        ccname = open("/tmp/kccache_" + user).read().split("=")[1].strip()
        os.environ['KRB5CCNAME'] = ccname
    except:
        sys.stderr.write("Unable to setup KRB5CCNAME!\nmaybe k5start not running?\n")
        sys.exit(1)


def download_url(driver, url, max_retries=100):
    retries = 0
    while retries < max_retries:
        try:
            driver.get(url)
            return  # Successfully loaded the URL, exit the function
        except TimeoutException:
            print(f"Timeout occurred while accessing {url}. Retrying... ({retries + 1}/{max_retries})")
            retries += 1
    print(f"Failed to access {url} after {max_retries} attempts.")

# Set up Kerberos authentication
setup_ccname()

chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--no-sandbox")  # Required when running as root

# Set Chrome to download files automatically to a specified directory without asking for a location each time
chrome_options.add_experimental_option("prefs", {
    "download.default_directory": "/net/polaris/storage/deeplearning/ntu",  # Change this to a desired path
    "download.prompt_for_download": False,
    "download.directory_upgrade": True,
    "safebrowsing_for_trusted_sources_enabled": False,
    "safebrowsing.enabled": False
})

with webdriver.Chrome(options=chrome_options) as driver:
    
    driver.get("https://rose1.ntu.edu.sg/login/")
    
    # Wait for username field to appear (you can adjust the timeout)
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.NAME, "username")))
    
    username_field = driver.find_element(By.NAME, "username")
    password_field = driver.find_element(By.NAME, "password")
    
    username_field.send_keys("Cappl")
    password_field.send_keys("@iBh5b4ET8nkcnM")
    
    password_field.submit()
    
    # Wait for a bit after login to ensure redirection or any other process completes
    time.sleep(5)
    
    # List of URLs to download
    urls = [f"https://rose1.ntu.edu.sg/dataset/actionRecognition/download/{i}" for i in [102, 158]]
    
    for url in urls:
        download_url(driver, url)
    
    # Wait again to ensure download starts
    time.sleep(5000000)
        

    