# imports
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup
import csv
import time
import parameters as parameters

# defining new variable passing two parameters
writer = csv.writer(open(parameters.linkedin_file, 'w'))

# writerow() method to the write to the file object
writer.writerow(['URL'])

# specifies the path to the chromedriver.exe
driver = webdriver.Chrome(executable_path='./chromedriver.exe')