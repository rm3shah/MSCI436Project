# imports
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup
import csv
import time
import parameters as parameters

# defining new variable passing two parameters
writer = csv.writer(open(parameters.file_name, 'w'))

# writerow() method to the write to the file object
writer.writerow(['URL'])

# specifies the path to the chromedriver.exe
driver = webdriver.Chrome(executable_path='./chromedriver.exe')

# driver.get method() will navigate to a page given by the URL address
driver.get('https://www.linkedin.com')

# driver.find() sign in button
driver.find_element_by_class_name('nav__button-secondary')

# locate email form by_class_name
username = driver.find_element_by_xpath('/html/body/nav/section[2]/form/div[1]/div[1]/input')
# username = driver.find_element_by_name('session_key')

# send_keys() to simulate key strokes
username.send_keys(parameters.linkedin_username)

# sleep for 0.5 seconds
time.sleep(0.5)

# locate password form by_class_name
# password = driver.find_element_by_xpath('/html/body/nav/section[2]/form/div[1]/div[2]/input')
password = driver.find_element_by_name('session_password')

# send_keys() to simulate key strokes
password.send_keys(parameters.linkedin_password)
time.sleep(0.5)

# locate submit button by_xpath
# sign_in_button = driver.find_element_by_xpath('/html/body/nav/section[2]/form/div[2]/button')
sign_in_button = driver.find_element_by_class_name('sign-in-form__submit-btn')

# .click() to mimic button click
sign_in_button.click()
time.sleep(0.5)

# go back to google
driver.get('http://google.com')

# locate search form by_name
search_query = driver.find_element_by_name('q')

# send_keys() to simulate the search text key strokes
search_query.send_keys(parameters.search_query)

# .send_keys() to simulate the return key 
search_query.send_keys(Keys.RETURN)
time.sleep(0.5)

# set max page count for number of searches
max_page_count = 30
linkedin_urls = []

for page in range(max_page_count):
    # locate URL by_class_name
    results = driver.find_elements_by_css_selector('div.g')

    for i in range(len(results)):
        link = results[i].find_elements_by_tag_name("a")
        linkedin_urls.append(link[0].get_attribute("href"))

    driver.find_element_by_id('pnnext').send_keys(Keys.ENTER)
    time.sleep(2)

print(linkedin_urls)
for url in linkedin_urls:
    writer.writerow([url])



driver.close()
  
