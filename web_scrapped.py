from selenium import webdriver

driver = webdriver.Chrome()
driver.get("https://library.iitgn.ac.in/faqs.php")

html = driver.page_source
with open("website.html", "w", encoding="utf-8") as file:
    file.write(html)

driver.quit()
print("HTML downloaded successfully!")
