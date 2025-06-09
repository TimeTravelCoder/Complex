from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import time

# 初始化WebDriver，这里以Chrome为例，如果你使用的是其他浏览器，请相应更改
driver = webdriver.Chrome()

try:
    # 打开百度首页
    driver.get("https://www.baidu.com")

    # 显式等待，直到页面加载完成（可以调整等待秒数的大小）
    time.sleep(5)

    # 查找搜索框并输入搜索词
    search_box = driver.find_element(By.ID, 'kw')
    search_box.send_keys('李白')

    # 提交搜索请求
    search_box.send_keys(Keys.RETURN)

    # 等待页面加载
    time.sleep(5)

    # 获取搜索结果页面的标题
    title = driver.title
    print(f"搜索结果页面标题是: {title}")

finally:
    # 关闭浏览器
    driver.quit()
