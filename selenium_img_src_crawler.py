import os

from selenium import webdriver
from selenium.common import TimeoutException
# from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
import pandas as pd
import time
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
import undetected_chromedriver as uc
from selenium.webdriver.support import expected_conditions as e_c


def click_accept_cookies(driver):
    try:
        # Espere até 10 segundos pelo botão de aceitar cookies e clique nele
        # Você precisará inspecionar a página para encontrar o seletor correto (ID, XPATH, etc.)
        accept_button = WebDriverWait(driver, 10).until(
            e_c.element_to_be_clickable((By.XPATH, "//button[contains(., 'Aceitar') or contains(., 'Accept')]"))
        )
        accept_button.click()
        print("Cookies aceitos.")
        return True
    except TimeoutException:
        print("Botão de aceitar cookies não encontrado ou não apareceu.")
    except Exception as e:
        print(f"Ocorreu um erro ao clicar nos cookies: {e}")
    return False

def search_url_images_google(query, df, num_images=10, seq=1):

    keyword = query.replace(" ", "_")
    search_url = f"https://www.google.com/search?site=&tbm=isch&source=hp&biw=1873&bih=990&q={query}"

    # options = webdriver.ChromeOptions()
    options = uc.ChromeOptions()
    # options.add_argument('--headless')
    # --> options.add_argument('--no-sandbox')
    options.add_argument('--disable-gpu')
    options.add_argument('--disable-web-security')
    options.add_argument('--allow-running-insecure-content')
    options.add_argument('--allow-cross-origin-auth-prompt')

    # Configura o driver do Selenium
    # driver = webdriver.Chrome(options=options)
    # Cria uma instância do navegador Chrome de forma indetectável
    driver = uc.Chrome()  # options=options
    time.sleep(1)
    driver.maximize_window()
    time.sleep(1)
    # Open browser to begin search
    driver.get(search_url)
    time.sleep(3)

    if not click_accept_cookies(driver):
        return df

    # Rola a página para carregar mais imagens
    rolar = True
    if rolar:
        last_height = driver.execute_script("return document.body.scrollHeight")
        while True:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)  # Aguarda o carregamento
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height

    seletor = "img[id^='dimg_'][src^='data:image'][class='YQ4gaf']"
    # images_box = driver.find_elements(By.CSS_SELECTOR, seletor)
    miniaturas = WebDriverWait(driver, 10).until(
        e_c.presence_of_all_elements_located((By.CSS_SELECTOR, seletor))
    )
    actions = ActionChains(driver)
    for img_box in miniaturas:
        if seq > num_images:
            break
        # Click on the thumbnail
        try:
            print(f"{seq}: Clicando na imagem miniatura")
            alt = img_box.get_attribute("alt")
            actions.move_to_element(img_box).perform()
            time.sleep(1)
            img_box.click()
            time.sleep(1)

            # Selector of the image display
            seletor = f"img[class^='sFlh5c'][src^='https://'][alt='{alt}'][jsname='kn3ccd']"
            # orig_images = driver.find_elements(By.CSS_SELECTOR, seletor)
            orig_images = WebDriverWait(driver, 10).until(
                e_c.presence_of_all_elements_located((By.CSS_SELECTOR, seletor))
            )
            # Wait between interaction
            time.sleep(2)
            for img_orig in orig_images:
                try:
                    print(f'{seq}: Salvando url da imagem para o df')
                    # img_orig.click()
                    actions.move_to_element(img_orig).perform()
                    time.sleep(1)
                    # Retrieve attribute of src from the element
                    img_src = img_orig.get_attribute('src')
                    if img_src not in df['src_link'].values:
                        df.loc[len(df)] = [keyword, img_src, 0, '', '', '', '']
                        seq += 1
                        print('OK:', img_src)
                        break
                except Exception as e:
                    print(f"Erro ao recuperar url da imagem: {e}")
        except Exception as e:
            print(f"Erro ao clicar na thumbnail: {e}")

    driver.close()
    # driver.quit()

    return df

def search_google(search_query):
    img_src = ''
    search_url = f"https://www.google.com/search?site=&tbm=isch&source=hp&biw=1873&bih=990&q={search_query}"

    options = webdriver.ChromeOptions()
    # options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-gpu')
    options.add_argument('--disable-web-security')
    options.add_argument('--allow-running-insecure-content')
    options.add_argument('--allow-cross-origin-auth-prompt')

    browser = webdriver.Chrome(options=options)
    browser.maximize_window()

    # Open browser to begin search
    browser.get(search_url)

    # XPath for the 1st image that appears in Google:
    seletor = "img[id^='dimg_'][src^='data:image'][class='YQ4gaf']"
    img_box = browser.find_element(By.CSS_SELECTOR, seletor)
    alt = img_box.get_attribute("alt")
    # Click on the thumbnail
    img_box.click()
    time.sleep(2)

    # Selector of the image display
    seletor = f"img[class^='sFlh5c'][src^='https://'][alt='{alt}'][jsname='kn3ccd']"
    orig_img = browser.find_element(By.CSS_SELECTOR, seletor)
    # Wait between interaction
    time.sleep(3)
    # orig_img.click()

    # Retrieve attribute of src from the element
    img_src = orig_img.get_attribute('src')

    return img_src


def search_image_fruits():
    keywords = pd.read_csv('output/scientific_botanical_names_veggies_fruits.csv', sep=",")
    # Creating header for file containing image source link
    with open("output/links/img_src_links.csv", "w") as outfile:
        outfile.write("search_terms|src_link\n")
    # Loops through the list of search input
    for keyword in keywords['scientific_names']:
        try:
            link = search_google(keyword)
            keyword = keyword.replace(" ", "_")
            with open("output/links/img_src_links.csv", "a") as outfile:
                outfile.write(f"{keyword}|{link}\n")
        except Exception as e:
            print(e)

def search_image_google(keywords):
    # Creating header for file containing image source link
    df: pd.DataFrame
    input_file = "output/links/img_forms_links.csv"
    output_file = "output/links/img_forms_links.csv"
    if os.path.exists(input_file):
        df = pd.read_csv(input_file, sep='|')
    else:
        df = pd.DataFrame(columns=['search_terms', 'src_link', 'status', 'reason', 'image_file', 'new_image', 'note'])

    seq = len(df) + 2
    # Loops through the list of search input
    for keyword in keywords:
        df = search_url_images_google(keyword, df, num_images=1090, seq=seq)
    df.to_csv(output_file, index=False, sep='|', columns=df.columns)
    print(df)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # main()
    # chromedriver = ChromeDriverManager().install()
    # search_keywords = ["tela de cadastro"]  # ficha dados de entrada, janela entrada de dados -porta -vidro
    # search_keywords = ["tela entrada dados"]   # ["formulario entrada dados"]
    search_keywords = ["registration window"]   # tkinter widgets, data entry design form system program box
    #  ["data entry window"]  #   ["data entry screen"]   # ["input data window... screen"]

    search_image_google(search_keywords)
