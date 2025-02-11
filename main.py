import argparse
import logging
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from bs4 import BeautifulSoup
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# -------------------- WEB SCRAPING FUNCTIONS -------------------- #

def scrape_website(url):
    logging.info("Launching Chrome browser...")
    
    chrome_driver_path = "./chromedriver"  
    options = webdriver.ChromeOptions()
    options.add_argument("--headless") 
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    
    driver = webdriver.Chrome(service=Service(chrome_driver_path), options=options)

    try:
        logging.info(f"Loading website: {url}")
        driver.get(url)
        html_content = driver.page_source
        logging.info("Page successfully loaded.")
        return html_content
    except Exception as e:
        logging.error(f"Error scraping website: {e}")
        return ""
    finally:
        driver.quit()


def extract_body_content(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    return str(soup.body) if soup.body else ""


def clean_body_content(body_content):
    soup = BeautifulSoup(body_content, "html.parser")
    
    for tag in soup(["script", "style"]):
        tag.extract()
    
    cleaned_text = "\n".join(line.strip() for line in soup.get_text(separator="\n").splitlines() if line.strip())
    return cleaned_text


def split_dom_content(dom_content, max_length=6000):
    return [dom_content[i : i + max_length] for i in range(0, len(dom_content), max_length)]


# -------------------- PARSING FUNCTIONS -------------------- #

# Initialize Ollama LLM
model = OllamaLLM(model="llama3.2")

# Define parsing template
TEMPLATE = (
    "You are tasked with extracting specific information from the following text content: {dom_content}. "
    "Please follow these instructions carefully:\n\n"
    "1. **Extract Information:** Only extract the information that directly matches the provided description: {parse_description}.\n"
    "2. **No Extra Content:** Do not include any additional text, comments, or explanations in your response.\n"
    "3. **Empty Response:** If no information matches the description, return an empty string ('').\n"
    "4. **Direct Data Only:** Your output should contain only the data that is explicitly requested, with no other text."
)

def parse(dom_chunks, parse_description):
    prompt = ChatPromptTemplate.from_template(TEMPLATE)
    chain = prompt | model
    parsed_results = []

    for chunk in dom_chunks:
        response = chain.invoke({"dom_content": chunk, "parse_description": parse_description})
        parsed_results.append(response)

    return "\n".join(parsed_results)


# -------------------- MAIN SCRIPT -------------------- #

def main():
    parser = argparse.ArgumentParser(description="AI Web Scraper")
    parser.add_argument("url", type=str, help="Enter the website URL to scrape")
    args = parser.parse_args()

    logging.info("Starting web scraping process...")

    html_content = scrape_website(args.url)
    if not html_content:
        logging.error("Failed to retrieve website content. Exiting.")
        return

    body_content = extract_body_content(html_content)
    cleaned_content = clean_body_content(body_content)

    while True:
        parse_description = input("Describe what you want to parse (or type 'q' to quit): ")
        if parse_description.lower() == 'q':
            logging.info("Exiting program.")
            break

        logging.info("Parsing the content...")
        dom_chunks = split_dom_content(cleaned_content)
        parsed_result = parse(dom_chunks, parse_description)

        print("\n--- Parsed Result ---")
        print(parsed_result)
        print("----------------------")


if __name__ == "__main__":
    main()
