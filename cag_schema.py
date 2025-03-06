import os
import time
import csv
import requests

import pandas as pd

from io import BytesIO
from PIL import Image
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from typing import List, Optional
from dotenv import load_dotenv

API_KEY = os.getenv("API_KEY")

class Product(BaseModel):
    product_description: Optional[str] = Field(None, description="Product description")
    product_search_keywords: Optional[List[str]] = Field(None, description="Possible search keywords")
    product_attributes: Optional[List[str]] = Field(None, description="Product attributes")
    product_categories: Optional[List[str]] = Field(None, description="Product categories")
    product_tags: Optional[List[str]] = Field(None, description="Product tags")

class CachedKnowledgeBase:
    def __init__(self, api_key, model="models/gemini-1.5-flash-002", cache_hours=24):
        print("Initializing CachedKnowledgeBase")
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.cache = None
        self.ttl_seconds = int(cache_hours * 3600)
        self.cache_created = False
        print("CachedKnowledgeBase initialized.")

    def load_knowledge_base(self, kb_file_path, system_instruction=None):
        print(f"Loading knowledge base from: {kb_file_path}")
        with open(kb_file_path, 'r', encoding='utf-8') as file:
            kb_content = file.read()
            print("Knowledge base file read successfully.")

        if not system_instruction:
            system_instruction = (
                "You are a customer support specialist. Answer customer questions based ONLY on the information in the product catalogue and the tags catalogue"
                "If you are unable to extract any information, return 'unknown', but try your best to find the closest category / tags."
                "Always be polite, concise, and helpful."
            )

        try:
            self.cache = self.client.caches.create(
                model=self.model,
                config=types.CreateCachedContentConfig(
                    display_name=f"support_kb_{os.path.basename(kb_file_path)}",
                    system_instruction=system_instruction,
                    contents=[kb_content],
                    ttl=f"{self.ttl_seconds}s",
                )
            )
            self.cache_created = True
            print(f"Knowledge base cached successfully! (ID: {self.cache.name})")
            print(f"Cache will expire in {self.ttl_seconds / 3600} hours")
            return True
        except Exception as e:
            print(f"Failed to cache knowledge base: {e}")
            return False

    def answer_question(self, question, temperature=0.2):
        print(f"Answering question: '{question}'")
        if not self.cache_created:
            raise Exception("Knowledge base not cached. Call load_knowledge_base first.")

        try:
            start_time = time.time()
            response = self.client.models.generate_content(
                model=self.model,
                contents=question,
                config=types.GenerateContentConfig(
                    cached_content=self.cache.name,
                    temperature=temperature,
                    response_mime_type='application/json',
                    response_schema=Product
                )
            )
            end_time = time.time()
            print(f"Response received in {end_time - start_time:.2f} seconds.")

            gemini_json_response = Product.model_validate_json(response.text)
            print(f"gemini_response:{gemini_json_response.product_description}")

            candidates_token_count = response.usage_metadata.candidates_token_count
            prompt_token_count = response.usage_metadata.prompt_token_count
            total_token_count = response.usage_metadata.total_token_count

            return gemini_json_response, prompt_token_count, candidates_token_count, total_token_count

        except Exception as e:
            print(f"Error answering question: {e}")
            return None, None, None, None

    def cleanup(self):
        if self.cache_created:
            try:
                self.client.caches.delete(name=self.cache.name)
                print("Cache deleted successfully")
                self.cache_created = False
                return True
            except Exception as e:
                print(f"Failed to delete cache: {e}")
                return False

def save_product_to_csv(product: Product, product_id: int, product_title: str, product_brand: str, product_offer_price: float,  
                        product_offer_percentage: float, product_rating: float, product_num_interactions: int, product_postcode: str, product_lat: float, product_long: float, product_geohash: str, product_link: str, product_image_link: str, product_forced: bool, product_spider_name: str, product_entity_type: str, input_token, output_token, total_token):
    file_exists = os.path.isfile("products.csv")

    with open("products.csv", mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow(["product_id", "product_title", "product_brand", "product_offer_price", "product_offer_percentage", "product_rating", "product_num_interactions", "product_postcode", "product_lat", "product_long", "product_geohash", "product_link", "product_image_link", "product_forced", "product_spider_name", "product_entity_type", "product_description", "product_search_keywords", "product_attributes", "product_categories", "product_tags", "input_token", "output_token", "total_token"])

        writer.writerow([product_id, product_title, product_brand, product_offer_price, product_offer_percentage,
                        product_rating, product_num_interactions, product_postcode, product_lat, product_long,
                        product_geohash, product_link, product_image_link, product_forced, product_spider_name,
                        product_entity_type, product.product_description, '; '.join(product.product_search_keywords or []),
                        '; '.join(product.product_attributes or []), '; '.join(product.product_categories or []),
                        '; '.join(product.product_tags or []), input_token, output_token, total_token])

    print(f"Product {product_id} saved to CSV.")

def create_question(description, image=None):
    base_question = (
        f"Summarize the product using the description and image. "
        f"Description: {description}. "
        "If you are unable to write the description, then return the original description. If you are not able to write the attributes or keywords, write 'unknown'."
        "Return:"
        "1. Product Description - concise but not too short)"
        "2. Product Search Keywords - for search purposes (return in a list separated by '; ')"
        "3. Product Attributes - For shopping items, return colour, material, style, type. For food items, return cuisine, type, style. For accommodation items, return type, rooms, style, colour. For event items, return type, time, venue. For service items, return type, time, location, availability. For job items, return type, wages, experience level, qualifications, location. For other miscellaneous items, return whatever attributes you think are appropriate (return in a list separated by '; ')"
        "4. Product Categories - choose the best and closest from the category catalogue"
        "5. Product Tags - in what scenario are these products most suited for? For example, time of day, weather, what point of interest is it related to. Add as many as you think is suitable from the tags catalogue (return these tags in a list separated by '; ')."
    )
    if image:
        return base_question.replace("Description:", f"Image: {image}, Description:")
    return base_question

if __name__ == "__main__":

    df = pd.read_parquet("data/tokens.parquet")

    batch_size = 1

    support_bot = CachedKnowledgeBase(
        api_key=API_KEY,
        model="models/gemini-1.5-flash-002",
        cache_hours=48
    )

    if support_bot.load_knowledge_base("data/products.txt"):
        print("Knowledge base loaded successfully.")

        total_products = len(df)
        for start_index in range(0, total_products, batch_size):
            print(f"Processing batch starting at index: {start_index}")
            batch = df.iloc[start_index:start_index + batch_size]

            for index, products in batch.iterrows():
                product_id = products['product_id']
                product_title = products['product_title']
                product_description = products['product_description_cleaned']
                product_brand = products['product_brand']
                product_offer_price = products['product_offer_price']
                product_offer_percentage = products['product_offer_percentage']
                product_rating = products['product_rating']
                product_num_interactions = products['product_num_interactions']
                product_postcode = products['product_postcode']
                product_lat = products['product_lat']
                product_long = products['product_long']
                product_geohash = products['product_geohash']
                product_link = products['product_link']
                product_image_link = products['product_image_link']
                product_categories = products['product_categories']
                product_tags = products['product_tags']
                product_forced = products['product_forced']
                product_spider_name = products['product_spider_name']
                product_entity_type = products['product_entity_type']

                char_limit = 200
                description = product_description[:char_limit] + '...' if len(product_description) > char_limit else product_description

                response = requests.get(product_image_link)
                image = None

                if response.status_code == 200:
                    try:
                        image = Image.open(BytesIO(response.content))
                        image = image.resize((640, 480))
                    except Exception as e:
                        print(f"Error opening image: {e}. Proceeding with description only.")

                question = create_question(description, image=image)

                product_response, input_token, output_token, total_token = support_bot.answer_question(question)

                save_product_to_csv(product_response, product_id, product_title, product_brand, product_offer_price, product_offer_percentage, product_rating, product_num_interactions, product_postcode, product_lat, product_long, product_geohash, product_link, product_image_link, product_forced, product_spider_name, product_entity_type, input_token, output_token, total_token)

            print(f"Results after batch starting at index {start_index} saved.")

        support_bot.cleanup()
        print("Cleanup completed.")