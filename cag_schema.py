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

load_dotenv()

API_KEY = os.getenv("API_KEY")

class Product(BaseModel):
    product_description: Optional[str] = Field(None, description="Product description")
    product_entity_type: Optional[str] = Field(None, description="Product entity type")
    product_search_keywords: Optional[List[str]] = Field(None, description="Possible search keywords")
    product_categories: Optional[List[str]] = Field(None, description="Product categories")
    product_tags: Optional[List[str]] = Field(None, description="Product tags")
    product_type: Optional[str] = Field(None, description="Product type")
    product_size: Optional[str] = Field(None, description="Product size")
    product_material: Optional[str] = Field(None, description="Product material")
    product_style: Optional[str] = Field(None, description="Product style")
    product_colour: Optional[str] = Field(None, description="Product colour")
    product_cuisine: Optional[str] = Field(None, description="Food cuisine")
    product_venue: Optional[str] = Field(None, description="Event venue")
    product_salary: Optional[str] = Field(None, description="Job salary")
    product_education: Optional[str] = Field(None, description="Job education required")
    product_location: Optional[str] = Field(None, description="Job location")
    product_skills: Optional[str] = Field(None, description="Job skills required")
    product_duration: Optional[int] = Field(None, description="Course duration")
    product_deposit: Optional[float] = Field(None, description="Accommodation deposit")
    product_rooms: Optional[int] = Field(None, description="Accommodation room number")

# class ShoppingProduct(Product):
#     product_size: Optional[str] = Field(None, description="Product size")
#     product_material: Optional[str] = Field(None, description="Product material")
#     product_style: Optional[str] = Field(None, description="Product style")
#     product_colour: Optional[str] = Field(None, description="Product colour")

# class FoodProduct(Product):
#     product_cuisine: Optional[str] = Field(None, description="Food cuisine")
#     product_style: Optional[str] = Field(None, description="Food style")

# class EventProduct(Product):
#     product_venue: Optional[str] = Field(None, description="Event venue")

# class JobProduct(Product):
#     product_salary: Optional[str] = Field(None, description="Job salary")
#     product_education: Optional[str] = Field(None, description="Job education required")
#     product_location: Optional[str] = Field(None, description="Job location")
#     product_skills: Optional[str] = Field(None, description="Job skills required")

# class DiscountCardProduct(Product):
#     product_category: Optional[str] = Field(None, description="Discount Card category")

# class EducationProduct(Product):
#     product_duration: Optional[int] = Field(None, description="Course duration")
#     product_skills: Optional[str] = Field(None, description="Course skills acquired")
#     product_location: Optional[str] = Field(None, description="Course location")

# class AccommodationProduct(Product):
#     product_location: Optional[str] = Field(None, description="Accommodation location")
#     product_deposit: Optional[float] = Field(None, description="Accommodation deposit")
#     product_rooms: Optional[int] = Field(None, description="Accommodation room number")
#     product_style: Optional[str] = Field(None, description="Accommodation style")

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

        system_instruction = system_instruction or (
            "You are a customer support specialist. Answer customer questions based ONLY on the information in the product catalogue and the tags catalogue."
            " If you are unable to extract any information, return 'unknown', but try your best to find the closest category / tags."
            " Always be polite, concise, and helpful."
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
        # print(f"Answering question: '{question}'")
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
            # print(f"gemini_response: {gemini_json_response}")

            candidates_token_count = response.usage_metadata.candidates_token_count
            prompt_token_count = response.usage_metadata.prompt_token_count
            total_token_count = response.usage_metadata.total_token_count

            return gemini_json_response, prompt_token_count, candidates_token_count, total_token_count

        except Exception as e:
            print(f"Error answering question: {e}")
            return None, None

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

def save_product_to_csv(product, product_id: int, product_title: str, product_brand: str,
                        product_offer_price: float, product_offer_percentage: float, product_rating: float,
                        product_num_interactions: int, product_postcode: str, product_lat: float,
                        product_long: float, product_geohash: str, product_link: str,
                        product_image_link: str, product_forced: bool, product_spider_name: str,
                        product_entity_type: str, input_token, output_token, total_token):
    file_exists = os.path.isfile("output/products.csv")

    with open("output/products.csv", mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow(["product_id", "product_title", "product_brand", "product_offer_price",
                             "product_offer_percentage", "product_rating", "product_num_interactions",
                             "product_postcode", "product_lat", "product_long", "product_geohash",
                             "product_link", "product_image_link", "product_forced", "product_spider_name",
                             "product_entity_type", "product_description", "product_search_keywords",
                             "product_categories", "product_tags", "product_type", "product_size",
                             "product_material", "product_colour", "product_style", "product_cuisine",
                             "product_venue", "product_salary", "product_education", "product_location",
                             "product_skills", "product_category", "product_duration", "product_deposit",
                             "product_rooms", "input_token", "output_token", "total_token"])

        def get_attribute(product: Product, attr_name: str, default=None):
            """Get attribute from product safely, returning default if not found."""
            return getattr(product, attr_name, default)

        product_data = [
            product_id, product_title, product_brand, product_offer_price,
            product_offer_percentage, product_rating, product_num_interactions,
            product_postcode, product_lat, product_long, product_geohash,
            product_link, product_image_link, product_forced, product_spider_name,
            product_entity_type,
            get_attribute(product, 'product_description'),
            '; '.join(product.product_search_keywords or []),
            '; '.join(product.product_categories or []),
            '; '.join(product.product_tags or []),
            get_attribute(product, 'product_type'),
            get_attribute(product, 'product_size'),
            get_attribute(product, 'product_material'),
            get_attribute(product, 'product_colour'),
            get_attribute(product, 'product_style'),
            get_attribute(product, 'product_cuisine'),
            get_attribute(product, 'product_venue'),
            get_attribute(product, 'product_salary'),
            get_attribute(product, 'product_education'),
            get_attribute(product, 'product_location'),
            get_attribute(product, 'product_skills'),
            get_attribute(product, 'product_category'),
            get_attribute(product, 'product_duration'),
            get_attribute(product, 'product_deposit'),
            get_attribute(product, 'product_rooms'),
            input_token, output_token, total_token
        ]

        writer.writerow(product_data)
    print(f"Product {product_id} saved to CSV.")

def create_question(product, image=None):
    base_question = (
        f"Summarize the product using the description and image. "
        "If you are unable to write the description, then return the original description. If you are not able to write the attributes or keywords, write 'unknown'. "
        "Return: "
        "1. Product Description - concise but not too short. "
        "2. Product Search Keywords - for search purposes (return in a list separated by '; '). "
        "3. Product Categories - choose the best and closest from the category catalogue. "
        "4. Product Tags - in what scenario are these products most suited for? For example, time of day, weather, what point of interest is it related to? Add as many as you think is suitable from the tags catalogue (return these tags in a list separated by '; '). "
        "5. Product Attributes - Return type of product. "
    )

    attributes_mapping = {
        "shopping": "From the description and image, 6. Product Attributes - Return size of product. 7. Product Attributes - Return material of product. 8. Product Attributes - Return style of product. 9. Product Attributes - Return colour of product. Return None only if these attributes cannot be determined. Otherwise, they are a requirement.",

        "food": "From the description and image, 6. Product Attributes - Return cuisine of product. 7. Product Attributes - Return style of product. Return None only if these attributes cannot be determined. Otherwise, they are a requirement.",

        "event": "From the description and image, 6. Product Attributes - Return venue of product. Return None only if these attributes cannot be determined. Otherwise, they are a requirement.",

        "job": "From the description and image, 6. Product Attributes - Return salary of product. 7. Product Attributes - Return education of product. 8. Product Attributes - Return location of product. 9. Product Attributes - Return skills of product. Return None only if these attributes cannot be determined. Otherwise, they are a requirement.",

        "discount card": "From the description and image, 6. Product Attributes - Return category of product. Return None only if these attributes cannot be determined. Otherwise, they are a requirement.",

        "education": "From the description and image, 6. Product Attributes - Return duration of product. 7. Product Attributes - Return location of product. 8. Product Attributes - Return duration of product. 9. Product Attributes - Return skills of product. Return None only if these attributes cannot be determined. Otherwise, they are a requirement.",

        "accommodation": "From the description and image, 6. Product Attributes - Return location of product. 7. Product Attributes - Return deposit of product. 8. Product Attributes - Return rooms of product. 9. Product Attributes - Return style of product. Return None only if these attributes cannot be determined. Otherwise, they are a requirement.",

        "services": "From the description and image, 6. Product Attributes - Return location of product. 7. Product Attributes - Return style of product. Return None only if these attributes cannot be determined. Otherwise, they are a requirement.",

        "travel": "From the description and image, 6. Product Attributes - Return location of product. 7. Product Attributes - Return style of product. Return None only if these attributes cannot be determined. Otherwise, they are a requirement.",

        "recommendations": "From the description and image, 6. Product Attributes - Return size of product. 7. Product Attributes - Return material of product. 8. Product Attributes - Return style of product. 9. Product Attributes - Return colour of product. 10. Product Attributes - Return cuisine of product. 11. Product Attributes - Return venue of product. 12. Product Attributes - Return salary of product. 13. Product Attributes - Return education of product. 14. Product Attributes - Return location of product. 15. Product Attributes - Return skills of product. 16. Product Attributes - Return duration of product. 17. Product Attributes - Return category of product. 18. Product Attributes - Return deposit of product. 19. Product Attributes - Return rooms of product. Return None only if these attributes cannot be determined. Otherwise, they are a requirement."
    }

    entity_type = product.product_entity_type
    if entity_type in attributes_mapping:
        base_question += attributes_mapping[entity_type]

    base_question += f"Description: {product.product_description}. "

    if image:
        return base_question.replace("Description:", f"Image: {image}, Description:")
    
    return base_question

if __name__ == "__main__":
    df = pd.read_parquet("data/test_df.parquet")
    batch_size = 1024

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
                product_forced = products['product_forced']
                product_spider_name = products['product_spider_name']
                product_entity_type = products['product_entity_type']

                # Fetch and process the product image
                response = requests.get(product_image_link)
                image = None
                if response.status_code == 200:
                    try:
                        image = Image.open(BytesIO(response.content)).resize((640, 480))
                    except Exception as e:
                        print(f"Error opening image: {e}. Proceeding with description only.")

                # if product_entity_type == "shopping":
                #     product = ShoppingProduct(product_description=product_description)
                # elif product_entity_type == "food":
                #     product = FoodProduct(product_description=product_description)
                # elif product_entity_type == "event":
                #     product = EventProduct(product_description=product_description)
                # elif product_entity_type == "job":
                #     product = JobProduct(product_description=product_description)
                # elif product_entity_type == "discount card":
                #     product = DiscountCardProduct(product_description=product_description)
                # elif product_entity_type == "education":
                #     product = EducationProduct(product_description=product_description)
                # elif product_entity_type == "accommodation":
                #     product = AccommodationProduct(product_description=product_description)

                product = Product(product_description=product_description, product_entity_type=product_entity_type)
                question = create_question(product, image=image)

                # Get the response from the knowledge base (AI model)
                product_response, input_token, output_token, total_token = support_bot.answer_question(question)

                if product_response:
                    # Save the product response to CSV
                    save_product_to_csv(
                        product_response, product_id, product_title, product_brand, product_offer_price, 
                        product_offer_percentage, product_rating, product_num_interactions, product_postcode, 
                        product_lat, product_long, product_geohash, product_link, product_image_link, 
                        product_forced, product_spider_name, product_entity_type, input_token, output_token, total_token
                    )

            print(f"Results after batch starting at index {start_index} saved.")

        support_bot.cleanup()
        print("Cleanup completed.")