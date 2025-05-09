from flask import Flask, request, jsonify
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict
import heapq
import json
import requests
import logging
from flask_cors import CORS
from dotenv import load_dotenv
import os


load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)


BACKEND_BASE_URL = os.getenv('BACKEND_BASE_URL', 'https://6041-2405-201-5004-7027-35e5-eaef-49e2-ccf7.ngrok-free.app')
FILTERED_VARIANTS_ENDPOINT = os.getenv('FILTERED_VARIANTS_ENDPOINT', '/api/v1/traceVenue/variant/filteredVariants')
USER_REQUIREMENTS_BASE_ENDPOINT = os.getenv('USER_REQUIREMENTS_BASE_ENDPOINT', '/api/v1/traceVenue/jobs')
PORT = int(os.getenv('PORT', 5001))
DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

# Set logging level based on environment variable
logging_level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
logger.setLevel(logging_level)

logger.info(f"Starting application with backend URL: {BACKEND_BASE_URL}")
logger.info(f"Filtered variants endpoint: {FILTERED_VARIANTS_ENDPOINT}")
logger.info(f"User requirements base endpoint: {USER_REQUIREMENTS_BASE_ENDPOINT}")



class MenuItem:
    def __init__(self, item_type: str, count: int):
        self.item_type = item_type
        self.count = count

class Subcategory:
    def __init__(self, name: str, items: Dict[str, int]):
        self.name = name
        self.items = items

class Cuisine:
    def __init__(self, name: str, subcategories: Dict[str, Subcategory]):
        self.name = name
        self.subcategories = subcategories
        self.contains_egg = False

class Category:
    def __init__(self, name: str, cuisines: Dict[str, Cuisine]):
        self.name = name
        self.cuisines = cuisines

class UserRequirement:
    def __init__(self, categories: Dict[str, Category]):
        self.categories = categories

class RestaurantPackage:
    def __init__(self, id: str, name: str, categories: Dict[str, Category], price: float, rating: float, package_id: str = "", venue_id: str = ""):
        self.id = id
        self.name = name
        self.categories = categories
        self.price = price
        self.rating = rating
        self.package_id = package_id
        self.venue_id = venue_id 
        

class MatchResult:
    def __init__(self, restaurant_id: str, restaurant_name: str, 
                 overall_match: float, category_matches: Dict[str, float],
                 unmet_requirements: List[Dict[str, Any]], price: float, rating: float, 
                 package_id: str = "", venue_id: str = ""):
        self.restaurant_id = restaurant_id  
        self.restaurant_name = restaurant_name
        self.overall_match = overall_match
        self.category_matches = category_matches
        self.unmet_requirements = unmet_requirements
        self.price = price
        self.rating = rating
        self.package_id = package_id
        self.venue_id = venue_id  # Added venue_id
    
    def to_dict(self):
        """Convert MatchResult to dictionary for JSON serialization"""
        return {
            "variant_id": self.restaurant_id,  
            "variant_name": self.restaurant_name,
            "overall_match": self.overall_match,
            "match_percentage": round(self.overall_match * 100, 2), 
            "category_matches": {k: round(v * 100, 2) for k, v in self.category_matches.items()}, 
            "unmet_requirements": self.unmet_requirements,
            "price": self.price,
            "rating": self.rating,
            "package_id": self.package_id,
            "venue_id": self.venue_id  # Include venue_id in the dictionary
        }

class OptimizedRestaurantMatcher:
    def __init__(self, threshold: float = 0.7, top_n: int = 10):
        self.threshold = threshold
        self.top_n = top_n
    
    def flatten_requirements(self, user_req):
        """Convert hierarchical user requirements to a flat dictionary"""
        flat_req = {}
        total_items = 0
        
        for cat_name, category in user_req.categories.items():
            for cuisine_name, cuisine in category.cuisines.items():
                for subcat_name, subcategory in cuisine.subcategories.items():
                    for item_type, count in subcategory.items.items():
                        if count > 0: 
                            key = f"{cat_name}|{cuisine_name}|{subcat_name}|{item_type}"
                            flat_req[key] = count
                            total_items += count
        
        return flat_req, total_items
    
    def flatten_restaurant(self, restaurant):
        """Convert hierarchical restaurant offerings to a flat dictionary"""
        flat_offerings = {}
        
        for cat_name, category in restaurant.categories.items():
            for cuisine_name, cuisine in category.cuisines.items():
                for subcat_name, subcategory in cuisine.subcategories.items():
                    for item_type, count in subcategory.items.items():
                        if count > 0: 
                            key = f"{cat_name}|{cuisine_name}|{subcat_name}|{item_type}"
                            flat_offerings[key] = count
        
        return flat_offerings
    
    def calculate_match(self, flat_user_req, flat_restaurant, total_user_items):
        """Calculate match score using flattened representations"""
        matched_items = 0
        category_matches = defaultdict(float)
        category_totals = defaultdict(int)
        unmet_requirements = []
        
        for key, user_count in flat_user_req.items():
            rest_count = flat_restaurant.get(key, 0)
            
            
            cat_name, cuisine_name, subcat_name, item_type = key.split('|')
            
            if rest_count >= user_count:
                matched_items += user_count
                item_match = 1.0
                
            else:
                matched_items += rest_count
                item_match = rest_count / user_count
                
                if rest_count < user_count:
                    unmet_requirements.append({
                        "level": "item",
                        "category": cat_name,
                        "cuisine": cuisine_name,
                        "subcategory": subcat_name,
                        "item_type": item_type,
                        "requested": user_count,
                        "available": rest_count,
                        "shortfall": user_count - rest_count,
                        "message": f"Insufficient {item_type} in {subcat_name}: need {user_count}, have {rest_count}"
                    })
            
            category_key = cat_name
            category_totals[category_key] += user_count
            category_matches[category_key] += item_match * user_count
            
            cuisine_key = f"{cat_name}|{cuisine_name}"
            category_totals[cuisine_key] += user_count
            category_matches[cuisine_key] += item_match * user_count
            
            subcat_key = f"{cat_name}|{cuisine_name}|{subcat_name}"
            category_totals[subcat_key] += user_count
            category_matches[subcat_key] += item_match * user_count
        
        for key in category_matches:
            if category_totals[key] > 0:
                category_matches[key] = category_matches[key] / category_totals[key]
        
        overall_match = matched_items / total_user_items if total_user_items > 0 else 0
        
        
        return overall_match, dict(category_matches), unmet_requirements
    
    def score_restaurants(self, user_req, restaurant_packages):
            """Score and rank restaurant packages against user requirements"""
            flat_user_req, total_user_items = self.flatten_requirements(user_req)
    
            all_matches = []
            
            for restaurant in restaurant_packages:
                flat_restaurant = self.flatten_restaurant(restaurant)
                
                overall_match, category_matches, unmet_requirements = self.calculate_match(
                    flat_user_req, flat_restaurant, total_user_items
                )
                
                # Get venue_id from the restaurant object
                venue_id = getattr(restaurant, 'venue_id', "")
                
                match_result = MatchResult(
                    restaurant_id=restaurant.id,
                    restaurant_name=restaurant.name,
                    overall_match=overall_match,
                    category_matches=category_matches,
                    unmet_requirements=unmet_requirements,
                    price=restaurant.price,
                    rating=restaurant.rating,
                    package_id=restaurant.package_id,
                    venue_id=venue_id  # Pass venue_id to the match result
                )
                
                all_matches.append((overall_match, restaurant.id, match_result))
            
            results = []
            sorted_matches = sorted(all_matches, key=lambda x: x[0], reverse=True)
            for _, _, match_result in sorted_matches:
                results.append(match_result)
            
            return results

def parse_user_requirements(user_requirements_data, count_field="count"):
    """Convert API JSON data to UserRequirement object
    Handles both menuSections format and availableMenuCount/count format
    
    Args:
        user_requirements_data: User requirements data in the format provided in the example
        count_field: Field name to look for counts ("count" for user requirements, 
                    "availableMenuCount" for restaurant variants)
        
    Returns:
        UserRequirement object that can be used by the matcher
    """
    logger.info(f"Parsing user requirements with count field: {count_field}")
    categories = {}

    
    if isinstance(user_requirements_data, list) and len(user_requirements_data) > 0:
        
        for section in user_requirements_data:
            if not isinstance(section, dict):
                continue
                
            cat_name = section.get("name", "Uncategorized")
            cuisines = {}
            
            subcategories_by_cuisine = section.get("subcategoriesByCuisine", {})
            for cuisine_name, subcategory_list in subcategories_by_cuisine.items():
                cuisine_subcategories = {}
                
                for subcategory_data in subcategory_list:
                    subcat_name = subcategory_data.get("name", "General")
                    
                    item_counts_data = subcategory_data.get(count_field, {})
                    items = {}
                    
                    if isinstance(item_counts_data, dict):
                        for item_type, count in item_counts_data.items():
                            if count > 0:
                                items[item_type] = count
                    elif isinstance(item_counts_data, list):
                        for item_data in item_counts_data:
                            if isinstance(item_data, dict):
                                item_type = item_data.get("name", "Unknown")
                                count = item_data.get("count", 0)
                                if count > 0:
                                    items[item_type] = count
                    
                    if items:
                        cuisine_subcategories[subcat_name] = Subcategory(subcat_name, items)
                
                if cuisine_subcategories:
                    cuisine = Cuisine(cuisine_name, cuisine_subcategories)
                    if isinstance(item_counts_data, dict):
                        cuisine.contains_egg = "Egg" in item_counts_data and item_counts_data.get("Egg", 0) > 0
                    elif isinstance(item_counts_data, list):
                        cuisine.contains_egg = any(
                            item.get("name") == "Egg" and item.get("count", 0) > 0
                            for item in item_counts_data if isinstance(item, dict)
                        )
                    cuisines[cuisine_name] = cuisine
            
            if cuisines:
                categories[cat_name] = Category(cat_name, cuisines)
    
    elif isinstance(user_requirements_data, dict):
        if count_field in user_requirements_data:
            count_data = user_requirements_data.get(count_field, [])
            
            if isinstance(count_data, list):
                
                for category_count in count_data:
                    if not isinstance(category_count, dict):
                        continue
                        
                    cat_name = category_count.get("name", "Uncategorized")
                    counts_data = category_count.get(count_field, {})
                    
                    cuisine_name = "General"
                    subcat_name = "General"
                    
                    items = {}
                    if isinstance(counts_data, dict):
                        for item_type, count in counts_data.items():
                            if count > 0:
                                items[item_type] = count
                    elif isinstance(counts_data, list):
                        for item_data in counts_data:
                            if isinstance(item_data, dict):
                                item_type = item_data.get("name", "Unknown")
                                count = item_data.get("count", 0)
                                if count > 0:
                                    items[item_type] = count
                    
                    if items:
                        if cat_name not in categories:
                            categories[cat_name] = Category(cat_name, {})
                        
                        cuisine_subcategories = {subcat_name: Subcategory(subcat_name, items)}
                        cuisine = Cuisine(cuisine_name, cuisine_subcategories)
                        
                        if isinstance(counts_data, dict):
                            cuisine.contains_egg = "Egg" in counts_data and counts_data.get("Egg", 0) > 0
                        elif isinstance(counts_data, list):
                            cuisine.contains_egg = any(
                                item.get("name") == "Egg" and item.get("count", 0) > 0
                                for item in counts_data if isinstance(item, dict)
                            )
                        
                        categories[cat_name].cuisines[cuisine_name] = cuisine
            
            elif isinstance(count_data, dict):
                cat_name = "Menu Items"
                cuisine_name = "General"
                subcat_name = "General"
                
                items = {}
                for item_type, count in count_data.items():
                    if count > 0:
                        items[item_type] = count
                
                if items:
                    if cat_name not in categories:
                        categories[cat_name] = Category(cat_name, {})
                    
                    cuisine_subcategories = {subcat_name: Subcategory(subcat_name, items)}
                    cuisine = Cuisine(cuisine_name, cuisine_subcategories)
                    cuisine.contains_egg = "Egg" in count_data and count_data.get("Egg", 0) > 0
                    
                    categories[cat_name].cuisines[cuisine_name] = cuisine
        
        elif "data" in user_requirements_data:
            if "menuSections" in user_requirements_data["data"]:
                return parse_user_requirements(user_requirements_data["data"]["menuSections"], count_field)
            elif count_field in user_requirements_data["data"]:
                return parse_user_requirements({"count_field": user_requirements_data["data"][count_field]}, count_field)
            for key, value in user_requirements_data["data"].items():
                if isinstance(value, list) and len(value) > 0:
                    result = parse_user_requirements(value, count_field)
                    if result.categories:
                        return result
                elif isinstance(value, dict):
                    result = parse_user_requirements(value, count_field)
                    if result.categories:
                        return result
        
        else:
            for key, value in user_requirements_data.items():
                if key == count_field:
                    return parse_user_requirements({count_field: value}, count_field)
                elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict) and "name" in value[0]:
                    return parse_user_requirements(value, count_field)
                elif isinstance(value, dict) and count_field in value:
                    return parse_user_requirements({count_field: value[count_field]}, count_field)
                elif isinstance(value, dict):
                    result = parse_user_requirements(value, count_field)
                    if result.categories:
                        return result
    
    return UserRequirement(categories)
def get_cuisine_name_by_id(cuisine_id):
    """
    Maps cuisine IDs to cuisine names
    In a real implementation, this could fetch the cuisine name from a database or cache
    
    Args:
        cuisine_id: Cuisine ID from the API
    
    Returns:
        Cuisine name
    """

    cuisine_mapping = {
    "67ac7d222ee4b070bd485694": "Indian",
    "67ac7d292ee4b070bd485696": "Italian",
    "67ad9f22af8c34b3272d8cb2": "Continental",
    "67aeda63d3af1523c01ef462": "Chinese",
    "67dba24c024b2035f1d88ec0": "North Indian / Punjabi",
    "67dba279024b2035f1d88ec9": "Mughlai / North Indian",
    "67dba33c024b2035f1d88f32": "Maharashtrian / South India",
    "67dba444024b2035f1d88f95": "Asian / Chinese",
    "67dba51f024b2035f1d89005": "Japanese / Fusion",
    "67dba648024b2035f1d89134": "Continental / European",
    "67e6853b4fc3da47168b4845": "American"
    }

    return cuisine_mapping.get(cuisine_id, "Other")

def api_to_user_requirements(api_response, is_user_requirement=True):
    """
    Convert the API response to the format expected by the matcher
    Handles different possible structures of the API response
    
    Args:
        api_response: Full API response 
        is_user_requirement: Boolean indicating if this is user requirement data (True) 
                            or restaurant variant data (False)
        
    Returns:
        UserRequirement object
    """
    logger.info(f"Processing {'user requirements' if is_user_requirement else 'restaurant variants'}")
    
    count_field = "count" if is_user_requirement else "availableMenuCount"
    logger.info(f"Looking for '{count_field}' field in data structure")
    
    if not isinstance(api_response, dict):
        logger.warning(f"API response is not a dictionary, but a {type(api_response)}")
        return UserRequirement({})
    
    if "menuSections" in api_response:
        return parse_user_requirements(api_response.get("menuSections", []), count_field)
    
    elif "data" in api_response and isinstance(api_response["data"], dict) and "menuSections" in api_response.get("data", {}):
        return parse_user_requirements(api_response.get("data", {}).get("menuSections", []), count_field)
    
    elif count_field in api_response:
        return parse_user_requirements({count_field: api_response.get(count_field, [])}, count_field)
    
    elif "data" in api_response and isinstance(api_response["data"], dict) and count_field in api_response.get("data", {}):
        return parse_user_requirements({count_field: api_response.get("data", {}).get(count_field, [])}, count_field)
    
    elif "variants" in api_response and len(api_response.get("variants", [])) > 0:
        first_variant = api_response.get("variants", [])[0]
        if isinstance(first_variant, dict) and count_field in first_variant:
            return parse_user_requirements({count_field: first_variant.get(count_field, [])}, count_field)
    
    logger.warning(f"Could not find menuSections or {count_field} in API response")
    return UserRequirement({})

def adapt_restaurant_data_updated(api_response):
    """
    Adapts the API response format to match what the restaurant matcher expects
    Handles both dictionary and list formats for availableMenuCount
    Now also extracts and stores venueId
    
    Debug version with more logging
    """
    adapted_data = []
    
    if not isinstance(api_response, dict):
        logger.warning("API response is not a dictionary")
        return adapted_data
    
    logger.info(f"Processing {len(api_response.get('variants', []))} variants")
    
    for variant in api_response.get('variants', []):
        if not isinstance(variant, dict):
            logger.warning(f"Skipping variant because it's not a dictionary: {type(variant)}")
            continue
        variant_data = variant  
        if hasattr(variant, '_doc') and isinstance(variant._doc, dict):
            variant_data = variant._doc
        elif "$__" in variant and "_doc" in variant:
            variant_data = variant.get("_doc", variant)
        
        venue_id = variant_data.get("venueId", "")
        logger.info(f"Extracted venue_id: {venue_id}")
        restaurant = {
            "id": variant_data.get("_id", ""),
            "name": variant_data.get("name", ""),
            "price": float(variant_data.get("cost", 0.0)),
            "rating": 0.0,  
            "package_id": variant_data.get("packageId", ""),
            "venue_id": venue_id,  
            "categories": {}
        }
        
        logger.debug(f"Processing variant: {variant.get('name', 'Unknown')} (ID: {variant.get('_id', 'Unknown')})")
        
        found_menu_items = False
        
        if "availableMenuCount" in variant:
            logger.debug(f"availableMenuCount type: {type(variant.get('availableMenuCount'))}")
            
            menu_sections = variant.get("availableMenuCount", [])
            
            if isinstance(menu_sections, dict):
                menu_sections = [{"name": "Menu Items", "availableMenuCount": menu_sections}]
            
            for menu_section in menu_sections:
                if isinstance(menu_section, dict):
                    cat_name = menu_section.get("name", "Uncategorized")
                    
                    if cat_name not in restaurant["categories"]:
                        restaurant["categories"][cat_name] = {"cuisines": {}}
                    
                    # Check for subcategoriesByCuisine
                    if "subcategoriesByCuisine" in menu_section:
                        for cuisine_name, subcategory_list in menu_section.get("subcategoriesByCuisine", {}).items():
                            # Initialize cuisine if needed
                            if cuisine_name not in restaurant["categories"][cat_name]["cuisines"]:
                                restaurant["categories"][cat_name]["cuisines"][cuisine_name] = {
                                    "subcategories": {},
                                    "contains_egg": False
                                }
                            
                            # Process each subcategory
                            for subcategory_data in subcategory_list:
                                subcat_name = subcategory_data.get("name", "General")
                                
                                # Initialize subcategory if needed
                                if subcat_name not in restaurant["categories"][cat_name]["cuisines"][cuisine_name]["subcategories"]:
                                    restaurant["categories"][cat_name]["cuisines"][cuisine_name]["subcategories"][subcat_name] = {
                                        "items": {}
                                    }
                                
                                # Get count data - handle both formats
                                count_data = subcategory_data.get("availableMenuCount", subcategory_data.get("count", {}))
                                
                                # Process counts
                                if isinstance(count_data, dict):
                                    for item_type, count in count_data.items():
                                        if count > 0:
                                            restaurant["categories"][cat_name]["cuisines"][cuisine_name]["subcategories"][subcat_name]["items"][item_type] = count
                                            found_menu_items = True
                                            
                                            # Check for eggs
                                            if item_type == "Egg" and count > 0:
                                                restaurant["categories"][cat_name]["cuisines"][cuisine_name]["contains_egg"] = True
                                elif isinstance(count_data, list):
                                    # Handle list format
                                    for item in count_data:
                                        if isinstance(item, dict):
                                            item_type = item.get("name", "Unknown")
                                            count = item.get("count", 0)
                                            if count > 0:
                                                restaurant["categories"][cat_name]["cuisines"][cuisine_name]["subcategories"][subcat_name]["items"][item_type] = count
                                                found_menu_items = True
                                                
                                                # Check for eggs
                                                if item_type == "Egg" and count > 0:
                                                    restaurant["categories"][cat_name]["cuisines"][cuisine_name]["contains_egg"] = True
                    # Check for direct count field (simplified format)
                    elif "availableMenuCount" in menu_section or "count" in menu_section:
                        count_data = menu_section.get("availableMenuCount", menu_section.get("count", {}))
                        cuisine_name = "General"
                        subcat_name = "General"
                        
                        # Initialize necessary structures
                        if cuisine_name not in restaurant["categories"][cat_name]["cuisines"]:
                            restaurant["categories"][cat_name]["cuisines"][cuisine_name] = {
                                "subcategories": {},
                                "contains_egg": False
                            }
                            
                        if subcat_name not in restaurant["categories"][cat_name]["cuisines"][cuisine_name]["subcategories"]:
                            restaurant["categories"][cat_name]["cuisines"][cuisine_name]["subcategories"][subcat_name] = {
                                "items": {}
                            }
                        
                        # Process counts
                        if isinstance(count_data, dict):
                            for item_type, count in count_data.items():
                                if count > 0:
                                    restaurant["categories"][cat_name]["cuisines"][cuisine_name]["subcategories"][subcat_name]["items"][item_type] = count
                                    found_menu_items = True
                                    
                                    # Check for eggs
                                    if item_type == "Egg" and count > 0:
                                        restaurant["categories"][cat_name]["cuisines"][cuisine_name]["contains_egg"] = True
                        elif isinstance(count_data, list):
                            # Handle list format
                            for item in count_data:
                                if isinstance(item, dict):
                                    item_type = item.get("name", "Unknown")
                                    count = item.get("count", 0)
                                    if count > 0:
                                        restaurant["categories"][cat_name]["cuisines"][cuisine_name]["subcategories"][subcat_name]["items"][item_type] = count
                                        found_menu_items = True
                                        
                                        # Check for eggs
                                        if item_type == "Egg" and count > 0:
                                            restaurant["categories"][cat_name]["cuisines"][cuisine_name]["contains_egg"] = True
        
        # If we found menu items, add the restaurant to our results
        if found_menu_items:
            logger.debug(f"Added restaurant: {restaurant['name']} with menu items")
            adapted_data.append(restaurant)
        else:
            logger.warning(f"No menu items found for restaurant: {restaurant['name']}")
            
            # As a fallback, if the restaurant has no menu items but has a name and ID,
            # add it with a default menu item structure
            if restaurant["name"] and restaurant["id"]:
                cat_name = "Menu Items"
                cuisine_name = "General"
                subcat_name = "General"
                
                if cat_name not in restaurant["categories"]:
                    restaurant["categories"][cat_name] = {"cuisines": {}}
                
                if cuisine_name not in restaurant["categories"][cat_name]["cuisines"]:
                    restaurant["categories"][cat_name]["cuisines"][cuisine_name] = {
                        "subcategories": {},
                        "contains_egg": False
                    }
                
                if subcat_name not in restaurant["categories"][cat_name]["cuisines"][cuisine_name]["subcategories"]:
                    restaurant["categories"][cat_name]["cuisines"][cuisine_name]["subcategories"][subcat_name] = {
                        "items": {"Default": 1}
                    }
                
                logger.debug(f"Added restaurant: {restaurant['name']} with default menu item")
                adapted_data.append(restaurant)
    
    logger.info(f"Successfully adapted {len(adapted_data)} restaurants out of {len(api_response.get('variants', []))} variants")
    return adapted_data

def _add_menu_item_to_restaurant(restaurant, item_type, count):
    """Helper function to add a menu item to a restaurant structure"""
    cat_name = "Menu Items"  
    cuisine_name = "General"  
    subcat_name = "General"   
    
    if cat_name not in restaurant["categories"]:
        restaurant["categories"][cat_name] = {"cuisines": {}}
    
    if cuisine_name not in restaurant["categories"][cat_name]["cuisines"]:
        restaurant["categories"][cat_name]["cuisines"][cuisine_name] = {
            "subcategories": {},
            "contains_egg": False
        }
    
    if subcat_name not in restaurant["categories"][cat_name]["cuisines"][cuisine_name]["subcategories"]:
        restaurant["categories"][cat_name]["cuisines"][cuisine_name]["subcategories"][subcat_name] = {
            "items": {}
        }
    
    restaurant["categories"][cat_name]["cuisines"][cuisine_name]["subcategories"][subcat_name]["items"][item_type] = count
    
    if item_type == "Egg" and count > 0:
        restaurant["categories"][cat_name]["cuisines"][cuisine_name]["contains_egg"] = True

def parse_restaurant_packages(restaurant_data):
    """Convert API JSON data to RestaurantPackage objects"""
    restaurant_packages = []
    
    for idx, rest_data in enumerate(restaurant_data):
        if not rest_data.get("categories"):
            logger.warning(f"Restaurant {rest_data.get('name', f'#{idx}')} has no categories")
            
        categories = {}
        
        for cat_name, cat_data in rest_data.get("categories", {}).items():
            cuisines = {}
            
            for cuisine_name, cuisine_data in cat_data.get("cuisines", {}).items():
                subcategories = {}
                
                for subcat_name, subcat_data in cuisine_data.get("subcategories", {}).items():
                    items = subcat_data.get("items", {})
                    
                    if not items:
                        logger.debug(f"Empty items in {rest_data.get('name')}/{cat_name}/{cuisine_name}/{subcat_name}")
                    
                    subcategories[subcat_name] = Subcategory(subcat_name, items)
                
                cuisine = Cuisine(cuisine_name, subcategories)
                cuisine.contains_egg = cuisine_data.get("contains_egg", False)
                cuisines[cuisine_name] = cuisine
            
            categories[cat_name] = Category(cat_name, cuisines)
        
        restaurant = RestaurantPackage(
            id=rest_data.get("id", ""),
            name=rest_data.get("name", ""),
            categories=categories,
            price=float(rest_data.get("price", 0.0)),
            rating=float(rest_data.get("rating", 0.0)),
            package_id=rest_data.get("package_id", "")
        )
        
        venue_id = rest_data.get("venue_id", "")
        setattr(restaurant, 'venue_id', venue_id)
        logger.info(f"Setting venue_id for {restaurant.name}: {venue_id}")
        
        restaurant_packages.append(restaurant)
    
    return restaurant_packages

def fetch_filtered_variants(filter_data):
    """
    Fetch filtered restaurant variants from the backend API
    Include maxPerson parameter if provided
    """
    try:
        url = f"{BACKEND_BASE_URL}{FILTERED_VARIANTS_ENDPOINT}"
        
        if 'maxPerson' in filter_data:
            logger.info(f"Fetching filtered variants with maxPerson: {filter_data['maxPerson']}")
        else:
            logger.info(f"Fetching filtered variants without maxPerson parameter")
        
        logger.info(f"Fetching filtered variants with data: {filter_data}")
        response = requests.post(url, json=filter_data)
        
        if response.status_code != 200:
            logger.error(f"Failed to fetch filtered variants: {response.text}")
            raise Exception(f"Failed to fetch filtered variants: {response.status_code} - {response.text}")
        
        data = response.json()
        logger.info(f"Successfully fetched {len(data)} filtered variants")
        
        if 'variants' in data and len(data['variants']) > 0:
            first_variant = data['variants'][0]
            logger.info(f"First variant structure: {json.dumps({k: type(v).__name__ for k, v in first_variant.items()})}")
             
        return data
        
    except Exception as e:
        logger.error(f"Error fetching filtered variants: {str(e)}")
        raise    

def debug_restaurant_data(restaurant_packages):
    """Helper function to debug restaurant data structure"""
    for idx, restaurant in enumerate(restaurant_packages):
        
        if not restaurant.categories:
            continue
            
        for cat_name, category in restaurant.categories.items():
            
            for cuisine_name, cuisine in category.cuisines.items():
                
                for subcat_name, subcategory in cuisine.subcategories.items():
                    
                    item_list = list(subcategory.items.items())
                    sample = item_list[:5]



def fetch_user_requirements(job_id):
    """
    Fetch user requirements/customizations from the backend API
    Token is no longer required
    
    Args:
        job_id: ID of the job to fetch requirements for
        
    Returns:
        User requirement data
    """
    try:
        url = f"{BACKEND_BASE_URL}{USER_REQUIREMENTS_BASE_ENDPOINT}/{job_id}"
        logger.info(f"Fetching user requirements for job ID: {job_id}")
        
        # No headers with token needed anymore
        response = requests.get(url)
        
        if response.status_code != 200:
            logger.error(f"Failed to fetch user requirements: {response.text}")
            raise Exception(f"Failed to fetch user requirements: {response.status_code} - {response.text}")
        
        data = response.json()
        logger.info(f"Successfully fetched user requirements")
        return data
        
    except Exception as e:
        logger.error(f"Error fetching user requirements: {str(e)}")
        raise

@app.route('/api/match-restaurants-integrated', methods=['POST'])
def match_restaurants_integrated():
    """
    Integrated API endpoint that fetches data from backend APIs 
    and performs matching with updated parser functions
    Now uses heapq for more efficient processing of best matches
    """
    try:
        data = request.json
        
        filter_data = data.get('filter_data', {})
        job_id = data.get('job_id', None)
        threshold = data.get('threshold', 0.75)
        
        # Add maxPerson to filter_data if provided
        if 'maxPerson' in data:
            filter_data['maxPerson'] = data.get('maxPerson')
            
        if not filter_data:
            return jsonify({
                'status': 'error',
                'message': 'Missing filter_data in request'
            }), 400
        
        if not job_id:
            return jsonify({
                'status': 'error',
                'message': 'Missing job_id in request'
            }), 400
        
        restaurant_packages_data = fetch_filtered_variants(filter_data)
        logger.info(f"Raw restaurant data structure: {json.dumps({k: type(v).__name__ for k, v in restaurant_packages_data.items()})}")
    
        if 'variants' in restaurant_packages_data and len(restaurant_packages_data['variants']) > 0:
            sample_variant = restaurant_packages_data['variants'][0]
            logger.info(f"Sample variant keys: {list(sample_variant.keys() if isinstance(sample_variant, dict) else ['Not a dict'])}")
        
        adapted_restaurant_data = adapt_restaurant_data_updated(restaurant_packages_data)
        logger.info(f"Adapted {len(adapted_restaurant_data)} restaurants")
        if not restaurant_packages_data or not restaurant_packages_data.get('variants'):
            return jsonify({
                'status': 'success',
                'message': 'No restaurants found matching your criteria',
                'matches': [],
                'total_restaurants': 0,
                'matched_restaurants': 0,
                'venue_matches': []
            })
        
        adapted_restaurant_data = adapt_restaurant_data_updated(restaurant_packages_data)
        if not adapted_restaurant_data:
            return jsonify({
                'status': 'success',
                'message': 'Could not adapt restaurant data',
                'matches': [],
                'total_restaurants': 0, 
                'matched_restaurants': 0,
                'venue_matches': []
            })

        try:
            user_requirements_data = fetch_user_requirements(job_id)
            print("DEBUG - User requirements response type:", type(user_requirements_data))
            if isinstance(user_requirements_data, dict):
                print("DEBUG - User requirements keys:", user_requirements_data.keys())
                if "data" in user_requirements_data:
                    print("DEBUG - User requirements data keys:", user_requirements_data["data"].keys() if isinstance(user_requirements_data["data"], dict) else "Not a dict")
        except Exception as e:
            logger.error(f"Error fetching user requirements: {str(e)}")
            return jsonify({
                'status': 'error',
                'message': f'Error fetching user requirements: {str(e)}'
            }), 500
        
        user_requirements = api_to_user_requirements(user_requirements_data, is_user_requirement=True)
        
        if not user_requirements.categories:
            logger.warning("No categories found in user requirements")
            return jsonify({
                'status': 'error',
                'message': 'No valid user requirements found',
            }), 500
        
        restaurant_packages = parse_restaurant_packages(adapted_restaurant_data)
        
        matcher = OptimizedRestaurantMatcher(threshold=threshold)
        match_results = matcher.score_restaurants(user_requirements, restaurant_packages)
        
        venue_heaps = {}
        simplified_results = []
        
        for result in match_results:
            match_percentage = round(result.overall_match * 100, 2)
            venue_id = result.venue_id
            
            simplified_result = {
                'variant_id': result.restaurant_id,  
                'variant_name': result.restaurant_name,
                'match_percentage': match_percentage,
                'price': result.price,
                'unmet_requirements': result.unmet_requirements,
                'package_id': result.package_id,
                'venue_id': venue_id
            }
            
            simplified_results.append(simplified_result)
            
            # For each venue, maintain a max heap of match percentages
            # We use negative match_percentage because heapq is a min-heap
            if venue_id:
                if venue_id not in venue_heaps:
                    venue_heaps[venue_id] = []
                
                # Push this result into the heap for its venue
                heapq.heappush(venue_heaps[venue_id], (-match_percentage, simplified_result))
        
        # Extract the best match for each venue (highest percentage = lowest negative value in heap)
        venue_matches = []
        for venue_id, heap in venue_heaps.items():
            if heap:
                # Get the best match (top of heap)
                best_match = heapq.heappop(heap)
                
                # Extract the actual match percentage (removing the negative)
                match_percentage = -best_match[0]
                
                venue_matches.append({
                    'venue_id': venue_id,
                    'match_percentage': match_percentage,
                    # Optional: include other details from the best-matching variant
                    'best_variant_id': best_match[1]['variant_id'],
                    'best_variant_name': best_match[1]['variant_name']
                })
        
        # Sort venue matches by match percentage (highest first)
        venue_matches.sort(key=lambda x: x['match_percentage'], reverse=True)
        
        return jsonify({
            'status': 'success',
            'matches': simplified_results,
            'venue_matches': venue_matches,
            'total_variants': len(restaurant_packages),
            'matched_variants': len(simplified_results)
        })
    
    except Exception as e:
        import traceback
        logger.error(f"Error in match_restaurants_integrated: {str(e)}\n{traceback.format_exc()}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

 
@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint"""
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT, debug=False)
    
    
    
    
    
    
    
    
    