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
import functools


def handle_api_error(func):
    """
    Decorator for API endpoints to catch and handle all exceptions gracefully
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except requests.RequestException as e:
            logger.error(f"API Request Error: {str(e)}")
            return jsonify({
                'status': 'error',
                'message': 'Unable to connect to backend service',
                'details': str(e) if DEBUG else "Backend service unavailable"
            }), 503
        except json.JSONDecodeError as e:
            logger.error(f"JSON Parsing Error: {str(e)}")
            return jsonify({
                'status': 'error',
                'message': 'Invalid response format from backend service',
                'details': str(e) if DEBUG else "Data format error"
            }), 502
        except ValueError as e:
            logger.error(f"Value Error: {str(e)}")
            return jsonify({
                'status': 'error',
                'message': 'Invalid input or processing error',
                'details': str(e) if DEBUG else "Processing error"
            }), 400
        except Exception as e:
            import traceback
            error_details = str(e)
            trace = traceback.format_exc()
            logger.error(f"Unexpected Error: {error_details}")
            logger.error(f"Traceback: {trace}")
            
            return jsonify({
                'status': 'error',
                'message': 'An unexpected error occurred',
                'details': error_details if DEBUG else "Internal server error"
            }), 500
    return wrapper

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)


BACKEND_BASE_URL = os.getenv('BACKEND_BASE_URL',"https://api.staging.tracevenue.com")
FILTERED_VARIANTS_ENDPOINT = os.getenv('FILTERED_VARIANTS_ENDPOINT', '/api/v1/traceVenue/variant/filteredVariants')
USER_REQUIREMENTS_BASE_ENDPOINT = os.getenv('USER_REQUIREMENTS_BASE_ENDPOINT', '/api/v1/traceVenue/jobs')
PORT = int(os.getenv('PORT', 5001))
DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

logging_level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
logger.setLevel(logging_level)

logger.info(f"Starting application with backend URL: {BACKEND_BASE_URL}")
logger.info(f"Filtered variants endpoint: {FILTERED_VARIANTS_ENDPOINT}")
logger.info(f"User requirements base endpoint: {USER_REQUIREMENTS_BASE_ENDPOINT}")




class ServiceMatcher:
    def __init__(self):
        pass
    
    def extract_venue_services(self, variant_data):
        """
        Extract services from a venue variant
        
        Args:
            variant_data: Variant data from the API response
            
        Returns:
            Dictionary with service names and their details
        """
        venue_services = {}
        
        # Extract free services
        if "freeServices" in variant_data and isinstance(variant_data["freeServices"], list):
            for service in variant_data["freeServices"]:
                # Add type check to ensure service is a dictionary
                if not isinstance(service, dict):
                    continue
                    
                service_name = service.get("serviceName", "").lower()
                if service_name:
                    service_category = service.get("serviceCategory", "")
                    service_variant = service.get("Variant", "")
                    service_variant_type = service.get("VariantType", "")
                    
                    venue_services[service_name] = {
                        "category": service_category,
                        "variant": service_variant,
                        "variant_type": service_variant_type,
                        "is_paid": False,
                        "price": 0
                    }
        
        if "paidServices" in variant_data and isinstance(variant_data["paidServices"], list):
            for service in variant_data["paidServices"]:
                # Add type check to ensure service is a dictionary
                if not isinstance(service, dict):
                    continue
                    
                service_name = service.get("serviceName", "").lower()
                if service_name:
                    service_category = service.get("serviceCategory", "")
                    service_variant = service.get("Variant", "")
                    service_variant_type = service.get("VariantType", "")
                    price = service.get("Price", "0")
                    
                    try:
                        price_value = float(price)
                    except ValueError:
                        price_value = 0
                    
                    venue_services[service_name] = {
                        "category": service_category,
                        "variant": service_variant,
                        "variant_type": service_variant_type,
                        "is_paid": True,
                        "price": price_value
                    }
        
        return venue_services
    
    def extract_user_services(self, user_requirements_data):
        """
        Extract services from user requirements
        
        Args:
            user_requirements_data: User requirements data from the API
            
        Returns:
            Dictionary with service names and their details
        """
        user_services = {}
        
        if not isinstance(user_requirements_data, dict):
            return user_services
        
        if "data" in user_requirements_data and isinstance(user_requirements_data["data"], dict) and "services" in user_requirements_data["data"]:
            services_list = user_requirements_data["data"]["services"]
            
            if isinstance(services_list, list):
                for service in services_list:
                    # Add type check to ensure service is a dictionary
                    if not isinstance(service, dict):
                        continue
                        
                    service_name = service.get("serviceName", "").lower()
                    if service_name:
                        price = service.get("Price", "0")
                        
                        is_paid = True
                        if isinstance(price, str) and price.lower() == "free":
                            is_paid = False
                            price_value = 0
                        else:
                            try:
                                price_value = float(price)
                            except ValueError:
                                price_value = 0
                        
                        service_category = service.get("serviceCategory", "")
                        service_variant = service.get("Variant", "")
                        service_variant_type = service.get("VariantType", "")
                        
                        user_services[service_name] = {
                            "category": service_category,
                            "variant": service_variant,
                            "variant_type": service_variant_type,
                            "is_paid": is_paid,
                            "price": price_value
                        }
        
        return user_services
    
    def calculate_service_match(self, venue_services, user_services):
        """
        Calculate service match percentage between venue and user services
        
        Args:
            venue_services: Dictionary of venue services
            user_services: Dictionary of user services
            
        Returns:
            Dictionary with match details
        """
        if not user_services:
            return {
                "match_percentage": 100.0,
                "matched_services": [],
                "unmatched_services": []
            }
        
        matched_services = []
        unmatched_services = []
        total_services = len(user_services)
        matched_count = 0
        
        for service_name, user_service in user_services.items():
            if service_name in venue_services:
                venue_service = venue_services[service_name]
                
                
                payment_match = (user_service["is_paid"] == venue_service["is_paid"])
                
                
                variant_match = (not user_service["variant"] or 
                                user_service["variant"].lower() == venue_service["variant"].lower())
                
                variant_type_match = (not user_service["variant_type"] or 
                                     user_service["variant_type"].lower() == venue_service["variant_type"].lower())
                
                if payment_match or variant_match or variant_type_match:
                    matched_count += 1
                    matched_services.append({
                        "name": service_name,
                        "variant": user_service["variant"],
                        "variant_type": user_service["variant_type"],
                        "is_paid": user_service["is_paid"],
                        "payment_match": payment_match,
                        "variant_match": variant_match,
                        "variant_type_match": variant_type_match
                    })
                else:
                    unmatched_services.append({
                        "name": service_name,
                        "variant": user_service["variant"],
                        "variant_type": user_service["variant_type"],
                        "is_paid": user_service["is_paid"],
                        "reason": "Service found but details do not match"
                    })
            else:
                unmatched_services.append({
                    "name": service_name,
                    "variant": user_service["variant"],
                    "variant_type": user_service["variant_type"],
                    "is_paid": user_service["is_paid"],
                    "reason": "Service not available at venue"
                })
        
        match_percentage = (matched_count / total_services * 100) if total_services > 0 else 100.0
        
        return {
            "match_percentage": round(match_percentage, 2),
            "matched_services": matched_services,
            "unmatched_services": unmatched_services
        }

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
                 package_id: str = "", venue_id: str = "",
                 service_match: Dict[str, Any] = None,
                 over_100_categories: Dict[str, Any] = None):  
        self.restaurant_id = restaurant_id  
        self.restaurant_name = restaurant_name
        self.overall_match = overall_match
        self.category_matches = category_matches
        self.unmet_requirements = unmet_requirements
        self.price = price
        self.rating = rating
        self.package_id = package_id
        self.venue_id = venue_id
        self.service_match = service_match or {"match_percentage": 100.0, "matched_services": [], "unmatched_services": []}
        self.over_100_categories = over_100_categories or {}  
    
    def to_dict(self):
        """Convert MatchResult to dictionary for JSON serialization"""
        return {
            "variant_id": self.restaurant_id,  
            "variant_name": self.restaurant_name,
            "overall_match": self.overall_match,
            "match_percentage": round(self.overall_match * 100, 2), 
            "service_match_percentage": round(self.service_match.get("match_percentage", 100.0), 2),
            "category_matches": {k: round(v * 100, 2) for k, v in self.category_matches.items()}, 
            "unmet_requirements": self.unmet_requirements,
            "unmet_services": self.service_match.get("unmatched_services", []),
            "over_100_categories": self.over_100_categories,  
            "price": self.price,
            "rating": self.rating,
            "package_id": self.package_id,
            "venue_id": self.venue_id
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
        category_restaurant_totals = defaultdict(int)  
        unmet_requirements = []
        over_100_categories = {}  
        
        for key, user_count in flat_user_req.items():
            rest_count = flat_restaurant.get(key, 0)
            
            cat_name, cuisine_name, subcat_name, item_type = key.split('|')
            
            if rest_count >= user_count:
                matched_items += user_count
                item_match = 1.0
            else:
                matched_items += rest_count
                item_match = rest_count / user_count if user_count > 0 else 0
                
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
            category_restaurant_totals[category_key] += rest_count
            category_matches[category_key] += item_match * user_count
            
            cuisine_key = f"{cat_name}|{cuisine_name}"
            category_totals[cuisine_key] += user_count
            category_restaurant_totals[cuisine_key] += rest_count  
            category_matches[cuisine_key] += item_match * user_count
            
            subcat_key = f"{cat_name}|{cuisine_name}|{subcat_name}"
            category_totals[subcat_key] += user_count
            category_restaurant_totals[subcat_key] += rest_count  
            category_matches[subcat_key] += item_match * user_count
        
        for key, rest_count in flat_restaurant.items():
            if key not in flat_user_req and rest_count > 0:
                cat_name, cuisine_name, subcat_name, item_type = key.split('|')
                
                category_key = cat_name
                if category_key not in category_restaurant_totals:
                    category_restaurant_totals[category_key] = 0
                category_restaurant_totals[category_key] += rest_count
                
                cuisine_key = f"{cat_name}|{cuisine_name}"
                if cuisine_key not in category_restaurant_totals:
                    category_restaurant_totals[cuisine_key] = 0
                category_restaurant_totals[cuisine_key] += rest_count
                
                subcat_key = f"{cat_name}|{cuisine_name}|{subcat_name}"
                if subcat_key not in category_restaurant_totals:
                    category_restaurant_totals[subcat_key] = 0
                category_restaurant_totals[subcat_key] += rest_count
        
        for key in category_matches:
            if category_totals[key] > 0:
                category_matches[key] = category_matches[key] / category_totals[key]
        
        for key in category_totals:  
            if '|' not in key:
                user_total = category_totals.get(key, 0)
                rest_total = category_restaurant_totals.get(key, 0)
                
                if user_total > 0:
                    if rest_total > user_total:
                        actual_percentage = (rest_total / user_total) * 100
                        over_100_categories[key] = {
                            "category_name": key,
                            "user_requested": user_total,
                            "restaurant_offers": rest_total,
                            "match_percentage": round(actual_percentage, 2),
                            "additional_items": rest_total - user_total
                        }
        
        overall_match = matched_items / total_user_items if total_user_items > 0 else 0
        
        return overall_match, dict(category_matches), unmet_requirements, over_100_categories

    def score_restaurants(self, user_req, restaurant_packages):
        """Score and rank restaurant packages against user requirements"""
        flat_user_req, total_user_items = self.flatten_requirements(user_req)

        all_matches = []
        
        for restaurant in restaurant_packages:
            flat_restaurant = self.flatten_restaurant(restaurant)
            
            overall_match, category_matches, unmet_requirements, over_100_categories = self.calculate_match(
                flat_user_req, flat_restaurant, total_user_items
            )
            
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
                venue_id=venue_id,
                over_100_categories=over_100_categories  
            )
            
            all_matches.append((overall_match, restaurant.id, match_result))
        
        results = []
        sorted_matches = sorted(all_matches, key=lambda x: x[0], reverse=True)
        for _, _, match_result in sorted_matches:
            results.append(match_result)
        
        return results
    
class ItemPopularityAnalyzer:
    """
    Enhanced version that handles various data structures more robustly
    Efficiently calculates item popularity across restaurant variants
    Time Complexity: O(n*m) where n = variants, m = avg items per variant
    Space Complexity: O(k) where k = unique items
    """
    
    def __init__(self):
        self.item_stats = {}
        self.total_variants = 0
        self.debug_info = []  # For debugging data extraction issues
    
    def analyze_variants(self, variants_data):
        """
        Analyze all variants and calculate item popularity
        
        Args:
            variants_data: List of variant dictionaries from API
            
        Returns:
            Dictionary with item statistics and popularity percentages
        """
        self.item_stats = {}
        self.debug_info = []
        self.total_variants = len(variants_data)
        
        if self.total_variants == 0:
            return {"items": [], "total_variants": 0, "debug_info": ["No variants provided"]}
        
        variants_processed = 0
        
        for i, variant in enumerate(variants_data):
            if not isinstance(variant, dict):
                self.debug_info.append(f"Variant {i}: Not a dictionary - {type(variant)}")
                continue
                
            variant_items = self._extract_items_from_variant(variant)
            
            if variant_items:
                variants_processed += 1
                for item_key, item_data in variant_items.items():
                    if item_key not in self.item_stats:
                        self.item_stats[item_key] = {
                            "count": 0,
                            "total_quantity": 0,
                            "item_name": item_data["name"],
                            "category": item_data["category"],
                            "cuisine": item_data["cuisine"],
                            "subcategory": item_data["subcategory"]
                        }
                    
                    self.item_stats[item_key]["count"] += 1
                    self.item_stats[item_key]["total_quantity"] += item_data["quantity"]
            else:
                variant_id = variant.get("_id", f"variant_{i}")
                self.debug_info.append(f"Variant {variant_id}: No items extracted")
        
        self.debug_info.append(f"Processed {variants_processed} out of {self.total_variants} variants")
        
        return self._prepare_popularity_response()
    
    def _extract_items_from_variant(self, variant):
        """
        Enhanced extraction that handles multiple data structures
        """
        variant_items = {}
        variant_id = variant.get("_id", "unknown")
        
        try:
            # Method 1: Direct availableMenuCount
            if "availableMenuCount" in variant:
                self._process_available_menu_count(variant["availableMenuCount"], variant_items, variant_id)
            
            # Method 2: menuSections structure
            if "menuSections" in variant and isinstance(variant["menuSections"], list):
                for section in variant["menuSections"]:
                    if isinstance(section, dict):
                        self._process_menu_section(section, variant_items, variant_id)
            
            # Method 3: Nested data structure
            if "data" in variant and isinstance(variant["data"], dict):
                data = variant["data"]
                
                if "availableMenuCount" in data:
                    self._process_available_menu_count(data["availableMenuCount"], variant_items, variant_id)
                
                if "menuSections" in data and isinstance(data["menuSections"], list):
                    for section in data["menuSections"]:
                        if isinstance(section, dict):
                            self._process_menu_section(section, variant_items, variant_id)
            
            # Method 4: Direct count field (fallback)
            if "count" in variant:
                self._process_count_data(variant["count"], variant_items, variant_id, "General", "General", "General")
            
            # Log what we found for this variant
            if variant_items:
                self.debug_info.append(f"Variant {variant_id}: Found {len(variant_items)} unique items")
            else:
                # Log the structure for debugging
                keys = list(variant.keys())[:5]  # First 5 keys
                self.debug_info.append(f"Variant {variant_id}: No items found. Keys: {keys}")
                
        except Exception as e:
            self.debug_info.append(f"Variant {variant_id}: Error during extraction - {str(e)}")
        
        return variant_items
    
    def _process_available_menu_count(self, menu_count_data, variant_items, variant_id):
        """Process availableMenuCount data in various formats"""
        if isinstance(menu_count_data, dict):
            # Direct dictionary of items
            for item_name, count in menu_count_data.items():
                if isinstance(count, (int, float)) and count > 0:
                    self._add_item_to_stats(
                        item_name, count, variant_items,
                        "Menu Items", "General", "General"
                    )
                elif isinstance(count, dict):
                    # Nested structure - process recursively
                    self._process_nested_count_structure(item_name, count, variant_items, "Menu Items", "General")
        
        elif isinstance(menu_count_data, list):
            # List of menu sections or items
            for item in menu_count_data:
                if isinstance(item, dict):
                    if "name" in item and "count" in item:
                        # Item with name and count
                        item_name = item.get("name")
                        count = item.get("count", 0)
                        if isinstance(count, (int, float)) and count > 0:
                            self._add_item_to_stats(
                                item_name, count, variant_items,
                                "Menu Items", "General", "General"
                            )
                    else:
                        # Might be a section - process as menu section
                        self._process_menu_section(item, variant_items, variant_id)
    
    def _process_menu_section(self, section, variant_items, variant_id):
        """Process a menu section structure"""
        section_name = section.get("name", "General")
        
        # Check for subcategoriesByCuisine
        if "subcategoriesByCuisine" in section:
            subcategories_by_cuisine = section["subcategoriesByCuisine"]
            if isinstance(subcategories_by_cuisine, dict):
                for cuisine_name, subcategories in subcategories_by_cuisine.items():
                    if isinstance(subcategories, list):
                        for subcat in subcategories:
                            if isinstance(subcat, dict):
                                subcat_name = subcat.get("name", "General")
                                
                                # Check for availableMenuCount or count in subcategory
                                count_data = subcat.get("availableMenuCount") or subcat.get("count")
                                if count_data:
                                    self._process_count_data(
                                        count_data, variant_items, variant_id,
                                        section_name, cuisine_name, subcat_name
                                    )
        
        # Check for direct availableMenuCount in section
        if "availableMenuCount" in section:
            self._process_count_data(
                section["availableMenuCount"], variant_items, variant_id,
                section_name, "General", "General"
            )
        
        # Check for direct count in section
        if "count" in section:
            self._process_count_data(
                section["count"], variant_items, variant_id,
                section_name, "General", "General"
            )
    
    def _process_count_data(self, count_data, variant_items, variant_id, category, cuisine, subcategory):
        """Process count data in various formats"""
        if isinstance(count_data, dict):
            for item_name, count in count_data.items():
                if isinstance(count, (int, float)) and count > 0:
                    self._add_item_to_stats(
                        item_name, count, variant_items,
                        category, cuisine, subcategory
                    )
                elif isinstance(count, dict):
                    self._process_nested_count_structure(item_name, count, variant_items, category, cuisine)
                
        elif isinstance(count_data, list):
            for item in count_data:
                if isinstance(item, dict):
                    item_name = item.get("name", "Unknown")
                    count = item.get("count", 0)
                    if isinstance(count, (int, float)) and count > 0:
                        self._add_item_to_stats(
                            item_name, count, variant_items,
                            category, cuisine, subcategory
                        )
    
    def _process_nested_count_structure(self, parent_name, count_structure, variant_items, category, cuisine):
        """Handle nested count structures"""
        if isinstance(count_structure, dict):
            if "count" in count_structure:
                # Has a count field
                count = count_structure.get("count", 0)
                if isinstance(count, (int, float)) and count > 0:
                    self._add_item_to_stats(
                        parent_name, count, variant_items,
                        category, cuisine, "General"
                    )
            else:
                # Process as nested items
                for sub_item, sub_count in count_structure.items():
                    if isinstance(sub_count, (int, float)) and sub_count > 0:
                        item_name = f"{parent_name}_{sub_item}"
                        self._add_item_to_stats(
                            item_name, sub_count, variant_items,
                            category, cuisine, "General"
                        )
    
    def _add_item_to_stats(self, item_name, quantity, variant_items, category, cuisine, subcategory):
        """Add an item to the variant's item collection"""
        if not item_name or not isinstance(quantity, (int, float)) or quantity <= 0:
            return
            
        item_key = f"{category}|{cuisine}|{subcategory}|{item_name}"
        
        if item_key not in variant_items:
            variant_items[item_key] = {
                "name": item_name,
                "category": category,
                "cuisine": cuisine, 
                "subcategory": subcategory,
                "quantity": quantity
            }
        else:
            variant_items[item_key]["quantity"] += quantity
    
    def _prepare_popularity_response(self):
        """
        Prepare the final response with popularity percentages
        Sorted by popularity (most common first)
        """
        items_list = []
        
        for item_key, stats in self.item_stats.items():
            popularity_percentage = (stats["count"] / self.total_variants) * 100
            avg_quantity = stats["total_quantity"] / stats["count"] if stats["count"] > 0 else 0
            
            items_list.append({
                "item_id": item_key,
                "item_name": stats["item_name"],
                "category": stats["category"],
                "cuisine": stats["cuisine"],
                "subcategory": stats["subcategory"],
                "variants_count": stats["count"],
                "popularity_percentage": round(popularity_percentage, 2),
                "total_quantity_across_variants": stats["total_quantity"],
                "average_quantity_per_variant": round(avg_quantity, 2)
            })
        
        items_list.sort(key=lambda x: x["popularity_percentage"], reverse=True)
        
        return {
            "items": items_list,
            "total_variants": self.total_variants,
            "total_unique_items": len(items_list),
            "debug_info": self.debug_info  # Include debug info for troubleshooting
        }

def add_item_popularity_to_response(restaurant_packages_data):
    """
    Add item popularity analysis to your existing response
    
    Args:
        restaurant_packages_data: The data returned from fetch_filtered_variants()
        
    Returns:
        Dictionary with item popularity statistics and debug info
    """
    if not isinstance(restaurant_packages_data, dict) or 'variants' not in restaurant_packages_data:
        return {
            "items": [], 
            "total_variants": 0, 
            "total_unique_items": 0,
            "debug_info": ["No variants data provided"]
        }
    
    analyzer = ItemPopularityAnalyzer()
    result = analyzer.analyze_variants(restaurant_packages_data['variants'])
    return result
    
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
    try:
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
                    
                    if not isinstance(subcategory_list, list):
                        continue
                        
                    for subcategory_data in subcategory_list:
                        if not isinstance(subcategory_data, dict):
                            continue
                            
                        subcat_name = subcategory_data.get("name", "General")
                        
                        item_counts_data = subcategory_data.get(count_field, {})
                        items = {}
                        
                        if isinstance(item_counts_data, dict):
                            for item_type, count in item_counts_data.items():
                                try:
                                    count_val = int(count) if count else 0
                                    if count_val > 0:
                                        items[item_type] = count_val
                                except (ValueError, TypeError):
                                    logger.warning(f"Invalid count value for item {item_type}: {count}")
                                    continue
                                    
                        elif isinstance(item_counts_data, list):
                            for item_data in item_counts_data:
                                if isinstance(item_data, dict):
                                    item_type = item_data.get("name", "Unknown")
                                    try:
                                        count = int(item_data.get("count", 0))
                                        if count > 0:
                                            items[item_type] = count
                                    except (ValueError, TypeError):
                                        logger.warning(f"Invalid count value for item {item_type}")
                                        continue
                        
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
    except Exception as e:
        logger.error(f"Error parsing user requirements: {str(e)}")
        return UserRequirement({})    
    
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
    
    count_field = "count" if is_user_requirement else "availableMenuCount"
    
    # Add check for string type
    if isinstance(api_response, str):
        logger.warning(f"api_to_user_requirements received string instead of dict: {api_response[:100]}...")
        try:
            # Try to parse the string as JSON if it looks like JSON
            if api_response.strip().startswith('{') and api_response.strip().endswith('}'):
                api_response = json.loads(api_response)
            else:
                return UserRequirement({})
        except:
            return UserRequirement({})
    
    if not isinstance(api_response, dict):
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
    
    return UserRequirement({})


def adapt_restaurant_data_updated(api_response):
    """
    Adapts the API response format to match what the restaurant matcher expects
    Handles both dictionary and list formats for availableMenuCount
    Now also extracts and stores venueId which is directly in the variant object
    Simplified to handle JSON data only, no Mongoose objects
    """
    adapted_data = []
    
    if not isinstance(api_response, dict):
        return adapted_data
    
    original_variants = {}
    
    variants = api_response.get('variants', [])
    if not isinstance(variants, list):
        return adapted_data
    
    for variant in variants:
        if not isinstance(variant, dict):
            continue
            
        venue_id = variant.get("venueId", "")
        variant_id = variant.get("_id", "")
        
        if variant_id:
            original_variants[variant_id] = variant
        
        restaurant = {
            "id": variant_id,
            "name": variant.get("name", ""),
            "price": float(variant.get("cost", 0.0)),
            "rating": 0.0,  
            "package_id": variant.get("packageId", ""),
            "venue_id": venue_id,  
            "categories": {},
            "paid_services": variant.get("paidServices", []),
            "free_services": variant.get("freeServices", [])
        }
        
        found_menu_items = False
        available_menu_count = variant.get("availableMenuCount")
        
        if available_menu_count is None and "data" in variant and isinstance(variant["data"], dict):
            available_menu_count = variant["data"].get("availableMenuCount")
        
        if available_menu_count is not None:
            menu_sections = []
            
            if isinstance(available_menu_count, dict):
                menu_sections = [{"name": "Menu Items", "availableMenuCount": available_menu_count}]
            elif isinstance(available_menu_count, list):
                menu_sections = available_menu_count
            
            for menu_section in menu_sections:
                if isinstance(menu_section, dict):
                    cat_name = menu_section.get("name", "Uncategorized")
                    
                    if cat_name not in restaurant["categories"]:
                        restaurant["categories"][cat_name] = {"cuisines": {}}
                    
                    if "subcategoriesByCuisine" in menu_section:
                        for cuisine_name, subcategory_list in menu_section.get("subcategoriesByCuisine", {}).items():
                            if cuisine_name not in restaurant["categories"][cat_name]["cuisines"]:
                                restaurant["categories"][cat_name]["cuisines"][cuisine_name] = {
                                    "subcategories": {},
                                    "contains_egg": False
                                }
                            
                            for subcategory_data in subcategory_list:
                                subcat_name = subcategory_data.get("name", "General")
                                
                                if subcat_name not in restaurant["categories"][cat_name]["cuisines"][cuisine_name]["subcategories"]:
                                    restaurant["categories"][cat_name]["cuisines"][cuisine_name]["subcategories"][subcat_name] = {
                                        "items": {}
                                    }
                                
                                count_data = subcategory_data.get("availableMenuCount", subcategory_data.get("count", {}))
                                
                                if isinstance(count_data, dict):
                                    for item_type, count in count_data.items():
                                        if count > 0:
                                            restaurant["categories"][cat_name]["cuisines"][cuisine_name]["subcategories"][subcat_name]["items"][item_type] = count
                                            found_menu_items = True
                                            
                                            if item_type == "Egg" and count > 0:
                                                restaurant["categories"][cat_name]["cuisines"][cuisine_name]["contains_egg"] = True
                                elif isinstance(count_data, list):
                                    for item in count_data:
                                        if isinstance(item, dict):
                                            item_type = item.get("name", "Unknown")
                                            count = item.get("count", 0)
                                            if count > 0:
                                                restaurant["categories"][cat_name]["cuisines"][cuisine_name]["subcategories"][subcat_name]["items"][item_type] = count
                                                found_menu_items = True
                                                
                                                if item_type == "Egg" and count > 0:
                                                    restaurant["categories"][cat_name]["cuisines"][cuisine_name]["contains_egg"] = True
                    elif "availableMenuCount" in menu_section or "count" in menu_section:
                        count_data = menu_section.get("availableMenuCount", menu_section.get("count", {}))
                        cuisine_name = "General"
                        subcat_name = "General"
                        
                        if cuisine_name not in restaurant["categories"][cat_name]["cuisines"]:
                            restaurant["categories"][cat_name]["cuisines"][cuisine_name] = {
                                "subcategories": {},
                                "contains_egg": False
                            }
                            
                        if subcat_name not in restaurant["categories"][cat_name]["cuisines"][cuisine_name]["subcategories"]:
                            restaurant["categories"][cat_name]["cuisines"][cuisine_name]["subcategories"][subcat_name] = {
                                "items": {}
                            }
                        
                        if isinstance(count_data, dict):
                            for item_type, count in count_data.items():
                                if count > 0:
                                    restaurant["categories"][cat_name]["cuisines"][cuisine_name]["subcategories"][subcat_name]["items"][item_type] = count
                                    found_menu_items = True
                                    
                                    if item_type == "Egg" and count > 0:
                                        restaurant["categories"][cat_name]["cuisines"][cuisine_name]["contains_egg"] = True
                        elif isinstance(count_data, list):
                            for item in count_data:
                                if isinstance(item, dict):
                                    item_type = item.get("name", "Unknown")
                                    count = item.get("count", 0)
                                    if count > 0:
                                        restaurant["categories"][cat_name]["cuisines"][cuisine_name]["subcategories"][subcat_name]["items"][item_type] = count
                                        found_menu_items = True
                                        
                                        if item_type == "Egg" and count > 0:
                                            restaurant["categories"][cat_name]["cuisines"][cuisine_name]["contains_egg"] = True
        
        if not found_menu_items:
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
    
            restaurant["categories"][cat_name]["cuisines"][cuisine_name]["subcategories"][subcat_name]["items"] = {}
            found_menu_items = True
            
        if restaurant["name"] and restaurant["id"]:
            adapted_data.append(restaurant)
    
    api_response["_original_variants"] = original_variants
    
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
       
        categories = {}
        
        for cat_name, cat_data in rest_data.get("categories", {}).items():
            cuisines = {}
            
            for cuisine_name, cuisine_data in cat_data.get("cuisines", {}).items():
                subcategories = {}
                
                for subcat_name, subcat_data in cuisine_data.get("subcategories", {}).items():
                    items = subcat_data.get("items", {})
                
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
        
        restaurant_packages.append(restaurant)
    
    return restaurant_packages

def fetch_filtered_variants(filter_data):
    """
    Fetch filtered restaurant variants from the backend API
    Include maxPerson parameter if provided
    """
    try:
        base_url = BACKEND_BASE_URL.strip().rstrip('/')
        endpoint = FILTERED_VARIANTS_ENDPOINT.strip().lstrip('/')
        url = f"{base_url}/{endpoint}"
        
        response = requests.post(url, json=filter_data, timeout=120)  
        
        if response.status_code != 200:
            logger.error(f"Failed API response: {response.status_code} - {response.text}")
            return {'variants': [], 'error': f"Backend API error: {response.status_code}"}
        
        # Check if response is valid JSON
        try:
            data = response.json()
        except Exception as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.error(f"Response content: {response.text[:200]}...")  # Log first 200 chars
            return {'variants': [], 'error': "Invalid JSON response from server"}
        
        # Validate that the response is a dictionary
        if not isinstance(data, dict):
            logger.error(f"Unexpected response format: {type(data).__name__}")
            return {'variants': [], 'error': f"Unexpected response format: expected dictionary, got {type(data).__name__}"}
        
        # Validate that variants exist in the response
        if 'variants' not in data:
            logger.warning("No 'variants' key in API response")
            return {'variants': []}
            
        # Validate that variants is a list
        if not isinstance(data['variants'], list):
            logger.error(f"'variants' is not a list: {type(data['variants']).__name__}")
            data['variants'] = []
            
        return data
        
    except requests.exceptions.Timeout:
        logger.error("Timeout error connecting to backend API")
        return {'variants': [], 'error': "Backend API request timed out"}
    except requests.exceptions.ConnectionError:
        logger.error("Connection error to backend API")
        return {'variants': [], 'error': "Could not connect to backend API"}
    except Exception as e:
        logger.error(f"Error in fetch_filtered_variants: {str(e)}")
        return {'variants': [], 'error': str(e)}

    
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
        
        response = requests.get(url)
        
        if response.status_code != 200:
            raise Exception(f"Failed to fetch user requirements: {response.status_code} - {response.text}")
        
        try:
            data = response.json()
            
            # Validate the response is a dictionary
            if not isinstance(data, dict):
                logger.error(f"User requirements API returned non-dictionary: {type(data).__name__}")
                return {"data": {}}
                
            return data
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode user requirements JSON: {e}")
            logger.error(f"Response content: {response.text[:200]}...")  
            return {"data": {}}
        
    except Exception as e:
        logger.error(f"Error in fetch_user_requirements: {str(e)}")
        return {"data": {}}
    

@app.route('/api/match-restaurants-integrated', methods=['POST'])
@handle_api_error  
def match_restaurants_integrated():
    """
    Integrated API endpoint that fetches data from backend APIs 
    and performs matching with updated parser functions
    Now includes service matching and handles JSON data only with improved error handling
    """
    data = request.json
    

    if not isinstance(data, dict):
        return jsonify({
            'status': 'error',
            'message': 'Invalid request format'
        }), 400
    
    filter_data = data.get('filter_data', {})
    job_id = data.get('job_id', None)
    threshold = data.get('threshold', 0.75)
        
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
    
    if 'error' in restaurant_packages_data:
        return jsonify({
            'status': 'error',
            'message': 'Error fetching restaurant data',
            'details': restaurant_packages_data.get('error')
        }), 502
    
    if not isinstance(restaurant_packages_data, dict):
        return jsonify({
            'status': 'error',
            'message': f'Expected dictionary from fetch_filtered_variants, got {type(restaurant_packages_data).__name__}'
        }), 500

    adapted_restaurant_data = adapt_restaurant_data_updated(restaurant_packages_data)
    
    if not restaurant_packages_data or not restaurant_packages_data.get('variants'):
        return jsonify({
            'status': 'success',
            'message': 'No restaurants found matching your criteria',
            'matches': [],
            'total_restaurants': 0,
            'matched_restaurants': 0,
            'venue_matches': []
        })
   
    user_requirements_data = fetch_user_requirements(job_id)
    
    if 'error' in user_requirements_data:
        return jsonify({
            'status': 'error',
            'message': 'Error fetching user requirements',
            'details': user_requirements_data.get('error')
        }), 502
    
    if not isinstance(user_requirements_data, dict):
        return jsonify({
            'status': 'error',
            'message': f'Expected dictionary from fetch_user_requirements, got {type(user_requirements_data).__name__}'
        }), 500
    
    user_requirements = api_to_user_requirements(user_requirements_data, is_user_requirement=True)
    item_popularity = add_item_popularity_to_response(restaurant_packages_data)
    restaurant_packages = parse_restaurant_packages(adapted_restaurant_data)
    
    service_matcher = ServiceMatcher()
    
    user_services = service_matcher.extract_user_services(user_requirements_data)
    
    matcher = OptimizedRestaurantMatcher(threshold=threshold)
    match_results = matcher.score_restaurants(user_requirements, restaurant_packages)
    
    venue_heaps = {}
    simplified_results = []
    
    for result in match_results:
        match_percentage = round(result.overall_match * 100, 2)  
        venue_id = result.venue_id
        variant_id = result.restaurant_id
        
        variant_data = None
        for variant in restaurant_packages_data.get('variants', []):
            if isinstance(variant, dict) and variant.get("_id", "") == variant_id:
                variant_data = variant
                break
        
        service_match_result = {"match_percentage": 100.0, "matched_services": [], "unmatched_services": []}
        if variant_data and isinstance(variant_data, dict):
            try:
                venue_services = service_matcher.extract_venue_services(variant_data)
                service_match_result = service_matcher.calculate_service_match(venue_services, user_services)
            except Exception as e:
                logger.error(f"Error processing services for variant {variant_id}: {str(e)}")
        
        simplified_result = {
        'variant_id': result.restaurant_id,  
        'variant_name': result.restaurant_name,
        'match_percentage': match_percentage,  
        'service_match_percentage': service_match_result["match_percentage"],  
        'price': result.price,
        'unmet_requirements': result.unmet_requirements,
        'unmet_services': service_match_result["unmatched_services"],
        'over_100_categories': result.over_100_categories,  # NEW field
        'package_id': result.package_id,
        'venue_id': venue_id
    }
        
        simplified_results.append(simplified_result)

        if venue_id:
            if venue_id not in venue_heaps:
                venue_heaps[venue_id] = []
    
            heapq.heappush(venue_heaps[venue_id], (-match_percentage, result.restaurant_id, simplified_result))
    
    venue_matches = []
    venue_matches = []
    for venue_id, heap in venue_heaps.items():
        if heap:
            best_match = heapq.heappop(heap)
            
            venue_matches.append({
                'venue_id': venue_id,
                'match_percentage': round(best_match[2]['match_percentage'], 2), 
                'service_match_percentage': round(best_match[2]['service_match_percentage'], 2), 
                'best_variant_id': best_match[2]['variant_id'],
                'best_variant_name': best_match[2]['variant_name']
            })
    
    venue_matches.sort(key=lambda x: x['match_percentage'], reverse=True)
    
    return jsonify({
        'status': 'success',
        'matches': simplified_results,
        'venue_matches': venue_matches,
        'total_variants': len(restaurant_packages),
        'matched_variants': len(simplified_results),
        'item_popularity': item_popularity 
    })

                                
@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint"""
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT, debug=False)
    
    
    
    
    
    
    
    
    