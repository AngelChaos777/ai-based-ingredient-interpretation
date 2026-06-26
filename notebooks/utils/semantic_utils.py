"""
Semantic ingredient classification utilities (Thesis Plan Deliverable 1)

This module provides tools for classifying individual ingredients into semantic
categories (e.g., MSG → Flavor Enhancer, Sodium Compound, Food Additive). It
supports both rule-based and ML-based approaches.

Key Features:
- Ingredient list parsing (split product-level ingredient text into tokens)
- Rule-based semantic category lookup for known ingredients
- Binary matrix conversion for ML training
- Category taxonomy management

Usage:
    >>> from utils.semantic_utils import parse_ingredient_list, ingredient_to_categories
    >>> ingredients = parse_ingredient_list("milk powder, sugar, cocoa butter")
    >>> categories = ingredient_to_categories("MSG")
    >>> categories  # ['food_additive', 'flavor_enhancer', 'sodium_compound']
"""

import json
import os
import re
from typing import Dict, List, Optional, Tuple, Set

# ──────────────────────────────────────────────
# Semantic category taxonomy
# ──────────────────────────────────────────────

SEMANTIC_CATEGORIES = [
    "food_additive",
    "preservative",
    "flavor_enhancer",
    "sweetener",
    "emulsifier",
    "stabilizer",
    "thickener",
    "antioxidant",
    "acidulant",
    "colorant",
    "fat_source",
    "oil_source",
    "protein_source",
    "carbohydrate_source",
    "animal_derived",
    "plant_derived",
    "milk_derivative",
    "egg_derivative",
    "soy_derivative",
    "wheat_derivative",
    "sodium_compound",
    "sugar",
    "added_sugar",
    "fiber",
    "vitamin",
    "mineral",
    "salt",
    "yeast",
    "leavening_agent",
    "gelling_agent",
    "humectant",
    "flavoring",
    "spice",
    "fruit_derived",
    "vegetable_derived",
    "fermented",
    "cured",
    "enzyme",
    "culture",
    "salt_substitute",
]

# ──────────────────────────────────────────────
# Rule-based ingredient → category mapping
# ──────────────────────────────────────────────

# Known ingredient-to-category mappings for the most common ingredients
# Extend this list as new ingredients are annotated.
INGREDIENT_SEMANTIC_MAP: Dict[str, List[str]] = {
    # Sweeteners
    "sugar": ["sweetener", "sugar", "added_sugar", "carbohydrate_source"],
    "sucrose": ["sweetener", "sugar", "added_sugar", "carbohydrate_source"],
    "dextrose": ["sweetener", "sugar", "added_sugar", "carbohydrate_source"],
    "fructose": ["sweetener", "sugar", "added_sugar", "carbohydrate_source"],
    "glucose": ["sweetener", "sugar", "added_sugar", "carbohydrate_source"],
    "glucose syrup": ["sweetener", "added_sugar", "carbohydrate_source"],
    "corn syrup": ["sweetener", "added_sugar", "carbohydrate_source"],
    "high fructose corn syrup": ["sweetener", "added_sugar", "carbohydrate_source"],
    "honey": ["sweetener", "added_sugar", "animal_derived"],
    "maple syrup": ["sweetener", "added_sugar", "plant_derived"],
    "aspartame": ["sweetener", "food_additive", "added_sugar"],
    "saccharin": ["sweetener", "food_additive", "added_sugar"],
    "stevia": ["sweetener", "plant_derived"],
    "splenda": ["sweetener", "food_additive"],
    "agave nectar": ["sweetener", "added_sugar", "plant_derived"],
    "molasses": ["sweetener", "added_sugar", "plant_derived"],
    "maltose": ["sweetener", "added_sugar", "carbohydrate_source"],
    "lactose": ["sweetener", "milk_derivative", "sugar"],
    "maltodextrin": ["carbohydrate_source", "food_additive", "stabilizer"],
    "sorbitol": ["sweetener", "food_additive", "humectant"],
    "xylitol": ["sweetener", "food_additive"],
    "mannitol": ["sweetener", "food_additive"],

    # Flavor enhancers
    "msg": ["food_additive", "flavor_enhancer", "sodium_compound"],
    "monosodium glutamate": ["food_additive", "flavor_enhancer", "sodium_compound"],
    "yeast extract": ["yeast", "flavor_enhancer", "fermented"],
    "hydrolyzed vegetable protein": ["flavor_enhancer", "protein_source", "plant_derived"],
    "hydrolyzed soy protein": ["flavor_enhancer", "protein_source", "soy_derivative"],
    "disodium inosinate": ["food_additive", "flavor_enhancer", "sodium_compound"],
    "disodium guanylate": ["food_additive", "flavor_enhancer", "sodium_compound"],
    "hydrolyzed protein": ["flavor_enhancer", "protein_source"],

    # Preservatives
    "sodium benzoate": ["preservative", "sodium_compound", "food_additive"],
    "potassium sorbate": ["preservative", "food_additive"],
    "sodium metabisulfite": ["preservative", "antioxidant", "sodium_compound"],
    "calcium propionate": ["preservative", "food_additive"],
    "sodium nitrite": ["preservative", "sodium_compound", "cured"],
    "sodium nitrate": ["preservative", "sodium_compound", "cured"],
    "sulfur dioxide": ["preservative", "antioxidant", "food_additive"],
    "sorbic acid": ["preservative", "food_additive"],
    "benzoic acid": ["preservative", "food_additive"],
    "acetic acid": ["preservative", "acidulant", "food_additive"],
    "vinegar": ["preservative", "acidulant", "fermented"],
    "bht": ["preservative", "antioxidant", "food_additive"],
    "bha": ["preservative", "antioxidant", "food_additive"],
    "tocopherols": ["antioxidant", "vitamin"],
    "vitamin e": ["antioxidant", "vitamin"],

    # Emulsifiers
    "lecithin": ["emulsifier", "food_additive"],
    "soy lecithin": ["emulsifier", "soy_derivative", "food_additive", "plant_derived"],
    "sunflower lecithin": ["emulsifier", "plant_derived", "food_additive"],
    "monoglycerides": ["emulsifier", "food_additive"],
    "diglycerides": ["emulsifier", "food_additive"],
    "mono and diglycerides": ["emulsifier", "food_additive"],
    "polysorbate 80": ["emulsifier", "food_additive"],
    "polysorbate 60": ["emulsifier", "food_additive"],
    "carrageenan": ["thickener", "stabilizer", "plant_derived", "gelling_agent"],
    "guar gum": ["thickener", "stabilizer", "plant_derived", "fiber"],
    "xanthan gum": ["thickener", "stabilizer", "fermented"],
    "locust bean gum": ["thickener", "stabilizer", "plant_derived"],
    "gum arabic": ["thickener", "stabilizer", "plant_derived", "fiber"],
    "pectin": ["gelling_agent", "thickener", "fruit_derived", "fiber"],
    "agar": ["gelling_agent", "thickener", "plant_derived"],

    # Fats and oils
    "palm oil": ["fat_source", "oil_source", "plant_derived"],
    "palm kernel oil": ["fat_source", "oil_source", "plant_derived"],
    "coconut oil": ["fat_source", "oil_source", "plant_derived"],
    "soybean oil": ["fat_source", "oil_source", "plant_derived", "soy_derivative"],
    "canola oil": ["fat_source", "oil_source", "plant_derived"],
    "sunflower oil": ["fat_source", "oil_source", "plant_derived"],
    "vegetable oil": ["fat_source", "oil_source", "plant_derived"],
    "olive oil": ["fat_source", "oil_source", "plant_derived"],
    "butter": ["fat_source", "milk_derivative", "animal_derived"],
    "milk fat": ["fat_source", "milk_derivative", "animal_derived"],
    "shortening": ["fat_source", "oil_source"],
    "lard": ["fat_source", "animal_derived"],
    "margarine": ["fat_source", "oil_source", "emulsifier"],

    # Milk derivatives
    "milk": ["milk_derivative", "animal_derived"],
    "milk powder": ["milk_derivative", "animal_derived"],
    "skim milk": ["milk_derivative", "animal_derived"],
    "whole milk": ["milk_derivative", "animal_derived"],
    "whey": ["milk_derivative", "protein_source", "animal_derived"],
    "whey protein": ["milk_derivative", "protein_source", "animal_derived"],
    "casein": ["milk_derivative", "protein_source", "animal_derived"],
    "caseinate": ["milk_derivative", "protein_source", "animal_derived"],
    "calcium caseinate": ["milk_derivative", "protein_source", "animal_derived"],
    "cream": ["milk_derivative", "fat_source", "animal_derived"],
    "cheese": ["milk_derivative", "animal_derived", "fermented"],
    "yogurt": ["milk_derivative", "animal_derived", "fermented"],
    "ghee": ["milk_derivative", "fat_source", "animal_derived"],

    # Egg derivatives
    "egg": ["egg_derivative", "animal_derived", "protein_source"],
    "egg white": ["egg_derivative", "animal_derived", "protein_source"],
    "egg yolk": ["egg_derivative", "animal_derived", "fat_source"],
    "albumin": ["egg_derivative", "protein_source", "animal_derived"],
    "whole egg": ["egg_derivative", "animal_derived", "protein_source"],
    "egg powder": ["egg_derivative", "animal_derived", "protein_source"],
    "mayonnaise": ["egg_derivative", "emulsifier", "fat_source"],

    # Soy derivatives
    "soy": ["soy_derivative", "plant_derived", "protein_source"],
    "soya": ["soy_derivative", "plant_derived", "protein_source"],
    "soybean": ["soy_derivative", "plant_derived", "protein_source"],
    "tofu": ["soy_derivative", "plant_derived", "protein_source"],
    "soy protein": ["soy_derivative", "plant_derived", "protein_source"],
    "soy protein isolate": ["soy_derivative", "plant_derived", "protein_source"],
    "textured vegetable protein": ["soy_derivative", "plant_derived", "protein_source"],
    "tempeh": ["soy_derivative", "plant_derived", "protein_source", "fermented"],
    "miso": ["soy_derivative", "plant_derived", "fermented", "flavor_enhancer"],

    # Wheat derivatives
    "wheat": ["wheat_derivative", "plant_derived", "carbohydrate_source"],
    "wheat flour": ["wheat_derivative", "plant_derived", "carbohydrate_source"],
    "flour": ["wheat_derivative", "plant_derived", "carbohydrate_source"],
    "gluten": ["wheat_derivative", "protein_source", "plant_derived"],
    "semolina": ["wheat_derivative", "plant_derived", "carbohydrate_source"],
    "durum wheat": ["wheat_derivative", "plant_derived", "carbohydrate_source"],
    "couscous": ["wheat_derivative", "plant_derived", "carbohydrate_source"],
    "pasta": ["wheat_derivative", "plant_derived", "carbohydrate_source"],
    "breadcrumbs": ["wheat_derivative", "plant_derived", "carbohydrate_source"],
    "spelt": ["wheat_derivative", "plant_derived", "carbohydrate_source"],
    "triticale": ["wheat_derivative", "plant_derived", "carbohydrate_source"],

    # Starches and carbs
    "corn starch": ["carbohydrate_source", "thickener", "plant_derived"],
    "cornstarch": ["carbohydrate_source", "thickener", "plant_derived"],
    "potato starch": ["carbohydrate_source", "thickener", "plant_derived"],
    "tapioca starch": ["carbohydrate_source", "thickener", "plant_derived"],
    "rice flour": ["carbohydrate_source", "plant_derived"],
    "modified starch": ["carbohydrate_source", "thickener", "food_additive"],
    "modified corn starch": ["carbohydrate_source", "thickener", "food_additive"],
    "inulin": ["fiber", "plant_derived"],
    "oat fiber": ["fiber", "plant_derived"],

    # Acids and acidity regulators
    "citric acid": ["acidulant", "antioxidant", "fruit_derived", "food_additive"],
    "lactic acid": ["acidulant", "preservative", "fermented", "food_additive"],
    "phosphoric acid": ["acidulant", "food_additive"],
    "malic acid": ["acidulant", "food_additive"],
    "tartaric acid": ["acidulant", "food_additive"],

    # Leavening agents
    "baking soda": ["leavening_agent", "mineral"],
    "sodium bicarbonate": ["leavening_agent", "mineral", "sodium_compound"],
    "baking powder": ["leavening_agent"],
    "yeast": ["yeast", "leavening_agent", "fermented"],

    # Colors
    "caramel color": ["colorant", "food_additive"],
    "titanium dioxide": ["colorant", "food_additive", "mineral"],
    "annatto": ["colorant", "plant_derived", "food_additive"],
    "turmeric": ["colorant", "spice", "plant_derived"],
    # paprika listed under Spices section

    # Gelling agents
    "gelatin": ["gelling_agent", "animal_derived", "protein_source"],
    "gelatine": ["gelling_agent", "animal_derived", "protein_source"],
    "agar agar": ["gelling_agent", "plant_derived"],
    "gelatinized starch": ["thickener", "stabilizer", "carbohydrate_source"],

    # Proteins
    "whey protein concentrate": ["protein_source", "milk_derivative", "animal_derived"],
    "whey protein isolate": ["protein_source", "milk_derivative", "animal_derived"],
    "soy protein concentrate": ["protein_source", "soy_derivative", "plant_derived"],
    "pea protein": ["protein_source", "plant_derived"],
    "pea protein isolate": ["protein_source", "plant_derived"],
    "rice protein": ["protein_source", "plant_derived"],
    "collagen": ["protein_source", "animal_derived", "gelling_agent"],

    # Salt
    "salt": ["salt", "mineral"],
    "sea salt": ["salt", "mineral"],
    "table salt": ["salt", "mineral"],
    "kosher salt": ["salt", "mineral"],
    "iodized salt": ["salt", "mineral"],
    "sodium chloride": ["salt", "mineral", "sodium_compound"],

    # Spices
    "garlic": ["spice", "vegetable_derived", "flavoring"],
    "onion": ["vegetable_derived", "flavoring"],
    "onion powder": ["spice", "vegetable_derived", "flavoring"],
    "garlic powder": ["spice", "vegetable_derived", "flavoring"],
    "black pepper": ["spice", "plant_derived", "flavoring"],
    "paprika": ["spice", "colorant", "plant_derived", "vegetable_derived"],
    "oregano": ["spice", "plant_derived", "flavoring"],
    "basil": ["spice", "plant_derived", "flavoring"],
    "cinnamon": ["spice", "plant_derived", "flavoring"],

    # Nuts
    "almond": ["plant_derived", "protein_source", "fat_source"],
    "cashew": ["plant_derived", "protein_source", "fat_source"],
    "peanut": ["plant_derived", "protein_source", "fat_source"],

    # Vitamins and minerals
    "vitamin a": ["vitamin"],
    "vitamin c": ["vitamin", "antioxidant"],
    "ascorbic acid": ["vitamin", "antioxidant", "acidulant", "food_additive"],
    "vitamin d": ["vitamin"],
    "vitamin b": ["vitamin"],
    "calcium carbonate": ["mineral", "food_additive"],
    "calcium": ["mineral"],
    "iron": ["mineral"],
    "zinc": ["mineral"],
    "potassium chloride": ["mineral", "salt_substitute"],
    "magnesium": ["mineral"],

    # Fruit and vegetable derivatives
    "fruit juice": ["fruit_derived", "flavoring", "sweetener"],
    "apple juice": ["fruit_derived", "flavoring", "sweetener"],
    "lemon juice": ["fruit_derived", "acidulant", "flavoring"],
    "tomato paste": ["vegetable_derived", "colorant", "flavoring"],
    "tomato": ["vegetable_derived", "fruit_derived", "flavoring"],
    "spinach": ["vegetable_derived"],
    "carrot": ["vegetable_derived", "colorant"],
    "beetroot": ["vegetable_derived", "colorant"],

    # Enzymes
    "rennet": ["enzyme", "animal_derived"],
    "lipase": ["enzyme"],
    "protease": ["enzyme"],
    "amylase": ["enzyme"],
    "invertase": ["enzyme"],

    # Cultures
    "lactobacillus": ["culture", "fermented"],
    "bifidobacterium": ["culture", "fermented"],
    "probiotics": ["culture", "fermented"],
    "lactic cultures": ["culture", "fermented"],
    "starter culture": ["culture", "fermented"],

    # Humectants
    "glycerol": ["humectant", "food_additive"],
    "glycerin": ["humectant", "food_additive"],
    "propylene glycol": ["humectant", "food_additive"],
    # (sorbitol already listed under Sweeteners section)

    # Other common
    "water": ["mineral"],
    "carbonated water": ["mineral"],
    "natural flavor": ["flavoring", "food_additive"],
    "natural flavors": ["flavoring", "food_additive"],
    "artificial flavor": ["flavoring", "food_additive"],
    "artificial flavors": ["flavoring", "food_additive"],
    "coffee": ["flavoring", "plant_derived"],
    "cocoa": ["flavoring", "colorant", "plant_derived"],
    "chocolate": ["flavoring", "sweetener", "fat_source"],
    "cocoa butter": ["fat_source", "plant_derived", "flavoring"],
    "vanilla": ["flavoring", "plant_derived"],
    "vanilla extract": ["flavoring", "plant_derived"],
    "malt": ["flavoring", "fermented", "carbohydrate_source"],
    "barley malt": ["flavoring", "fermented", "carbohydrate_source"],

    # ── Phase 3: Expanded mappings (from unknown_ingredients_for_labeling.csv) ──

    # Self-identifying functional terms
    "spices": ["spice", "plant_derived", "flavoring"],
    "emulsifier": ["emulsifier", "food_additive"],
    "emulsifiers": ["emulsifier", "food_additive"],
    "stabilizer": ["stabilizer", "food_additive"],
    "stabilizers": ["stabilizer", "food_additive"],
    "antioxidant": ["antioxidant", "food_additive"],
    "antioxidants": ["antioxidant", "food_additive"],
    "acidity regulator": ["acidulant", "food_additive"],
    "acidity regulators": ["acidulant", "food_additive"],
    "preservative": ["preservative", "food_additive"],
    "preservatives": ["preservative", "food_additive"],
    "flavor enhancer": ["flavor_enhancer", "food_additive"],
    "flavor enhancers": ["flavor_enhancer", "food_additive"],
    "sweetener": ["sweetener", "added_sugar"],
    "sweeteners": ["sweetener", "added_sugar"],
    "thickener": ["thickener", "food_additive"],
    "thickeners": ["thickener", "food_additive"],
    "acidulant": ["acidulant", "food_additive"],
    "acidulants": ["acidulant", "food_additive"],
    "humectant": ["humectant", "food_additive"],
    "humectants": ["humectant", "food_additive"],
    "colorant": ["colorant", "food_additive"],
    "colorants": ["colorant", "food_additive"],
    "spice": ["spice", "plant_derived", "flavoring"],
    "seasoning": ["flavoring", "spice", "food_additive"],
    "seasonings": ["flavoring", "spice", "food_additive"],
    "flavoring": ["flavoring", "food_additive"],
    "flavorings": ["flavoring", "food_additive"],
    "gelling agent": ["gelling_agent", "food_additive"],
    "leavening agent": ["leavening_agent"],
    "leavening agents": ["leavening_agent"],
    "raising agent": ["leavening_agent"],
    "raising agents": ["leavening_agent"],
    "anticaking agent": ["food_additive", "mineral"],
    "anti-caking agent": ["food_additive", "mineral"],
    "nature-identical flavor": ["flavoring", "food_additive"],
    "nature-identical flavors": ["flavoring", "food_additive"],
    "permitted flavouring": ["flavoring", "food_additive"],
    "dough improver": ["food_additive", "emulsifier"],
    "dough improvers": ["food_additive", "emulsifier"],
    "emulsifying agent": ["emulsifier", "food_additive"],
    "coloring": ["colorant", "food_additive"],
    "colouring": ["colorant", "food_additive"],

    # Specific chemicals and additives
    "silicon dioxide": ["mineral", "food_additive"],
    "sodium citrate": ["sodium_compound", "acidulant", "food_additive"],
    "sucralose": ["sweetener", "added_sugar", "food_additive"],
    "tartrazine": ["colorant", "food_additive"],
    "sunset yellow": ["colorant", "food_additive"],
    "tbhq": ["antioxidant", "preservative", "food_additive"],
    "ammonium bicarbonate": ["leavening_agent", "mineral"],
    "sodium carbonate": ["mineral", "food_additive"],
    "dipotassium phosphate": ["mineral", "food_additive", "stabilizer"],
    "sodium polyphosphate": ["stabilizer", "food_additive", "sodium_compound"],
    "potassium carbonate": ["mineral", "food_additive"],
    "acesulfame potassium": ["sweetener", "added_sugar", "food_additive"],
    "steviol glycosides": ["sweetener", "plant_derived"],
    "vanillin": ["flavoring", "food_additive"],
    "beta-carotene": ["colorant", "vitamin", "food_additive"],
    "beta carotene": ["colorant", "vitamin", "food_additive"],
    "datem": ["emulsifier", "food_additive"],
    "sodium carboxymethyl cellulose": ["thickener", "stabilizer", "food_additive"],
    "sodium stearoyl lactylate": ["emulsifier", "food_additive"],
    "disodium 5'-guanylate": ["flavor_enhancer", "food_additive", "sodium_compound"],
    "disodium 5'-inosinate": ["flavor_enhancer", "food_additive", "sodium_compound"],
    "sodium erythorbate": ["antioxidant", "sodium_compound", "food_additive"],
    "sodium diacetate": ["preservative", "acidulant", "food_additive"],
    "sodium phosphate": ["mineral", "food_additive"],
    "trisodium citrate": ["acidulant", "food_additive", "sodium_compound"],
    "allura red": ["colorant", "food_additive"],
    "brilliant blue": ["colorant", "food_additive"],
    "disodium phosphate": ["mineral", "food_additive", "stabilizer"],

    # Vitamins
    "riboflavin": ["vitamin", "colorant"],
    "niacin": ["vitamin"],
    "niacinamide": ["vitamin"],
    "cyanocobalamin": ["vitamin"],
    "pyridoxine hydrochloride": ["vitamin"],
    "folic acid": ["vitamin"],
    "biotin": ["vitamin"],
    "pantothenic acid": ["vitamin"],
    "vitamins": ["vitamin"],
    "minerals": ["mineral"],
    "vitamin b2": ["vitamin"],
    "vitamin b3": ["vitamin"],
    "vitamin b6": ["vitamin"],
    "vitamin b12": ["vitamin"],
    "vitamin e": ["vitamin", "antioxidant"],
    "vitamin k": ["vitamin"],

    # Allergen-relevant ingredients (also helps allergen detection mapping)
    "fish": ["animal_derived", "protein_source"],
    "shrimp": ["animal_derived", "protein_source"],
    "shrimps": ["animal_derived", "protein_source"],
    "tuna flakes": ["animal_derived", "protein_source"],
    "beef": ["animal_derived", "protein_source"],
    "pork": ["animal_derived", "protein_source"],
    "chicken": ["animal_derived", "protein_source"],
    "chicken meat": ["animal_derived", "protein_source"],
    "chicken broth": ["animal_derived", "flavoring"],
    "sesame": ["plant_derived", "fat_source", "oil_source", "protein_source"],
    "sesame seeds": ["plant_derived", "fat_source", "oil_source"],
    "mustard": ["spice", "plant_derived", "flavoring"],
    "oats": ["plant_derived", "carbohydrate_source", "fiber"],
    "crustacean": ["animal_derived"],
    "crustaceans": ["animal_derived"],
    "molluscs": ["animal_derived"],
    "mollusks": ["animal_derived"],
    "squid": ["animal_derived", "protein_source"],
    "crab": ["animal_derived", "protein_source"],
    "tree nuts": ["plant_derived", "fat_source", "protein_source"],
    "tree nut": ["plant_derived", "fat_source", "protein_source"],
    "sulphites": ["preservative", "food_additive"],
    "sulfites": ["preservative", "food_additive"],
    "celery": ["vegetable_derived", "flavoring"],
    "dairy solids": ["milk_derivative", "animal_derived"],
    "fish oil": ["oil_source", "animal_derived"],

    # Oils and fats
    "vegetable fat": ["fat_source", "plant_derived"],
    "palm olein": ["oil_source", "plant_derived", "fat_source"],
    "hydrogenated vegetable fat": ["fat_source", "plant_derived"],
    "hydrogenated vegetable oil": ["oil_source", "fat_source", "plant_derived"],
    "palm fat": ["fat_source", "oil_source", "plant_derived"],
    "corn oil": ["oil_source", "plant_derived", "fat_source"],
    "rapeseed oil": ["oil_source", "plant_derived", "fat_source"],
    "non-hydrogenated vegetable fat": ["fat_source", "plant_derived"],
    "palm oil fraction": ["oil_source", "fat_source", "plant_derived"],
    "interesterified vegetable fat": ["fat_source", "plant_derived"],

    # Food ingredients — produce, etc.
    "chili": ["spice", "vegetable_derived", "flavoring"],
    "chili powder": ["spice", "vegetable_derived", "flavoring"],
    "dehydrated vegetables": ["vegetable_derived"],
    "dehydrated potatoes": ["vegetable_derived", "carbohydrate_source"],
    "cabbage": ["vegetable_derived"],
    "chives": ["vegetable_derived", "flavoring"],
    "potatoes": ["vegetable_derived", "carbohydrate_source"],
    "nata de coco": ["plant_derived", "fermented", "fiber"],
    "enzymes": ["enzyme"],
    "fish sauce": ["animal_derived", "fermented", "flavor_enhancer"],
    "caramel iv": ["colorant", "food_additive"],
    "caramel colour": ["colorant", "food_additive"],
    "beet red": ["colorant", "vegetable_derived", "food_additive"],
    "rosemary extract": ["antioxidant", "spice", "plant_derived"],
    "strawberry": ["fruit_derived", "flavoring"],
    "pineapple": ["fruit_derived", "flavoring", "sweetener"],
    "raisins": ["fruit_derived", "sweetener"],
    "cherries": ["fruit_derived", "flavoring"],
    "orange juice concentrate": ["fruit_derived", "flavoring", "sweetener"],
    "tea": ["plant_derived", "flavoring"],
    "black tea": ["plant_derived", "flavoring"],
    "green tea extract": ["antioxidant", "plant_derived", "flavoring"],

    # More spices and seasonings
    "ginger": ["spice", "plant_derived", "flavoring"],
    "clove": ["spice", "plant_derived", "flavoring"],
    "nutmeg": ["spice", "plant_derived", "flavoring"],
    "cumin": ["spice", "plant_derived", "flavoring"],
    "coriander": ["spice", "plant_derived", "flavoring"],
    "turmeric": ["spice", "colorant", "plant_derived"],
    "red pepper": ["spice", "vegetable_derived", "flavoring"],
    "red pepper flakes": ["spice", "vegetable_derived", "flavoring"],
    "white pepper": ["spice", "plant_derived", "flavoring"],
    "bay leaf": ["spice", "plant_derived", "flavoring"],

    # Additional common ingredients
    "modified food starch": ["thickener", "carbohydrate_source", "food_additive"],
    "cellulose gum": ["thickener", "stabilizer", "fiber"],
    "potato": ["vegetable_derived", "carbohydrate_source"],
    "sweet potato": ["vegetable_derived", "carbohydrate_source"],
    "coconut milk": ["plant_derived", "fat_source", "flavoring"],
    "coconut cream": ["plant_derived", "fat_source", "flavoring"],
    "coconut water": ["plant_derived", "mineral"],
    "wheat starch": ["wheat_derivative", "carbohydrate_source", "thickener"],
    "corn syrup solids": ["sweetener", "carbohydrate_source", "added_sugar"],
    "palm kernel stearin": ["fat_source", "oil_source", "plant_derived"],
    "milk protein concentrate": ["protein_source", "milk_derivative", "animal_derived"],
    "milk solids": ["milk_derivative", "animal_derived"],
    "skim milk powder": ["milk_derivative", "animal_derived"],
    "whole milk powder": ["milk_derivative", "animal_derived"],
    "buttermilk": ["milk_derivative", "animal_derived", "fermented"],
    "sweetened condensed milk": ["milk_derivative", "sweetener", "added_sugar"],
    "evaporated milk": ["milk_derivative", "animal_derived"],
    "whipped cream": ["milk_derivative", "fat_source", "animal_derived"],
    "mono -": ["emulsifier", "food_additive"],  # truncated "mono- and diglycerides"
    "dextrose monohydrate": ["sweetener", "added_sugar", "carbohydrate_source"],
    "milk chocolate": ["flavoring", "sweetener", "milk_derivative", "fat_source"],
    "white chocolate": ["flavoring", "sweetener", "fat_source"],

    # Fruit-derived
    "orange juice": ["fruit_derived", "flavoring", "sweetener"],
    "lemon juice concentrate": ["fruit_derived", "acidulant", "flavoring"],
    "apple juice concentrate": ["fruit_derived", "sweetener", "flavoring"],
    "grape juice": ["fruit_derived", "sweetener", "flavoring"],
    "cranberry": ["fruit_derived", "flavoring"],
    "blueberry": ["fruit_derived", "flavoring"],
    "banana": ["fruit_derived", "flavoring", "sweetener"],
    "mango": ["fruit_derived", "flavoring", "sweetener"],
    "papaya": ["fruit_derived", "flavoring"],
    "lychee": ["fruit_derived", "flavoring", "sweetener"],
}

# Separator pattern for ingredient list parsing (compiled once at module load)
_SEPARATOR_PATTERN = re.compile(r'\s*[,;]\s*|\s*\(|\)\s*|\s+and\s+|\s*\+\s*|\s*\(\s*|\s*\)\s*')

# ──────────────────────────────────────────────
# Public functions
# ──────────────────────────────────────────────


def get_semantic_category_list() -> List[str]:
    """Return the full list of semantic categories (in canonical order).

    Returns:
        List of category strings.
    """
    return SEMANTIC_CATEGORIES.copy()


def load_semantic_config(config_path: str) -> dict:
    """Load the semantic categories config from JSON.

    Args:
        config_path: Path to semantic_categories.json.

    Returns:
        Dictionary with categories, examples, and category_groups.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_ingredient_list(ingredient_text: str) -> List[str]:
    """Split a full ingredient list string into individual ingredient tokens.

    Handles common separators: commas, semicolons, parentheses, "and", "+".

    Args:
        ingredient_text: Raw ingredient list string.
            Example: "sugar, wheat flour, soy lecithin, salt"

    Returns:
        List of individual ingredient strings (lowercased, stripped).
    """
    if not isinstance(ingredient_text, str) or not ingredient_text.strip():
        return []

    # Normalize separators
    text = ingredient_text.lower().strip()

    # Split on common separators
    parts = _SEPARATOR_PATTERN.split(text)

    # Clean, filter, and return
    ingredients = []
    for part in parts:
        part = part.strip().strip(".,")
        if part and not part.isdigit() and len(part) > 1:
            if part not in ingredients:
                ingredients.append(part)

    return ingredients


def _compile_patterns():
    """Stub: pre-compile regex patterns from the ingredient map for bulk matching.
    Currently unused — dict + substring matching suffices for the current map size.
    """
    pass


# Compiled patterns for E-number, FDC color, and INS code matching
_ENUMBER_PATTERN = re.compile(r'^e([0-9]{3,4})$')
_FDC_COLOR_PATTERN = re.compile(
    r'^fd&c\s+(red|yellow|blue|green)\s+(no\.?\s*)?[0-9]+$',
    re.IGNORECASE
)
_INS_PATTERN = re.compile(r'^ins\s+([0-9]{3,4})$', re.IGNORECASE)


def _normalize_plural(ingredient: str) -> str:
    """Strip trailing 's' from known functional-role words for lookup.

    Handles common plural forms of additive/functional terms that map directly
    to singular keys. Does NOT normalize words whose singular form has a
    different semantic meaning (e.g., 'peanuts', 'eggs', 'spices', 'oats').

    Args:
        ingredient: Lowercased ingredient string.

    Returns:
        Normalized ingredient string (plural → singular where applicable).
    """
    # Words that should NOT be normalized (keep plural form)
    _no_normalize = {
        "peanuts", "almonds", "cashews", "walnuts", "hazelnuts",
        "pecans", "peanuts", "eggs", "spices", "oats", "cultures",
        "probiotics", "vitamins",
    }

    if ingredient in _no_normalize:
        return ingredient

    # Apply plural → singular for known functional terms
    _plural_map = {
        "emulsifiers": "emulsifier",
        "stabilizers": "stabilizer",
        "thickeners": "thickener",
        "preservatives": "preservative",
        "acidulants": "acidulant",
        "sweeteners": "sweetener",
        "flavorings": "flavoring",
        "flavourings": "flavoring",
        "seasonings": "seasoning",
        "colorants": "colorant",
        "humectants": "humectant",
        "enzymes": "enzyme",
        "gelling agents": "gelling agent",
        "raising agents": "raising agent",
        "leavening agents": "leavening agent",
        "colors": "colorant",
        "colours": "colorant",
        "flavours": "flavoring",
        "flavors": "flavoring",
        "minerals": "mineral",
        "sweetener": "sweetener",
        "antioxidants": "antioxidant",
    }

    return _plural_map.get(ingredient, ingredient)


# E-number range → category mapping
_ENUMBER_RANGES = [
    (100, 199, ["colorant", "food_additive"]),
    (200, 299, ["preservative", "food_additive"]),
    (300, 399, ["antioxidant", "food_additive"]),
    (400, 499, ["thickener", "stabilizer", "emulsifier", "food_additive"]),
    (500, 599, ["acidulant", "food_additive", "mineral"]),
    (600, 699, ["flavor_enhancer", "food_additive"]),
    (900, 999, ["sweetener", "food_additive"]),
    (1000, 1599, ["food_additive"]),
]


def _match_enumber(ingredient: str) -> Optional[List[str]]:
    """Check if ingredient is an E-number and return corresponding categories.

    Args:
        ingredient: Lowercased, stripped ingredient string.

    Returns:
        List of categories if matched, None otherwise.
    """
    m = _ENUMBER_PATTERN.match(ingredient)
    if not m:
        return None
    num = int(m.group(1))
    for lo, hi, cats in _ENUMBER_RANGES:
        if lo <= num <= hi:
            return list(cats)
    return ["food_additive"]


def _match_fdc_color(ingredient: str) -> Optional[List[str]]:
    """Check if ingredient is an FDC color name and return category.

    Args:
        ingredient: Lowercased, stripped ingredient string.

    Returns:
        ['colorant', 'food_additive'] if matched, None otherwise.
    """
    if _FDC_COLOR_PATTERN.match(ingredient):
        return ["colorant", "food_additive"]
    return None


def _match_ins_code(ingredient: str) -> Optional[List[str]]:
    """Check if ingredient is an INS code and return category.

    INS codes correspond to E-number ranges (e.g., INS 330 == E330 == citric acid),
    so the same range-based mapping is used.

    Args:
        ingredient: Lowercased, stripped ingredient string.

    Returns:
        List of categories if matched, None otherwise.
    """
    m = _INS_PATTERN.match(ingredient)
    if not m:
        return None
    num = int(m.group(1))
    for lo, hi, cats in _ENUMBER_RANGES:
        if lo <= num <= hi:
            return list(cats)
    return ["food_additive"]


def ingredient_to_categories(
    ingredient: str,
    categories: Optional[List[str]] = None,
) -> List[str]:
    """Map a single ingredient to its semantic categories.

    Supports both exact lookup (faster) and substring matching (fallback).
    Also normalizes plural forms of functional terms and recognizes
    E-numbers, FDC colors, and INS codes.

    Args:
        ingredient: An individual ingredient term (e.g., "MSG", "soy lecithin").
        categories: Optional subset of categories to restrict results to
                    (not yet implemented — returns all matched categories).

    Returns:
        List of semantic category strings.
    """
    if not ingredient or not isinstance(ingredient, str):
        return []

    ingredient_lower = ingredient.lower().strip()

    # 1. Try exact match
    if ingredient_lower in INGREDIENT_SEMANTIC_MAP:
        return INGREDIENT_SEMANTIC_MAP[ingredient_lower].copy()

    # 2. Try plural-normalized match (emulsifiers → emulsifier)
    normalized = _normalize_plural(ingredient_lower)
    if normalized != ingredient_lower and normalized in INGREDIENT_SEMANTIC_MAP:
        return INGREDIENT_SEMANTIC_MAP[normalized].copy()

    # 3. Try E-number match (e330 → acidulant)
    enumber_cats = _match_enumber(ingredient_lower)
    if enumber_cats:
        return enumber_cats

    # 4. Try FDC color match (fd&c red no. 40 → colorant)
    fdc_cats = _match_fdc_color(ingredient_lower)
    if fdc_cats:
        return fdc_cats

    # 5. Try INS code match
    ins_cats = _match_ins_code(ingredient_lower)
    if ins_cats:
        return ins_cats

    # 6. Try partial match — check if any known ingredient is a substring
    matches = []
    for known_ing, cats in INGREDIENT_SEMANTIC_MAP.items():
        if known_ing in ingredient_lower or ingredient_lower in known_ing:
            matches.extend(cats)

    # Deduplicate while preserving order
    seen: Set[str] = set()
    unique_matches = []
    for cat in matches:
        if cat not in seen:
            seen.add(cat)
            unique_matches.append(cat)

    return unique_matches


def classify_ingredient_list(
    ingredients: List[str],
    categories: Optional[List[str]] = None,
) -> Dict[str, List[str]]:
    """Classify a list of ingredients into semantic categories.

    Args:
        ingredients: List of individual ingredient strings.
        categories: Optional list of category columns to restrict to.

    Returns:
        Dictionary mapping each ingredient to its list of category labels.
    """
    return {ing: ingredient_to_categories(ing, categories) for ing in ingredients}


def semantic_labels_to_binary(
    label_list: List[str],
    categories: Optional[List[str]] = None,
) -> List[int]:
    """Convert a list of semantic category strings to a binary vector.

    Args:
        label_list: List of category strings (e.g., ['sweetener', 'added_sugar']).
        categories: Category order for the binary vector. Uses SEMANTIC_CATEGORIES
                    if not provided.

    Returns:
        Binary list of length len(categories).
    """
    cats = categories if categories is not None else SEMANTIC_CATEGORIES
    label_set = set(label_list)
    return [1 if cat in label_set else 0 for cat in cats]


def binary_to_semantic_labels(
    binary_vector: List[int],
    categories: Optional[List[str]] = None,
) -> List[str]:
    """Convert a binary vector back to a list of semantic category labels.

    Args:
        binary_vector: Binary list of length len(categories).
        categories: Category order. Uses SEMANTIC_CATEGORIES if not provided.

    Returns:
        List of category strings where the binary value is 1.
    """
    cats = categories if categories is not None else SEMANTIC_CATEGORIES
    return [cats[i] for i, val in enumerate(binary_vector) if val == 1]


def build_semantic_label_matrix(
    ingredient_texts: List[str],
    ingredient_parser: callable = parse_ingredient_list,
    label_func: callable = ingredient_to_categories,
    categories: Optional[List[str]] = None,
) -> List[List[int]]:
    """Build a multi-label binary matrix from ingredient texts.

    For each product's ingredient list, this:
    1. Parses the ingredient text into individual ingredients
    2. Looks up semantic categories for each ingredient
    3. Aggregates categories across all ingredients
    4. Encodes as a binary vector

    Args:
        ingredient_texts: List of ingredient text strings (one per product).
        ingredient_parser: Function to split text into ingredient tokens.
        label_func: Function to map ingredient → label list.
        categories: Category order for output. Uses SEMANTIC_CATEGORIES if None.

    Returns:
        List of binary lists, shape (n_products, n_categories).
    """
    cats = categories if categories is not None else SEMANTIC_CATEGORIES
    result = []

    for text in ingredient_texts:
        ingredients = ingredient_parser(text)
        all_labels: Set[str] = set()
        for ing in ingredients:
            ing_labels = label_func(ing)
            all_labels.update(ing_labels)

        binary = semantic_labels_to_binary(list(all_labels), cats)
        result.append(binary)

    return result


def get_category_groups() -> Dict[str, List[str]]:
    """Return the category grouping taxonomy.

    Returns:
        Dictionary mapping group names to lists of category strings.
    """
    return {
        "additives_and_preservatives": [
            "food_additive", "preservative", "antioxidant", "acidulant", "colorant",
        ],
        "flavor_and_sweeteners": [
            "flavor_enhancer", "sweetener", "sugar", "added_sugar",
            "flavoring", "spice", "salt", "yeast",
        ],
        "functional_ingredients": [
            "emulsifier", "stabilizer", "thickener", "gelling_agent",
            "leavening_agent", "humectant",
        ],
        "macronutrients": [
            "fat_source", "oil_source", "protein_source", "carbohydrate_source", "fiber",
        ],
        "origin_and_derivation": [
            "animal_derived", "plant_derived", "milk_derivative", "egg_derivative",
            "soy_derivative", "wheat_derivative", "fermented", "cured",
        ],
        "micronutrients": ["vitamin", "mineral"],
        "biological": ["enzyme", "culture"],
    }



# Compile regex patterns at module load time for efficient ingredient matching
# (Stub kept for future regex-based lookup optimization)
