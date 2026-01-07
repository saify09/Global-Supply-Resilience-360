from dataclasses import dataclass
from typing import List, Dict

# Constants for Scale
NUM_COMPANIES = 250
NUM_TIER_1 = 1200
NUM_TIER_2 = 6000
NUM_TIER_3 = 12000
NUM_PRODUCTS = 3500
NUM_COMPONENTS = 9000
NUM_PORTS = 350
NUM_ROUTES = 25000
NUM_EVENTS = 100000

# Categories
INDUSTRIES = ['Electronics', 'Automotive', 'Pharma', 'Textiles', 'Energy', 'Aerospace']
REGIONS = ['North America', 'Europe', 'APAC', 'LATAM', 'EMEA']
RISK_LEVELS = ['Low', 'Medium', 'High', 'Critical']

# Node Labels
LABEL_COMPANY = 'Company'
LABEL_SUPPLIER = 'Supplier'
LABEL_PRODUCT = 'Product'
LABEL_COMPONENT = 'Component'
LABEL_PORT = 'Port'
LABEL_COUNTRY = 'Country'
LABEL_EVENT = 'Event'

# Rel Types
REL_SOURCES_FROM = 'SOURCES_FROM'
REL_PRODUCES = 'PRODUCES'
REL_PART_OF = 'PART_OF'
REL_LOCATED_IN = 'LOCATED_IN'
REL_SHIPS_VIA = 'SHIPS_VIA'
REL_IMPACTED_BY = 'IMPACTED_BY'
