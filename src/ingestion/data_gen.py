import pandas as pd
import numpy as np
from faker import Faker
import random
import os
import uuid
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.ingestion.data_schema import *

fake = Faker()
Faker.seed(42)
np.random.seed(42)
random.seed(42)

OUTPUT_DIR = Path("data/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def generate_countries_ports():
    print("Generating Countries and Ports...")
    countries = []
    # Generate ~80 countries
    for _ in range(85):
        c_code = fake.country_code()
        c_name = fake.country()
        countries.append({
            'country_id': c_code,
            'name': c_name,
            'region': random.choice(REGIONS),
            'risk_score': round(random.uniform(0.1, 0.9), 2)
        })
    df_countries = pd.DataFrame(countries).drop_duplicates(subset=['country_id'])
    
    ports = []
    country_list = df_countries['country_id'].tolist()
    for _ in range(NUM_PORTS):
        ports.append({
            'port_id': f"PORT_{uuid.uuid4().hex[:8]}",
            'name': f"{fake.city()} Port",
            'country_id': random.choice(country_list),
            'capacity_teu': random.randint(10000, 5000000),
            'status': 'Operational'
        })
    df_ports = pd.DataFrame(ports)
    
    df_countries.to_csv(OUTPUT_DIR / "countries.csv", index=False)
    df_ports.to_csv(OUTPUT_DIR / "ports.csv", index=False)
    return df_countries, df_ports

def generate_companies_suppliers(df_countries):
    print("Generating Companies and Suppliers...")
    country_list = df_countries['country_id'].tolist()
    
    # Buyers
    companies = []
    for i in range(NUM_COMPANIES):
        companies.append({
            'id': f"COM_{i:04d}",
            'name': fake.company(),
            'industry': random.choice(INDUSTRIES),
            'revenue_b': round(random.uniform(1.0, 100.0), 2),
            'country_id': random.choice(country_list)
        })
    df_companies = pd.DataFrame(companies)
    
    # Suppliers (Tier 1, 2, 3)
    suppliers = []
    # Tier 1
    for i in range(NUM_TIER_1):
        suppliers.append({
            'id': f"SUP_T1_{i:05d}",
            'name': fake.company() + " Mfg",
            'tier': 1,
            'country_id': random.choice(country_list),
            'esg_score': round(random.uniform(50, 100), 1),
            'carbon_intensity': round(random.uniform(10, 500), 1)
        })
    # Tier 2
    for i in range(NUM_TIER_2):
        suppliers.append({
            'id': f"SUP_T2_{i:05d}",
            'name': fake.company() + " Parts",
            'tier': 2,
            'country_id': random.choice(country_list),
            'esg_score': round(random.uniform(40, 95), 1),
            'carbon_intensity': round(random.uniform(20, 600), 1)
        })
    # Tier 3
    for i in range(NUM_TIER_3):
        suppliers.append({
            'id': f"SUP_T3_{i:05d}",
            'name': fake.company() + " Raw",
            'tier': 3,
            'country_id': random.choice(country_list),
            'esg_score': round(random.uniform(30, 90), 1),
            'carbon_intensity': round(random.uniform(50, 800), 1)
        })
        
    df_suppliers = pd.DataFrame(suppliers)
    
    df_companies.to_csv(OUTPUT_DIR / "companies.csv", index=False)
    df_suppliers.to_csv(OUTPUT_DIR / "suppliers.csv", index=False)
    return df_companies, df_suppliers

def generate_products_components():
    print("Generating Products and Components...")
    products = []
    for i in range(NUM_PRODUCTS):
        products.append({
            'id': f"PROD_{i:05d}",
            'name': f"Product {fake.word().title()} {random.randint(100,999)}",
            'category': random.choice(INDUSTRIES),
            'price': round(random.uniform(100, 5000), 2)
        })
    df_products = pd.DataFrame(products)
    
    components = []
    for i in range(NUM_COMPONENTS):
        components.append({
            'id': f"COMP_{i:05d}",
            'name': f"Component {fake.word().upper()} {random.randint(10,99)}",
            'type': random.choice(['Electronic', 'Metal', 'Plastic', 'Chemical']),
            'criticality': random.choice(['Low', 'Medium', 'High'])
        })
    df_components = pd.DataFrame(components)
    
    df_products.to_csv(OUTPUT_DIR / "products.csv", index=False)
    df_components.to_csv(OUTPUT_DIR / "components.csv", index=False)
    return df_products, df_components

def generate_relationships(df_companies, df_suppliers, df_products, df_components):
    print("Generating Relationships...")
    
    # Sources From: Company -> Tier 1
    rels_c_t1 = []
    t1_ids = df_suppliers[df_suppliers['tier'] == 1]['id'].tolist()
    for c_id in df_companies['id']:
        # Each company has 5-20 Tier 1 suppliers
        my_suppliers = random.sample(t1_ids, k=random.randint(5, 20))
        for s_id in my_suppliers:
            rels_c_t1.append({
                'from_id': c_id,
                'to_id': s_id,
                'type': REL_SOURCES_FROM,
                'annual_spend_m': round(random.uniform(0.5, 50), 2)
            })
            
    # Sources From: Tier 1 -> Tier 2
    rels_t1_t2 = []
    t2_ids = df_suppliers[df_suppliers['tier'] == 2]['id'].tolist()
    for s1_id in t1_ids:
         # Each T1 has 3-10 Tier 2 suppliers
        my_suppliers = random.sample(t2_ids, k=random.randint(3, 10))
        for s2_id in my_suppliers:
            rels_t1_t2.append({
                'from_id': s1_id,
                'to_id': s2_id,
                'type': REL_SOURCES_FROM,
                'dependency_weight': round(random.uniform(0.1, 1.0), 2)
            })

    # Sources From: Tier 2 -> Tier 3
    rels_t2_t3 = []
    t3_ids = df_suppliers[df_suppliers['tier'] == 3]['id'].tolist()
    for s2_id in t2_ids:
         # Each T2 has 2-8 Tier 3 suppliers
        my_suppliers = random.sample(t3_ids, k=random.randint(2, 8))
        for s3_id in my_suppliers:
            rels_t2_t3.append({
                'from_id': s2_id,
                'to_id': s3_id,
                'type': REL_SOURCES_FROM,
                'dependency_weight': round(random.uniform(0.1, 1.0), 2)
            })

    # Produces: Supplier -> Component
    # We'll assign components to suppliers randomly
    rels_sup_comp = []
    comp_ids = df_components['id'].tolist()
    # Assume T2 and T3 produce components
    producers = df_suppliers[df_suppliers['tier'].isin([2, 3])]['id'].tolist()
    
    for c_id in comp_ids:
        # Each component produced by 1-3 suppliers
        makers = random.sample(producers, k=random.randint(1, 3))
        for s_id in makers:
            rels_sup_comp.append({
                'from_id': s_id,
                'to_id': c_id,
                'type': REL_PRODUCES,
                'capacity_per_week': random.randint(100, 10000)
            })

    # Part Of: Component -> Product
    rels_comp_prod = []
    prod_ids = df_products['id'].tolist()
    for p_id in prod_ids:
        # Each product made of 10-30 components
        parts = random.sample(comp_ids, k=random.randint(10, 30))
        for c_id in parts:
            rels_comp_prod.append({
                'from_id': c_id,
                'to_id': p_id,
                'type': REL_PART_OF,
                'quantity': random.randint(1, 4)
            })

    all_rels = rels_c_t1 + rels_t1_t2 + rels_t2_t3 + rels_sup_comp + rels_comp_prod
    df_rels = pd.DataFrame(all_rels)
    df_rels.to_csv(OUTPUT_DIR / "relationships.csv", index=False)
    return df_rels

if __name__ == "__main__":
    df_c, df_p = generate_countries_ports()
    df_comp, df_sup = generate_companies_suppliers(df_c)
    df_prod, df_parts = generate_products_components()
    generate_relationships(df_comp, df_sup, df_prod, df_parts)
    print("Data Generation Complete.")
