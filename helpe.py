import pandas as pd


def contains_single_letter(name):
    words = name.split()
    for word in words:
        if len(word) == 1 and word.isalpha():
            return 1
    return 0

company_keywords = [
    "LLC", "INC", "CORP", "CO", "LTD", "LP", "PLC", "GROUP", "INVESTORS", "Equity", "Lending", "loan",
    "PARTNERS", "HOLDINGS", "ENTERPRISES", "VENTURES", "INVESTMENTS", "trust", "finance", "ADMINISTRATOR",
    "CAPITAL", "FUND", "PROPERTIES", "MANAGEMENT", "SERVICES", "MORTGAGE", "funding", "foundation", "network",
    "TECHNOLOGIES", "SOLUTIONS", "INDUSTRIES", "INTERNATIONAL", "City",
    "DEVELOPMENT", "CONSULTING", "TRADING", "SYSTEMS", "ASSOCIATES", 
    "RESOURCES", "FINANCIAL", "REALTY", "DISTRIBUTION", "LOGISTICS", 
    "PRODUCTS", "COMMUNICATIONS", "MEDIA", "NETWORKS", "INNOVATIONS", 
    "MARKETING", "DESIGN", "HOLDINGS", "HOSPITALITY", "ENERGY", 
    "CONSTRUCTION", "RETAIL", "AUTOMOTIVE", "PHARMA", "BIOTECH", 
    "HEALTHCARE", "EDUCATION", "RESEARCH", "LABS", "PLAN"]

def contains_keyword(company_name):
    for keyword in company_keywords:
        if " " + keyword.lower() in company_name.lower():
            return 1
        if keyword.lower() + " " in company_name.lower():
            return 1
    return 0

def target(df):
    if df['contains_keyword'] == 1:
        return 0
    elif df["digit_count"] > 0: 
        return 0
    elif df['two_words'] > 0:
        return 0
    else:
        return 1
    

def parse(df):
    df['two_words'] = df['Lender'].apply(lambda x: 0 if len(x.split()) == 2 or len(x.split()) == 3 else 1)
    df['contains_keyword'] = df['Lender'].apply(contains_keyword)
    df['char_count'] = df['Lender'].apply(len)
    df['digit_count'] = df['Lender'].apply(lambda x: sum(c.isdigit() for c in x))
    df['special_char_count'] = df['Lender'].apply(lambda x: sum(c in '-&.' for c in x))
    #df['uppercase_word_count'] = df['Lender'].apply(lambda x: sum(word.isupper() for word in x.split()))
    df['title_case_word_count'] = df['Lender'].apply(lambda x: sum(word.istitle() for word in x.split()))
    df['avg_word_length'] = df['Lender'].apply(lambda x: sum(len(word) for word in x.split()) / len(x.split()))
    df['contains_single_letter'] = df['Lender'].apply(contains_single_letter)
    return df

features = ['two_words', 'contains_keyword', 'char_count', 'digit_count', 'special_char_count', 
            'title_case_word_count', 'avg_word_length', 'contains_single_letter']

def additional(df):
    return df[['two_words', 'contains_keyword', 'char_count', 'digit_count', 
               'special_char_count', 'title_case_word_count', 
                'avg_word_length', 'contains_single_letter']].values

def preprocess_name(name):
    # Create a DataFrame from the single name
    df = pd.DataFrame([{'Lender': name}])
    df = parse(df)  # Assuming this function processes a single name
    df[features] = df[features].apply(pd.to_numeric, errors='coerce')
    df.fillna(0, inplace=True)  # Handle any missing values by filling with 0
    return df[features]