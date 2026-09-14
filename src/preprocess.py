import pandas as pd
import re


def group_rare(df, feature, threshold=10000):
    counts = df[feature].value_counts()
    rare = counts[counts < threshold].index.tolist()
    return df[feature].replace(rare, 'Other')


def extract_car_info(url):
    if not isinstance(url, str):
        return {'brand': None, 'model': None}

    pattern = r'/cars/all/([^/]+)/([^/]+)'
    match = re.search(pattern, url)
    if match:
        return {'brand': match.group(1), 'model': match.group(2).split('?')[0].split('#')[0]}

    pattern_simple = r'/cars/([^/?]+)'
    match = re.search(pattern_simple, url)
    if match:
        brand = match.group(1)
        if brand and brand not in ['all', '']:
            return {'brand': brand, 'model': None}

    return {'brand': None, 'model': None}


def preprocess_hits(hits_data: list, targets: list = None) -> pd.DataFrame:
    if not hits_data:
        return pd.DataFrame([{
            'total_events_before': 0,
            'unique_brands_before': 0,
            'car_detail_views_before': 0
        }])

    df_hits = pd.DataFrame(hits_data)

    car_info = df_hits['hit_page_path'].apply(extract_car_info)
    df_hits['car_brand'] = car_info.apply(lambda x: x['brand'])

    if targets is None:
        targets = []

    df_hits['is_target'] = df_hits['event_action'].isin(targets)

    first_target = df_hits[df_hits['is_target']].groupby('session_id')['hit_number'].min().rename('first_target_hit')
    df_hits_with_first = df_hits.merge(first_target, on='session_id', how='left')

    df_hits_before = df_hits_with_first[
        (df_hits_with_first['first_target_hit'].isna()) |
        (df_hits_with_first['hit_number'] < df_hits_with_first['first_target_hit'])
    ]

    df_hits_before['is_car_detail'] = df_hits_before['hit_page_path'].str.contains('/cars/all/', regex=False).astype(int)

    return pd.DataFrame([{
        'total_events_before': len(df_hits_before),
        'unique_brands_before': df_hits_before['car_brand'].nunique(),
        'car_detail_views_before': df_hits_before['is_car_detail'].sum()
    }])


def preprocess_session(data: dict) -> pd.DataFrame:
    df = pd.DataFrame([data])

    df['device_brand'] = df['device_brand'].fillna('unknow')
    df['device_brand'] = df['device_brand'].replace('(not set)', 'Not Set')
    df['device_brand'] = group_rare(df, 'device_brand', threshold=9964)

    if 'utm_adcontent' in df.columns:
        df['utm_adcontent'] = group_rare(df, 'utm_adcontent', threshold=100000)
        df['utm_adcontent'] = df['utm_adcontent'].fillna('unknow')

    if 'utm_campaign' in df.columns:
        df['utm_campaign'] = df['utm_campaign'].fillna('unknow')
        df['utm_campaign'] = group_rare(df, 'utm_campaign', threshold=200000)

    df['utm_keyword'] = df['utm_keyword'].notna().astype(int)

    def group_device_os(os_name):
        if pd.isna(os_name) or str(os_name) == '(not set)':
            return 'unknown'
        mobile_os = ['Android', 'iOS', 'Samsung', 'Windows Phone', 'Firefox OS', 'Tizen', 'BlackBerry', 'Nokia']
        desktop_os = ['Windows', 'Macintosh', 'Linux', 'Chrome OS']
        if os_name in mobile_os:
            return 'mobile'
        elif os_name in desktop_os:
            return 'desktop'
        return 'unknown'

    df['device_os'] = df['device_os'].apply(group_device_os)

    df['visit_date'] = pd.to_datetime(df['visit_date'])
    df['weekday'] = df['visit_date'].dt.weekday
    df['month'] = df['visit_date'].dt.month

    def get_season(month):
        if month in [12, 1, 2]:
            return 'winter'
        elif month in [3, 4, 5]:
            return 'spring'
        elif month in [6, 7, 8]:
            return 'summer'
        return 'autumn'

    df['season'] = df['month'].apply(get_season)

    df['visit_time'] = pd.to_datetime(df['visit_time'], format='%H:%M:%S')
    df['hour'] = df['visit_time'].dt.hour

    def get_time_period(hour):
        if 6 <= hour < 12:
            return 'morning'
        elif 12 <= hour < 18:
            return 'afternoon'
        elif 18 <= hour < 24:
            return 'evening'
        return 'night'

    df['time_period'] = df['hour'].apply(get_time_period)

    df['visit_number_group'] = df['visit_number'].apply(lambda x: x if x <= 4 else 5)

    social_sources = [
        'QxAxdyPLuQMEcrdZWdWb', 'MvfHsxITijuriZxsqZqt', 'ISrKoXQCxqqYvAZICvjs',
        'IZEXUFLARCUMynmHNBGo', 'PlbkrSYoHuZBWfYjYnfw', 'gVRrcxiDQubJiljoTbGm'
    ]
    df['is_social'] = df['utm_source'].isin(social_sources).astype(int)

    organic_mediums = ['organic', 'referral', '(none)']
    df['traffic_type'] = df['utm_medium'].apply(lambda x: 'organic' if x in organic_mediums else 'paid')

    cols_to_drop = [
        'session_id', 'client_id', 'visit_date', 'visit_time', 'visit_number',
        'utm_source', 'utm_medium', 'device_model', 'device_screen_resolution',
        'device_browser', 'geo_country', 'geo_city', 'month', 'hour'
    ]
    for col in cols_to_drop:
        if col in df.columns:
            df = df.drop(col, axis=1)

    return df


def preprocess_input(session_data: dict, hits_data: list = None, targets: list = None) -> pd.DataFrame:
    df_session = preprocess_session(session_data)
    df_hits = preprocess_hits(hits_data or [], targets)

    result = pd.concat([df_session, df_hits], axis=1)

    features = [
        'visit_number_group', 'is_social', 'traffic_type', 'device_category',
        'device_os', 'device_brand', 'utm_adcontent', 'utm_campaign',
        'utm_keyword', 'weekday', 'season', 'time_period',
        'total_events_before', 'unique_brands_before', 'car_detail_views_before'
    ]

    for col in features:
        if col not in result.columns:
            result[col] = 0

    return result[features]