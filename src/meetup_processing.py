import json
import datetime
import pandas as pd
from data_getters.meetup import select_meetup
import pickle

def parse_topics(group_id, topics):
    topics = json.loads(topics)
    df = pd.DataFrame(topics)
    df['group_id'] = group_id
    return df

def milliseconds2datetime(input_time):
    """Transform time from milliseconds to Year-Month-Day."""
    input_time = input_time / 1000.0
    return datetime.datetime.fromtimestamp(input_time).strftime('%Y-%m-%d')

def keep_year(input_time):
    return input_time[:4]

def flatten_lists(lst):
    """Remove nested lists."""
    return [item for sublist in lst for item in sublist]

def read_member_data(category, african_country_code, database):
    m = select_meetup(("../innovation-mapping.config"), category, african_country_code)
    members = m[database].copy()
    if members.shape[0] > 0:
        return members
    else:
        return None

def read_group_data(category, african_country_code, database):
    dfs = select_meetup(("../innovation-mapping.config"), category, african_country_code)

    groups = dfs[database].copy()

    if groups.shape[0] > 0:

        groups.rename(index=str, inplace=True, columns={'id':'group_id', 'name':'group_name'})
        groups.set_index('group_id', inplace=True)
        groups['country_code'] = african_country_code
        groups['category'] = category
        groups['db'] = database
        # Find creation date and keep years
        groups['created_at'] = groups.created.apply(milliseconds2datetime)
        groups['year'] = groups.created_at.apply(keep_year)

        # Parse every group's topics and create a dataframe
        topics = pd.concat([parse_topics(idx, row['topics']) for idx, row in groups.iterrows()])

        # Group by group_id and aggregate values
        topics['id'] = topics['id'].astype(str)
        grouped_topics = topics.groupby('group_id').agg(lambda col: ', '.join(col))

        # Group groups with topics
        groups = groups.merge(grouped_topics, left_index=True, right_index=True)

    return groups

def main():
    african_country_codes = ['DZ', 'AO', 'SH', 'BJ', 'BW', 'BF', 'BI', 'CM', 'CV', 'CF', 'TD', 'KM', 'CG', 'CD', 'DJ', 'EG', 'GQ',
                         'ER', 'SZ', 'ET', 'GA', 'GM', 'GH', 'GN', 'GW', 'CI', 'KE', 'LS', 'LR', 'LY', 'MG', 'MW', 'ML', 'MR',
                         'MU', 'YT', 'MA', 'MZ', 'NA', 'NE', 'NG', 'ST', 'RE' 'RW', 'ST', 'SN', 'SC', 'SL', 'SO', 'ZA', 'SH',
                         'SD', 'SZ', 'TZ', 'TG', 'TN', 'UG', 'CD', 'ZM', 'TZ', 'ZW']
    categories = [34, 2]

    data = []
    meetup_users = []
    for category in categories:
        for african_country_code in african_country_codes:
            for db in ['core_members_groups', 'extended_members_groups']:
                print('Members -- {} -- {} -- {}'.format(category, african_country_code, db))
                meetup_users.append(read_member_data(category, african_country_code, db))

            for db in ['core_groups', 'extended_groups']:
                print('Groups -- {} -- {} -- {}'.format(category, african_country_code, db))
                data.append(read_group_data(category, african_country_code, db))

    with open('../data/raw/data.pickle', 'wb') as h:
        pickle.dump(data, h)

    with open('../data/raw/meetup_users.pickle', 'wb') as h:
        pickle.dump(meetup_users, h)

if __name__ == '__main__':
    main()
