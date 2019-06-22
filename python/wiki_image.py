import pandas as pd
import requests

from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import re
import unidecode

def clean_state(state):
    m = re.sub("( [Ss]pecial)*( [Ee]lection){1}", "",state)
    return(m)

def clean_cand(name):
    m = name.rstrip()
    m = unidecode.unidecode(m)
    m = re.sub('[^A-Za-z]', "_", m)
    m = re.sub('[^A-Za-z]$', "", m)
    return(m)

def wikimg_filter(img_link):
    '''Determines if wiki img is a portrait.'''
    if 'svg' in img_link or 'static/images' in img_link:
        return False
    elif 'upload.wikimedia.org' not in img_link:
        return False
    return True

def extract_senate_links(url_senate):
    '''Extracts links to senator candidates wiki pages.'''
    html = requests.get(url_senate).text
    soup = BeautifulSoup(html, 'html.parser')
    state_sums = soup.select('.infobox') # box contains summaries

    senate_cands = {
        'state': [],
        'party': [],
        'cand_name': [],
        'wiki_page': []
    }

    for state_sum in state_sums:
        state_name = state_sum.find('caption').text
        state_name = clean_state(state_name)
        box = state_sum.find_all('tr')[3] # 3rd table row contains names
        candidates = box.find_all('tr')[1:3] # name & links + party affilications

        for idx, cand in enumerate(candidates):
            link_tags = cand.find_all('td')
            if idx == 0: # names
                a_tags = [l.find('a') for l in link_tags]
                text = [clean_cand(l.text) for l in link_tags]
                links = ["" if a is None else f"https://en.wikipedia.org{a['href']}" for a in a_tags]
                senate_cands['state'] += [state_name] * len(text)
                senate_cands['cand_name'] += text
                senate_cands['wiki_page'] += links
            elif idx == 1: # party affiliations
                party = [l.text.rstrip() for l in link_tags]
                if len(party) < 3: # expecting 3 party values
                    party = party + ['']
                senate_cands['party'] += party

    return senate_cands

def img_download(img_url, file_name):
    '''Downloads image and saves specified by file name.'''
    img_data = None
    try:
        img_data = requests.get(img_url).content 
    except (ConnectionError) as e:
        print(f'--- {img_url} had {e}')
    except Exception as e:
        print(f'--- {img_url} had {e}')

    if img_data != None:
        with open(file_name, 'wb+') as handler:
            handler.write(img_data)
    print(f'Finished img download: {file_name}')

def get_wiki_portraits(wiki_link, name, img_dir):
    '''Downloads all portrait images from 1 wiki page into specified file location.
    
    Args:
        wiki_link: URI of wiki page.
        name: Name of page, to be used to name downloaded image.
        img_dir: Directory to save images into.
    
    Returns:
        A list of tuples (image link, file name) that images 
        originated from and downloaded as.
    '''
    html = requests.get(wiki_link).text
    soup = BeautifulSoup(html, 'html.parser')
    
    imgs = soup.select('img')
    imgs_links = [img['src'] for img in imgs if wikimg_filter(img['src'])]
    img_fps = []

    for idx, img_link in enumerate(imgs_links):
        full_link = f"https:{img_link}"
        img_name = f"{name}{idx:02}.jpg" # filename of img
        img_fp = f"{img_dir}{name}{idx:02}.jpg" # filepath to save img
        # print(img_fp, full_link)
        img_download(full_link, img_fp)
        img_fps.append((full_link, img_name))

    return img_fps


def get_all_candidates(cand_df, imgs_dir):
    '''Downloads all potrait image from all wiki pages into directory.
    
    Args:
        cand_df: Pandas df w/ headers [state, party, cand_name, wiki_page].
        img_dir: Directory to save images into.
    
    Returns:
        A pandas df where each row is a single portrait image/wiki
        image link. Each row contains the meta data from the input
        dataframe. 
    '''
    cand_df = cand_df.dropna()
    cands_meta = cand_df.to_dict("records")

    state, party, cand_name = [], [], []
    wiki_page, img_link, my_image = [], [], []
    for cand_meta in cands_meta:
        img_fps = get_wiki_portraits(
            cand_meta['wiki_page'],
            cand_meta['cand_name'],
            imgs_dir
        )

        for (full_link, img_fp) in img_fps:
            state.append(cand_meta['state'])
            party.append(cand_meta['party'])
            cand_name.append(cand_meta['cand_name'])
            wiki_page.append(cand_meta['wiki_page'])
            img_link.append(full_link)
            my_image.append(img_fp)

    return pd.DataFrame.from_dict({
        'state': state,
        'party': party,
        'cand_name': cand_name,
        'wiki_page': wiki_page,
        'img_link': img_link,
        'img_fp': my_image,
    })


def main():
    url = u'https://en.wikipedia.org/wiki/2018_United_States_Senate_elections'
    senate_df_fp = "./data/candidate-images/wiki-senate/senate_candidates.csv"
    senate_img_fp = "./data/candidate-images/wiki-senate/senate_images.csv"
    imgs_dir = "./data/candidate-images/wiki-senate/"

    senate_dict = extract_senate_links(url)
    df = pd.DataFrame.from_dict(senate_dict)
    df = df.replace(r'^\s*$', np.nan, regex=True)
    df.to_csv(senate_df_fp,index=False)
    # [state, party, cand_name, wiki_page, img_link, img_fp]
    
    df_senate_images = get_all_candidates(df, imgs_dir)
    df_senate_images.to_csv(senate_img_fp, index=False)

    
if __name__ == "__main__":
    main()
