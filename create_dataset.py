import spotipy
import sys
import urllib.request
import csv
import sqlite3
import numpy as np

## SPOTIFY.DB
# 'album_name' = TEXT
# 'imageURL' = TEXT
# 'danceability' = TEXT
# 'energy' = TEXT
# 'speechiness' = TEXT
# 'acousticness' = TEXT
# 'liveness' = TEXT
# 'imageLoc' = TEXT
# 'instrumentalness' = TEXT
# 'tempo' = TEXT
## 

def ave_feat(album_id):
    tracks = sp.album(album_id)['tracks']['items']
    ids = [x['id'] for x in tracks]
    af = sp.audio_features(ids)
    if af == None:
        return None, None, None, None, None, None, None
    d = get_ave_feat(af, 'danceability')
    e = get_ave_feat(af, 'energy')
    s = get_ave_feat(af, 'speechiness')
    a = get_ave_feat(af, 'acousticness')
    l = get_ave_feat(af, 'liveness')
    i = get_ave_feat(af, 'instrumentalness')
    t = get_ave_feat(af, 'tempo')
    return d, e, s, a, l, i, t

def get_ave_feat(features_list, feature):
    feat_lst = []
    for x in features_list:
        if x is not None:
            feat = x[feature]
            if feat is not None:
                feat_lst.append(feat)
    if not feat_lst:
        return None
    else:
        return np.average(feat_lst)


from spotipy.oauth2 import SpotifyClientCredentials
client_credentials_manager = SpotifyClientCredentials(client_id='INSERT ID', client_secret='INSERET SECRET')
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

database_dir = "C:/Users/Alex/Documents/PhD/spotify-dataset2"
conn = sqlite3.connect("C:/Users/Alex/Documents/PhD/spotify-dataset2/spotify.db")
c = conn.cursor()
num_songs_per_genre = 10000
genres = ['o', 'p', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p']
album_dict = {}
counter = 0
with open('dataset-links.tsv', 'w') as tsvfile:
    writer = csv.writer(tsvfile, delimiter='\t', lineterminator='\n')
    for genre in genres:
        print(genre)
        off = 0
        urls = {}
        more_albums = True
        while more_albums:
            try:
                results = sp.search(q=genre, type='track', limit=50, offset=off)['tracks']['items']
            except Exception as e:
                print(e)
                more_albums = False
            if more_albums:
                off += 50
                for item in results:
                    album = item['album']
                    a_name = album['name']
                    if a_name not in album_dict:
                        album_dict[a_name] = counter
                        d, e, s, a, l , i, t = ave_feat(album['id'])
                        if (d != None):
                            print(genre, counter)
                            counter += 1
                            image_indx = 2
                            if len(album['images']) < 2:
                                image_indx = 0
                            try:
                                url = album['images'][image_indx]['url']
                                loc = database_dir  + '/' + str(counter%10000) + '/' + a_name + '.jpg' 
                                sqcomm = (a_name, url, d, e, s, a, l, loc, i, t)
                                c.execute("REPLACE INTO spotify VALUES (?, ?, ?, ?, ?, ?, ?, ? ,? ,?)", sqcomm)
                                conn.commit()
                            except Exception as e:
                                print(e)
