import pandas as pd
import io
import requests

def get_dataset(key):
    urls = {
        "one-source":"https://figshare.com/ndownloader/files/26648885",
        "hoax":"https://figshare.com/ndownloader/files/26648861"
    }
    response = requests.get(urls[key], allow_redirects=True)
    df = pd.read_csv(io.StringIO(response.content.decode('utf-8')))
    return df

