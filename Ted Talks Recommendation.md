```python
import numpy as np
import pandas as pd
from sklearn.feature_extraction import text
from sklearn.metrics.pairwise import cosine_similarity
```


```python
data=pd.read_csv('C:/Users/Rakesh/Datasets/ted_talks.csv')
```


```python
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>transcript</th>
      <th>url</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Good morning. How are you?(Laughter)It's been ...</td>
      <td>https://www.ted.com/talks/ken_robinson_says_sc...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Thank you so much, Chris. And it's truly a gre...</td>
      <td>https://www.ted.com/talks/al_gore_on_averting_...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>(Music: "The Sound of Silence," Simon &amp; Garfun...</td>
      <td>https://www.ted.com/talks/david_pogue_says_sim...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>If you're here today — and I'm very happy that...</td>
      <td>https://www.ted.com/talks/majora_carter_s_tale...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>About 10 years ago, I took on the task to teac...</td>
      <td>https://www.ted.com/talks/hans_rosling_shows_t...</td>
    </tr>
  </tbody>
</table>
</div>




```python
data['title'] = data['url'].map(lambda x:x.split('/')[-1])
```


```python
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>transcript</th>
      <th>url</th>
      <th>title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Good morning. How are you?(Laughter)It's been ...</td>
      <td>https://www.ted.com/talks/ken_robinson_says_sc...</td>
      <td>ken_robinson_says_schools_kill_creativity\n</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Thank you so much, Chris. And it's truly a gre...</td>
      <td>https://www.ted.com/talks/al_gore_on_averting_...</td>
      <td>al_gore_on_averting_climate_crisis\n</td>
    </tr>
    <tr>
      <th>2</th>
      <td>(Music: "The Sound of Silence," Simon &amp; Garfun...</td>
      <td>https://www.ted.com/talks/david_pogue_says_sim...</td>
      <td>david_pogue_says_simplicity_sells\n</td>
    </tr>
    <tr>
      <th>3</th>
      <td>If you're here today — and I'm very happy that...</td>
      <td>https://www.ted.com/talks/majora_carter_s_tale...</td>
      <td>majora_carter_s_tale_of_urban_renewal\n</td>
    </tr>
    <tr>
      <th>4</th>
      <td>About 10 years ago, I took on the task to teac...</td>
      <td>https://www.ted.com/talks/hans_rosling_shows_t...</td>
      <td>hans_rosling_shows_the_best_stats_you_ve_ever_...</td>
    </tr>
  </tbody>
</table>
</div>




```python
ted_talks = data['transcript'].tolist()
bi_tfidf = text.TfidfVectorizer(input=ted_talks, stop_words='english', ngram_range=(1,2))
bi_matrix = bi_tfidf.fit_transform(ted_talks)

uni_tfidf = text.TfidfVectorizer(input=ted_talks, stop_words='english')
uni_matrix = uni_tfidf.fit_transform(ted_talks)

bi_sim=cosine_similarity(bi_matrix)
uni_sim=cosine_similarity(uni_matrix)
```


```python
def recommend_ted_talks(x):
    return ". ".join(data['title'].loc[x.argsort()[-5:-1]])

data['ted_talks_uni']=[recommend_ted_talks(x) for x in uni_sim]
data['ted_talks_bi']=[recommend_ted_talks(x) for x in bi_sim]
print(data['ted_talks_uni'].str.replace("_"," ").str.upper().str.strip().str.split('\n')[1])
```

    ['RORY BREMNER S ONE MAN WORLD SUMMIT', '. ALICE BOWS LARKIN WE RE TOO LATE TO PREVENT CLIMATE CHANGE HERE S HOW WE ADAPT', '. TED HALSTEAD A CLIMATE SOLUTION WHERE ALL SIDES CAN WIN', '. AL GORE S NEW THINKING ON THE CLIMATE CRISIS']
    
