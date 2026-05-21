import pandas as pd
import requests
from bs4 import BeautifulSoup

Data = pd.DataFrame()
for i in range(1,4):
  webpage = requests.get(f'https://stackoverflow.com/questions?tab=newest&page={i}').text

  soup = BeautifulSoup(webpage, 'lxml')
  #print(soup.prettify())
  #soup.find_all('h1')[0].text

  questions = soup.find_all('div', class_='s-post-summary')

  ques = []
  desc = []
  meta_tags = []
  for j in questions:
    ques.append(j.find('a', class_='s-link').text)
    desc.append(j.find('div', class_='s-post-summary--content-excerpt').text.strip())
    # meta_tags.append(i.find('div', class_='s-post-summary--meta-tags').text.strip())
    meta_tags.append(", ".join(tag.text.strip() for tag in j.find('div', class_='s-post-summary--meta-tags').find_all('a')))



  question_dict = {
    'question' : ques,
    'Description' : desc,
    'Meta Tags' : meta_tags,
  }
  df = pd.DataFrame(question_dict)
  Data=pd.concat([Data,df])


Data.to_csv("Question_data_from_stackoverflow.csv",index=False)
Data.to_excel("Question_data_from_stackoverflow.xlsx",index=False)