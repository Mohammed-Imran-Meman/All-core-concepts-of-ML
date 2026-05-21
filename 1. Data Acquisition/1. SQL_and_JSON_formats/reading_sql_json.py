import pandas as pd
import mysql.connector

conn = mysql.connector.connect(host = 'localhost',user = 'root', password = '', database = 'world')
df = pd.read_json("train.json")
df2 = pd.read_sql_query("SELECT * FROM CITY WHERE CountryCode LIKE 'IND'", conn)