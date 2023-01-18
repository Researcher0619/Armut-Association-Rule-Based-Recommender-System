#!pip install mlxtend
import pandas as pd
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
# çıktının tek bir satırda olmasını sağlar.
pd.set_option('display.expand_frame_repr', False)
from mlxtend.frequent_patterns import apriori, association_rules

df = pd.read_csv("Datasets/armut_data.csv")
df_ = df.copy()
df.head(10)
df.columns
df.shape
df.describe().T
df.isnull().sum()
df.info()
df.nunique()

df["Hizmet"] = df["ServiceId"].astype(str) + "_" + df["CategoryId"].astype(str)
df["Hizmet"].head()
df.nunique()
df.head()

df["New_Date"]=pd.DatetimeIndex(df['CreateDate']).year.astype(str)+"-"+pd.DatetimeIndex(df['CreateDate']).month.astype(str)
df["SepetID"] = df['UserId'].astype(str) +"_"+ df["New_Date"].astype(str)

df.head()
df_recommend = df.groupby(["SepetID","Hizmet"]).agg("Hizmet").sum().unstack().fillna(0).applymap(lambda x: 1 if x != 0 else 0)

# Adım 2: Birliktelik kurallarını oluşturunuz.

frequent_itemsets = apriori(df_recommend,
                            min_support=0.01,
                            use_colnames=True)

rules = association_rules(frequent_itemsets,
                          metric="support",
                          min_threshold=0.01)
rules.head()
rules[(rules["support"] > 0.01) & (rules["confidence"] > 0.01) & (rules["lift"] > 1)]. \
sort_values("confidence", ascending=False)
# Confidence, lift değerlerini bulmak için yukarıda apriori ile bulduğumuz support değerlerini kullanarak association rule uyguladık.
rules = association_rules(frequent_itemsets, metric='support', min_threshold=0.01)
rules.sort_values("support", ascending=False).head(50)

### Adım3: arl_recommender fonksiyonunu kullanarak son 1 ay içerisinde 2_0 hizmetini alan bir kullanıcıya hizmet önerisinde bulununuz.
# Recommendation için script yazalım.


def arl_recommender(rules_df, hizmet, rec_count=1):
    print(rules_df, hizmet, rec_count)
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    recommendations = []
    # print(sorted_rules)
    for k, product in sorted_rules["antecedents"].items():
        # print('k-proudct',k,"-",product)
        for j in list(product):
            # print('j == hizmet',j,hizmet,j == hizmet)
            if j[1] == hizmet:
                recommendations.append(list(sorted_rules.iloc[k]["consequents"]))

    recommendations = list(dict.fromkeys({item for item_list in recommendations for item in item_list})) # Tekrar eden itemleri tekilleştirme.
    return recommendations[:rec_count]

arl_recommender(rules, '15_1',3)
df.head()

