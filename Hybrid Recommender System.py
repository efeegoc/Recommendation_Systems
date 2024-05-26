
#############################################
# PROJE: Hybrid Recommender System
#############################################

# ID'si verilen kullanıcı için item-based ve user-based recomennder yöntemlerini kullanarak tahmin yapalım.
# 5 öneri user-based modelden 5 öneri de item-based modelden ele alalım ve nihai olarak 10 öneriyi 2 modelden yapalım.

#############################################
# Verinin Hazırlanması
#############################################

import pandas as pd
pd.pandas.set_option('display.max_columns', None)
pd.pandas.set_option('display.width', 300)

# Movie ve Rating veri setlerini okutalım.
# movieId, film adı ve filmin tür bilgilerini içeren veri seti
dosya_yolu = r'C:\Users\ExtraBT\Desktop\MASAUSTU\Yazılım\Python\Recommendation Systems\movie.csv'
movie = pd.read_csv(dosya_yolu)
movie.head()
movie.shape

# UserID, film adı, filme verilen oy ve zaman bilgisini içeren veri seti
dosya_yolu = r'C:\Users\ExtraBT\Desktop\MASAUSTU\Yazılım\Python\Recommendation Systems\rating.csv'
rating = pd.read_csv(dosya_yolu)
rating.head()
rating.shape
rating["userId"].nunique()


# Rating veri setine filmlerin isimlerini ve türünü movie film setini kullanrak ekleyelim.
# Ratingdeki kullanıcıların oy kullandıkları filmlerin sadece id'si var.
# Idlere ait film isimlerini ve türünü movie veri setinden ekliyoruz.
df = movie.merge(rating, how="left", on="movieId")
df.head()
df.shape


# Herbir film için toplam kaç kişinin oy kullandığını hesaplayalım.Toplam oy kullanılma sayısı 1000'un altında olan filmleri veri setinden çıkaralım.
# Herbir film için toplam kaç kişinin oy kullanıldığını hesaplıyoruz.
comment_counts = pd.DataFrame(df["title"].value_counts())
comment_counts
print(comment_counts.columns)

# Toplam oy kullanılma sayısı 1000'in altında olan filmlerin isimlerini rare_movies de tutuyoruz.
# Ve veri setinden çıkartıyoruz
rare_movies = comment_counts[comment_counts["title"] <= 1000].index
common_movies = df[~df["title"].isin(rare_movies)]
common_movies.shape


# index'te userID'lerin sutunlarda film isimlerinin ve değer olarakta ratinglerin bulunduğu
# dataframe için pivot table oluşturalım.

user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
user_movie_df.head()


# Yukarıda yapılan tüm işlemleri fonksiyonlaştıralım
def create_user_movie_df():
    import pandas as pd
    dosya_yolu = r'C:\Users\ExtraBT\Desktop\MASAUSTU\Yazılım\Python\Recommendation Systems\movie.csv'
    movie = pd.read_csv(dosya_yolu)
    dosya_yolu = r'C:\Users\ExtraBT\Desktop\MASAUSTU\Yazılım\Python\Recommendation Systems\rating.csv'
    rating = pd.read_csv(dosya_yolu)
    df = movie.merge(rating, how="left", on="movieId")
    comment_counts = pd.DataFrame(df["title"].value_counts())
    rare_movies = comment_counts[comment_counts["title"] <= 1000].index
    common_movies = df[~df["title"].isin(rare_movies)]
    user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
    return user_movie_df

user_movie_df = create_user_movie_df()


#############################################
# Öneri Yapılacak Kullanıcının İzlediği Filmlerin Belirlenmesi
#############################################

# Rastgele bir kullanıcı id'si seçelim.
random_user = 108170

# Seçilen kullanıcıya ait gözlem birimlerinden oluşan random_user_df adında yeni bir dataframe oluşturalım.
random_user_df = user_movie_df[user_movie_df.index == random_user]
random_user_df.head()
random_user_df.shape

# Seçilen kullanıcının oy kullandığı filmleri movies_watched adında bir listeye atayalım.
movies_watched = random_user_df.columns[random_user_df.notna().any()].to_list()
movies_watched

movie.columns[movie.notna().any()].to_list()

#############################################
# Aynı Filmleri İzleyen Diğer Kullanıcıların Verisine ve Id'lerine Erişmek
#############################################

# Seçilen kullanıcının izlediği fimlere ait sutunları user_movie_df'ten seçelim ve movies_watched_df adında yeni bir dataframe oluşturuyoruz.
movies_watched_df = user_movie_df[movies_watched]
movies_watched_df.head()
movies_watched_df.shape

# Herbir kullancının seçili user'in izlediği filmlerin kaçını izlediği bilgisini taşıyan user_movie_count adında yeni bir dataframe oluşturalım.
# Ve yeni bir df oluşturuyoruz.
user_movie_count = movies_watched_df.T.notnull().sum()
user_movie_count = user_movie_count.reset_index()
user_movie_count.columns = ["userId", "movie_count"]
user_movie_count.head(5)

# Seçilen kullanıcının oy verdiği filmlerin yüzde 60 ve üstünü izleyenleri benzer kullanıcılar olarak görüyoruz.
# Bu kullanıcıların id’lerinden users_same_movies adında bir liste oluşturalım.
perc = len(movies_watched) * 60 / 100
users_same_movies = user_movie_count[user_movie_count["movie_count"] > perc]["userId"]
len(users_same_movies)


#############################################
# Öneri Yapılacak Kullanıcı ile En Benzer Kullanıcıların Belirlenmesi
#############################################

# user_same_movies listesi içerisindeki seçili user ile benzerlik gösteren kullanıcıların id’lerinin bulunacağı şekilde movies_watched_df dataframe’ini filtreleyelim.
final_df = movies_watched_df[movies_watched_df.index.isin(users_same_movies)]
final_df.head()
final_df.shape

# Kullanıcıların birbirleri ile olan korelasyonlarının bulunacağı yeni bir corr_df dataframe’i oluşturalım.
corr_df = final_df.T.corr().unstack().sort_values()
corr_df = pd.DataFrame(corr_df, columns=["corr"])
corr_df.index.names = ['user_id_1', 'user_id_2']
corr_df = corr_df.reset_index()

#corr_df[corr_df["user_id_1"] == random_user]



# Seçili kullanıcı ile yüksek korelasyona sahip (0.65’in üzerinde olan) kullanıcıları filtreleyerek top_users adında yeni bir dataframe oluşturalım.
top_users = corr_df[(corr_df["user_id_1"] == random_user) & (corr_df["corr"] >= 0.65)][["user_id_2", "corr"]].reset_index(drop=True)
top_users = top_users.sort_values(by='corr', ascending=False)
top_users.rename(columns={"user_id_2": "userId"}, inplace=True)
top_users.shape
top_users.head()

# top_users dataframe’ine rating veri seti ile merge edelim
top_users_ratings = top_users.merge(rating[["userId", "movieId", "rating"]], how='inner')
top_users_ratings = top_users_ratings[top_users_ratings["userId"] != random_user]
top_users_ratings["userId"].unique()
top_users_ratings.head()

#############################################
# Sadece korelasyona yönelik verilen tavsiyeler yeterli olmayacağı için Score tutuyoruz
# Çünkü kişinin korelasyonu fazla olup verdiği reyting düşük olabilir yada korelasyonu düşük olup verdiği reyting yüksek olabilir
# Weighted Average Recommendation Score'un Hesaplanması ve İlk 5 Filmin Tutulması
#############################################

# Her bir kullanıcının corr ve rating değerlerinin çarpımından oluşan weighted_rating adında yeni bir değişken oluşturalım.
top_users_ratings['weighted_rating'] = top_users_ratings['corr'] * top_users_ratings['rating']
top_users_ratings.head()

# Film id’sive her bir filme ait tüm kullanıcıların weighted rating’lerinin ortalama değerini içeren recommendation_df adında yeni bir
# dataframe oluşturalım.
recommendation_df = top_users_ratings.groupby('movieId').agg({"weighted_rating": "mean"})
recommendation_df = recommendation_df.reset_index()
recommendation_df.head()

# recommendation_df içerisinde weighted rating'i 3.5'ten büyük olan filmleri seçelim ve weighted rating’e göre sıralayalım.
# İlk 5 gözlemi movies_to_be_recommend olarak kaydedelim.
recommendation_df[recommendation_df["weighted_rating"] > 3.5]
movies_to_be_recommend = recommendation_df[recommendation_df["weighted_rating"] > 3.5].sort_values("weighted_rating", ascending=False)


# Tavsiye edilen 5 filmin isimlerini getirelim.
movies_to_be_recommend.merge(movie[["movieId", "title"]])["title"][:5]

# 0    Mystery Science Theater 3000: The Movie (1996)
# 1                               Natural, The (1984)
# 2                             Super Troopers (2001)
# 3                         Christmas Story, A (1983)
# 4                       Sonatine (Sonachine) (1993)

#############################################
#############################################
# Item-Based Recommendation
#############################################
#############################################

# Kullanıcının en son izlediği ve en yüksek puan verdiği filmin adına göre item-based öneri yapalım.
user = 108170

# movie,rating veri setlerini okutalım.
movie = pd.read_csv('Modül_4_Tavsiye_Sistemleri/datasets/movie.csv')
rating = pd.read_csv('datasets/movie_lens_dataset/rating.csv')

# Öneri yapılacak kullanıcının 5 puan verdiği filmlerden puanı en güncel olan filmin id'sinin alalım.
movie_id = rating[(rating["userId"] == user) & (rating["rating"] == 5.0)].sort_values(by="timestamp", ascending=False)["movieId"][0:1].values[0]

# User based recommendation bölümünde oluşturulan user_movie_df dataframe’ini seçilen film id’sine göre filtreleyelim.
movie[movie["movieId"] == movie_id]["title"].values[0]
movie_df = user_movie_df[movie[movie["movieId"] == movie_id]["title"].values[0]]
movie_df

# Filtrelenen dataframe’i kullanarak seçili filmle diğer filmlerin korelasyonunu bulalım ve sıralayalım.
user_movie_df.corrwith(movie_df).sort_values(ascending=False).head(10)

# Son iki adımı uygulayan fonksiyon
def item_based_recommender(movie_name, user_movie_df):
    movie = user_movie_df[movie_name]
    return user_movie_df.corrwith(movie).sort_values(ascending=False).head(10)

# Seçili film’in kendisi haricinde ilk 5 film’I öneri olarak verelim.
movies_from_item_based = item_based_recommender(movie[movie["movieId"] == movie_id]["title"].values[0], user_movie_df)
# 1'den 6'ya kadar. 0'da filmin kendisi var. Onu dışarda bıraktık.
movies_from_item_based[1:6].index

# 'My Science Project (1985)',
# 'Mediterraneo (1991)',
# 'Old Man and the Sea,
# The (1958)',
# 'National Lampoon's Senior Trip (1995)',
# 'Clockwatchers (1997)']



