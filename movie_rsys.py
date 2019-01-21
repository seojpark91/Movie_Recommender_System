import pandas as pd
import numpy as np
from scipy import spatial

class MovieRecommendSys:
    
    def __init__(self):
        metadata_df = pd.read_csv("movie_metadata.csv")
        rating_df = pd.read_csv("movie_rating.csv")
        links_df = pd.read_csv("movie_links.csv")
        
        self.rating_df = rating_df
        self.links_df = links_df
        self.metadata_df = metadata_df
        self.user_df = self.make_user_based_df()
        self.sm_df = None
        self.pred_df = None
        self.recommend = None

    def make_user_based_df(self, user_limit=100, movie_limit=100):
        
        user_counts_df = self.rating_df.groupby('userId').size().reset_index(name = 'user_rating_count')
        user_counts_df = user_counts_df.sort_values(by=["user_rating_count"], ascending=False)
        
        filtered_userId = user_counts_df[user_counts_df['user_rating_count'] > user_limit]["userId"]
        filtered_userId = list(filtered_userId)
        
        movie_counts_df = self.rating_df.groupby("movieId").size().reset_index(name="movie_rating_count")
        movie_counts_df = movie_counts_df.sort_values(by=["movie_rating_count"], ascending=False)
        
        filtered_movieId = movie_counts_df[movie_counts_df['movie_rating_count'] > movie_limit]["movieId"]
        filtered_movieId = list(filtered_movieId)
        
        filtered_df = self.rating_df[self.rating_df["userId"].isin(filtered_userId)]
        filtered_df = filtered_df[filtered_df["movieId"].isin(filtered_movieId)]
        
        self.user_df = filtered_df.pivot_table(values="rating", index = "userId", columns = "movieId",
                                  aggfunc=np.average, fill_value = 0, dropna=False)
        
        return self.user_df
    
    def euclidean_similarity(self, vector_1, vector_2):
        idx = vector_1.nonzero()[0]
        if len(idx) == 0:
            return 0
        vector_1, vector_2 = np.array(vector_1)[idx], np.array(vector_2)[idx]
    
        idx = vector_2.nonzero()[0]
        if len(idx) == 0:
            return 0
        vector_1, vector_2 = np.array(vector_1)[idx], np.array(vector_2)[idx]     
        return np.linalg.norm(vector_1 - vector_2)
    
    def cosine_similarity(self, vector_1, vector_2):
        idx = vector_1.nonzero()[0]
        if len(idx) == 0:
            return 0
        vector_1, vector_2 = np.array(vector_1)[idx], np.array(vector_2)[idx]
    
        idx = vector_2.nonzero()[0]
        if len(idx) == 0:
            return 0
        vector_1, vector_2 = np.array(vector_1)[idx], np.array(vector_2)[idx]       
        return 1- spatial.distance.cosine(vector_1, vector_2)
    
    def similarity_matrix(self, similarity_func):
        index = self.user_df.index   
        matrix = []
        for idx_1, value_1 in self.user_df.iterrows():
            row = []
            for idx_2, value_2 in self.user_df.iterrows(): 
                row.append(similarity_func(value_1, value_2))
            matrix.append(row)      
        self.sm_df = pd.DataFrame(matrix, columns = index, index = index)
        return self.sm_df
    
    def mean_score(self, target, closer_count): # prediction할 수 있는 matrix 만들기
        ms_df = self.sm_df.drop(target)
        ms_df = ms_df.sort_values(target, ascending=False)
        ms_df = ms_df[target][:closer_count]
        ms_df = self.user_df.loc[ms_df.index]

        self.pred_df = pd.DataFrame(columns = self.user_df.columns)
        self.pred_df.loc["user"] = self.user_df.loc[target]
        self.pred_df.loc["mean"] = ms_df.mean()
        return self.pred_df
    
    def _get_recommend_movie_ids(self, r_count=10):
        self.recommend_df = self.pred_df.T
        self.recommend_df = self.recommend_df[self.recommend_df["user"]==0] 
        self.recommend_df = self.recommend_df.sort_values("mean", ascending = False)
        return list(self.recommend_df[:r_count].index)
        
    def _id_to_movie(self, id_num):
        pd.options.display.float_format = '{:.0f}'.format
        tmdbId = self.links_df.loc[self.links_df["movieId"] == id_num]["tmdbId"].values[0]
        movie_info = self.metadata_df.loc[self.metadata_df["id"] == str(tmdbId)]
        pd.reset_option('display')
        return movie_info
    
    def make_movie_info(self):
        movie_ids = self._get_recommend_movie_ids()
        datas = []
        for movie_id in movie_ids:
            data = self._id_to_movie(movie_id).to_dict('records')[0] 
            datas.append(data)
        return pd.DataFrame(datas)
    
    def run(self, similarity_func, target, closer_count=5, r_count=5):
        self.sm_df = self.similarity_matrix(similarity_func)
        self.pred_df = self.mean_score(target, closer_count)
        movie_ids = self._get_recommend_movie_ids()
        result = self.make_movie_info()  
        return result
    
    def mae(self, value, pred):
        idx = value.nonzero()[0]
        value, pred = np.array(value)[idx], np.array(pred)[idx]
        idx = pred.nonzero()[0]
        value, pred = np.array(value)[idx], np.array(pred)[idx]
        return np.absolute(sum(value-pred)) / len(idx)
    
    def evaluate(self, df, sm_df, closer_count):
        users = df.index
        evaluate_list = []
    
        for target in users:
            self.pred_df = self.mean_score(target, closer_count)
            evaluate_list.append(self.mae(self.pred_df.loc["user"], self.pred_df.loc["mean"]))    
        return np.average(evaluate_list)