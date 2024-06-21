from typing import Tuple
from utils.bgg_api_client import BGGClient
from utils.comment_cleaning import clean_comments
from utils.aspect_extraction import aspect_extraction_for_game
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import torch
import os
import sys
sys.path.append('../')

## Download and prepare data
def prepapre_data(top_n: int | None = 2) -> None:
    """
    Main function to download and process comments of top n games from Board Game Geek API.
    
    Args:
    
    top_n : Optional[int]
        The number of top ranked games for which comments should be downloaded and processed, defaults to 2 (top 2 games). If set as None it will process up to the top 100 games.
    
    Returns:
        None
        
    """
    
    # Initialize the client and download comments of top n games
    bgg_client = BGGClient()
    
    # Download top 100 game's information
    # Commented out since already downloaded in the notebook
    # bgg_client.get_games() 
    
    if top_n == None:
        top_n = 100
        
    games_df = pd.read_csv('data/raw/top_100_games.csv')[:top_n]
    
    # Download, clean and extract aspects for the top n games
    # Be careful that higher numbers will cause OpenAI uasage costs
    for idx, row in games_df.iterrows():
        
        game_name = row['TITLE'][0]
        # Download comments
        bgg_client.download_comments(game_name)

        # Remove non-English comments
        clean_comments(game_name)

        # Get aspects for each comment of all games
        aspect_extraction_for_game(game_name)

