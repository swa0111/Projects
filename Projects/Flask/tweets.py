import tweepy
import time
import pandas as pd
pd.set_option('display.max_colwidth', 1000)

# api key
#api_key = "Enter API Key" 
api_key = 'xbMuxcJpRTiVGt2C2EYnA'
# api secret key
#api_secret_key = "Enter API Secret Key" 
api_secret_key = '2DbQTsvIptkPTdaUcos8DDvQH9fzO0hNjJpUT2uVzQ'
# access token
#access_token = "Enter Access Token" 
access_token = '7319442-EDm4CPxL7W4KkZcGWRMJNVHp88W5OH9vgblu898fg'
# access token secret
#access_token_secret = "Enter Access Token Secret" 
access_token_secret = '5ZxJSbqXhG7uhgXzTFWf9XhkfsxxinlPRXyDTzbA9w'

authentication = tweepy.OAuthHandler(api_key, api_secret_key)
authentication.set_access_token(access_token, access_token_secret)
api = tweepy.API(authentication, wait_on_rate_limit=True)

def get_related_tweets(text_query):
    # list to store tweets
    tweets_list = []
    # no of tweets
    count = 50
    try:
        # Pulling individual tweets from query
        for tweet in api.search(q=text_query, count=count):
            print(tweet.text)
            # Adding to list that contains all tweets
            tweets_list.append({'created_at': tweet.created_at,
                                'tweet_id': tweet.id,
                                'tweet_text': tweet.text})
        return pd.DataFrame.from_dict(tweets_list)

    except BaseException as e:
        print('failed on_status,', str(e))
        time.sleep(3)
