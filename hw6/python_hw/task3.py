from collections import defaultdict
import time
import sys
import random
import tweepy

API_KEY = "raY5X4WEHzATD2peK4HrnVLFG"
API_SECRET_KEY = "uJqnTDn29BYrGc5SVKyhvfhARzcah7xo2356wpkxRHYPSWa60L"
ACCESS_TOKEN = "1387963678369140740-sj1VxJQm32TIPk139p0wZeh2JAdW47"
ACCESS_TOKEN_SECRET = "awaWjr0map8rM8uh1avmJodVP1DACQYkqMDp8xlMOTqpH"

KEY_WORDS = ["COVID", "India", "Bitcoin", "Lakers", "NFL", "NFLDraft", "Ethereum", "Dehli", "Dogecoin",
             "Crypto", "COVID-19"]


class TwitterStreamListener(tweepy.StreamListener):
    def __init__(self, output_fp):
        tweepy.StreamListener.__init__(self)
        self.output_fp = output_fp
        self.tags = defaultdict(list)
        self.count = 0
        with open(self.output_fp, "w") as w:
            w.write("")

    def on_status(self, status):
        words = status.text.split()
        tags = [x[1:] for x in words if len(x) > 1 and x[0] == "#"]
        if len(tags) > 0:
            self.count += 1
            if self.count > 100:
                prob_ = random.randint(1, self.count)
                if prob_ < 100:
                    self.tags[prob_-1] = tags

                rank_d = {}
                for k, tag_lst in self.tags.items():
                    for tag in tag_lst:
                        if tag in rank_d:
                            rank_d[tag] += 1
                        else:
                            rank_d[tag] = 1
                ranked_lst = sorted(rank_d.items(), key=lambda item: (-item[1], item[0]))

                with open(self.output_fp, "a") as w:
                    freq_lst = []
                    w.write("The number of tweets with tags from the beginning: " + str(self.count))
                    w.write("\n")
                    for tag_cnt in ranked_lst:
                        freq_ = tag_cnt[1]
                        if freq_ not in freq_lst:
                            if len(freq_lst) == 3:
                                break
                            else:
                                freq_lst.append(freq_)
                                w.write(str(tag_cnt[0]) + " : " + str(freq_))
                                w.write("\n")
                        else:
                            w.write(str(tag_cnt[0]) + " : " + str(freq_))
                            w.write("\n")

                    w.write("\n")
            else:
                #print(status.text)
                self.tags[self.count-1] = tags

if __name__ == "__main__":
    port_num = int(sys.argv[1])
    output_fp = sys.argv[2]

    # port_num = 9999
    # output_fp = "./output/output3.csv"

    start = time.time()
    auth = tweepy.OAuthHandler(API_KEY, API_SECRET_KEY)
    auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)

    TwitterStreamListener = TwitterStreamListener(output_fp)
    myStream = tweepy.Stream(auth=auth, listener=TwitterStreamListener)
    myStream.filter(track=KEY_WORDS, languages=["en"])

    end = time.time()
    print("Duration:", round((end - start), 2))
