Clustering with BERT sentence embeddings presents us with a total of (only) three clusters. There are 3,218 noise points. These points could not be fitted into another cluster and are therefore excluded. The table shows the results of the clustering. Notably, cluster one contains many more items (5,117) in comparison to the other two clusters (respectively 20 and 25).


| Cluster  | Items | Description |
| ------------- | ------------- | ------------- |
| -1  | 3,218  | Noise points  |
| 0  | 20 | Single characters  |
| 1  | 5,117 | - |
| 2  | 25 | Typos and abbreviations  |

Cluster zero contains single characters (and in one case two characters) and could be considered typos or utterances that were started but not continued by the user before sending the message. Cluster two is similar, these utterances are also very short and often contain typos (`ke' should probably be `oke'). Next to typos we also see abbreviations (`nvm' is `nevermind') and acknowledgements (`yeah' and `yes'). The first cluster contains many of the longer messages while the noise points contain many shorter (words < 10) messages. At first glance it seems the algorithm based the clustering mostly on the length of the utterances.
