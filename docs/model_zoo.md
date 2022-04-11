## Model Zoo

#### EPIC-KITCHENS-100

| Modality      | Method      | Backbone    | Verb        |  Noun       |  Action     |  
| -----------   | ----------- | ----------- | ----------- | ----------- | ----------- |  
| RGB   | RULSTM    | TSN       |   27.5    | 29.0  |  13.3 |
| RGB   | AVT       | TSN       |   27.2    | 30.7  |  13.6 |
| RGB   | AVT       | irCSN-52  |   25.5    | 28.1  |  12.8 |
| RGB   | AVT       | ViT*      |   28.7    | 32.3  |  14.9 |
| RGB   | DCR(LSTM) | TSN       |   27.9    | 28.0  |  14.5 |
| RGB   | DCR(LSTM) | TSM       |   28.4    | 28.5  |  15.2 |
| RGB   | DCR       | TSN       |   31.0    | 31.1  |  14.6 |
| RGB   | DCR       | TSM       |   32.6    | 32.7  |  16.1 |
| FLOW  | RULSTM    | TSN       |   19.1    | 16.7  |  7.2  |
| FLOW  | AVT       | TSN       |   20.9    | 16.9  |  6.6  |
| FLOW  | DCR(LSTM) | TSN       |   21.6    | 15.3  |  7.8  |
| FLOW  | DCR       | TSN       |   25.9    | 17.6  |  8.4  |
| OBJ   | RULSTM    | TSN       |   17.9    | 23.3  |  7.8  |
| OBJ   | AVT       | FRCNN     |   18.0    | 24.3  |  8.7  |
| OBJ   | DCR(LSTM) | FRCNN     |   16.1    | 19.6  |  7.5  |
| OBJ   | DCR       | FRCNN     |   22.2    | 24.2  |  9.7  |

#### EPIC-KITCHENS-55

| Modality      | Method      | Backbone    | Top-1 Action  |  Top-5 Action       |  
| -----------   | ----------- | ----------- | -----------   | -----------         |  
| RGB   | RULSTM        | TSN       |   13.1    | 30.8 |
| RGB   | ActionBanks   | TSN       |   12.7    | 28.6 |
| RGB   | AVT           | TSN       |   13.1    | 28.1 |
| RGB   | AVT           | ViT*      |   12.5    | 30.1 |
| RGB   | AVT           | irCSN-152 |   14.4    | 31.7 |
| RGB   | DCR           | TSN       |   13.6    | 30.8 |
| RGB   | DCR           | irCSN-152 |   15.1    | 34.0 |
| RGB   | DCR           | TSM       |   16.1    | 33.1 |
| FLOW  | RULSTM        | TSN       |   8.7     | 21.4 |
| FLOW  | ActionBanks   | TSN       |   8.4     | 19.8 |
| FLOW  | DCR           | TSN       |   8.9     | 22.7 |
| OBJ   | RULSTM        | FRCNN     |   10.0    | 29.8 |
| OBJ   | ActionBanks   | FRCNN     |   10.2    | 29.1 |
| OBJ   | DCR           | FRCNN     |   11.5    | 30.5 |


The EPIC-KITCHENS test server submission files are at [here](https://drive.google.com/drive/folders/129uG7kI1IbsHLPwvVCLHPLBacLSUf1sk?usp=sharing).