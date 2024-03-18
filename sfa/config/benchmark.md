int16 (sort and unique) + out float32 (insteadof 64) see set elems speed
Pcd message delay: -14186063.073100183
Bevmap discretization latency: 0.0009170320008706767 s
Bevmap sorting latency: 0.005868328000360634 s
Bevmap unique elems latency: 0.005224748998443829 s
Bevmap set elems latency: 0.0024780499988992233 s
PCD filtering latency: 0.0032677900017006323 s
PCD to bevmap latency: 0.014601540999137796 s
[DEBUG] [1710772847.817191]: Deserialization latency: 0.0019064699990849476 s
[DEBUG] [1710772847.818676]: Pre-processing latency: 0.007655368000996532 s
[DEBUG] [1710772847.819476]: To bevmap latency: 0.017934079998667585 s
[DEBUG] [1710772847.820119]: Inference latency: 0.07653348500025459 s
[DEBUG] [1710772847.820715]: Post-processing latency: 0.000489891001052456 s
[DEBUG] [1710772847.828340]: Message publishing latency: 6.69879991619382e-05 s
[DEBUG] [1710772847.829383]: Total latency: 0.10458628199921804 s


int16 (sort and unique)
Pcd message delay: -14185262.052552044
Bevmap discretization latency: 0.0010079160001623677 s
Bevmap sorting latency: 0.005664845000865171 s
Bevmap unique elems latency: 0.0054887930000404594 s
Bevmap set elems latency: 0.003932649999114801 s
PCD filtering latency: 0.002807003998896107 s
PCD to bevmap latency: 0.016157504000148037 s
[DEBUG] [1710772046.597298]: Deserialization latency: 0.0016947090007306542 s
[DEBUG] [1710772046.598596]: Pre-processing latency: 0.007222917998660705 s
[DEBUG] [1710772046.599404]: To bevmap latency: 0.01901592700050969 s
[DEBUG] [1710772046.600031]: Inference latency: 0.07671487100014929 s
[DEBUG] [1710772046.600534]: Post-processing latency: 0.0005576210005528992 s
[DEBUG] [1710772046.601168]: Message publishing latency: 7.166199975472409e-05 s
[DEBUG] [1710772046.601678]: Total latency: 0.10527770800035796 s


int16 (only unique)
Pcd message delay: -14185125.23179714
Bevmap discretization latency: 0.000621536999460659 s
Bevmap sorting latency: 7.110011210897937e-07 s
Bevmap unique elems latency: 0.03052903800016793 s
Bevmap set elems latency: 0.003617076999944402 s
PCD filtering latency: 0.003106805999777862 s
PCD to bevmap latency: 0.03485149699918111 s
[DEBUG] [1710771907.596432]: Deserialization latency: 0.0012983270007680403 s
[DEBUG] [1710771907.598112]: Pre-processing latency: 0.006626352998864604 s
[DEBUG] [1710771907.598861]: To bevmap latency: 0.03802541500044754 s
[DEBUG] [1710771907.599538]: Inference latency: 0.07843017800041707 s
[DEBUG] [1710771907.600157]: Post-processing latency: 0.0005612280001514591 s
[DEBUG] [1710771907.600682]: Message publishing latency: 7.818399899406359e-05 s
[DEBUG] [1710771907.601173]: Total latency: 0.12501968499964278 s


int64 (only iunique)
Pcd message delay: -14185067.85670822
Bevmap discretization latency: 0.0011858439993375214 s
Bevmap sorting latency: 6.629998097196221e-07 s
Bevmap unique elems latency: 0.035687315999894054 s
Bevmap set elems latency: 0.004093659001227934 s
PCD filtering latency: 0.0029987859998072963 s
PCD to bevmap latency: 0.04104900900165376 s
[DEBUG] [1710771849.421406]: Deserialization latency: 0.00177093800084549 s
[DEBUG] [1710771849.422708]: Pre-processing latency: 0.004945631000737194 s
[DEBUG] [1710771849.423549]: To bevmap latency: 0.0440959509996901 s
[DEBUG] [1710771849.424234]: Inference latency: 0.07395358499888971 s
[DEBUG] [1710771849.424807]: Post-processing latency: 0.0006159069998830091 s
[DEBUG] [1710771849.425391]: Message publishing latency: 7.886200000939425e-05 s
[DEBUG] [1710771849.425934]: Total latency: 0.1254608740000549 s



Bevmap discretization latency: 0.0010423450003145263 s
Bevmap sorting latency: 0.013822371999594907 s
Bevmap unique elems latency: 0.006423761999940325 s
Bevmap set elems latency: 0.005017188000238093 s
PCD filtering latency: 0.003439527999944403 s
PCD to bevmap latency: 0.026391394999336626 s
[DEBUG] [1710498647.160138]: Deserialization latency: 0.001282536999497097 s
[DEBUG] [1710498647.161310]: Pre-processing latency: 0.00840589400013414 s
[DEBUG] [1710498647.162148]: To bevmap latency: 0.029899934999775724 s
[DEBUG] [1710498647.162774]: Inference latency: 0.06720552900060284 s
[DEBUG] [1710498647.163342]: Post-processing latency: 0.00035585599925980205 s
[DEBUG] [1710498647.163903]: Message publishing latency: 5.116400006954791e-05 s
[DEBUG] [1710498647.164455]: Total latency: 0.10720091499933915 s

# without lexsort
Pcd message delay: -13914600.817471713
Bevmap discretization latency: 0.0004620489999069832 s
Bevmap sorting latency: 2.220003807451576e-07 s
Bevmap unique elems latency: 0.029696034000153304 s
Bevmap set elems latency: 0.004279034999854048 s
PCD filtering latency: 0.0020034190001751995 s
PCD to bevmap latency: 0.03451239700007136 s
[DEBUG] [1710501389.579253]: Deserialization latency: 0.0015734039998278604 s
[DEBUG] [1710501389.580461]: Pre-processing latency: 0.0045210360003693495 s
[DEBUG] [1710501389.581242]: To bevmap latency: 0.03657097599989356 s
[DEBUG] [1710501389.581800]: Inference latency: 0.07922475499981374 s
[DEBUG] [1710501389.582317]: Post-processing latency: 0.0005388820000007399 s
[DEBUG] [1710501389.582804]: Message publishing latency: 6.596500043087872e-05 s
[DEBUG] [1710501389.583280]: Total latency: 0.12249501800033613 s



# multi BEV threaded
[DEBUG] [1710496537.361499]: Deserialization latency: 0.001212139000017487 s
[DEBUG] [1710496537.362726]: Pre-processing latency: 0.004201540999929421 s
[DEBUG] [1710496537.363464]: To bevmap latency: 0.04083452300028512 s
[DEBUG] [1710496537.364056]: Inference latency: 0.2013420819998828 s
[DEBUG] [1710496537.364620]: Post-processing latency: 0.0010159650000787224 s
[DEBUG] [1710496537.365164]: Message publishing latency: 1.1327999800414545e-05 s
[DEBUG] [1710496537.365722]: Total latency: 0.24861757799999396 s



Default:
delay: -13685142.474364566
delay: -13685145.143248199
delay: -13685145.629199678
delay: -13685146.043305924
delay: -13685146.433989353
delay: -13685146.872323198

Run model once on dummy data and perform njit compile:
delay: -13686394.28696373
delay: -13686395.492621299
delay: -13686395.876047857
delay: -13686396.23436041
delay: -13686396.595257012
delay: -13686396.967169814

Run model once on dummy data and remove njit entirely:
delay: -13686722.438959233
delay: -13686722.982733164
delay: -13686723.371293828
delay: -13686723.765575944
delay: -13686724.147571499
delay: -13686724.549853452
delay: -13686724.935740376
delay: -13686725.319354355


With process pool:
Pre-processing latency: 0.17817026199963948
Inference latency: 0.2491132350000953
Post-processing latency: 0.0004902630003016384
Total latency: 0.4277737600000364

Without:
Pre-processing latency: 0.07202909800025736
Inference latency: 0.27396004899992477
Post-processing latency: 0.0006706359999952838
Total latency: 0.3466597830001774


with pool of 3 workers:
Pre-processing latency: 0.11648121299958802
Inference latency: 0.1950780330002999
Post-processing latency: 0.0005951899997853616
Total latency: 0.3121544359996733

with thread-pool of 3:
Pre-processing latency: 0.04417497499980527
Inference latency: 0.20938094299981458
Post-processing latency: 0.000441806000253564
Total latency: 0.2539977239998734

with unlimited thread pool:
Pre-processing latency: 0.05071891500028869
Inference latency: 0.22175539099953312
Post-processing latency: 0.0004191269999864744
Total latency: 0.2728934329998083