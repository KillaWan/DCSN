# DCSN

[English](README.md) | **Nederlands** | [中文（简体）](README.zh-CN.md)

**Difference-Conditioned Shape Normalization** is een normalisatiemethode voor het voorspellen van tijdreeksen. Wie van elk blok het gemiddelde aftrekt, verwijdert het niveau van dat blok. Toch kan de vorm die overblijft verschillen naargelang het niveau tussen twee blokken stijgt of daalt. DCSN schat dit verband voor elk invoervenster, trekt de vorm af die de afbeelding voorspelt en gebruikt dezelfde afbeelding om de voorspelling terug te brengen naar de oorspronkelijke coördinaten.

Deze repository reproduceert de hoofdresultaten van `norm_backbone50_20261023`. Er zijn drie voorspelmodellen (**DLinear, FEDformer en Autoformer**), acht datasets om te downloaden en vier voorspelhorizonten (96, 192, 336 en 720 tijdstappen). De [96 configuraties](dcsn/configs.json) gebruiken de oorspronkelijke instellingen en seed 2021. De datasets, checkpoints en normalisatiemethoden uit de vergelijkende experimenten zijn niet meegeleverd.

## Gegevens downloaden

Voer vanuit de hoofdmap een van de volgende opdrachten uit. De eerste downloadt alle acht datasets (ongeveer 253 MiB na uitpakken); de tweede alleen de twee genoemde datasets. Het downloadscript gebruikt uitsluitend de standaardbibliotheek van Python 3.11 en heeft PyTorch niet nodig.

```bash
python scripts/download_datasets.py
python scripts/download_datasets.py Electricity Traffic
```

De bestanden komen in `datasets/`, een map die Git negeert. Het script controleert de bestandsgrootte en de SHA-256-hash aan de hand van het [gegevensmanifest](scripts/dataset_manifest.json). Een bestaand bestand dat overeenkomt, wordt overgeslagen. Wijkt het af, dan meldt het script een fout zonder het bestand te vervangen.

| Dataset | Bron | Pad ten opzichte van de datamap |
| --- | --- | --- |
| ETTh1, ETTh2, ETTm1, ETTm2 | [ETDataset / ETT-small](https://github.com/zhouhaoyi/ETDataset/tree/1d16c8f4f943005d613b5bc962e9eeb06058cf07/ETT-small) | `<naam>/<naam>.csv` |
| Electricity, ExchangeRate | [Spiegel van de multivariate benchmarkdatasets](https://github.com/wwy155/multivariate_timeseries_datasets/tree/89e003fc15d20a310c337f73363e1a07a4beb532) | `electricity/electricity.csv`, `exchange_rate/exchange_rate.csv` |
| Weather | [weather.zip](https://github.com/wwy155/multivariate_timeseries_datasets/blob/89e003fc15d20a310c337f73363e1a07a4beb532/weather.zip) | `weather/weather/weather.csv` |
| Traffic | [Multivariate time series data / traffic](https://github.com/laiguokun/multivariate-time-series-data/tree/7f402f185cc2435b5e66aed13a3b560ed142e023/traffic) | `traffic/traffic.txt` (zonder kopregel of datumkolom) |

Heeft de rekennode geen internetverbinding, download de bestanden dan op een andere machine en kopieer de datamap. Gebruik bij gedeelde opslag hetzelfde pad voor downloaden en trainen:

```bash
python scripts/download_datasets.py Weather --data-root /path/to/data
python -m dcsn train --backbone DLinear --dataset Weather --horizon 336 \
  --data-root /path/to/data --device cuda:0 --output runs/dlinear_weather_336
```

## Trainen en evalueren

Gebruik Python 3.11 en de versies in [requirements.txt](requirements.txt). Kies voor GPU-berekeningen een PyTorch-versie die bij de CUDA-omgeving past. Voer de onderstaande opdrachten uit vanuit de hoofdmap. Voor een bewerkbare installatie kun je na het installeren van de afhankelijkheden `pip install --no-deps --no-build-isolation -e .` uitvoeren.

```bash
python -m unittest discover -s tests -v

python -m dcsn train --backbone DLinear --dataset Weather --horizon 336 \
  --device cuda:0 --output runs/dlinear_weather_336

python -m dcsn evaluate --checkpoint runs/dlinear_weather_336/selected.pt \
  --device cuda:0
```

Voor elke training is een nieuwe uitvoermap nodig. De gebruikte configuratie komt in `config.json`, de op basis van de validatie-MSE geselecteerde gewichten in `selected.pt` en de leercurves en het geselecteerde epochnummer in `results.json`. Gebruik je een andere datamap dan de standaardmap, geef `--data-root` dan zowel bij het trainen als bij het evalueren op.

### Een eerder checkpoint laden

Geef bij een checkpoint uit de oorspronkelijke hoofdexperimenten expliciet het model, de dataset en de voorspelhorizon op. De meegeleverde configuraties gelden niet voor ablatie-experimenten of andere seeds.

```bash
python -m dcsn evaluate --checkpoint /path/to/selected_s2021.pt \
  --backbone FEDformer --dataset Electricity --horizon 336 --device cuda:0

python -m dcsn convert --checkpoint /path/to/selected_s2021.pt \
  --backbone FEDformer --dataset Electricity --horizon 336 \
  --output checkpoints/fedformer_electricity_336.pt --device cpu
```

Een geconverteerd checkpoint bevat de configuratie, de gewichten en de indices van de Fouriermodi. Het oorspronkelijke project is daarna niet meer nodig voor de evaluatie.

## Werking

1. DCSN schat per invoervenster met ridgeregressie een afbeelding van verschillen tussen blokgemiddelden naar de vorm binnen de blokken. De spectrale norm is begrensd op één. De afbeelding wordt voor dit venster bewaard. Bij het schatten worden geen toekomstige waarnemingen gebruikt en worden geen gradiënten berekend.
2. DCSN transformeert de invoer tot `(I − B_X)X`. Na centrering per blok gaat deze reeks naar het voorspelmodel, dat `F` levert. Een afzonderlijke voorspeller leest de getransformeerde invoer vóór het centreren en voorspelt de toekomstige blokgemiddelden `M`.
3. De uiteindelijke voorspelling is `(I + B_X)(M + F)`, met dezelfde afbeelding als bij de invoer. Gradiënten bereiken beide voorspellers via de coördinatentransformaties. De uitvoer van het voorspelmodel hoeft per blok geen gemiddelde van nul te hebben.

Begin bij `ForecastModel.forward` in [model.py](dcsn/model.py). De coördinatentransformatie en de bijbehorende inverse staan in [normalization.py](dcsn/normalization.py). De [voorspeller van de blokgemiddelden](dcsn/reference.py) gebruikt een gedeelde MLP, een embedding per variabele en twee lineaire snelkoppelingen. De invoer en uitvoer hebben de volgorde `[batch, tijd, variabelen]`; intern gebruikt de normalisatiemodule `[batch, variabelen, tijd]`.

## Reproduceerbaarheid

De [trainingscode](dcsn/training.py) traint twee kandidaat-voorspellers voor de blokgemiddelden vooraf, vanuit dezelfde initialisatie (maximaal tien epochs in totaal). Na de keuze op basis van de validatiefout wordt het voorspelmodel twee epochs afzonderlijk getraind. Daarna worden beide voorspellers gezamenlijk getraind. Bloklengte, breedte van de blokgemiddeldenvoorspeller, leersnelheden, batchgrootte en maximaal aantal epochs liggen per configuratie vast.

De implementatie behoudt de oorspronkelijke volgorde van de toevalstrekkingen bij initialisatie, het herstel van de toevalstoestand, de datasplitsingen, de tijdkenmerken, de markeerwaarden in Weather en de verwerking van de laatste batch. Bij MSE en MAE telt elk geëvalueerd element even zwaar. De [validatieresultaten](VALIDATION.md) beschrijven de vergelijking binnen dezelfde rekenomgeving. Bij opnieuw trainen op andere hardware is gelijkheid tot op bitniveau niet gegarandeerd.

Zie de [licentievermeldingen](LICENSES/README.txt) voor de herkomst van de externe code. Het [bronmanifest](LICENSES/SOURCE_REPORT.json) vermeldt de vastgelegde bronversies en hun controlesommen.
