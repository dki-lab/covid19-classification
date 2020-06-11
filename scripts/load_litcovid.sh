#!/bin/bash

cd ../data

wget -O litcovid_source_general.tsv https://www.ncbi.nlm.nih.gov/research/coronavirus-api/export/tsv?filters=%7B%22topics%22%3A%5B%22General%20Info%22%5D%7D
wget -O litcovid_source_mechanism.tsv https://www.ncbi.nlm.nih.gov/research/coronavirus-api/export/tsv?filters=%7B%22topics%22%3A%5B%22Mechanism%22%5D%7D
wget -O litcovid_source_transmission.tsv https://www.ncbi.nlm.nih.gov/research/coronavirus-api/export/tsv?filters=%7B%22topics%22%3A%5B%22Transmission%22%5D%7D
wget -O litcovid_source_diagnosis.tsv https://www.ncbi.nlm.nih.gov/research/coronavirus-api/export/tsv?filters=%7B%22topics%22%3A%5B%22Diagnosis%22%5D%7D
wget -O litcovid_source_treatment.tsv https://www.ncbi.nlm.nih.gov/research/coronavirus-api/export/tsv?filters=%7B%22topics%22%3A%5B%22Treatment%22%5D%7D
wget -O litcovid_source_prevention.tsv https://www.ncbi.nlm.nih.gov/research/coronavirus-api/export/tsv?filters=%7B%22topics%22%3A%5B%22Prevention%22%5D%7D
wget -O litcovid_source_forecasting.tsv https://www.ncbi.nlm.nih.gov/research/coronavirus-api/export/tsv?filters=%7B%22topics%22%3A%5B%22Epidemic%20Forecasting%22%5D%7D
wget -O litcovid_source_case_report.tsv https://www.ncbi.nlm.nih.gov/research/coronavirus-api/export/tsv?filters=%7B%22topics%22%3A%5B%22Case%20Report%22%5D%7D

wget https://ftp.ncbi.nlm.nih.gov/pub/lu/LitCovid/litcovid2BioCXML.gz

gunzip litcovid2BioCXML.gz

cd ../scripts
python litcovid_dataset_from_xml.py