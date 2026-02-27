# Execute all code 
source .venv/bin/activate
pip install requirements.txt

# Tables 1 and A11 ✔️
stata code/summary_stats.do

# Tables 2 and 3, Figure A1 ✔️
stata code/estimate_variance.do
python code/vcov_main_part1.py
python code/vcov_main_part2.py

# Tables 4 and A8
python code/vcov_implied_reg.py

# Table 5 ✔️
stata code/entry_ivs_part1.do

# Figure 1 ✔️
python code/vcov_main_part3.py

# Figure 2 and A3, Tables A4 and A5 ✔️
stata code/estimate_ovb.do

# Figures 3-4, A5 ✔️
stata code/estimate_heterog_variance.do 
python code/vcov_hetero.py

# Figure 5 ✔️
python code/bottomXpercent.py

# Figure A2 ✔️
python code/vcov_main_part4.py

# Figure A4
stata code/estimate_variance_robustness.do
python code/vcov_robustness.py

# Tables A1 and A12 ✔️
stata code/teacher_chars.do

# Table A2-A3 ✔️
stata code/regression_version.do

# Tables A6 and A7 ✔️
stata code/entry_ivs_part2.do
stata code/entry_ivs_part3.do

# Table A9
python code/vcov_within_schl.py

# Table A10 ✔️
python code/vcov_t_sensitivity.py

# Table A13 ✔️
python code/vcov_hetero_subgroups.py
stata code/vcov_hetero_subgroups_table.do

# Calculations cited in-text only ✔️
stata code/covariate_correlation_part1.do
python code/covariate_correlation_part2.py

