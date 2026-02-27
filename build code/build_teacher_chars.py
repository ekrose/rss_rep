# Build teacher Xs
import pandas as pd
import numpy as np


### 0) Teacher resids
resids = pd.read_stata("temp/teach_mean_resids.dta")


### 1) Education
educs = []
for year in range(12,25):
    educ = pd.read_sas(f"../ncerdc_data/Education/educ_pub{year}.sas7bdat", encoding='latin-1')
    educ['year'] = 2000 + year
    educs += [educ.copy()]
for year in range(1995,2009):
    educ = pd.read_sas(f"../ncerdc_data/Education/lsprseduc{year}.zip", encoding='latin-1',
            compression='zip', format="sas7bdat")
    educ['year'] = year
    educs += [educ.copy()]
for year in range(2009,2012):
    educ = pd.read_sas(f"../ncerdc_data/Education/lsprseduc{year}.sas7bdat", encoding='latin-1')
    educ['year'] = year
    educs += [educ.copy()]

educs = pd.concat(educs, ignore_index=True)

# Formatting fixes
educs['educ_lvl_cd'] = pd.to_numeric(educs.educ_lvl_cd)
educs['grad_date'] = pd.to_datetime(educs.grad_date)
educs['grad_year'] = educs.grad_date.dt.year

'''
4 - bachelors
5 - masters
6 - advanced
7 - doctorate 
'''

# Diagnostics
print(educs.groupby('year').educ_lvl_cd.value_counts().unstack(level=1))
print(resids.teachid.isin(educs.teachid).mean())

# Final
educs = educs[['teachid','year','educ_lvl_cd','grad_year']].drop_duplicates()
educs = educs.sort_values(['teachid','year','educ_lvl_cd'], ascending=True)
educs = educs.groupby(['teachid','year'], as_index=False).last()



### 3) Demographics
demos = pd.read_sas(f"../ncerdc_data/Pay/personnel_through24.sas7bdat", encoding='latin-1')
print(resids.teachid.isin(demos.teachid).mean())
assert demos.duplicated('teachid').sum() == 0


### 4) Testing
tests = []
for year in range(12,25):
    test = pd.read_sas(f"../ncerdc_data/Testing/test_pub{year}.sas7bdat", encoding='latin-1')
    test['year'] = 2000 + year
    tests += [test.copy()]
for year in range(1995,2012):
    test = pd.read_sas(f"../ncerdc_data/Testing/lstestsnap{year}.zip", encoding='latin-1',
                    compression='zip', format="sas7bdat")
    test['year'] = year
    tests += [test.copy()]
tests = pd.concat(tests, ignore_index=True)

# Keep the PRAXIS tests (by far the most common)
tests = tests.loc[tests.test_type_desc == "PRAXIS"]

# Norm tests
tests['score_normed'] = ((tests.tst_score_num - tests.groupby(['year','tst_cd']).tst_score_num.transform('mean'))/
                            tests.groupby(['year','tst_cd']).tst_score_num.transform('std'))

# Average by teacher-year
tests = tests.groupby(['teachid','year'], as_index=False).score_normed.mean()


### 5) Pay
salaries = []
# 1995-2011
for year in range(1995,2012):
    salary = pd.read_sas(f"../ncerdc_data/Pay/lspaysnap{year}.sas7bdat", encoding='latin-1')
    pay = salary.groupby('teachid', as_index=False)[['cert_sal_amt']].sum().rename(
        columns={'cert_sal_amt':'total_gross_pay'})
    pay['year'] = year
    salaries += [pay.copy()]

# 2012
salary = pd.read_sas(f"../ncerdc_data/Pay/certsalpub2012.sas7bdat", encoding='latin-1')
pay = salary.groupby('teachid', as_index=False)[['payline_gross_amt']].sum().rename(
        columns={'payline_gross_amt':'total_gross_pay'})
pay['year'] = 2012
pay = pay.merge(salary.groupby('teachid', as_index=False).tchr_exp.max(), how='left', on='teachid')
salaries += [pay.copy()]

# 2013-2014
for year in range(2013,2024):
    salary = pd.read_sas(f"../ncerdc_data/Pay/certsalpub{year}.sas7bdat", encoding='latin-1')
    pay = salary.groupby('teachid', as_index=False)[['total_gross_pay']].sum()
    pay['year'] = year
    pay = pay.merge(salary.groupby('teachid', as_index=False).tchr_exp.max(), how='left', on='teachid')
    salaries += [pay.copy()]

salaries = pd.concat(salaries, ignore_index=True)
assert salaries.duplicated(['teachid','year']).sum() == 0

# Winsorize
for yr in salaries.year.unique():
    ub = salaries.loc[salaries.year == yr, 'total_gross_pay'].quantile(.99)
    lb = salaries.loc[salaries.year == yr, 'total_gross_pay'].quantile(.01)
    salaries.loc[(salaries.year == yr) & (salaries.total_gross_pay > ub), 'total_gross_pay'] = ub
    salaries.loc[(salaries.year == yr) & (salaries.total_gross_pay < lb), 'total_gross_pay'] = lb

# Convert to relative pay
salaries['relative_pay'] = ((salaries.total_gross_pay - salaries.groupby('year').total_gross_pay.transform('mean'))/
                                salaries.groupby('year').total_gross_pay.transform('std'))

# experience
salaries['tchr_exp'] = pd.to_numeric(salaries.tchr_exp)
print(resids.teachid.isin(salaries.teachid).mean())


### 6) Combine and save
teacher_covars = educs.merge(demos, how='outer', on='teachid').merge(
        tests, how='outer', on=['teachid','year']).merge(
        salaries, how='outer', on=['teachid','year'])
assert teacher_covars.duplicated(['teachid','year']).sum() == 0

teacher_covars.to_stata('data/teacher_covars.dta', write_index=False, convert_dates={'bdate':'td'})
