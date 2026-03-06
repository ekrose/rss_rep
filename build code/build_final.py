# Basic packages
import numpy as np
import pandas as pd
import gzip, gc, re

# Pandas display options
pd.options.display.max_columns = 200
pd.options.display.max_rows = 1000
pd.set_option('max_info_columns', 200)
pd.set_option('max_colwidth',1000)
pd.set_option('display.width',None)

# Numpy display options
np.set_printoptions(suppress=True)
np.set_printoptions(precision=5)

# Paths
home = "/"
data_path = "/data/NCERDC"
firstyr = 2006
lastyr = 2013

### 1) Build CMB files directly
if 0:
    print("\n\nWorking on CMB files")
    finals = []
    for year in range(2006,2014):
        print("Loading course membership for {}".format(year))
        with gzip.open(data_path + "/Student/Course Membership/crs_memb_pub{}.dta.gz".format(year,str(year)[-2:]), 'rb') as inF:
            cmb = pd.read_stata(inF)

        # Clean coursetitle to make more uniform over years
        cmb['coursetitle'] = cmb.coursetitle.str.lower()
        cmb['coursetitle'] = cmb.coursetitle.str.strip()
        cmb['coursetitle'] = cmb.coursetitle.str.replace(r'(ma)$','math')
        cmb['coursetitle'] = cmb.coursetitle.str.replace(r'(eng)$','english')
        cmb['coursetitle'] = cmb.coursetitle.str.replace(r'( adv )',' advanced ')
        cmb['coursetitle'] = cmb.coursetitle.str.replace(r'(self contained)','self-contained')
        cmb['coursetitle'] = cmb.coursetitle.str.replace(r'([\s]+)',' ')

        # Subset to the state courses we want
        cmb['coursecode'] = pd.to_numeric(cmb.statecourse.str.replace('[^0-9]','').apply(lambda x: x[:4]))
        cmb['localccode'] = cmb.localcourse.str.extract('^([0-9]{4,5})')
        courses = [0,1010,1021,1022,1023,1024,2001,2003,2021,2022,2023,2024,2030,2025]
        cmb = cmb.loc[cmb.coursecode.isin(courses)]

        # Subset to the grades we want
        cmb = cmb.loc[pd.to_numeric(cmb.grade, errors='coerce').between(3,12)]

        # Drop unnecessary columns
        for col in ['numstudents','nstudents','reporting_year','collection_code'] + ['localcourse','statecourse','course_cycle','meetingcode']:
            try:
                cmb.drop(col, axis=1, inplace=True)
            except:
                pass

        # Add to aggregator
        cmb['year'] = year
        cmb = cmb.drop_duplicates()
        finals += [cmb.copy(),]
        del cmb
        gc.collect()

    # Put together CMB build
    data = pd.concat(finals, sort=False, ignore_index=True)
    data = data.loc[data.mastid.notnull()]
    data['grade'] = pd.to_numeric(data.grade)

    # Fix coding of ethnicity
    data['ethnic'] = data.ethnic.replace({1:"I",2:"A",3:"H",4:"B",5:"W",6:"M"})
    data['ethnic'] = data.ethnic.replace({'1':"I",'2':"A",'3':"H",'4':"B",'5':"W",'6':"M"})

    # Subset data
    data = data[['mastid','year','coursetitle','lea','schlcode','teachid','grade','coursecode','localccode',
                    'sex','ethnic','birthdt','semester','section']] # 'unique_courses','unique_courses_id'
    data['classroom_id'] = data.groupby(['lea','schlcode','year','grade','coursetitle','semester','section']).grouper.group_info[0]
    data['school_course_grade_year_fe'] = data.groupby(['lea','schlcode','year','grade','coursetitle']).grouper.group_info[0]

    # Save full dataset
    data.to_pickle("data/cmb.pkl.gz", compression='gzip')


### 1b) Add p values
if 1:
    from scipy.stats import chi2_contingency
    data = pd.read_pickle("data/cmb.pkl.gz", compression='gzip')

    # Add pvalues for grades 4-8 and 2007 on
    def chi2(table):
        for col in table.columns:
            if table[col].sum() == 0:
                table = table.drop(col, axis=1)
        table = table.values
        r, p, dof, e = chi2_contingency(table)
        return p

    def classroom_chi2(c):
        current = data.loc[data.school_course_grade_year_fe == c]
        lag = data.loc[(data.year == current.year.max() - 1) &
                            data.mastid.isin(current.mastid), ['mastid','classroom_id']].rename(
                    columns={'classroom_id':'lag_classrom_id'})
        current = current[['mastid','classroom_id']].merge(lag, how='inner', on='mastid')
        table = pd.crosstab(current.classroom_id, current.lag_classrom_id)
        try:
            p = chi2(table)
        except:
            p = None
        print(c,table,p,"\n", flush=True)
        return c, p

    gc.collect()
    import multiprocessing
    if __name__ == "__main__":
        p = multiprocessing.Pool(8)
        ans = p.map(classroom_chi2, data.loc[(data.year >= 2007) & data.grade.between(4,12)].school_course_grade_year_fe.unique())

    pvals = pd.DataFrame(ans, columns=['school_course_grade_year_fe','pval'])
    data = data.merge(pvals, how='left', on=['school_course_grade_year_fe'])
    data.loc[data.pval.notnull()].groupby(['lea','schlcode','year','grade','coursetitle']).pval.first().reset_index().to_pickle(
        "data/pvals.pkl.gz", compression='gzip')

### 3) Build EOG data
if 1:
    print("\n\nAdding EOG files")
    eogs = []
    for year in range(1996,2014):
        for grade in range(3,9):
            if ((year == 1996) & (grade >= 5)):
                continue
            print("Loading EOG for {} grade {}".format(year,grade))
            with gzip.open(data_path + "/Student/End of Grade/{}/eog{}pub{}.dta.gz".format(year,grade,str(year)[-2:]), 'rb') as inF:
                eog = pd.read_stata(inF)

            # Subset to non duplicated key fars
            tmp = eog[['mastid','lea','schlcode','sex','ethnic','bdate']].drop_duplicates()
            tmp['year'] = year
            tmp['grade'] = grade

            # Find best math
            if 'administ' not in eog.columns:
                eog['administ'] = "0"
            tokeep = ['mastid','mathscal','tchmtjdg','mathach','antgrdm','teachid','classpd']
            tokeep = [c for c in tokeep if c in eog.columns]
            math = eog.loc[eog.mastid.notnull() & eog.mathscal.notnull()
                        & (eog.administ.astype(str) == "0"), tokeep].rename(
                                columns={'teachid':'teachid_eogm','classpd':'classpd_eogm'})
            math = math.loc[~math.duplicated('mastid', keep=False)]

            # Find best reading
            tokeep = ['mastid','readscal','tchrdjdg','antgrdr','readach','teachid','classpd',]
            tokeep = [c for c in tokeep if c in eog.columns]
            read = eog.loc[eog.mastid.notnull() & eog.readscal.notnull()
                        & (eog.administ == "0"), tokeep].rename(
                                columns={'teachid':'teachid_eogr','classpd':'classpd_eogr'})
            read = read.loc[~read.duplicated('mastid', keep=False)]

            # Find skills stuff
            tokeep = ['mastid', 'homework', 'freeread', 'elctrnic', 'lookover', 'mathnotes', 'tv']   
            tokeep = [c for c in tokeep if c in eog.columns]
            if len(tokeep) > 0:
                noncog = eog.loc[eog.mastid.notnull(), tokeep]
                noncog = noncog.drop_duplicates('mastid', keep='first')
            else:
                noncog = None

            # Keep relevant vars
            eog = tmp.merge(math, how='left', on='mastid')
            eog = eog.merge(read, how='left', on='mastid')
            if noncog is not None:
                eog = eog.merge(noncog, how='left', on='mastid')

            # Add to lsit
            eogs += [eog,]

    # Put together
    eog = pd.concat(eogs, sort=False, ignore_index=True)

    # Standardize
    for col in ['mathscal','readscal']:
        eog[col] = (eog[col] - eog.groupby(['grade','year'])[col].transform('mean')
                        ) / eog.groupby(['grade','year'])[col].transform('std')

    # Reformat noncog variables
    eog['homework'] = pd.to_numeric(eog.homework.replace({'G':-1,'A':0,'B':1,'C':2,'D':3,'E':4,'F':5})) # time spent on homework   
    eog['freeread'] = pd.to_numeric(eog.freeread.replace({'A':0,'B':1,'C':2,'D':3,'E':4}))  # time spent free reading
    eog['tv'] = pd.to_numeric(eog.tv.replace({'A':0,'B':1,'C':2,'D':3,'E':4,'F':5}))    # time spent watching TV
    for col in ['homework','freeread','tv']:
        eog[col] = (eog[col] - eog.groupby(['grade','year'])[col].transform('mean')
                        ) / eog.groupby(['grade','year'])[col].transform('std')  
    eog.rename(columns={'tv':'watchtv'}, inplace=True)      

    # Save
    assert eog[['mastid','year','grade']].duplicated().sum() == 0
    eog.to_pickle("data/eog.pkl.gz", compression='gzip')


### 4) Add EOC data
if 1:
    print("\n\nAdding EOC files")
    eocs = []
    for year in range(firstyr-2,lastyr+1):
        tmp = []
        for subject in ['eng1','alg1','alg2','geom']:
            print("Loading EOC for {} subject {}".format(year,subject))
            try:
                with gzip.open(data_path + "/Student/End of Course/{}/{}pub{}.dta.gz".format(year,subject,str(year)[-2:]), 'rb') as inF:
                    eoc = pd.read_stata(inF)
            except Exception as e:
                print(e)
                continue

            # Find entry
            if 'administ' not in eoc.columns:
                eoc['administ'] = "0"
            if 'tjudge' in eoc.columns:
                eoc.rename(columns={'tjudge':'teachach'}, inplace=True)

            tokeep = ['mastid','antgrade'] + [c for c in eoc.columns if any([
                                a in c for a in ['scal','ach']]) 
                                and c not in ['teachid','comachom']] 

            eoc = eoc.loc[eoc.mastid.notnull() & eoc['{}scal'.format(subject)].notnull()
                    & (eoc.administ.astype(str) == "0"), tokeep]
            eoc = eoc.rename(columns={'antgrade':'{}antgrade'.format(subject)})
            if 'teachach' in eoc.columns:
                eoc = eoc.rename(columns={'teachach':'{}teachach'.format(subject)})

            # De-dup
            eoc = eoc.loc[~eoc.duplicated('mastid', keep=False)].set_index('mastid')
            tmp += [eoc,]

        # Add to lsit
        eoc = pd.concat(tmp, axis=1, sort=False, ignore_index=True)
        eoc['year'] = year
        eocs += [eoc,]

    # Put together
    eoc = pd.concat(eocs, sort=False)

    # Standardize
    for col in ['eng1scal','alg1scal','alg2scal','geomscal']:
        eoc[col] = (eoc[col] - eoc.groupby(['year'])[col].transform('mean')
                        ) / eoc.groupby(['year'])[col].transform('std')
    
    # Save
    eoc.to_pickle("data/eoc.pkl.gz", compression='gzip')


### 5) Add demographic data from various files
if 1:
    '''
    EOG coding changes:
    pared:
    1996-1999, 2001-2002:   1 = Did not finish high school
                            2 = High school graduate
                            3 = Trade or business school graduate
                            4 = Community, technical or junior college graduate
                            5 = Four-year college graduate
                            6 = Graduate school degree

    2003-2004:          1 = Did not finish high school
                        2 = High school graduate
                        3 = Some education after high school but did not graduate
                        4 = Community, technical or junior college graduate
                        5 = Trade or business school graduate
                        6 = Four-year college graduate
                        7 = Graduate school degree

    2000,2005-:         1 = Did not finish high school
                        2 = High school graduate
                        3 = Some education after high school but did not graduate
                        4 = Trade or business school graduate
                        5 = Community, technical or junior college graduate
                        6 = Four-year college graduate
                        7 = Graduate school degree

    except:
    1996-2000:    1 = Not identified as an Exceptional Student
                  2 = Academically/Intellectually Gifted (AIG)
                  3 = Behaviorally-Emotionally Handicapped
                  4 = Hearing Impaired
                  5 = Educable Mentally Handicapped
                  6 = Specific Learning Disabled
                  7 = Speech-Language Impaired
                  8 = Visually Impaired 
                  9 = Other Health Impaired
                10 = Orthopedically Impaired
                11 = Traumatic Brain Injured
                12 = Other Exceptional Classifications

    2001-2002:    1 = Not identified as an Exceptional Student
                  2 = Academically/Intellectually Gifted (AIG)
                  3 = Behaviorally-Emotionally Handicapped
                  4 = Hearing Impaired
                  5 = Educable Mentally Handicapped
                  6 = Specific Learning Disabled
                  7 = Speech-Language Impaired
                  8 = Visually Impaired 
                  9 = Other Health Impaired
                10 = Orthopedically Impaired
                11 = Traumatic Brain Injured
                12 = Autistic
                13 = Severe/Profound Mentally Disabled
                14 = Multihandicapped
                15 = Deaf-Blind
                16 = Trainable Mentally Handicapped

    2003-:        1 = Not identified as an Exceptional Student
                  2 = Academically/Intellectually Gifted (AIG)
                  3 = Behaviorally-Emotionally Handicapped
                  4 = Hearing Impaired
                  5 = Educable Mentally Handicapped
                  6 = Deaf-Blind
                  7 = Visually Impaired
                  8 = Other Health Impaired 
                  9 = Orthopedically Impaired
                10 = Traumatic Brain Injured
                11 = Severe/Profound Mentally Disabled 
                12 = Multihandicapped
                13 = Speech-Language Impaired
                14 = Trainable Mentally Handicapped
                15 = Specific Leaning Disabled
                16 = Autistic

    schllunch:
    1997-2002:  0 = Student not eligible for National School Lunch Program
                1 = Reduced price lunch
                2 = Free lunch
                3 = Information not available
                4 = School not participating

    2003-       2 = Free lunch
                3 = Reduced price lunch
                4 = Full pay; School not participating; Student not eligible

    
    EOC coding:
    pared:
    2003-:      A = Did not finish high school
                B = High school graduate
                C = Some additional education after high school, but did not graduate
                D = Trade or business school graduate
                E = Community, technical or junior college graduate
                F = Four-year college graduate
                G = Graduate school degree

    except:
    2003:        1 = Not identified as an Exceptional Student
                  2 = Academically/Intellectually Gifted (AIG)
                  3 = Behaviorally-Emotionally Handicapped
                  4 = Hearing Impaired
                  5 = Educable Mentally Handicapped
                  6 = Specific Learning Disabled
                  7 = Speech-Language Impaired
                  8 = Visually Impaired
                  9 = Other Health Impaired
                10 = Orthopedically Impaired
                11 = Traumatic Brain Injured
                12 = Autistic
                13 = Severe/Profound Mentally Disabled
                14 = Multihandicapped
                15 = Deaf-Blind
                16 = Trainable Mentally Handicapped

    2004-:      same as EOG

    MBuild coding:
    pared:
    2006-2008:      1 = Less than high school
                    2 = High school graduate
                    3 = Some education after high school
                    4 = Trade school
                    5 = Junior college 
                    6 = 4-year college
                    7 = Graduate school

    '''
    print("\n\nAdding demos from EOG / EOC / Mbuild files")
    demos = []
    for year in range(1996,lastyr+1):
        # EOG
        for grade in range(3,9):
            if ((year == 1996) & (grade >= 5)):
                continue
            print("Loading EOG for {} grade {}".format(year,grade))
            with gzip.open(data_path + "/Student/End of Grade/{}/eog{}pub{}.dta.gz".format(year,grade,str(year)[-2:]), 'rb') as inF:
                eog = pd.read_stata(inF)

            # Check if present
            cols = ['mastid','sex','ethnic','bdate','pared','except','id504','schlunch','frl','ed',
                        'limeng','lep','lang','lepstat','aigmath','aigread','twin']
            for col in cols:
                if col not in eog.columns:
                    eog[col] = None

            # Add to list
            eog = eog.loc[eog.mastid.notnull(), cols].drop_duplicates()
            eog['year'] = year
            eog['source'] = 'eog'
            demos += [eog,]

        # EOC
        for subject in ['eng1','alg1','alg2','ush','civ','geom','bio']:
            print("Loading EOC for {} subject {}".format(year,subject))
            try:
                with gzip.open(data_path + "/Student/End of Course/{}/{}pub{}.dta.gz".format(year,subject,str(year)[-2:]), 'rb') as inF:
                    eoc = pd.read_stata(inF)
            except Exception as e:
                print(e)
                continue

            # Check if present
            cols = ['mastid','sex','ethnic','pared','bdate','except','id504','schlunch','frl','lepstat',
                        'aigmath','aigread','twin']
            for col in cols:
                if col not in eoc.columns:
                    eoc[col] = None

            # Add to list
            eoc = eoc.loc[eoc.mastid.notnull(), cols].drop_duplicates()
            eoc['year'] = year
            eoc['source'] = 'eoc'

            demos += [eoc,]

        # Mbuild
        if year == 1996:
            continue        
        print("Loading Mbuild for {}".format(year))
        with gzip.open(data_path + "/Student/MBuild/mb_{}_pub.dta.gz".format(year), 'rb') as inF:
            mb = pd.read_stata(inF)

        # Check if present
        cols = ['mastid','sex','ethnic','bdate','pared','eds','frl',
                        'lep','lepflag','aig','aig_m','aig_math','aig_r']
        for col in cols:
            if col not in mb.columns:
                mb[col] = None

        # Add to list
        mb = mb.loc[mb.mastid.notnull(), cols].drop_duplicates()
        mb['year'] = year
        mb['source'] = 'mbuild'
        demos += [mb,]

    # Combine
    demo = pd.concat(demos, sort=False, ignore_index=True).drop_duplicates()

    # Recode twin indicators, which are year specific
    twin = demo.loc[demo.twin.notnull(), ['mastid','year','source','twin']].sort_values(['twin','mastid'])
    twin['twin_id'] = twin.groupby(['year','source','twin']).grouper.group_info[0]
    twin = twin.loc[twin.groupby('twin_id').mastid.transform('nunique') > 1]
    twin = twin.groupby('mastid').twin_id.max()

    # Recode parents education
    demo.loc[(demo.source == "eoc") & (demo.year < 2000), 'pared'] = None
    demo['pared_nohs'] = demo.pared.astype(str).isin(["1","A"]).astype(int)
    demo['pared_hsorless'] = demo.pared.astype(str).isin(["1","2","A","B"]).astype(int)
    demo['pared_somecol'] = demo.pared.astype(str).isin(["3","4","5","6","7","C","D","E","F","G"]).astype(int)
    demo['pared_baormore'] = demo.pared.astype(str).isin(["6","7","F","G"]).astype(int)
    demo.loc[demo.year.isin([1996,1997,1998,1999,2001,2002]) & (demo.source == 'eog'), 
                        'pared_baormore'] = demo.pared.astype(str).isin(["5","6"]).astype(int)
    for col in ['pared_nohs','pared_hsorless','pared_somecol','pared_baormore']:
        demo.loc[demo.pared.isnull() | (demo.pared == ""), col] = None

    # Recode exceptionality
    demo['exc_not'] = (pd.to_numeric(demo['except']) == 1).astype(int)
    demo['exc_aig'] = (pd.to_numeric(demo['except']) == 2).astype(int)
    demo['exc_behav'] = (pd.to_numeric(demo['except']) == 3).astype(int)
    demo['exc_educ'] = (pd.to_numeric(demo['except']) == 5).astype(int)
    for col in ['exc_not','exc_aig','exc_behav','exc_educ']:
        demo.loc[pd.to_numeric(demo['except']).isnull(), col] = None
    
    # id504 is sometimes 1 and sometimes "Y"
    demo.loc[demo.id504.notnull(), 'id504'] = demo.id504.isin(['1','Y',1]).astype(int)

    # Free lunches
    demo['fr_lunch'] = (demo.frl.isin(['F','R']) | demo.schlunch.isin([2,3,'2','3'])).astype(int)
    demo.loc[demo.schlunch.notnull() & (demo.year <= 2002), 'fr_lunch'] = demo.schlunch.isin(['1','2',1,2]).astype(int)
    demo.loc[demo.frl.isnull() & demo.schlunch.isnull(), 'fr_lunch'] = None

    # Gifted indicators
    demo.loc[demo.aigmath.notnull(), 'aigmath'] = demo.aigmath.isin(['1','Y',1]).astype(int)
    demo.loc[demo.aigread.notnull(), 'aigread'] = demo.aigread.isin(['1','Y',1]).astype(int)

    # Disadvantaged indicator
    demo.loc[demo.fr_lunch.notnull() | demo.eds.notnull(), 'disadv'] = ((demo.fr_lunch == 1) | (demo.eds == "Y")).astype(int)

    # LEP status
    demo['lim_eng'] = (demo.limeng.isin(['1',1,'Y']) | 
                        ((demo.source == "Mbuild") & demo.lep.isin(['Y','1',1])) |
                        demo.lepstat.isin([2,3,4,5,'2','3','4','5']) |
                        demo.lepflag.isin([1,2,3,4,'1','2','3','4'])
                        ).astype(int)
    demo.loc[demo.limeng.isnull() & demo.lep.isnull() & demo.lepstat.isnull() & demo.lepflag.isnull(), 'lim_eng'] = None

    # Make bdates datetime
    demo['bdate'] = pd.to_datetime(demo.bdate, errors='coerce')

    # Fix any collisions
    tmp = [twin,]
    for col in ['sex','ethnic','bdate']:
        print("Fixing collisions for {}".format(col))
        demo.loc[demo[col] == "", col] = None
        demo['{}_nuni'.format(col)] = demo.groupby(['mastid',col])['year'].transform('nunique')
        tmp += [demo.loc[demo['{}_nuni'.format(col)] == demo.groupby('mastid')['{}_nuni'.format(col)].transform('max'),
                ['mastid',col]].drop_duplicates('mastid').set_index('mastid'),]

    for col in ['pared_nohs','pared_hsorless','pared_somecol','pared_baormore']:
        print("Fixing collisions for {}".format(col))
        demo.loc[demo[col] == "", col] = None
        demo['{}_nuni'.format(col)] = demo.groupby(['mastid',col])['year'].transform('nunique')
        tmp += [demo.loc[demo['{}_nuni'.format(col)] == demo.groupby('mastid')['{}_nuni'.format(col)].transform('max'),
                ].groupby('mastid')[col].max(),]

    for col in ['exc_not','exc_aig','exc_behav','exc_educ','id504','fr_lunch','aigmath','aigread','disadv','lim_eng']:
        print("Taking max for {}".format(col))
        tmp += [demo.groupby('mastid')[col].max(),]
    
    tosave = pd.concat(tmp, axis=1, sort=False).reset_index()
    assert tosave.mastid.duplicated().sum() == 0

    # Save 
    tosave.to_pickle("data/demos.pkl.gz", compression='gzip')


### 6) Add disciplinary data
    susps = []
    for year in range(2002,2017):
        if year == 2005:
            continue
        print('Working on suspensions in year {}'.format(year), flush=True)

        if year in [2002,2003,2004]:
            with gzip.open(data_path + "/Student/Suspension/mastsusp{}r.dta.gz".format(str(year)[-2:]), 'rb') as inF:
                _tmp = pd.read_stata(inF, convert_dates = False)
        else:
            with gzip.open(data_path + "/Student/Suspension/mastsusp{}.dta.gz".format(year), 'rb') as inF:
                _tmp = pd.read_stata(inF, convert_dates = False)
        
        consequence_columns = [c for c in _tmp.columns if 'c_conseq' in c or 'consqtype' in c]
        if year <= 2004:
            _tmp['any_discp'] = 1
            _tmp['iss'] = _tmp[consequence_columns].apply(lambda x: any([a == 3 for a in x]), axis=1).astype(int)
            _tmp['oss'] = _tmp[consequence_columns].apply(lambda x: any([a == 4 for a in x]), axis=1).astype(int)
            _tmp['explusion'] = _tmp[consequence_columns].apply(lambda x: any([a == 5 for a in x]), axis=1).astype(int)
            _tmp['detention'] = (_tmp.c_deten > 0).astype(int)          
        elif year >= 2006:
            _tmp['any_discp'] = 1
            _tmp['iss'] = _tmp[consequence_columns].apply(lambda x: any([(a == "002") | (a == "105") for a in x]), axis=1).astype(int)
            _tmp['oss'] = _tmp[consequence_columns].apply(lambda x: any([a in ["003","004","005","035"] for a in x]), axis=1).astype(int)
            _tmp['oss10'] = _tmp[consequence_columns].apply(lambda x: any([a in ["003","035"] for a in x]), axis=1).astype(int)
            _tmp['oss11'] = _tmp[consequence_columns].apply(lambda x: any([a == "004" for a in x]), axis=1).astype(int)
            _tmp['oss365'] = _tmp[consequence_columns].apply(lambda x: any([a == "005" for a in x]), axis=1).astype(int)
            _tmp['explusion'] = _tmp[consequence_columns].apply(lambda x: any([a == "006" for a in x]), axis=1).astype(int)
            _tmp['conf_parent'] = _tmp[consequence_columns].apply(lambda x: any([a == "030" for a in x]), axis=1).astype(int)
            _tmp['report_law'] = _tmp[consequence_columns].apply(lambda x: any([a == "012" for a in x]), axis=1).astype(int)
            _tmp['detention'] = _tmp[consequence_columns].apply(lambda x: any([a == "021" for a in x]), axis=1).astype(int)

        # Fix formats and keep students in grades 1-12
        _tmp['grade'] = pd.to_numeric(_tmp['grade'], errors='coerce')
        _tmp = _tmp.loc[_tmp.mastid.notnull() & _tmp.grade.notnull()]

        # Collapse to sums
        keepcols = ['any_discp','iss','oss','detention','oss10','oss11','oss365','explusion','conf_parent','report_law']
        s = _tmp.groupby('mastid')[[c for c in keepcols if c in _tmp.columns]].max().reset_index()
        s['year'] = year
        susps += [s,]

    susp = pd.concat(susps)
    susp.drop_duplicates(inplace=True)
    assert susp[['mastid','year']].duplicated().sum() == 0
    susp.to_pickle("data/susp.pkl.gz", compression='gzip')


### 6b) Add absences data from MBuild files
if 1:
    '''
    In mbuild, days daysabs exists only in 2004-2005m 2013-, days member exists in 2006- in mbuild files
    In accdemo, daysabs exists 2006-2015, also has times_tardy, unexc_abs, times_in_susp, times_out_susp, days_in_susp, days_out_susp
        at least in 2006-2011
    '''
    acc = []
    for yr in range(2004,2017):
        print("Loading attendence data for {}".format(yr))
        if yr <= 2005:
            with gzip.open(data_path + "/Student/MBuild/mb_{}_pub.dta.gz".format(yr), 'rb') as inF:
                _acc = pd.read_stata(inF)
        elif (yr>2005) & (yr <=2012):
            with gzip.open(data_path + "/Student/MBuild/ACCDEMO/accdemopub{}.dta.gz".format(yr), 'rb') as inF:
                _acc = pd.read_stata(inF)
        elif (yr > 2012) & (yr < 2016):
            with gzip.open(data_path + "/Student/MBuild/ACCDEMO/accdemo_pub{}.dta.gz".format(yr), 'rb') as inF:
                _acc = pd.read_stata(inF)
        elif yr == 2016:
            with gzip.open(data_path + "/Student/MBuild/pcaudit_pub{}.dta.gz".format(yr), 'rb') as inF:
                _acc = pd.read_stata(inF)

        # Fix formats and keep students in grades 1-12
        _acc['grade'] = pd.to_numeric(_acc['grade'], errors='coerce')
        _acc = _acc.loc[_acc.mastid.notnull() & _acc.grade.notnull()] 

        # Collect relevant information
        keepvars = ['daysabs', 'daysmem', 'times_tardy', 'days_in_susp', 'days_out_susp', 'times_in_susp', 'times_out_susp']
        _acc = _acc[[l for l in keepvars if l in _acc.columns] + ['mastid']]
        for k in _acc.columns:
            _acc[k] = pd.to_numeric(_acc[k], errors='coerce')
        for col in [l for l in keepvars if l in _acc.columns]:
            if _acc[col].isnull().min() == 1:
                _acc.drop(col, axis=1, inplace=True)
        _acc = _acc.groupby(['mastid'])[[l for l in keepvars if l in _acc.columns]].sum().reset_index()
        _acc['year'] = int(yr)
        acc += [_acc, ]

    acc = pd.concat(acc)
    acc.drop_duplicates(inplace=True)
    assert acc[['mastid','year']].duplicated().sum() == 0
    acc.to_pickle("data/absences.pkl.gz", compression='gzip')


### 7) Add graduation data
if 1:
    ls_files = []
    for yr in range(2006,2016+1):
        print('Working on Student Exit file from year {}'.format(yr))
        with gzip.open(data_path + "/Student/School Exit/exit_pub{}.dta.gz".format(str(yr)), 'rb') as inF:
            _tmp = pd.read_stata(inF, convert_dates = False)
        _tmp = _tmp.loc[_tmp.mastid.notnull(), ['mastid', 'effective_exit_code', 'grad_year', 'dropout_year']].drop_duplicates()

        _tmp = _tmp.loc[_tmp.effective_exit_code.isin(['4','6','9'])].drop_duplicates() # Keep if student dropped out, died, or graduated
        _tmp['grad']  = (_tmp['effective_exit_code'] == '9').astype(int)
        _tmp['dead']  = (_tmp['effective_exit_code'] == '6').astype(int)
        _tmp['dropout']  = (_tmp['effective_exit_code'] == '4').astype(int)

        for ll in ['grad', 'dead', 'dropout']:
            _tmp[ll] = _tmp.groupby(['mastid'])[ll].transform('max')
        _tmp['grad'].where((_tmp['dropout'] == 0) | (_tmp['dead'] == 0), 0, inplace=True)


        for ll in ['grad', 'dropout']:
            _tmp[ll+'_year'] = pd.to_numeric(_tmp[ll+'_year'], errors='coerce')
            _tmp[ll+'_year'].where(_tmp[ll] == 1, np.nan, inplace=True)
            _tmp[ll+'_year'] = _tmp.groupby(['mastid'])[ll+'_year'].transform('max')
        _tmp['year_stuExit_file'] = yr

        del _tmp['effective_exit_code']
        _tmp.drop_duplicates(inplace=True)

        assert _tmp.mastid.duplicated().sum() == 0
        ls_files += [_tmp,]
        del _tmp
    grd = pd.concat(ls_files)
    del ls_files

    # Fix multiple mastid enteries
    for ll in ['grad', 'dead', 'dropout']:
        grd[ll] = grd.groupby(['mastid'])[ll].transform('max')
    grd['grad'].where(grd['dead'] == 0, 0, inplace=True)
    grd['dropout'].where(grd['dead'] == 0, 0, inplace=True)

    for ll in ['grad', 'dropout']:
        grd[ll+'_year'] = grd.groupby(['mastid'])[ll+'_year'].transform('max')

    # Max year student appear's
    grd['max_year_stuExit_file'] = grd.groupby(['mastid'])['year_stuExit_file'].transform('max')
    del grd['year_stuExit_file']

    # Too little students die (about 1300) so dropping that variable
    del grd['dead']
    grd.drop_duplicates(inplace=True)

    # save
    assert grd.mastid.duplicated().sum() == 0
    grd[['mastid','grad','dropout']].to_pickle(
        "data/schoolExit.pkl.gz", compression='gzip')


### 9) Add 12th grade GPA data, college plans, etc.
if 1:
    print("\n\nWorking on GPA files")
    finals = []
    for year in range(2005,2011):
        print("Loading GPA data for {}".format(year))
        with gzip.open(data_path + "/Student/GPA/gpa{}.dta.gz".format(year), 'rb') as inF:
            gpa = pd.read_stata(inF)
            gpa['class_rank_uw'] = gpa.rank_unweighted / gpa.rank_number
            gpa['class_rank_w'] = gpa.rank_weighted / gpa.rank_number
            gpa = gpa[['mastid','bound_for_code','gpa_unweighted','gpa_weighted','class_rank_uw','class_rank_w']]
            gpa['year'] = year
        finals += [gpa,]

    for year in range(2012,2014):
        print("Loading GPA data for {}".format(year))
        with gzip.open(data_path + "/Student/GPA/gpa{}.dta.gz".format(year), 'rb') as inF:
            gpa = pd.read_stata(inF)
            gpa['class_rank_uw'] = gpa.rank_unweighted / gpa.rank_number
            gpa['class_rank_w'] = gpa.rank_weighted / gpa.rank_number
            gpa = gpa[['mastid','bound_for_code','gpa_unweighted','gpa_weighted','class_rank_uw','class_rank_w']]
            gpa['bound_for_code'] = gpa.bound_for_code.replace({'9':'1','10':'2','11':'3','12':'3','13':'5'})
            gpa['year'] = year
        finals += [gpa,]

    for year in range(2014,2017):
        print("Loading GPA data for {}".format(year))
        with gzip.open(data_path + "/Student/GPA/gpa{}.dta.gz".format(year), 'rb') as inF:
            gpa = pd.read_stata(inF)
            gpa['class_rank_uw'] = gpa.rank_unweighted / gpa.rank_number_unweight
            gpa['class_rank_w'] = gpa.rank_weighted / gpa.rank_number_weighted
            gpa = gpa[['mastid','bound_for','gpa_unweighted','gpa_weighted','class_rank_uw','class_rank_w']]
            gpa['bound_for_code'] = gpa.bound_for.replace({
                'COMMUNITY & TECHNICAL COLLEGES':'3',
                'PUBLIC SENIOR INSTITUTIONS':'1',
                'EMPLOYMENT':'7',                       
                'PRIVATE SENIOR INSTITUTIONS':'2',        
                'MILITARY':'6',                           
                'PUBLIC SENIOR INST -OUT OF NC':'1',      
                'PRIVATE SENIOR INST -OUT OF NC':'2',    
                'OTHER':'8',                            
                'TRADE, BUSINESS, NURSING SCHS':'5',       
                'COMM & TECH COLLEGES-OUT OF NC':'3',      
                'PRIVATE JUNIOR COLLEGE':'4',              
                'TRADE, BUS, NURS SCH-OUT OF NC':'5',
                'PRIVATE JR COLLEGE -OUT OF NC':'4',
                })
            gpa['year'] = year
        finals += [gpa.drop('bound_for',axis=1),]        
    
    # concat
    gpa = pd.concat(finals)
    gpa = gpa.drop_duplicates()
    gpa = gpa.loc[gpa.gpa_unweighted.notnull() | gpa.gpa_weighted.notnull()]
    gpa = gpa.loc[~gpa.duplicated(['mastid','year'], keep=False)]
    gpa = gpa.loc[gpa.year == gpa.groupby('mastid').year.transform('min')]

    # save
    assert gpa.mastid.duplicated().sum() == 0
    gpa.to_pickle("data/gpa.pkl.gz", compression='gzip')


### 8) Add school data
if 1:
    print("\n\nWorking on school files")
    finals = []
    for year in range(2006,2013+1):
        print("Loading school data for {}".format(year))
        with gzip.open(data_path + "/School/Public School Universe/ccdpsu{}.dta.gz".format(year), 'rb') as inF:
            schl = pd.read_stata(inF)
            schl['year'] = year

        schl = schl.rename(columns=lambda x: re.sub("[0-9]{2}$","",x))
        schl = schl[['lea','schlcode','chartr','magnet','type','year']].rename(columns={'type':'schl_type'})       # 1 = yes, 2 = no
        finals += [schl,]

    # Combine, collapse, and save
    schl = pd.concat(finals)
    schl['charter_schl'] = (schl.chartr.astype(str) == "1").astype(int)
    schl['magnet_schl'] = (schl.magnet.astype(str) == "1").astype(int)
    schl['voc_schl'] = (schl.schl_type.astype(str) == "3").astype(int)
    schl['spec_ed_schl'] = (schl.schl_type.astype(str) == "2").astype(int)
    schl['alt_schl'] = (schl.schl_type.astype(str) == "4").astype(int)
    schl = schl.groupby(['lea','schlcode'])[['charter_schl','magnet_schl','voc_schl','spec_ed_schl','alt_schl']].max().reset_index()
    schl.to_pickle("data/schls.pkl.gz", compression='gzip')

### 10) SAR file data and student counts
if 1:
    print("\n\nWorking on SAR files")
    finals = []
    for year in range(1997,2014):
        print("Loading teacher data for {}".format(year))
        with gzip.open(data_path + "/Classroom/Personnel/pers{}.dta.gz".format(year), 'rb') as inF:
            sar = pd.read_stata(inF)
            sar['year'] = year    
        finals += [sar[['teachid','lea','schlcode','year','acadlvl','astype','subjct','crsnum','secnum','semstr']],]

    sar = pd.concat(finals, sort=False, ignore_index=True)
    sar['semstr'] = pd.to_numeric(sar.semstr)
    sar.to_pickle("data/sar.pkl.gz", compression='gzip')


### 10) Build crime indicators
if 1:
    # Latest AOC data (dates are basically through the end of 2019)
    aoc_data = pd.read_pickle("aoc_data/offense_newonly_2020-09-08.pkl", compression='gzip')

    # Remove motions --- rows with a charge code of "MOTIONS"
    aoc_data = aoc_data.loc[~aoc_data.off_charged_code.isin([5046])]   

    # Remove all habitual charges and other enhancements, 9901 is extradition, 9902 Habeas, 9904 is governors warrant
    aoc_data = aoc_data.loc[~aoc_data.off_charged_code.isin([9922,9923,9921,2316,5256,5527,1393,9901,9902,9904])]  

    # Drop show cause failures to comply, contempts, FTAs, bill of particulars
    aoc_data = aoc_data.loc[~aoc_data.off_charged_code.isin([5034,5022,5026,5028,5029,5020])]  

    # Add sentence data
    aoc_data.loc[aoc_data.min_sent_frame == "D", 'sent'] = pd.to_numeric(aoc_data.min_sent_length, errors='coerce')    
    aoc_data.loc[aoc_data.min_sent_frame == "M", 'sent'] = pd.to_numeric(aoc_data.min_sent_length, errors='coerce') * 30.42        
    aoc_data.loc[aoc_data.min_sent_frame == "Y", 'sent'] = pd.to_numeric(aoc_data.min_sent_length, errors='coerce') * 365.25   

    # 09/2020 pii and data
    if 1:
        ids = pd.read_pickle(home + '/data/Crime/match_09_2020/NC_ids_crosswalk_2020-09-08.pkl.gz', compression='gzip')
    # 07/2017 pii and data
    else:
        ids = pd.read_pickle(home + '/data/Crime/match_05_2017/person_identifiers_NCERDC.pkl.gz', compression='gzip').drop(['defssn','sbi_no','fbi_no','initial_group3','initial_group8','defadd_street','defadd_city','defadd_state','defadd_zip'], axis=1)

    aoc_data = aoc_data.merge(ids[['crrkey','group_id']], how='left', on='crrkey')

    # Load codes and categories
    codes = pd.read_csv(home + "/aux/valid_codes.csv")
    ucr_codes = pd.read_csv(home + "/aux/nc_offensecodes_categories.csv")

    # Any arrest, any valid arrest, ucr categories, same for conviction
    aoc_data['any'] = 1
    aoc_data['crim'] = aoc_data.off_arrg_code.isin(codes.loc[codes.Exclude != "Yes"].CODE.unique()).astype(int)
    aoc_data['index'] = aoc_data.off_arrg_code.isin(ucr_codes.loc[ucr_codes['UCR Code Classification'].str.strip().isin([
                            'Aggravated Assault','Forcible Rape','Robbery','Criminal Homicide','Arson','Burglary','Larceny','Motor Vehicle Theft']),
                                'NCAOC Criminal Code'].unique()).astype(int)
    aoc_data['viol'] = aoc_data.off_arrg_code.isin(ucr_codes.loc[ucr_codes['Case Category for SEP Outcome Study'].str.strip().isin([
                            'Aggravated Assault','Other Assaults','Criminal Homicide','Kidnapping','Forcible Rape','Robbery']),
                                'NCAOC Criminal Code'].unique()).astype(int)
    aoc_data['dwi'] = aoc_data.off_arrg_code.isin(ucr_codes.loc[ucr_codes['Case Category for SEP Outcome Study'].str.strip().isin([
                            'Driving Under the Influence']),
                                'NCAOC Criminal Code'].unique()).astype(int)
    aoc_data['drug'] = aoc_data.off_arrg_code.isin(ucr_codes.loc[ucr_codes['Case Category for SEP Outcome Study'].str.strip().isin([
                            'Drug Abuse Violations']),
                                'NCAOC Criminal Code'].unique()).astype(int)
    aoc_data['traff'] = aoc_data.off_arrg_code.isin(ucr_codes.loc[ucr_codes['Case Category for SEP Outcome Study'].str.strip().isin([
                            'Traffic Offenses']),
                                'NCAOC Criminal Code'].unique()).astype(int)
    aoc_data['infrac'] = aoc_data.off_arrg_code.isin(ucr_codes.loc[ucr_codes['Case Category for SEP Outcome Study'].str.strip().isin([
                            'Infractions']),
                                'NCAOC Criminal Code'].unique()).astype(int)

    # Additional crime categories added by YST to calculate costs of crime measures (namings are to match READI QJI Table Table A.XVIII in WP)
    aoc_data['cost_murder'] = aoc_data.off_arrg_code.isin(ucr_codes.loc[ucr_codes['Case Category for SEP Outcome Study'].str.strip().isin([
                        'Criminal Homicide']),
                            'NCAOC Criminal Code'].unique()).astype(int)
    aoc_data['cost_rape'] = aoc_data.off_arrg_code.isin(ucr_codes.loc[ucr_codes['Case Category for SEP Outcome Study'].str.strip().isin([
                        'Forcible Rape','Sex Offenses']),
                            'NCAOC Criminal Code'].unique()).astype(int)
    aoc_data['cost_roberry'] = aoc_data.off_arrg_code.isin(ucr_codes.loc[ucr_codes['Case Category for SEP Outcome Study'].str.strip().isin([
                        'Robbery']),
                            'NCAOC Criminal Code'].unique()).astype(int)
    aoc_data['cost_aggravated_assault'] = aoc_data.off_arrg_code.isin(ucr_codes.loc[ucr_codes['Case Category for SEP Outcome Study'].str.strip().isin([
                        'Aggravated Assault','Kidnapping','Abuse of Elderly/Blind/Disabled','Explosives, Terrorism, and Bombing Offenses']),
                            'NCAOC Criminal Code'].unique()).astype(int)
    aoc_data['cost_assault'] = aoc_data.off_arrg_code.isin(ucr_codes.loc[ucr_codes['Case Category for SEP Outcome Study'].str.strip().isin([
                        'Other Assaults','Offenses Against the Family and Children']),
                            'NCAOC Criminal Code'].unique()).astype(int)
    aoc_data['cost_burglary'] = aoc_data.off_arrg_code.isin(ucr_codes.loc[ucr_codes['Case Category for SEP Outcome Study'].str.strip().isin([
                        'Burglary']),
                            'NCAOC Criminal Code'].unique()).astype(int)
    aoc_data['cost_motor_vehicle_theft'] = aoc_data.off_arrg_code.isin(ucr_codes.loc[ucr_codes['Case Category for SEP Outcome Study'].str.strip().isin([
                        'Motor Vehicle Theft']),
                            'NCAOC Criminal Code'].unique()).astype(int)
    aoc_data['cost_larceny'] = aoc_data.off_arrg_code.isin(ucr_codes.loc[ucr_codes['Case Category for SEP Outcome Study'].str.strip().isin([
                        'Larceny/Theft (except motor vehicle theft)','Stolen Property: Buying, Receiving, Possessing']),
                            'NCAOC Criminal Code'].unique()).astype(int)
    aoc_data['cost_drunk_driving_crash'] = aoc_data.off_arrg_code.isin(ucr_codes.loc[ucr_codes['Case Category for SEP Outcome Study'].str.strip().isin([
                        'Driving Under the Influence']),
                            'NCAOC Criminal Code'].unique()).astype(int)
    aoc_data['cost_arson'] = aoc_data.off_arrg_code.isin(ucr_codes.loc[ucr_codes['Case Category for SEP Outcome Study'].str.strip().isin([
                        'Arson']),
                            'NCAOC Criminal Code'].unique()).astype(int)
    aoc_data['cost_vandalism'] = aoc_data.off_arrg_code.isin(ucr_codes.loc[ucr_codes['Case Category for SEP Outcome Study'].str.strip().isin([
                        'Damage Property/Vandalism']),
                            'NCAOC Criminal Code'].unique()).astype(int)
    aoc_data['cost_fraud'] = aoc_data.off_arrg_code.isin(ucr_codes.loc[ucr_codes['Case Category for SEP Outcome Study'].str.strip().isin([
                        'Fraud','Forgery and Counterfeiting']),
                            'NCAOC Criminal Code'].unique()).astype(int)
    aoc_data['cost_other_offense'] = 1 -  aoc_data.filter(regex=("cost_.*")).max(axis=1)
    assert aoc_data.filter(regex=("cost_.*")).max(axis=1).min() == 1

    # We inflate the cost estimate for each crime type by the likelihood of reporting and then by the likelihood an arrest will be made 
    aoc_data['cost_wtp'] = (aoc_data['cost_murder']/1/0.37)*13429758 + (aoc_data['cost_rape']/0.28/0.32)*327705 + (aoc_data['cost_roberry']/0.09/0.52)*304179 + (aoc_data['cost_aggravated_assault']/0.15/0.56)*76961 + (aoc_data['cost_assault']/0.11/0.37)*15014  + (aoc_data['cost_burglary']/0.09/0.48)*37476 + (aoc_data['cost_motor_vehicle_theft']/0.12/0.78)*15487 + (aoc_data['cost_larceny']/0.11/0.28)*3310 + (aoc_data['cost_drunk_driving_crash']/0.09/1)*68095 + (aoc_data['cost_arson']/0.09/1)*133115 + (aoc_data['cost_vandalism']/0.09/0.28)*1182 + (aoc_data['cost_fraud']/0.09/0.28)*4138 + (aoc_data['cost_other_offense']/0.09/0.28)*591
    aoc_data['cost_bottomup'] = (aoc_data['cost_murder']/1/0.37)*5910985 + (aoc_data['cost_rape']/0.28/0.32)*177330 + (aoc_data['cost_roberry']/0.09/0.52)*59110 +  (aoc_data['cost_aggravated_assault']/0.15/0.56)*65021 + (aoc_data['cost_assault']/0.11/0.37)*13004 + (aoc_data['cost_burglary']/0.09/0.48)*5911 + (aoc_data['cost_motor_vehicle_theft']/0.12/0.78)*10640 + (aoc_data['cost_larceny']/0.11/0.28)*1892 + (aoc_data['cost_drunk_driving_crash']/0.09/1)*35466 + (aoc_data['cost_arson']/0.09/1)*70932 + (aoc_data['cost_vandalism']/0.09/0.28)*1620 + (aoc_data['cost_fraud']/0.09/0.28)*3665 + (aoc_data['cost_other_offense']/0.09/0.28)*591
    aoc_data = aoc_data.drop(['cost_murder','cost_rape','cost_roberry','cost_aggravated_assault','cost_assault','cost_burglary','cost_motor_vehicle_theft','cost_larceny','cost_drunk_driving_crash','cost_arson','cost_vandalism','cost_fraud','cost_other_offense'], axis=1)
    assert ((aoc_data['cost_bottomup']<=0).sum()==0) & (aoc_data['cost_wtp']<=0).sum()==0

    # Add convictions
    cols = ['any','crim','index','viol','dwi','drug','traff','infrac']
    for col in cols:
        aoc_data[col + '_conv'] = ((aoc_data[col] == 1) & (aoc_data.off_convic_code.notnull())).astype(int)

    # Add incar
    aoc_data['incar'] = aoc_data.sent.notnull().astype(int)

    # Subset 
    tosave = aoc_data[['group_id','off_date','incar'] + cols + [c + '_conv' for c in cols] + ['cost_wtp','cost_bottomup']]
    tosave = tosave.loc[tosave.group_id.notnull() & tosave.off_date.notnull()]
    tosave = tosave.drop_duplicates()   # This removes multiple charges for same categories stemming from offenses on the same day

    # Replace identifiers
    ident = pd.read_sas(home + "/data/Crime/match_09_2020/ncindent20_match.sas7bdat", encoding='ascii').rename(columns={'person_id':'group_id'})
    tosave = tosave.merge(ident[['group_id','mastid','matchtype']], how='inner', on='group_id')
    tosave = tosave.loc[tosave.mastid.notnull()]

    # Save
    tosave.to_pickle("data/aoc.pkl.gz", compression='gzip')

### 11) Combine into panel based on EOG data
if 1:
    # Load EOG files
    data = pd.read_pickle("data/eog.pkl.gz", compression='gzip').drop(['sex','ethnic','bdate'], axis=1).drop_duplicates()
    data = data.loc[~data.duplicated(['mastid','year'])]
        # Unique in mastid-year, drops ssmall share of sstudents in multipel schools

    # Add teacher assignments from CMB files
    cmb = pd.read_pickle("data/cmb.pkl.gz", compression='gzip').rename(
            columns={'sex':'sex_cmb','ethnic':'ethnic_cmb'})
    cmb = cmb.loc[cmb.grade.between(3,8)]
    cmb = cmb.loc[cmb.teachid.notnull()]
    for course, lb, ub in [('hr',0,0),('math',2000,2999),('eng',1000,1999)]:
        tmp = cmb.loc[cmb.coursecode.between(lb,ub), ['mastid','year','grade','teachid','classroom_id','semester']].drop_duplicates()
        tmp['semester'] = pd.to_numeric(tmp.semester, errors='coerce').fillna(0)
        tmp = tmp.loc[tmp.semester == tmp.groupby(['mastid','year','grade']).semester.transform('max')] # Take spring semester where not null
        tmp = tmp.loc[tmp.groupby(['mastid','year','grade']).teachid.transform('nunique') == 1]
        tmp = tmp.groupby(['mastid','year','grade'])[['teachid','classroom_id']].first().reset_index()
        tmp = tmp.rename(columns={'teachid':'cmb_teachid_{}'.format(course),'classroom_id':'cmb_classroom_id_{}'.format(course)})
        data = data.merge(tmp, how='left', on=['mastid','year','grade'])

    # Add teacher validity info from SAR files to each teacher ID
    sar = pd.read_pickle("data/sar.pkl.gz", compression='gzip')
    sar['sar_reg_hr'] = ((pd.to_numeric(sar.subjct, errors='coerce') == 0)*(sar['astype'] == "TE")*(pd.to_numeric(sar.acadlvl, errors='coerce') == 2)).astype(int)
    sar['sar_reg_math'] = ((pd.to_numeric(sar.subjct, errors='coerce').between(2000,2999))*(sar['astype'] == "TE")*(pd.to_numeric(sar.acadlvl, errors='coerce') == 2)).astype(int)
    sar['sar_reg_eng'] = ((pd.to_numeric(sar.subjct, errors='coerce').between(1000,1999))*(sar['astype'] == "TE")*(pd.to_numeric(sar.acadlvl, errors='coerce') == 2)).astype(int)
    for col in ['teachid_eogm','teachid_eogr','cmb_teachid_hr','cmb_teachid_math','cmb_teachid_eng']:
        sar[col] = sar.teachid
        sar['{}_sar_valid_hr'.format(col)] = sar.sar_reg_hr
        sar['{}_sar_valid_math'.format(col)] = sar.sar_reg_math
        sar['{}_sar_valid_eng'.format(col)] = sar.sar_reg_eng
        data = data.merge(sar.groupby([col,'lea','schlcode'])[['{}_sar_valid_hr'.format(col),'{}_sar_valid_math'.format(col),'{}_sar_valid_eng'.format(col)]].max().reset_index(), 
                    how='left', on=[col,'lea','schlcode']) # YST: changed here [] to [[]] on June 14, 2024 to match pandas updated version syntex

    ## Prioritize CMB matches
    # Define teacher indicators for homeroom teachers
    data['teachid_hr'] = data.cmb_teachid_hr.where((data.grade <= 5) & (data.cmb_teachid_hr_sar_valid_hr == 1), np.NaN)  
    data.loc[data.teachid_hr.isnull() & (data.grade <= 5) & (data.teachid_eogr_sar_valid_hr == 1), 'teachid_hr'] = data.teachid_eogr
    data.loc[data.teachid_hr.isnull() & (data.grade <= 5) & (data.teachid_eogm_sar_valid_hr == 1), 'teachid_hr'] = data.teachid_eogm
    print(data.groupby(['year','grade']).teachid_hr.apply(lambda x: x.isnull().mean()).unstack())
    print(data.loc[(data.grade <= 5) & data.year.between(1997,2013)].teachid_hr.isnull().mean())

    # Define teacher indicators for math
    data['teachid_math'] = data.cmb_teachid_math.where((data.cmb_teachid_math_sar_valid_math == 1), np.NaN)  
    data.loc[data.teachid_math.isnull() & (data.teachid_eogm_sar_valid_math == 1), 'teachid_math'] = data.teachid_eogm
    data.loc[data.teachid_math.isnull() & (data.teachid_eogr_sar_valid_math == 1), 'teachid_math'] = data.teachid_eogr
    print(data.groupby(['year','grade']).teachid_math.apply(lambda x: x.isnull().mean()).unstack())
    print(data.teachid_math.isnull().mean())

    # Define teacher indicators for eng
    data['teachid_eng'] = data.cmb_teachid_eng.where((data.cmb_teachid_eng_sar_valid_eng == 1), np.NaN)  
    data.loc[data.teachid_eng.isnull() & (data.teachid_eogr_sar_valid_eng == 1), 'teachid_eng'] = data.teachid_eogr
    data.loc[data.teachid_eng.isnull() & (data.teachid_eogm_sar_valid_eng == 1), 'teachid_eng'] = data.teachid_eogm
    print(data.groupby(['year','grade']).teachid_eng.apply(lambda x: x.isnull().mean()).unstack())
    print(data.teachid_eng.isnull().mean())

    # Non-cognitive stuff from EOG
    for col in ['homework','freeread','watchtv']:
        data['_tmp'] = data[col].notnull()
        data['_validSGY'] = data.groupby(['year', 'schlcode', 'lea'])['_tmp'].transform('mean') > 0.75
        data.loc[~data._validSGY, col] = np.NaN

    # Ensure valid teachers
    for col in ['teachid_hr','teachid_eng','teachid_math']:
        data.loc[~data[col].apply(lambda x: len(str(int(x))) if pd.notnull(x) else 0).between(6,7), col] = np.NaN

    # Add classroom indicators
        # TBC

    # Drop extraneous columns
    todrop = [c for c in data.columns if 'sar_valid' in c]
    data = data.drop(todrop, axis=1)

    # Add schools
    schl = pd.read_pickle("data/schls.pkl.gz", compression='gzip')
    data = data.merge(schl, how='left', on=['lea','schlcode'])

    # Add other data
    demos = pd.read_pickle("data/demos.pkl.gz", compression='gzip')
    susp = pd.read_pickle("data/susp.pkl.gz", compression='gzip')
    absences = pd.read_pickle("data/absences.pkl.gz", compression='gzip')
    grd = pd.read_pickle("data/schoolExit.pkl.gz", compression='gzip')
    gpa = pd.read_pickle("data/gpa.pkl.gz", compression='gzip')
    data = data.merge(demos, how='left', on='mastid')
    data = data.merge(susp, how='left', on=['mastid','year'])
    data = data.merge(absences, how='left', on=['mastid','year'])
    data = data.merge(grd, how='left', on='mastid')
    data = data.merge(gpa.drop('year', axis=1), how='left', on='mastid')

    # Leads and lags of suspensions and absences
    for df in [susp[['mastid','year','any_discp','iss','oss','detention']], absences[['mastid','year','daysabs']]]:
        for label, transform in [('lead4_',-4),('lead3_',-3),('lead2_',-2),('lead1_',-1),('lag1_',1)]:
            tmp = df.copy()
            tmp['year'] = tmp.year + transform
            tmp = tmp.set_index(['year','mastid'])
            tmp.columns = [label + c for c in tmp.columns]
            data = data.merge(tmp.reset_index(), how='left', on=['mastid','year'])

    # Set suspension data to missing in school-years where there are zero records
        # 2007 through 2016, 
    for var in ["","lead1_","lead2_","lead3_","lead4_","lag1_"]:
        # data['_tmp'] = data[var + 'any_discp'].notnull()
        # data['_validSGY'] = data.groupby(['year', 'schlcode', 'lea'])['_tmp'].transform('sum') > 0
        if var == "":
            adj = 0
        elif "lead" in var:
            adj = int(var[4])
        elif "lag" in var:
            adj = -int(var[3])
        data['_validSGY'] = (data.year + adj).between(2007,2016)
        for k in [var + l for l in susp.columns if l not in ['mastid', 'year']]:
            if k in data.columns:
                data[k] = data[k].fillna(0)
                data.loc[~data._validSGY, k] = np.NaN
    # data.drop(['_validSGY', '_tmp'], axis=1, inplace = True)     
    data.drop(['_validSGY'], axis=1, inplace = True)     

    # Fix missing years in gpa data
    data.loc[~((12-data.grade)+data.year).between(2005,2016) | ((12-data.grade)+data.year == 2011),
                ['gpa_unweighted','gpa_weighted','class_rank_uw','class_rank_w']] = np.NaN
    print(data.groupby(['year','grade']).gpa_unweighted.apply(lambda x: x.notnull().mean()).unstack())

    # Fix missing years in absences data | We have this from 2004 through 2016
    for var in ["","lead1_","lead2_","lead3_","lead4_","lag1_"]:
        # data['_tmp'] = data[var + 'daysabs'].notnull()
        # data['_validSGY'] = data.groupby(['year', 'schlcode', 'lea'])['_tmp'].transform('sum') > 0
        if var == "":
            adj = 0
        elif "lead" in var:
            adj = int(var[4])
        elif "lag" in var:
            adj = -int(var[3])
        data['_validSGY'] = (data.year + adj).between(2004,2016)
        data[var + 'daysabs'] = data[var + 'daysabs'].fillna(0)
        data.loc[~data._validSGY, var + 'daysabs'] = np.NaN
    # data.drop(['_validSGY', '_tmp'], axis=1, inplace = True)     
    data.drop(['_validSGY'], axis=1, inplace = True)     

    # Fix missing years in grad data | Observed from 2007-2016, 
    data.loc[~(12-data.grade + data.year).between(2007,2016), 'grad'] = np.NaN

    # Code sex / ethnicity
    data['female'] = (data['sex'] == 'F').astype(int)
    data['female'].where(data['sex'].notnull(), np.nan, inplace=True)
    data['black'] = (data['ethnic'] == 'B').astype(int)
    data['white'] = (data['ethnic'] == 'W').astype(int)
    data['hispanic'] = (data['ethnic'] == 'H').astype(int)
    for ll in ['black', 'white', 'hispanic']:
        data[ll].where(data['ethnic'].notnull(), np.nan, inplace=True)

    # Add lag reading / math scores and study skills
    eog = pd.read_pickle("data/eog.pkl.gz", compression='gzip').drop(['sex','ethnic','bdate'], axis=1).drop_duplicates()
    eog['year_orig'] = eog.year
    eog['year'] = eog.year + 1
    eog = eog.loc[~eog.duplicated(['mastid','year'], keep=False)]       # about 800 obs. Can only happen if a mastid appears in multiple grades in the same year.
    data = data.merge(eog[['mastid','year','mathscal','readscal','watchtv','homework','freeread']].rename(
                    columns={'mathscal':'lag1_mathscal','readscal':'lag1_readscal',
                            'watchtv':'lag1_watchtv','homework':'lag1_homework','freeread':'lag1_freeread'}),
                    how='left', on=['mastid','year'])

    eog['year'] = eog.year_orig + 2
    data = data.merge(eog[['mastid','year','mathscal','readscal','watchtv','homework','freeread']].rename(
                    columns={'mathscal':'lag2_mathscal','readscal':'lag2_readscal',
                        'watchtv':'lag2_watchtv','homework':'lag2_homework','freeread':'lag2_freeread'}),
                    how='left', on=['mastid','year'])
    eog['year'] = eog.year_orig

    # Add leads of math / reading scoores
    eog['year'] = eog.year_orig - 1
    data = data.merge(eog[['mastid','year','mathscal','readscal','watchtv','homework','freeread']].rename(
                    columns={'mathscal':'lead1_mathscal','readscal':'lead1_readscal',
                            'watchtv':'lead1_watchtv','homework':'lead1_homework','freeread':'lead1_freeread'}),
                    how='left', on=['mastid','year'])
    eog['year'] = eog.year_orig - 2
    data = data.merge(eog[['mastid','year','mathscal','readscal','watchtv','homework','freeread']].rename(
                    columns={'mathscal':'lead2_mathscal','readscal':'lead2_readscal',
                            'watchtv':'lead2_watchtv','homework':'lead2_homework','freeread':'lead2_freeread'}),
                    how='left', on=['mastid','year'])
    eog['year'] = eog.year_orig - 3
    data = data.merge(eog[['mastid','year','mathscal','readscal','watchtv','homework','freeread']].rename(
                    columns={'mathscal':'lead3_mathscal','readscal':'lead3_readscal',
                            'watchtv':'lead3_watchtv','homework':'lead3_homework','freeread':'lead3_freeread'}),
                    how='left', on=['mastid','year'])
    eog['year'] = eog.year_orig - 4
    data = data.merge(eog[['mastid','year','mathscal','readscal','watchtv','homework','freeread']].rename(
                    columns={'mathscal':'lead4_mathscal','readscal':'lead4_readscal',
                            'watchtv':'lead4_watchtv','homework':'lead4_homework','freeread':'lead4_freeread'}),
                    how='left', on=['mastid','year'])

    # Remove leads where they shouldn't exist
    for lead in range(1,4):
        for col in ['mathscal','readscal','watchtv','homework','freeread']:
            data.loc[data.grade >= 8-(lead-1), 'lead{}_{}'.format(lead,col)] = np.NaN

    # Lag grade indicators
    tmp = data.groupby(['mastid','year']).grade.max().reset_index()
    tmp.rename(columns={'grade':'lag_grade'}, inplace=True)
    tmp['year'] = tmp.year + 1
    data = data.merge(tmp, how='left', on=['mastid','year'])

    # Grade repeitition indiciator
    data['grade_rep'] = (data.grade == data.lag_grade).astype(int)

    # Propspective grade repetition
    tmp = data.groupby(['mastid','year']).grade.max().reset_index()
    tmp.rename(columns={'grade':'lead_grade'}, inplace=True)
    tmp['year'] = tmp.year - 1
    data = data.merge(tmp, how='left', on=['mastid','year'])
    data['lead_grade_rep'] = (data.grade == data.lead_grade).astype(int)

    # Leads of grade repetition
    tmp = data.groupby(['mastid','year']).lead_grade_rep.max().reset_index()
    tmp['year'] = tmp.year - 1
    data = data.merge(tmp.rename(columns={'lead_grade_rep':'lead1_grade_rep'}), how='left', on=['mastid','year'])
    tmp['year'] = tmp.year - 1
    data = data.merge(tmp.rename(columns={'lead_grade_rep':'lead2_grade_rep'}), how='left', on=['mastid','year'])
    tmp['year'] = tmp.year - 1
    data = data.merge(tmp.rename(columns={'lead_grade_rep':'lead3_grade_rep'}), how='left', on=['mastid','year'])
    tmp['year'] = tmp.year - 1
    data = data.merge(tmp.rename(columns={'lead_grade_rep':'lead4_grade_rep'}), how='left', on=['mastid','year'])

    # Age at end of 2015, 2019
    data['age2005'] = (pd.to_datetime("2005-12-31") - pd.to_datetime(data.bdate, format="%Y-%m-%d")).dt.days / 365.25 
    data['age2015'] = (pd.to_datetime("2015-12-31") - pd.to_datetime(data.bdate, format="%Y-%m-%d")).dt.days / 365.25 
    data['age2019'] = (pd.to_datetime("2019-12-31") - pd.to_datetime(data.bdate, format="%Y-%m-%d")).dt.days / 365.25 

    # New AOC data
    aoc = pd.read_pickle("data/aoc.pkl.gz", compression='gzip')
    aoc = aoc.merge(data.groupby('mastid').bdate.min().reset_index(), how='inner', on='mastid')

    # Offenses age 16-21
    aoc['age'] = (aoc.off_date - pd.to_datetime(aoc.bdate, format="%Y-%m-%d")).dt.days / 365.25
    aoc = aoc.loc[aoc.age.between(16,21)]

    # Add to frame
    cols = ['incar','any','crim','index','viol','dwi','drug','traff','infrac']
    cols += [c + '_conv' for c in cols if c not in ['incar']] + ['cost_wtp','cost_bottomup']
    data = data.merge(aoc.groupby('mastid')[cols].max().rename(columns=lambda x: 'aoc_' + x).reset_index(), how='left', on='mastid')
    for col in ['aoc_' + c for c in cols]:
        data[col] = data[col].fillna(0)

    data = data.merge(aoc.groupby('mastid')[cols].sum().rename(columns=lambda x: 'aoc_total_' + x).reset_index(), how='left', on='mastid')
    for col in ['aoc_total_' + c for c in cols]:
        data[col] = data[col].fillna(0)

    # Set to missing where required
    data['cansee_aoc1621'] = ((data.age2019 >= 22) & (data.age2005 <= 16)).astype(int)
    data['cansee_aoc1621'] = (data.groupby(['grade','year']).cansee_aoc1621.transform('mean') >= 0.9).astype(int)
    data.loc[data.cansee_aoc1621 == 0, ['aoc_' + c for c in cols] + ['aoc_total_' + c for c in cols]] = np.NaN # YST: added "+ ['aoc_total_' + c for c in cols]"
    print(data.groupby(['year','grade']).aoc_any.apply(lambda x: x.notnull().mean()).unstack())
    print(data.groupby(['year','grade']).aoc_any.mean().unstack())

    # Other FEs and ID vars
    data['school_fe'] = data.groupby(['lea','schlcode']).grouper.group_info[0]
    data['school_year_fe'] = data.groupby(['lea','schlcode','year',]).grouper.group_info[0]
    data['school_year_grade_fe'] = data.groupby(['lea','schlcode','year','grade']).grouper.group_info[0]

    # Drop missing current math and reading scores
    data = data.loc[data.mathscal.notnull() & data.readscal.notnull()]

    # Lag score missing and square
    data['miss_lag1_mathscal'] = data.lag1_mathscal.isnull().astype(int)
    data['miss_lag2_mathscal'] = data.lag2_mathscal.isnull().astype(int)
    data['miss_lag1_readscal'] = data.lag1_readscal.isnull().astype(int)
    data['miss_lag2_readscal'] = data.lag2_readscal.isnull().astype(int)
    for k in range(2,4):
        for lag in range(1,3):
            for subj in ['mathscal','readscal']:
                data['lag{}_{}_{}'.format(lag,subj,k)] = data['lag{}_{}'.format(lag,subj)]**k

    # SGY ans S means of controls
    for control in ['black','white','female','disadv','lim_eng','pared_nohs','pared_hsorless','pared_somecol','pared_baormore',
                'exc_not','exc_aig','exc_behav','exc_educ','grade_rep',
                'lag1_mathscal','lag1_readscal','lag1_mathscal_2','lag1_readscal_2','lag1_mathscal_3','lag1_readscal_3',
                'miss_lag1_mathscal','miss_lag1_readscal',
                'lag2_mathscal','lag2_readscal','lag2_mathscal_2','lag2_readscal_2','lag2_mathscal_3','lag2_readscal_3',
                'miss_lag2_mathscal','miss_lag2_readscal']:
        print("Working on means for {}".format(control))
        data['sgy_mean_{}'.format(control)] = data.groupby('school_year_grade_fe')[control].transform('mean')
        data['s_mean_{}'.format(control)] = data.groupby('school_fe')[control].transform('mean')
            # NB these are missing if non-missing values for each control
            # will need to fill in means where all obs are missing control in stata using whatever method is preferred

    # Add year lags of sgy_means
    tolag = [c for c in data.columns if 'sgy_mean' in c]
    tmp = data[['mastid','year'] + tolag].rename(columns=lambda x: 'lag_' + x if 'sgy_mean' in x else x)
    tmp['year'] = tmp.year + 1
    data = data.merge(tmp, how='left', on=['mastid','year'])

    # Save
    data.to_stata('data/tmp.dta', write_index=False) 

### 12) Build final analysis dataset
if 1:
    # Load the data
    data = pd.read_stata('data/tmp.dta')

    ## Drop observations with no teacher of any type
    data = data.loc[data.teachid_hr.notnull() | data.teachid_math.notnull() | data.teachid_eng.notnull()]

    ## Convert to long format and relabel
    data['n_teach'] = data.teachid_hr.notnull().astype(int) + data.teachid_math.notnull().astype(int) + data.teachid_eng.notnull().astype(int)
    data = data.loc[data.index.repeat(data.n_teach)].reset_index(drop=True)
    data['teach_order'] = 1
    data['teach_order'] = data.groupby(['mastid','year']).teach_order.cumsum()

    data.loc[(data.teach_order == 1) & data.teachid_hr.notnull(), 'subject'] = 'hr'
    data.loc[(data.teach_order == 1) & data.teachid_hr.notnull(), 'teachid'] = data.teachid_hr

    data.loc[(data.teach_order == 1) & data.subject.isnull() & data.teachid_math.notnull(), 'subject'] = 'math'
    data.loc[(data.teach_order == 1) & data.teachid.isnull() & data.teachid_math.notnull(), 'teachid'] = data.teachid_math

    data.loc[(data.teach_order == 1) & data.subject.isnull() & data.teachid_eng.notnull(), 'subject'] = 'eng'
    data.loc[(data.teach_order == 1) & data.teachid.isnull() & data.teachid_eng.notnull(), 'teachid'] = data.teachid_eng

    data.loc[(data.teach_order == 2) & data.subject.isnull() & data.teachid_math.notnull() & data.teachid_hr.notnull(), 'subject'] = 'math'
    data.loc[(data.teach_order == 2) & data.teachid.isnull() & data.teachid_math.notnull() & data.teachid_hr.notnull(), 'teachid'] = data.teachid_math

    data.loc[(data.teach_order == 2) & data.subject.isnull() & data.teachid_eng.notnull(), 'subject'] = 'eng'
    data.loc[(data.teach_order == 2) & data.teachid.isnull() & data.teachid_eng.notnull(), 'teachid'] = data.teachid_eng

    data.loc[(data.teach_order == 3) & data.subject.isnull() & data.teachid_eng.notnull(), 'subject'] = 'eng'
    data.loc[(data.teach_order == 3) & data.teachid.isnull() & data.teachid_eng.notnull(), 'teachid'] = data.teachid_eng

    # Student-year-subject level data
    assert data.teachid.isnull().sum() == 0
    assert data.duplicated(['mastid','year','subject']).max() == 0
    print(data.loc[data.mastid == np.random.choice(data.mastid.unique()), ['mastid','year','teachid','subject','teachid_hr','teachid_eng','teachid_math']])
    data['teachid_year_subject_fe'] = data.groupby(['teachid','year','subject']).grouper.group_info[0]
    orig_shape = data.loc[(data.grade >= 4) & data.year.between(1997,2013)].shape[0]
    print("N for valid teachers {}".format(orig_shape))

    ## Sample selections
    # Teachers observed in multiple schools in single year
    data = data.loc[data.groupby(['teachid','year']).school_fe.transform('nunique') == 1]
    new_shape =data.loc[(data.grade >= 4) & data.year.between(1997,2013)].shape[0]
    print("N after single schools {}".format(new_shape))
    orig_shape = data.loc[(data.grade >= 4) & data.year.between(1997,2013)].shape[0]

    # Teachers observed in single grade
    data = data.loc[data.groupby(['teachid','year']).grade.transform('nunique') == 1]
    new_shape = data.loc[(data.grade >= 4) & data.year.between(1997,2013)].shape[0]
    print("N after single grade {}".format(new_shape))
    orig_shape = data.loc[(data.grade >= 4) & data.year.between(1997,2013)].shape[0]
    
    # Drop special schools
    for col in ['alt_schl','spec_ed_schl']:
        data = data.loc[data[col] == 0]
    new_shape = data.loc[(data.grade >= 4) & data.year.between(1997,2013)].shape[0]
    print("N after no special schools {}".format(new_shape))
    orig_shape = data.loc[(data.grade >= 4) & data.year.between(1997,2013)].shape[0]

    # Keep obs with non-missing lag scores or in grade 3
    data = data.loc[(data.lag1_mathscal.notnull() & data.lag1_readscal.notnull())
        | (data.grade == 3)]
    new_shape = data.loc[(data.grade >= 4) & data.year.between(1997,2013)].shape[0]
    print("N after has lag {}".format(new_shape))
    orig_shape = data.loc[(data.grade >= 4) & data.year.between(1997,2013)].shape[0]

    # Restrict to 15 to 100 students per teacher-year-grade-subejct
    data = data.loc[data.groupby(['teachid','year','grade','subject']).mastid.transform(
                'nunique').between(15,100)]
    new_shape = data.loc[(data.grade >= 4) & data.year.between(1997,2013)].shape[0]
    print("N after 15-100 students {}".format(new_shape))
    orig_shape = data.loc[(data.grade >= 4) & data.year.between(1997,2013)].shape[0]

    # Add pvalues for sorting
    if 1: 
        from scipy.stats import chi2_contingency

        # Add pvalues for grades 4-8 and 2007 on
        def chi2(table):
            for col in table.columns:
                if table[col].sum() == 0:
                    table = table.drop(col, axis=1)
            table = table.values
            r, p, dof, e = chi2_contingency(table)
            return p

        def classroom_chi2(c):
            current = data.loc[data.school_year_grade_fe == c]
            lea = data.loc[data.school_year_grade_fe == c].lea.values[0]
            schl = data.loc[data.school_year_grade_fe == c].schlcode.values[0]
            yr = data.loc[data.school_year_grade_fe == c].year.values[0]
            grd = data.loc[data.school_year_grade_fe == c].grade.values[0]
            lag = data.loc[(data.year == current.year.max() - 1) &
                                data.mastid.isin(current.mastid), ['mastid','teachid_year_subject_fe']].rename(
                        columns={'teachid_year_subject_fe':'lag_teachid_year_subject_fe'})
            current = current[['mastid','teachid_year_subject_fe']].merge(lag, how='inner', on='mastid')
            table = pd.crosstab(current.teachid_year_subject_fe, current.lag_teachid_year_subject_fe)
            try:
                p = chi2(table)
            except:
                p = None
            print(c,table,p,"\n", flush=True)
            return lea, schl, yr, grd, p

        gc.collect()
        import multiprocessing
        if __name__ == "__main__":
            p = multiprocessing.Pool(7)
            ans = p.map(classroom_chi2, data.school_year_grade_fe.unique())

        pvals = pd.DataFrame(ans, columns=['lea','schlcode','year','grade','pval'])
        pvals.to_pickle('data/pvals_post.pkl')

        # Make histogram
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set(style="white", color_codes=True)
        fig, axes = plt.subplots()
        sns.histplot(data=pvals, x="pval", stat='probability', ax=axes)
        axes.set_xlabel('P-value') 
        axes.set_ylabel('Density of school-grade-year observations') 
        fig.tight_layout()
        fig.savefig('figures/pvalue_histogram.pdf')
        fig, axes = plt.subplots()
        sns.histplot(data=pvals.loc[pvals.pval < 1], x="pval", stat='probability', ax=axes)
        axes.set_xlabel('P-value') 
        axes.set_ylabel('Density of school-grade-year observations') 
        fig.tight_layout()
        fig.savefig('figures/pvalue_histogram_l1.pdf')
    else:
        pvals = pd.read_pickle('data/pvals_post.pkl')

    data = data.merge(pvals, how='left', on=['lea','schlcode','year','grade'])

    # "Classroom" means of covariates
    for control in ['black','white','female','disadv','lim_eng','pared_nohs','pared_hsorless','pared_somecol','pared_baormore',
                'exc_not','exc_aig','exc_behav','exc_educ','grade_rep',
                'lag1_mathscal','lag1_readscal','lag1_mathscal_2','lag1_readscal_2','lag1_mathscal_3','lag1_readscal_3',
                'miss_lag1_mathscal','miss_lag1_readscal',
                'lag2_mathscal','lag2_readscal','lag2_mathscal_2','lag2_readscal_2','lag2_mathscal_3','lag2_readscal_3',
                'miss_lag2_mathscal','miss_lag2_readscal']:

        print("Working on control {}".format(control))
        data['sgyts_mean_{}'.format(control)] = data.groupby(['school_year_grade_fe','teachid','subject'])[control].transform('mean')

    # Teacher exit / entrance indicators
    data['teach_first_year'] = data.groupby(['teachid','school_fe','grade','subject']).year.transform('min')
    data['teacher_enter'] = (data.year == data.teach_first_year).astype(int)
    data['teach_last_year'] = data.groupby(['teachid','school_fe','grade','subject']).year.transform('max')
    data['teacher_exit'] = (data.year == data.teach_last_year).astype(int)
    data['teach_nyears'] = data.groupby(['teachid','school_fe','grade','subject']).year.transform('nunique')
    data.loc[data.year == data.year.min(), 'teacher_enter'] = 0
    data.loc[data.year == data.year.max(), 'teacher_exit'] = 0

    # Number of prior years in school-grade-subject 
    for name, grp in [('_schgrdsubj',['school_fe','grade','subject']),
                    ('_schgrd',['school_fe','grade']),
                    ('_school',['school_fe']),
                    ('',[]),]:
        # For teacher
        tmp = data[['teachid'] + grp + ['year']].drop_duplicates().sort_values(['teachid'] + grp + ['year'])
        tmp['nyears_teach{}'.format(name)] = 1
        tmp['nyears_teach{}'.format(name)] = tmp.groupby(['teachid'] + grp)['nyears_teach{}'.format(name)].cumsum() - 1
        data = data.merge(tmp, how='left', on=['teachid'] + grp +['year'])

        # For unit
        if len(grp) > 0:
            tmp = data[grp + ['year']].drop_duplicates().sort_values(grp + ['year'])
            tmp['nyears{}'.format(name)] = 1
            tmp['nyears{}'.format(name)] = tmp.groupby(grp)['nyears{}'.format(name)].cumsum() - 1
            data = data.merge(tmp, how='left', on=grp +['year'])

    # Save
    data.to_stata('data/analysis_data.dta', write_index=False, convert_dates = {'bdate':'td'}) 


